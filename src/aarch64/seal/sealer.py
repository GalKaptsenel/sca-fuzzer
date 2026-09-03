"""Sealing as composable per-location sealings (AArch64).

  * `Sealing` — one location; `seal(value)` returns its slot instructions (None -> arch-safe). Pure.
  * `Sealer.seal(tc) -> SealedTestCase` — walks the TC, creates sealings, inserts placeholders.
  * `SealedTestCase.resolve(input) -> ResolvedSealingTestCase` — runs the CE trace(s), computes values.
  * `ResolvedSealingTestCase` — `object_code` plus `genuine()` / `decoy()` relocations for it.
"""
import abc
import copy
import random
from typing import Dict, List, Optional, Set, Tuple

from ...config import CONF
from ...interfaces import (Instruction, TestCase, BasicBlock, GeneratorException,
                          RegisterOperand, ImmediateOperand)
from .primitives import (make_nop, index_instructions, inst_at, _SANDBOX_MASK,
                           _SandboxInstrumentationBase)
from .pac import (PacSign, PacSigner, build_pac_specs, _AUTH_TO_PAC, _AUTH_TO_XPAC, _read_reg)
from ..aarch64_target_desc import SANDBOX_BASE_REGISTER
from ..aarch64_printer import Aarch64ASMLayout
from ..aarch64_relocations import (NOP_WORD, xpac_word, addg_word, movk_word, aut_word, eor_imm_word,
                                    Relocation, apply_relocations)


M = (1 << 64) - 1


def _reg_num(reg: str) -> int:
    return int(reg[1:])   # seal value registers are x0..x30


def _encode(inst: Instruction) -> int:
    """The 32-bit machine-code word of one seal instruction."""
    mn = inst.name.lower()
    if mn == "nop":
        return NOP_WORD
    if mn.startswith("xpac"):
        return xpac_word(mn == "xpacd", _reg_num(inst.operands[0].value))
    if mn.startswith("aut"):
        rn = _reg_num(inst.operands[1].value) if len(inst.operands) > 1 else 31
        return aut_word(mn, _reg_num(inst.operands[0].value), rn)
    if mn == "movk":
        return movk_word(_reg_num(inst.operands[0].value),
                         int(inst.operands[1].value), int(inst.operands[2].value))
    if mn == "addg":
        return addg_word(_reg_num(inst.operands[0].value), int(inst.operands[3].value))
    if mn == "eor":
        return eor_imm_word(_reg_num(inst.operands[0].value), _reg_num(inst.operands[1].value),
                            int(inst.operands[2].value))
    raise GeneratorException(f"cannot encode seal instruction {inst.name!r}")


# ==================================================================================================
# Sealing — one location; pure seal(value); exposes the facts the resolver reads
# ==================================================================================================
class Sealing(abc.ABC):
    """One sealing site. `seal(value)` returns this slot's instructions for a runtime value (None ->
    the arch-safe placeholder). `slot_insts`/`slot_locs` are the placeholder instructions and positions."""
    value_reg: str

    def __init__(self) -> None:
        self.slot_insts: List[Instruction] = []
        self.slot_locs: List = []

    @abc.abstractmethod
    def seal(self, value: Optional[int], rng: Optional[random.Random]) -> List[Instruction]:
        """The instructions sealing this location with `value` (None -> the arch-safe placeholder).
        `rng` seeds any render choice; None only for the placeholder, which has none."""


class SandboxSealing(Sealing):
    """Clamp a base register into the input region: [AND reg, #mask; ADD reg, base]. Needs no runtime
    value — the clamp is identical every time, so `seal` ignores its argument."""

    def __init__(self, value_reg: str, mask: str, base_reg: str) -> None:
        super().__init__()
        self.value_reg = value_reg
        self._mask = mask
        self._base_reg = base_reg
        self.slot_insts = self.seal(None, None)   # seal itself with its placeholder = the slot to fill

    def seal(self, value: Optional[int], rng: Optional[random.Random]) -> List[Instruction]:
        return [Instruction("and", True, "", False,
                            template=f"AND {self.value_reg}, {self.value_reg}, {self._mask}"),
                Instruction("add", True, "", False,
                            template=f"ADD {self.value_reg}, {self.value_reg}, {self._base_reg}")]


class PacSealing(Sealing):
    """Authenticate a pointer register. `seal(sig)` emits the signature MOVK + the auth (or, prob
    `CONF.pac_strip_prob`, the arch-safe strip); `seal(None)` is the placeholder. `committed_inst` is
    exposed for the resolver."""

    def __init__(self, value_reg: str, committed_inst: Instruction, encoder: PacSign,
                 revert: bool = False) -> None:
        super().__init__()
        self.value_reg = value_reg
        self.committed_inst = committed_inst
        self._enc = encoder
        self._revert = revert
        self.n_before = encoder.n_sig_movks + 1   # signature MOVKs + the AUT*/strip (before the access)
        self.slot_insts = self.seal(None, None)   # placeholder: before [+ after XPAC]

    def seal(self, value: Optional[int], rng: Optional[random.Random]) -> List[Instruction]:
        auth_mn = self.committed_inst.name.lower()
        ctx_reg = self.committed_inst.operands[1].value if len(self.committed_inst.operands) > 1 else None
        xpac = self._enc.make_xpac_inst(_AUTH_TO_XPAC[auth_mn], self.value_reg)
        if value is None:
            before = [make_nop() for _ in range(self._enc.n_sig_movks)] + [xpac]
        else:
            movks = self._enc.make_sig_movks(self.value_reg, value)
            before = movks + ([xpac] if rng.random() < CONF.pac_strip_prob
                              else [self._enc.make_auth_inst(auth_mn, self.value_reg, ctx_reg)])
        if not self._revert:
            return before
        # After the access, strip the register back to a canonical pointer: XPAC clears the PAC field,
        # so genuine (already stripped by the AUT*) is unchanged and decoy (failed AUT* leaving FPAC
        # bits) is re-canonicalized -> both agree downstream. A separate XPAC instance (not the
        # before one) so slot positions stay one-for-one.
        return before + [self._enc.make_xpac_inst(_AUTH_TO_XPAC[auth_mn], self.value_reg)]


class MteSealing(Sealing):
    """Retag a pointer register onto its cell's allocation tag: `seal(delta)` emits one ADDG that adds
    the 4-bit delta to the pointer's tag right before the access (or [NOP] when 0/None), so the access
    is tag-checked against a matching granule. There is NO retag-back: the matched (checked) tag then
    propagates through the rest of the test-case logic — exactly as the CE's after-access tag correction
    models a genuine ADDG — so the placeholder and genuine traces agree on every later tag store's
    committed granule tag, and no pointer is ever left at the sandbox base's Unchecked tag. A decoy's
    wrong tag lives only on the speculative path it perturbs and is rolled back with that speculation.
    `access_inst` is exposed for the resolver (its accessed cell's tag is the genuine tag)."""

    def __init__(self, value_reg: str, access_inst: Instruction) -> None:
        super().__init__()
        self.value_reg = value_reg
        self.access_inst = access_inst
        self.n_before = 1
        self.slot_insts = self.seal(None, None)   # placeholder: [before]

    def _addg(self, delta: int) -> Instruction:
        return (Instruction("addg", True, "", False,
                            template=f"ADDG {self.value_reg}, {self.value_reg}, #0, #{delta}")
                .add_op(RegisterOperand(self.value_reg, 64, False, True))
                .add_op(RegisterOperand(self.value_reg, 64, True, False))
                .add_op(ImmediateOperand("0", 6))
                .add_op(ImmediateOperand(str(delta), 4)))

    def seal(self, value: Optional[int], rng: Optional[random.Random]) -> List[Instruction]:
        delta = 0 if value is None else value % 16
        return [self._addg(delta) if delta else make_nop()]


def _mask_runs(mask: int) -> List[int]:
    """Split `mask` into its maximal contiguous runs of set bits, low to high. A single run of ones
    is an encodable AArch64 logical (bitmask) immediate, so one EOR renders each run — a mask with a
    gap (e.g. bit 55 punched out) needs one EOR per run."""
    runs, i = [], 0
    while i < 64:
        if not (mask >> i) & 1:
            i += 1
            continue
        j = i
        while j < 64 and (mask >> j) & 1:
            j += 1
        runs.append(((1 << (j - i)) - 1) << i)   # bits [i:j-1]
        i = j
    return runs


# How many random non-canonical masks to offer per canonicality slot, and the sampling budget.
_CANON_POOL_SIZE = 6
_CANON_POOL_TRIES = 64


def _canon_va() -> int:
    va = CONF.va_size
    if va is None:
        raise GeneratorException("canonicality needs va_size set")
    if not 0 < va < 55:
        raise GeneratorException(f"va_size {va} out of range (expected 1..54)")
    return va


def _canon_mask_pool(salt: int) -> List[int]:
    """A deterministic pool of non-canonical flip masks for the decoy to choose from — the analogue of
    PAC's wrong-signature pool / MTE's wrong-tag set. Each mask is a RANDOM contiguous run of bits
    within the guaranteed-fault range [54:VA_SIZE]:

      * flipping any bit in [54:VA] makes it disagree with the (never-flipped) regime-selector bit 55,
        so the address is non-canonical in EITHER regime (TTBR0/TTBR1) and under TBI on or off — a
        guaranteed translation fault. Bit 55 and the VA bits (<VA_SIZE) are never touched, so the
        in-sandbox address and its cache-set index [11:6] are untouched (only canonicality changes);
      * a single contiguous run is one AArch64 logical (bitmask) immediate, i.e. one EOR;
      * randomising WHICH sub-run flips spreads the probe across the faulting field instead of always
        flipping the same bits, and is seeded by the salt so a sealing class forges identically.

    If canonicality_mask is set, the pool is that one fixed mask (must be a single [54:VA] run)."""
    va = _canon_va()
    fault = (((1 << (55 - va)) - 1) << va)   # bits [54:va]
    if CONF.canonicality_mask is not None:
        m = CONF.canonicality_mask & M
        if 0 == m or (m & ~fault) or _mask_runs(m) != [m]:
            raise GeneratorException(
                f"canonicality_mask 0x{m:016x} must be a single contiguous run within [54:{va}] "
                f"(the guaranteed-fault range; excludes bit 55, the top byte, and the VA bits)")
        return [m]
    rng = random.Random(hash(("canon-pool", salt)))
    pool: List[int] = []
    for _ in range(_CANON_POOL_TRIES):
        lo = rng.randrange(va, 55)          # run start in [VA, 54]
        hi = rng.randrange(lo, 55)          # run end   in [lo, 54]
        m = ((1 << (hi - lo + 1)) - 1) << lo
        if m and m not in pool:
            pool.append(m)
            if len(pool) >= _CANON_POOL_SIZE:
                break
    if not pool:
        raise GeneratorException(f"no canonicality flip mask for VA_SIZE {va}")
    return pool


class CanonSealing(Sealing):
    """Make a pointer register non-canonical for ONE speculative access, then restore it.
    `seal(mask)` flips `mask` into the base register with an EOR right before the access (a
    would-fault address in either translation regime) and, when the access preserves its base
    register, flips it back with the SAME EOR right after — so the decoy RE-CONVERGES with the
    baseline: only the sealed access differs, and the flipped high bits cannot cascade into a later
    access's cache set (e.g. via a downstream shift that would move them into the set-index bits).
    `seal(None/0)` is the canonical placeholder (NOPs). `pool` is the set of candidate non-canonical
    masks the resolver offers (the decoy picks one at random); `revert` is False when the access
    itself overwrites the base (a load into it, or write-back) — then the flipped bits are consumed
    and no restore is emitted. The seal lands only on speculative-only slots, so the architectural
    path stays canonical. `access_inst`/`pool` are exposed for the resolver."""

    def __init__(self, value_reg: str, access_inst: Instruction, pool: List[int], revert: bool) -> None:
        super().__init__()
        self.value_reg = value_reg
        self.access_inst = access_inst
        self.pool = pool
        self._revert = revert
        self.slot_insts = self.seal(None, None)   # placeholder: [before] or [before, after]

    def _eor(self, run: int) -> Instruction:
        return (Instruction("eor", True, "", False,
                            template=f"EOR {self.value_reg}, {self.value_reg}, #{run}")
                .add_op(RegisterOperand(self.value_reg, 64, False, True))
                .add_op(RegisterOperand(self.value_reg, 64, True, False))
                .add_op(ImmediateOperand(str(run), 64)))

    def seal(self, value: Optional[int], rng: Optional[random.Random]) -> List[Instruction]:
        # One op before the access (the flip), and — when the base survives the access — the SAME flip
        # after it (EOR is its own inverse) to restore the base, so the decoy matches the baseline
        # downstream. genuine (value None/0) fills a NOP at each position; the resolver offers exactly
        # one mask per decoy, so before and after always use the same value and cancel.
        emit = (lambda: self._eor(value)) if value else make_nop
        return [emit()] + ([emit()] if self._revert else [])


# ==================================================================================================
# ResolvedSealingTestCase — one input's resolution; mints genuine / decoy hardware test cases
# ==================================================================================================
class _Resolved:
    """A sealing paired with its resolved runtime value for one input: `value` is the correct value
    (None where the sealing needs none / was unreached); `alts` are alternative values that fail; the
    slot is speculative (decoy-eligible) when it never ran architecturally."""
    def __init__(self, sealing: Sealing, value: Optional[int], alts: List[int],
                 spec_nesting: Optional[int]) -> None:
        self.sealing = sealing
        self.value = value
        self.alts = alts
        self.speculative = spec_nesting != 0   # None (unreached) or >0 -> speculative; 0 -> arch


class ResolvedSealingTestCase:
    """One input's resolution. `object_code` is the assembled base; `genuine()`/`decoy()` are the
    relocations that turn it into a variant (apply with apply_relocations). `genuine()` seals every slot
    correctly; `decoy()` seals the architectural slots correctly, with no guarantees on the other slots."""

    def __init__(self, entries: List[_Resolved], object_code: bytes,
                 offsets: Dict[int, Tuple[int, ...]], salt: int) -> None:
        self._entries = entries
        self._object_code = object_code
        self._offsets = offsets
        self._salt = salt
        genuine_rng = random.Random(hash((self.collapse_key, salt)))   # class-invariant render choices
        self._genuine = self._solve_relocations(offsets, genuine_rng, decoy=False)

    @staticmethod
    def _decoy_subset(eligible: List[_Resolved], rng: random.Random) -> set:
        """A non-empty subset of the decoy-eligible slots to perturb (empty when none are eligible)."""
        if not eligible:
            return set()
        chosen = {r for r in eligible if rng.random() < 0.5}
        return chosen if chosen else {rng.choice(eligible)}

    @property
    def object_code(self) -> bytes:
        return self._object_code

    def genuine(self) -> Tuple[Relocation, ...]:
        return self._genuine

    def decoy(self, rng: random.Random) -> Tuple[Relocation, ...]:
        """The decoy relocations for one variant. `rng` (the caller seeds it from the sealing class,
        salt, and variant index) drives every choice, so the variant is a pure function of that seed:
        it reproduces across trace passes, and same-class inputs run the identical program."""
        return self._solve_relocations(self._offsets, rng, decoy=True)

    def _eligible(self) -> List[_Resolved]:
        """Slots a decoy may perturb: speculative-or-unreached (never architectural) and carrying
        alternative values. When empty, decoy() reproduces genuine() (a null decoy) and the input is
        not non-interference-testable."""
        return [r for r in self._entries if r.speculative and r.alts]

    def has_decoy(self) -> bool:
        return bool(self._eligible())

    def _solve_relocations(self, offsets: Dict[int, Tuple[int, ...]],
                           rng: random.Random, decoy: bool) -> Tuple[Relocation, ...]:
        perturb = self._decoy_subset(self._eligible(), rng) if decoy else set()
        relocs: List[Relocation] = []
        for r in self._entries:
            offs = offsets.get(id(r.sealing))
            if offs is None:
                continue
            value = rng.choice(r.alts) if r in perturb else r.value
            relocs += [Relocation(off, _encode(i)) for off, i in zip(offs, r.sealing.seal(value, rng))]
        return tuple(relocs)

    @property
    def collapse_key(self) -> Tuple:
        """The sealing class of the resolved input: per-entry (value, speculative). Two inputs share
        a sealed TC iff every slot resolves to the same value with the same speculative status (so the
        same genuine fill is arch-safe for both, and decoy perturbs the same slots). Sandbox entries
        are constant across inputs (value None, non-speculative) so they never split classes."""
        return tuple((r.value, r.speculative) for r in self._entries)


# ==================================================================================================
# Resolvers — concern-specific value computation from a CE trace (used by the SealedTestCase)
# ==================================================================================================
# Signature forgery pool: how many wrong signatures to offer per PAC slot, and the sampling budget.
_FORGERY_POOL_SIZE = 6
_FORGERY_TRIES = 64


def _resolve_pac(s: PacSealing, cer, layout, signer: PacSigner, salt: int
                 ) -> Tuple[Optional[int], List[int], Optional[int]]:
    """A PacSealing's value from a trace: sign the pointer that reaches the sealing's XPAC, plus a pool
    of wrong signatures that fail AUTH. When the XPAC is never reached, the slot is still decoy-eligible
    (HW may speculate deeper than the model): forge the pool over the sandbox base, whose PAC-field-window
    bits every clamped sandbox pointer shares, so the decoy MOVKs preserve the pointer's non-field bits
    and the after-XPAC re-converges — exactly as the reached case does. (value, alts, spec_nesting)."""
    correct_sig, alts, spec = None, [], None
    if not cer or s.committed_inst is None:
        return correct_sig, alts, spec
    xpac = next(i for i in s.slot_insts if i.name.lower() in ("xpaci", "xpacd"))
    xpac_off, code_base = layout.instruction_address[xpac], cer[0].cpu.pc
    pac_mn = _AUTH_TO_PAC[s.committed_inst.name.lower()]
    mask = signer.field_mask(pac_mn)
    value_reg = s.committed_inst.operands[0].value
    ctx_reg = s.committed_inst.operands[1].value if len(s.committed_inst.operands) > 1 else None
    for ite in cer:
        if ite.cpu.pc - code_base != xpac_off:
            continue
        depth = ite.metadata.speculation_nesting
        if spec is None or depth < spec:
            spec = depth
        if correct_sig is not None and depth != 0:   # architectural occurrence is authoritative
            continue
        ptr = _read_reg(ite.cpu, value_reg)
        cval = _read_reg(ite.cpu, ctx_reg) if ctx_reg is not None else 0
        correct_sig = signer.sign(ptr, cval, pac_mn)
        alts = _wrong_sigs(correct_sig, mask, salt)
    if correct_sig is None:                          # unreached: forge over the shared sandbox base
        base_ptr = _read_reg(cer[0].cpu, SANDBOX_BASE_REGISTER)
        alts = _wrong_sigs(signer.sign(base_ptr, 0, pac_mn), mask, salt)
    return correct_sig, alts, spec


def _wrong_sigs(correct_sig: int, mask: int, salt: int) -> List[int]:
    """A deterministic pool of wrong signatures: the correct signature with only its PAC field bits
    perturbed (so each is a genuine AUTH failure). Seeded by (correct_sig, mask, salt), so every member
    of a sealing class forges identically."""
    rng = random.Random(hash((correct_sig, mask, salt)))
    pool: List[int] = []
    for _ in range(_FORGERY_TRIES):
        sig = (correct_sig & ~mask) | (rng.randrange(1 << 64) & mask)
        if sig != correct_sig and sig not in pool:
            pool.append(sig)
            if len(pool) >= _FORGERY_POOL_SIZE:
                break
    if not pool:
        raise GeneratorException(f"no PAC forgery for field mask 0x{mask:016x}")
    return pool


def _resolve_mte(s: MteSealing, cer, layout) -> Tuple[Optional[int], List[int], Optional[int]]:
    """An MteSealing's tag delta from a trace: classify the accessed cell's allocation tag and the
    pointer's own tag at the guarded access; the genuine delta brings the pointer to the cell tag,
    alternatives are every other tag (a mismatch). (value, alts, spec_nesting)."""
    correct_tag, ptr_tag, spec = None, None, None
    if cer and s.access_inst is not None:
        access_off, code_base = layout.instruction_address[s.access_inst], cer[0].cpu.pc
        for ite in cer:
            if not ite.metadata.has_memory_access or ite.cpu.pc - code_base != access_off:
                continue
            nest = ite.metadata.speculation_nesting
            ma = ite.metadata.memory_access
            if spec is None or nest < spec:
                spec = int(nest)
            # The CE reports the granule's real allocation tag per access (seeded from the input tags,
            # updated by STGs, speculation rolled back) -- the same value the hardware holds. Read it
            # straight from the trace instead of re-deriving the tag memory in Python.
            if nest == 0 or correct_tag is None:
                correct_tag, ptr_tag = ma.allocation_tag, (ma.effective_address >> 56) & 0xF
    if correct_tag is None or ptr_tag is None:
        # unreached in the placeholder trace: genuine applies no retag (value None -> NOP), so any
        # nonzero delta is a valid decoy should HW speculate deeper than the model reached.
        return None, list(range(1, 16)), spec
    delta = (correct_tag - ptr_tag) % 16
    # A decoy whose retagged pointer lands on tag 0 or 15 cannot leak: the executor runs with
    # TCR_EL1.TCMA1 set, which makes an EL1 access through a tag-0b0000 or tag-0b1111 ("match-all")
    # pointer Tag-UNCHECKED — the hardware silently skips the tag comparison, so the intended mismatch
    # never happens and the decoy behaves identically to the genuine access. Offer only deltas whose
    # resulting tag (ptr_tag + d) is a genuinely-checked value in 1..14.
    alts = [d for d in range(16) if d != delta and (ptr_tag + d) % 16 not in (0, 15)]
    return delta, alts, spec


def _resolve_canon(s: "CanonSealing", cer, layout) -> Tuple[Optional[int], List[int], Optional[int]]:
    """A CanonSealing's value from a trace: the genuine value is canonical (None -> the NOP
    placeholders); the alternatives are the sealing's pool of random non-canonical flip masks (the
    decoy picks one). spec_nesting is the minimum depth the guarded access is reached at, so an access
    ever reached architecturally (min == 0) is non-speculative and never non-canonicalized.
    (value, alts, spec_nesting)."""
    spec = None
    if cer and s.access_inst is not None:
        access_off, code_base = layout.instruction_address[s.access_inst], cer[0].cpu.pc
        for ite in cer:
            if not ite.metadata.has_memory_access or ite.cpu.pc - code_base != access_off:
                continue
            nest = ite.metadata.speculation_nesting
            if spec is None or nest < spec:
                spec = int(nest)
    return None, list(s.pool), spec


def _slot_offsets(tc: TestCase, layout, sealings: List["Sealing"]) -> Dict[int, Tuple[int, ...]]:
    """{id(sealing): its slot byte offsets in the assembled template}. Positions are one-for-one across
    fills, so an offset taken from the placeholder holds for every variant."""
    return {id(s): tuple(layout.instruction_address[inst_at(tc, loc)[0]] for loc in s.slot_locs)
            for s in sealings}


# ==================================================================================================
# SealedTestCase — holds the unresolved sealings; resolve(input) orchestrates value computation
# ==================================================================================================
class SealedTestCase:
    """Owns the sealings for one test case end-to-end. During construction it PLACES each value
    sealing's slot in the order this concern requires (the sandbox walk supplies clamps + per-access
    sites but never decides value-seal order), then `resolve(input)` computes their values. The
    per-concern subclass owns both the placement order and the resolution."""

    def __init__(self, sealed_tc: TestCase, trace_fn, assemble, sandbox_sealings: List[SandboxSealing],
                 data_sites: List, trace_bytes_fn=None) -> None:
        self._tc = sealed_tc
        self._trace_fn = trace_fn
        self._trace_bytes_fn = trace_bytes_fn              # (machine code, input) -> cer, for a re-trace
        self._assemble = assemble                          # tc -> machine code, for the skeleton
        self._sandbox = sandbox_sealings
        self._salt = random.randrange(1 << 64)             # per-test-case; seeds deterministic forgery
        self._insert_slots(data_sites)                            # subclass inserts its value sealings, in order
        _record_positions(self._tc, self._sealings())      # AFTER every slot is inserted
        self._layout = Aarch64ASMLayout(self._tc)

    @property
    def salt(self) -> int:
        return self._salt

    @property
    def object_code(self) -> bytes:
        """The assembled skeleton (placeholder slots, no resolved values); input-independent."""
        return self._assemble(self._tc)

    def _insert_slots(self, data_sites: List) -> None:
        """Insert this concern's value sealings around each data-access site."""
        raise NotImplementedError

    def _sealings(self) -> List[Sealing]:
        raise NotImplementedError

    def resolve(self, inp) -> ResolvedSealingTestCase:
        raise NotImplementedError

    @staticmethod
    def _clamp_entries(sandbox_sealings: List[SandboxSealing]) -> List[_Resolved]:
        return [_Resolved(s, None, [], None) for s in sandbox_sealings]   # always seal(None); never decoyed



class PacSealedTestCase(SealedTestCase):
    """Sandbox clamp + a PAC auth per data access, plus a PAC auth per standalone AUT*."""

    def __init__(self, sealed_tc, trace_fn, assemble, sandbox_sealings, data_sites, signer, enc, auth_specs) -> None:
        self._signer = signer
        self._enc = enc
        self._auth_specs = auth_specs
        self._pac: List[PacSealing] = []
        super().__init__(sealed_tc, trace_fn, assemble, sandbox_sealings, data_sites)

    def _insert_slots(self, data_sites) -> None:
        # Per test case (seeded by the salt so every input shares one skeleton), each access is sealed
        # with probability CONF.pac_seal_prob. A skipped access still gets its offset cancellation
        # (offset_subs is sandbox safety, not seal machinery: it pulls base+offset back into the clamped
        # region), just no AUT* — leaving a raw, sandbox-clamped, unauthenticated pointer.
        rng = random.Random(self._salt)
        for inst, bb, mem_reg, offset_subs, base_preserved in data_sites:
            if rng.random() >= CONF.pac_seal_prob:
                _insert(bb, inst, offset_subs)
                continue
            s = PacSealing(mem_reg, self._enc._pick_mem_auth(mem_reg), self._enc, base_preserved)
            self._pac.append(s)
            _insert(bb, inst, s.slot_insts[:s.n_before], offset_subs)   # auth the base, then cancel offset
            _place_after(bb, inst, s.slot_insts[s.n_before:])           # XPAC back to canonical (re-converge)
        _seal_auths(self._tc, self._enc, self._auth_specs, self._pac)

    def _sealings(self) -> List[Sealing]:
        return self._sandbox + self._pac

    def resolve(self, inp) -> ResolvedSealingTestCase:
        cer = self._trace_fn(self._tc, inp)
        pac = [_Resolved(s, *_resolve_pac(s, cer, self._layout, self._signer, self._salt)) for s in self._pac]
        entries = self._clamp_entries(self._sandbox) + pac
        object_code = self._assemble(self._tc)
        return ResolvedSealingTestCase(entries, object_code,
                                       _slot_offsets(self._tc, self._layout, self._pac), self._salt)


class MteSealedTestCase(SealedTestCase):
    """Sandbox clamp + an MTE retag per data access — the retag is the last op before the access."""

    def __init__(self, sealed_tc, trace_fn, assemble, sandbox_sealings, data_sites) -> None:
        self._mte: List[MteSealing] = []
        super().__init__(sealed_tc, trace_fn, assemble, sandbox_sealings, data_sites)

    def _insert_slots(self, data_sites) -> None:
        for inst, bb, mem_reg, offset_subs, _ in data_sites:
            s = MteSealing(mem_reg, inst)
            self._mte.append(s)
            _insert(bb, inst, offset_subs, s.slot_insts)   # cancel offset, then retag onto the cell tag

    def _sealings(self) -> List[Sealing]:
        return self._sandbox + self._mte

    def resolve(self, inp) -> ResolvedSealingTestCase:
        cer = self._trace_fn(self._tc, inp)
        mte = [_Resolved(s, *_resolve_mte(s, cer, self._layout)) for s in self._mte]
        entries = self._clamp_entries(self._sandbox) + mte
        object_code = self._assemble(self._tc)
        return ResolvedSealingTestCase(entries, object_code,
                                       _slot_offsets(self._tc, self._layout, self._mte), self._salt)


class CanonSealedTestCase(SealedTestCase):
    """Sandbox clamp + a canonicality flip per data access. The flip EOR is the last op before the
    access (after the offset-cancel SUBs), so nothing re-canonicalizes the pointer before it is used;
    a matching EOR right AFTER the access restores the base (when the access preserves it), so the
    decoy re-converges with the baseline and the flipped bits cannot cascade downstream. No new
    instructions, no CE plugin, no per-input REIF section: the CE traces the canonical placeholder
    (NOP slots) and only supplies speculation depth; the decoy is ordinary EOR words spliced through
    the existing code-relocation path."""

    def __init__(self, sealed_tc, trace_fn, assemble, sandbox_sealings, data_sites) -> None:
        self._canon: List[CanonSealing] = []
        super().__init__(sealed_tc, trace_fn, assemble, sandbox_sealings, data_sites)

    def _insert_slots(self, data_sites) -> None:
        # Per test case (seeded by the salt so every input shares one skeleton), each access is sealed
        # with probability CONF.canonicality_seal_prob. A skipped access still gets its offset
        # cancellation (sandbox safety), just no canonicality slot. base_preserved says whether the
        # access leaves its base register intact (so the after-EOR can restore it).
        pool = _canon_mask_pool(self._salt)   # validated once; shared by every slot in this test case
        rng = random.Random(self._salt)
        for inst, bb, mem_reg, offset_subs, base_preserved in data_sites:
            if rng.random() >= CONF.canonicality_seal_prob:
                _insert(bb, inst, offset_subs)
                continue
            s = CanonSealing(mem_reg, inst, pool, base_preserved)
            self._canon.append(s)
            _insert(bb, inst, offset_subs, s.slot_insts[:1])   # cancel offset, then flip (last before)
            _place_after(bb, inst, s.slot_insts[1:])           # restore the base right after (re-converge)

    def _sealings(self) -> List[Sealing]:
        return self._sandbox + self._canon

    def resolve(self, inp) -> ResolvedSealingTestCase:
        cer = self._trace_fn(self._tc, inp)
        canon = [_Resolved(s, *_resolve_canon(s, cer, self._layout)) for s in self._canon]
        entries = self._clamp_entries(self._sandbox) + canon
        object_code = self._assemble(self._tc)
        return ResolvedSealingTestCase(entries, object_code,
                                       _slot_offsets(self._tc, self._layout, self._canon), self._salt)


class MtePacSealedTestCase(SealedTestCase):
    """Sandbox clamp + PAC auth + MTE retag per data access, plus a PAC auth per standalone AUT*.
    The MTE retag (ADDG) is placed LAST — after the offset-cancel SUBs, immediately before the access
    — so no address op runs between the retag and the access. The CE's after-access tag correction is
    then positionally equivalent to the genuine retag, so the placeholder trace already carries the
    genuine tag at every later AUT*; PAC resolves over that single trace, no genuine-tag re-trace."""

    def __init__(self, sealed_tc, trace_fn, assemble, sandbox_sealings, data_sites, signer, enc,
                 auth_specs, trace_bytes_fn=None) -> None:
        self._signer = signer
        self._enc = enc
        self._auth_specs = auth_specs
        self._pac: List[PacSealing] = []
        self._mte: List[MteSealing] = []
        super().__init__(sealed_tc, trace_fn, assemble, sandbox_sealings, data_sites, trace_bytes_fn)

    def _insert_slots(self, data_sites) -> None:
        for inst, bb, mem_reg, offset_subs, base_preserved in data_sites:
            p = PacSealing(mem_reg, self._enc._pick_mem_auth(mem_reg), self._enc, base_preserved)
            m = MteSealing(mem_reg, inst)
            self._pac.append(p)
            self._mte.append(m)
            # before: auth the base, cancel the offset, retag onto the cell tag (retag closest to the access)
            _insert(bb, inst, p.slot_insts[:p.n_before], offset_subs, m.slot_insts)
            # after: the PAC XPAC re-converges the decoy.
            _place_after(bb, inst, p.slot_insts[p.n_before:])
        _seal_auths(self._tc, self._enc, self._auth_specs, self._pac)

    def _sealings(self) -> List[Sealing]:
        return self._sandbox + self._pac + self._mte

    def resolve(self, inp) -> ResolvedSealingTestCase:
        # Pass A (placeholder): resolve the context-independent values — MTE tag deltas + PAC
        # speculation-nesting. PAC contexts must NOT be signed off this trace: an AUT*'s context
        # register may itself be MTE-retagged by another seal, and the placeholder renders those
        # retags as NOPs, so the register carries the wrong tag here.
        cer = self._trace_fn(self._tc, inp)
        mte = [_Resolved(s, *_resolve_mte(s, cer, self._layout)) for s in self._mte]
        object_code = self._assemble(self._tc)
        offsets = _slot_offsets(self._tc, self._layout, self._pac + self._mte)

        # Pass B (genuine-tag): re-trace with the genuine MTE retags spliced in (PAC left as the
        # placeholder XPAC, which equals a passing AUT*, so nothing FPACs). Now every AUT*'s context
        # register carries its real tag, so PAC signs over the true genuine (pointer, context).
        mte_reloc = [Relocation(off, _encode(i))
                     for r in mte for off, i in zip(offsets[id(r.sealing)], r.sealing.seal(r.value, None))]
        cer_b = self._trace_bytes_fn(apply_relocations(object_code, mte_reloc), inp)
        pac = [_Resolved(s, *_resolve_pac(s, cer_b, self._layout, self._signer, self._salt)) for s in self._pac]

        entries = self._clamp_entries(self._sandbox) + pac + mte
        return ResolvedSealingTestCase(entries, object_code, offsets, self._salt)


# ==================================================================================================
# Sealer — walk the TC, create the sealings, insert their placeholders
# ==================================================================================================
class _Addressing(_SandboxInstrumentationBase):
    """Composable wrapper over the shared AArch64 address/CFG helpers (a sealer *has-a* one)."""
    def __init__(self, norm: Dict[str, str], mask: str, base_reg: str) -> None:
        self._norm = norm
        self._sandbox_mask = mask
        self._sandbox_base_reg = base_reg

    def address_preserving_tag_op(self, inst: Instruction) -> Optional[Tuple[str, str]]:
        name = inst.name.lower()
        if name == "irg" and len(inst.operands) >= 2:
            return inst.operands[0].value, inst.operands[1].value
        if name in ("addg", "subg"):
            if len(inst.operands) < 4:
                return None
            try:
                imm6 = int(inst.operands[2].value)
            except (ValueError, IndexError):
                return None
            return None if imm6 != 0 else (inst.operands[0].value, inst.operands[1].value)
        return None


# The PAC field position depends on the VA/TCR config, not the key, so any sign mnemonic reveals it.
_PROBE_PAC_MN = "pacia"


def _pac_encoder(generator, field_mask):
    """The PacSign slot encoder + the AUT* spec table."""
    _, auth_specs, xpac_specs = build_pac_specs(generator)
    return PacSign(generator, auth_specs, xpac_specs, field_mask), auth_specs


def _seal_auths(tc: TestCase, enc: PacSign, auth_specs, pac: List) -> None:
    """Replace every generator-emitted AUT* with a PacSealing committing to it (a non-memory site the
    sandbox walk doesn't reach). Appends the new sealings to `pac`."""
    for func in tc.functions:
        replacements = []
        for bb in func:
            for inst in bb:
                if inst.name.lower() in auth_specs:
                    s = PacSealing(inst.operands[0].value, copy.deepcopy(inst), enc)
                    pac.append(s)
                    replacements.append((inst, bb, s.slot_insts))
        for old, bb, slot in replacements:
            for i in slot:
                bb.insert_before(old, i)
            bb.delete(old)


class Sealer:
    """Factory: runs the sandbox walk (mechanics only) and constructs the per-primitive
    SealedTestCase, which owns its value-seal placement order + resolution. The Sealer holds no
    knowledge of slot order itself."""

    def __init__(self, generator, trace_fn, assemble, primitives: Set[str], signer,
                 trace_bytes_fn=None) -> None:
        self._walk = SandboxWalk(generator)
        self._trace_fn = trace_fn
        self._trace_bytes_fn = trace_bytes_fn
        self._assemble = assemble
        self._primitives = frozenset(primitives)
        self._signer = signer
        if "pac" in self._primitives:
            self._enc, self._auth_specs = _pac_encoder(generator, signer.field_mask(_PROBE_PAC_MN))
        else:
            self._enc, self._auth_specs = None, None

    def seal(self, test_case: TestCase) -> SealedTestCase:
        tc = copy.deepcopy(test_case)
        sandbox, data_sites = self._walk.sandbox(tc)
        if self._primitives == frozenset({"canon"}):
            return CanonSealedTestCase(tc, self._trace_fn, self._assemble, sandbox, data_sites)
        if self._primitives == frozenset({"mte"}):
            return MteSealedTestCase(tc, self._trace_fn, self._assemble, sandbox, data_sites)
        if self._primitives == frozenset({"pac"}):
            return PacSealedTestCase(tc, self._trace_fn, self._assemble, sandbox, data_sites,
                                     self._signer, self._enc, self._auth_specs)
        return MtePacSealedTestCase(tc, self._trace_fn, self._assemble, sandbox, data_sites,
                                    self._signer, self._enc, self._auth_specs, self._trace_bytes_fn)


def _insert(bb, anchor, *groups) -> None:
    """Insert each group of instructions before `anchor`, in order — so they land in group order
    immediately before it (clamp, then value sealings, then the index/displacement cancellation)."""
    for group in groups:
        for inst in group:
            bb.insert_before(anchor, inst)


def _place_after(bb, anchor, insts) -> None:
    """Insert `insts` immediately after `anchor`, preserving their order — the seal's after-revert
    (restore the base register once the sealed access has run). insert_after puts each element right
    after the anchor, so iterate in reverse to keep list order. Composed reverts (MtePac) call this
    once per primitive; calling the outer primitive first lands the inner one closest to the access."""
    for inst in reversed(list(insts)):
        bb.insert_after(anchor, inst)


def _record_positions(tc: TestCase, sealings: List[Sealing]) -> None:
    """Once every slot is inserted, record each sealing's slot positions in the final TC so a
    structural copy can locate and fill them. Must run last — any later insertion shifts positions."""
    locs = index_instructions(tc)
    for s in sealings:
        s.slot_locs = [locs[id(i)] for i in s.slot_insts]


class SandboxWalk:
    """Mechanics only: the sandbox-taint walk. Decides which memory bases need clamping (the clamp is
    not idempotent, so a base already in-region is not re-clamped) and makes a SandboxSealing for each.
    Knows nothing about value sealings, PAC, MTE, slot placement, or position recording — `sandbox`
    returns the clamps plus a per-access site list the SealedTestCase uses to place its value sealings."""

    def __init__(self, generator) -> None:
        # Under MTE the sandbox is tagged one allocation tag per 16-byte granule, and an access is
        # tag-checked against every granule its bytes touch. A pointer carries a single tag, so an
        # unaligned/multi-byte data access straddling two differently-tagged granules faults on the one
        # the pointer does not match. Clamping the base 16-byte-aligned (as tag stores already are)
        # keeps each access within one granule, so its single tag covers it; the cache-set index
        # (bits [11:6]) is unaffected. Only when a tag store can make granules non-uniform: with no
        # tag-memory instruction in the pool the sandbox stays uniformly tagged and unaligned is safe.
        granule_align = any("MTE-TAG-MEM-STORE" in s.tags
                            for s in generator.instruction_set.instructions)
        stg_mask = f"#0x{_SANDBOX_MASK & ~0xF:x}"
        data_mask = stg_mask if granule_align else f"#0x{_SANDBOX_MASK:x}"
        self._addr = _Addressing(generator.target_desc.reg_normalized,
                                 data_mask, SANDBOX_BASE_REGISTER)
        self._mask = data_mask
        self._stg_mask = stg_mask

    def sandbox(self, tc: TestCase) -> Tuple[List[SandboxSealing], List]:
        """The sandbox-taint dataflow + clamping. Decides which memory bases need an in-region clamp
        (the clamp is not idempotent, so a base already in-region is not re-clamped) and inserts a
        SandboxSealing for each (16B-aligned for STG). STG/LDG are handled fully here (clamp + the
        index/displacement cancellation, since they take no value sealing). A data access only gets
        its clamp inserted here and yields a site — (access, bb, mem_reg, offset_subs) — so the host
        sealer can place its value sealings between the clamp and the cancellation. Returns
        (sandbox_sealings, data_sites)."""
        addr, base, sandbox, data_sites = self._addr, SANDBOX_BASE_REGISTER, [], []

        def clamp(reg, mask):
            s = SandboxSealing(reg, mask, base)   # seals its own placeholder on construction
            sandbox.append(s)
            return s.slot_insts

        for func in tc.functions:
            prefixes = []                       # (access, bb, insts) the walk inserts itself
            predecessors, topo = addr._topo_sort(func)
            taint_out: Dict[BasicBlock, frozenset] = {}
            for bb in topo:
                processed = [p for p in predecessors.get(bb, []) if p in taint_out]
                curr: frozenset = frozenset()
                if processed:
                    curr = taint_out[processed[0]]
                    for p in processed[1:]:
                        curr = curr & taint_out[p]
                for inst in bb:
                    if inst.has_memory_access:
                        if len(inst.get_mem_operands()) > 1:
                            raise GeneratorException(
                                f"the memory seal models one access per instruction; {inst.name!r} has several")
                        mem_reg = addr._get_mem_base_reg(inst)
                        if mem_reg is not None:
                            norm_mem = addr._norm_reg(mem_reg)
                            offset_subs = addr._make_offset_sub_insts(inst.get_mem_operands()[0])
                            modifies_base = bool(offset_subs)
                            if addr._is_tag_store(inst):             # STG: 16B-aligned clamp, no value
                                prefixes.append((inst, bb, clamp(mem_reg, self._stg_mask) + offset_subs))
                                curr = curr - frozenset([norm_mem])
                            elif addr._is_tag_load(inst):            # LDG: clamp, no value
                                if norm_mem in curr:
                                    prefixes.append((inst, bb, list(offset_subs)))
                                    if modifies_base:
                                        curr = curr - frozenset([norm_mem])
                                else:
                                    prefixes.append((inst, bb, clamp(mem_reg, self._mask) + offset_subs))
                                    if not modifies_base:
                                        curr = curr | frozenset([norm_mem])
                            else:                                    # data access: clamp now, value later
                                if norm_mem not in curr:
                                    prefixes.append((inst, bb, clamp(mem_reg, self._mask)))
                                    if not modifies_base:
                                        curr = curr | frozenset([norm_mem])
                                elif modifies_base:
                                    curr = curr - frozenset([norm_mem])
                                # base_preserved: the access leaves its base register intact (not a
                                # load destination, no write-back) -> a canonicality flip can be
                                # cleanly reverted after it. dest regs include write-back bases.
                                base_preserved = norm_mem not in addr._dest_regs(inst)
                                data_sites.append((inst, bb, mem_reg, offset_subs, base_preserved))
                    prop = addr.address_preserving_tag_op(inst)
                    for dreg in addr._dest_regs(inst):
                        if prop is not None and dreg == addr._norm_reg(prop[0]):
                            curr = curr | frozenset([dreg]) if addr._norm_reg(prop[1]) in curr \
                                else curr - frozenset([dreg])
                        else:
                            curr = curr - frozenset([dreg])
                taint_out[bb] = curr
            for anchor, bb, insts in prefixes:   # clamps (and STG/LDG cancellation) go in before value
                for i in insts:
                    bb.insert_before(anchor, i)
        return sandbox, data_sites


def make_sealer(generator, trace_fn, assemble, primitives, signer, trace_bytes_fn=None) -> Sealer:
    """The Sealer for the active primitives. `trace_fn(tc, input) -> cer` runs a CE trace; `assemble(tc)
    -> bytes` assembles the object code; `trace_bytes_fn(code, input) -> cer` re-traces already-assembled
    machine code (used by the PAC+MTE resolve to read AUT* contexts with the genuine tags applied);
    `signer` is the PAC signer used by resolve when 'pac' is active (None otherwise)."""
    prims = frozenset(primitives)
    if prims not in (frozenset({"canon"}), frozenset({"mte"}), frozenset({"pac"}),
                     frozenset({"pac", "mte"})):
        raise ValueError(f"unsupported seal primitives: {primitives!r}")
    return Sealer(generator, trace_fn, assemble, prims, signer, trace_bytes_fn)
