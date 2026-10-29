"""Sealing as composable per-location sealings (AArch64).

Each seal's per-input value lives in the GlobalSealTable (a reserved vector register, loaded from the
input); the seal's slot is fixed code that loads it and applies it. A variant is a GST image, not a
code change.

  * `Sealing` — one site; its fixed slot loads its value from the GST and applies it.
  * `Sealer.seal(tc) -> SealedTestCase` — walks the TC, creates sealings, inserts their slots.
  * `SealedTestCase.resolve(input) -> ResolvedSealingTestCase` — runs the CE trace, computes values.
  * `ResolvedSealingTestCase` — `object_code` plus `genuine()` / `decoy()` GST images for it.
"""
import abc
import copy
import random
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..aarch64_contract_executor import ContractExecutionResult

from ...config import CONF
from ...util import stable_rng
from ...interfaces import (Instruction, TestCase, BasicBlock, GeneratorException, RegisterOperand,
                          SIMD_SUBREGION_SIZE)
from .primitives import index_instructions, _SANDBOX_MASK, _SandboxInstrumentationBase
from .pac import PacSign, PacSigner, build_pac_specs, _AUTH_TO_PAC, _read_reg
from .global_seal_table import GlobalSealTable, VectorGlobalSealTable, Location
from .seal_values import SealValue, MaskFlip
from ..aarch64_target_desc import SANDBOX_BASE_REGISTER, SEAL_SCRATCH_REGISTER
from ..aarch64_printer import Aarch64ASMLayout


M = (1 << 64) - 1

# Bits reserved for the GST: the full SIMD input section (all CE-loaded high vector registers). The
# generated body emits no vector registers, so the whole section is free for seal values.
_GST_CAPACITY_BITS = SIMD_SUBREGION_SIZE * 8

_MIGRATION_PENDING = ("PAC/MTE register-indirection (GST) is not yet migrated: it needs the CE "
                      "op-emulator inject-value change plus HW validation; only canonicality and "
                      "branch-target seals are on the GST path")


# ==================================================================================================
# Sealing — one location; pure seal(value); exposes the facts the resolver reads
# ==================================================================================================
class Sealing(abc.ABC):
    """One sealing site. Its slot is FIXED code that loads the per-input value from the GST and applies
    it, so the genuine/decoy difference is the stored value, not the code. `value`/`location` are the
    GST binding (both None for the sandbox clamp, which carries no value); `slot_insts`/`slot_locs` are
    the slot instructions and their recorded positions."""
    value_reg: str
    value: SealValue        # a value seal sets both at construction; the sandbox clamp sets neither
    location: Location

    def __init__(self) -> None:
        self.slot_insts: List[Instruction] = []
        self.slot_locs: List = []


class SandboxSealing(Sealing):
    """Clamp a base register into the input region: [AND reg, #mask; ADD reg, base]. Carries no value."""

    def __init__(self, value_reg: str, mask: str, base_reg: str) -> None:
        super().__init__()
        self.value_reg = value_reg
        self.slot_insts = [
            Instruction("and", True, "", False, template=f"AND {value_reg}, {value_reg}, {mask}"),
            Instruction("add", True, "", False, template=f"ADD {value_reg}, {value_reg}, {base_reg}"),
        ]


class PacSealing(Sealing):
    """PAC seal: pending GST migration (needs the op-emulator inject-value change + HW validation)."""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(_MIGRATION_PENDING)


class MteSealing(Sealing):
    """MTE seal: pending GST migration (needs the op-emulator inject-value change + HW validation)."""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(_MIGRATION_PENDING)


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
    rng = stable_rng("canon-pool", salt)
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
    """Make a pointer register non-canonical for ONE speculative access via a GST mask, then restore it.
    The fixed slot loads the mask from the GST and EORs it into the base before the access; when the
    access preserves its base (`revert`), the same EOR after re-converges with the baseline so the
    flipped high bits cannot cascade into a later access's cache set. Genuine value is 0 (no flip);
    `pool` is the candidate non-canonical masks the decoy picks. `access_inst`/`pool` feed the resolver."""

    def __init__(self, value_reg: str, access_inst: Instruction, pool: List[int], revert: bool,
                 gst: GlobalSealTable) -> None:
        super().__init__()
        self.value_reg = value_reg
        self.access_inst = access_inst
        self.pool = pool
        self.value = MaskFlip()
        self.location = gst.allocate(self.value.width)
        pre = self.value.apply(self.location, value_reg, SEAL_SCRATCH_REGISTER)
        self.n_before = len(pre)
        post = self.value.apply(self.location, value_reg, SEAL_SCRATCH_REGISTER) if revert else []
        self.slot_insts = pre + post


def _branch_target_mask_pool(salt: int) -> List[int]:
    """Masks the decoy XORs into an indirect call's target register before the BLR. Two fault axes:
      * non-canonical: a contiguous run in [54:VA] (as in canonicality) -> translation fault;
      * misaligned (when branch_target_seal_misalign): a low-bit value {1,2,3} -> the target is no
        longer 4-byte aligned -> PC-alignment fault (an axis x86 lacks).
    Unlike CanonSealing the target is a dead scratch (never a data base), so any bits may flip.
    branch_target_canon_mask fixes the non-canonical run; otherwise a salt-seeded pool is used."""
    va = _canon_va()
    fault = (((1 << (55 - va)) - 1) << va)   # bits [54:va]
    pool: List[int] = []
    if CONF.branch_target_canon_mask is not None:
        m = CONF.branch_target_canon_mask & M
        if 0 == m or (m & ~fault) or _mask_runs(m) != [m]:
            raise GeneratorException(
                f"branch_target_canon_mask 0x{m:016x} must be a single contiguous run within [54:{va}]")
        pool.append(m)
    else:
        rng = stable_rng("bt-canon-pool", salt)
        for _ in range(_CANON_POOL_TRIES):
            lo = rng.randrange(va, 55)
            hi = rng.randrange(lo, 55)
            m = ((1 << (hi - lo + 1)) - 1) << lo
            if m and m not in pool:
                pool.append(m)
                if len(pool) >= _CANON_POOL_SIZE:
                    break
    if CONF.branch_target_seal_misalign:
        pool += [1, 2, 3]   # low-bit misalignments -> PC-alignment fault
    if not pool:
        raise GeneratorException(f"no branch-target seal mask for VA_SIZE {va}")
    return pool


class BranchTargetSealing(Sealing):
    """Corrupt an indirect call's target register for ONE speculative BLR via a GST mask: the fixed slot
    loads the mask and EORs it into the target before the branch (non-canonical high bits or [1:0]
    misalignment), then the same EOR after to re-converge — x28 is the shared indirect-call target reg,
    so a leftover flip would cascade into a later call's target. Genuine value is 0 (valid target);
    `pool` is the candidate masks the decoy picks; `branch_inst` feeds the resolver."""

    def __init__(self, value_reg: str, branch_inst: Instruction, pool: List[int],
                 gst: GlobalSealTable) -> None:
        super().__init__()
        self.value_reg = value_reg
        self.branch_inst = branch_inst
        self.pool = pool
        self.value = MaskFlip()
        self.location = gst.allocate(self.value.width)
        pre = self.value.apply(self.location, value_reg, SEAL_SCRATCH_REGISTER)
        self.n_before = len(pre)
        self.slot_insts = pre + self.value.apply(self.location, value_reg, SEAL_SCRATCH_REGISTER)


_SIGNEXT_BIT = 1 << 55   # the regime-selector / sign bit: the one bit non-canonical under TBI on or off


class SignExtSealing(CanonSealing):
    """Break a pointer's sign-extension for ONE speculative access, then restore it — like CanonSealing
    but flipping bit 55 (the TTBR0/TTBR1 regime selector that canonicality leaves untouched) rather than
    [54:VA]. `seal(mask)` EORs bit 55 in right before the access, so the regime selector disagrees with
    the never-flipped sign-extension bits [54:VA] — a translation fault under TBI on or off — and EORs it
    back after (when the base survives). The top byte [63:56] is NOT flipped: bit 55 alone already
    faults under TBI on or off, whereas a top-byte flip faults only with TBI off (it is ignored when TBI
    is on), so the pool keeps the one TBI-robust bit. `seal(None/0)` is the canonical placeholder (NOPs)."""


def _signext_mask_pool() -> List[int]:
    """The sign-extension flip mask: bit 55 alone. Flipping the regime selector makes it disagree with
    the sign-extension bits [54:VA], which are never flipped — a guaranteed translation fault under TBI
    on or off. The top byte [63:56] is excluded: bit 55 alone already faults either way, whereas a
    top-byte flip faults only with TBI off (it is ignored when TBI is on)."""
    return [_SIGNEXT_BIT]


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
    """One input's resolution. `object_code` is the input-independent assembled test case (every variant
    shares it); `genuine()`/`decoy()` return the GST image bytes (the SIMD section payload) that select a
    variant. `genuine()` stores every slot's correct value; `decoy()` perturbs a subset of the
    speculative slots. `cer` is the single resolution trace, reused as the genuine contract trace."""

    def __init__(self, entries: List[_Resolved], object_code: bytes, gst: GlobalSealTable, salt: int,
                 cer: Optional["ContractExecutionResult"]) -> None:
        self._entries = entries
        self._object_code = object_code
        self._gst = gst
        self._salt = salt
        self._cer = cer
        self._genuine = self._build_image(stable_rng(self.collapse_key, salt), decoy=False)

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

    @property
    def cer(self) -> Optional["ContractExecutionResult"]:
        """The resolution trace, reused as the genuine contract trace (see the class docstring)."""
        return self._cer

    def genuine(self) -> bytes:
        return self._genuine

    def decoy(self, rng: random.Random) -> bytes:
        """The decoy GST image for one variant. `rng` (seeded by the caller from the sealing class, salt,
        and variant index) drives every choice, so the variant is a pure function of that seed: it
        reproduces across trace passes, and same-class inputs run the identical program."""
        return self._build_image(rng, decoy=True)

    def _eligible(self) -> List[_Resolved]:
        """Slots a decoy may perturb: speculative-or-unreached (never architectural) and carrying
        alternative values. When empty, decoy() reproduces genuine() (a null decoy) and the input is
        not speculative-exception-testable."""
        return [r for r in self._entries if r.speculative and r.alts]

    def has_decoy(self) -> bool:
        return bool(self._eligible())

    def _build_image(self, rng: random.Random, decoy: bool) -> bytes:
        """The SIMD section bytes for one variant: store each value seal's value into the GST image at
        its Location — the genuine value, or a random alt for a perturbed speculative slot. The genuine
        value of a canonicality / branch-target seal is "no perturbation" (resolver value None), which
        is mask 0. Every entry here is a value seal, so its Location is always set."""
        perturb = self._decoy_subset(self._eligible(), rng) if decoy else set()
        image = self._gst.blank_image()
        for r in self._entries:
            value = rng.choice(r.alts) if r in perturb else r.value
            image.store(r.sealing.location, 0 if value is None else value)
        return image.to_section().content

    @property
    def collapse_key(self) -> Tuple:
        """The sealing class of the resolved input: per-entry (value, speculative). Two inputs share
        a sealed TC iff every slot resolves to the same value with the same speculative status (so the
        same genuine fill is arch-safe for both, and decoy perturbs the same slots). Sandbox entries
        carry the same value (None) for every input, so they never split classes."""
        return tuple((r.value, r.speculative) for r in self._entries)


# ==================================================================================================
# Resolvers — concern-specific value computation from a CE trace (used by the SealedTestCase)
# ==================================================================================================
# Signature forgery pool: how many wrong signatures to offer per PAC slot, and the sampling budget.
_FORGERY_POOL_SIZE = 6
_FORGERY_TRIES = 64


def _input_pac_keys(inp) -> Tuple[int, ...]:
    """The PAC keys the input runs under; a PAC sealing cannot be resolved without them."""
    if inp.pac_keys is None:
        raise GeneratorException(f"PAC input {inp.seed} carries no PAC keys")
    return inp.pac_keys


def _resolve_pac(s: PacSealing, cer, layout, signer: PacSigner, keys: Tuple[int, ...], salt: int
                 ) -> Tuple[Optional[int], List[int], Optional[int]]:
    """A PacSealing's value from a trace: sign the pointer that reaches the sealing's auth position, plus
    a pool of wrong signatures that fail AUTH. When that position is never reached, the slot is still
    decoy-eligible (HW may speculate deeper than the model): forge the pool over the sandbox base, whose
    PAC-field-window bits every clamped sandbox pointer shares, so the decoy MOVKs preserve the pointer's
    non-field bits and the after-XPAC re-converges — exactly as the reached case does.
    (value, alts, spec_nesting)."""
    correct_sig, alts, spec = None, [], None
    if not cer or s.committed_inst is None:
        return correct_sig, alts, spec
    auth_off, code_base = layout.instruction_address[s.auth_slot_inst], cer[0].cpu.pc
    pac_mn = _AUTH_TO_PAC[s.committed_inst.name.lower()]
    value_reg = s.committed_inst.operands[0].value
    ctx_reg = s.committed_inst.operands[1].value if len(s.committed_inst.operands) > 1 else None
    for ite in cer:
        if ite.cpu.pc - code_base != auth_off:
            continue
        depth = ite.metadata.speculation_nesting
        if spec is None or depth < spec:
            spec = depth
        if correct_sig is not None and depth != 0:   # architectural occurrence is authoritative
            continue
        ptr = _read_reg(ite.cpu, value_reg)
        cval = _read_reg(ite.cpu, ctx_reg) if ctx_reg is not None else 0
        correct_sig = signer.sign(ptr, cval, pac_mn, keys)
        alts = _wrong_sigs(correct_sig, signer.field_mask(ptr, pac_mn), salt)
    if correct_sig is None:                          # unreached: forge over the shared sandbox base
        base_ptr = _read_reg(cer[0].cpu, SANDBOX_BASE_REGISTER)
        alts = _wrong_sigs(signer.sign(base_ptr, 0, pac_mn, keys), signer.field_mask(base_ptr, pac_mn), salt)
    return correct_sig, alts, spec


def _wrong_sigs(correct_sig: int, mask: int, salt: int) -> List[int]:
    """A deterministic pool of wrong signatures: the correct signature with only its PAC field bits
    perturbed (so each is a genuine AUTH failure). Seeded by (correct_sig, mask, salt), so every member
    of a sealing class forges identically."""
    rng = stable_rng(correct_sig, mask, salt)
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
    """An MteSealing's tag delta from a trace: the retag the EMUL_OP_MTE_RETAG op-emulator materialized
    (it reads the granule's tag from the CE's tag memory and plants `ADDG #0, #delta` to reach it, same
    as the genuine binary) plus the alternatives that mismatch. The delta is read back from that planted
    ADDG; the effective address's unretagged tag is `cell - delta`. (value, alts, spec_nesting)."""
    if not cer:
        return None, list(range(1, 16)), None
    retag_off = layout.instruction_address[s.slot_insts[0]]      # the materialized retag ADDG slot
    access_off, code_base = layout.instruction_address[s.access_inst], cer[0].cpu.pc
    # The genuine render bakes ONE delta; prefer an architectural (nest==0) reach, falling back to the
    # first reach when the site is speculative-only. A site reached architecturally more than once with
    # disagreeing tags (a loop whose granule is retagged between iterations) cannot be sealed with a
    # single delta — fail loud rather than silently bake one that mismatches the others.
    arch_deltas, arch_cells, first_delta, first_cell, spec = set(), set(), None, None, None
    for ite in cer:
        off, nest = ite.cpu.pc - code_base, ite.metadata.speculation_nesting
        if off == retag_off:
            d = (ite.cpu.encoding >> 10) & 0xF                 # ADDG uimm4 = the retag delta
            first_delta = d if first_delta is None else first_delta
            if 0 == nest:
                arch_deltas.add(d)
        if ite.metadata.has_sandbox_memory_access and off == access_off:
            if spec is None or nest < spec:
                spec = int(nest)
            c = ite.metadata.memory_access.allocation_tag      # the granule's tag the CE holds
            first_cell = c if first_cell is None else first_cell
            if 0 == nest:
                arch_cells.add(c)
    if len(arch_deltas) > 1 or len(arch_cells) > 1:
        raise GeneratorException(
            f"MTE site at offset {retag_off} reached architecturally with disagreeing tags "
            f"(deltas={sorted(arch_deltas)}, cells={sorted(arch_cells)}); cannot bake one retag")
    delta = next(iter(arch_deltas), first_delta)
    cell = next(iter(arch_cells), first_cell)
    if delta is None or cell is None:
        # unreached in the trace: genuine applies no retag (value None -> NOP), so any nonzero delta is
        # a valid decoy should HW speculate deeper than the model reached.
        return None, list(range(1, 16)), spec
    ea_tag = (cell - delta) % 16
    # A decoy whose retagged access lands on tag 0 or 15 cannot leak: the executor runs with
    # TCR_EL1.TCMA1 set, which makes an EL1 access through a tag-0b0000 or tag-0b1111 ("match-all")
    # address Tag-UNCHECKED — the hardware silently skips the tag comparison, so the intended mismatch
    # never happens and the decoy behaves identically to the genuine access. Offer only deltas whose
    # resulting tag (ea_tag + d) is a genuinely-checked value in 1..14.
    alts = [d for d in range(16) if d != delta and (ea_tag + d) % 16 not in (0, 15)]
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
            if not ite.metadata.has_sandbox_memory_access or ite.cpu.pc - code_base != access_off:
                continue
            nest = ite.metadata.speculation_nesting
            if spec is None or nest < spec:
                spec = int(nest)
    return None, list(s.pool), spec


def _resolve_branch_target(s: "BranchTargetSealing", cer, layout) -> Tuple[Optional[int], List[int], Optional[int]]:
    """A BranchTargetSealing's value from a trace: the genuine value is a valid target (None -> the NOP
    placeholder); the alternatives are the pool of non-canonical / misaligned masks (the decoy picks
    one). spec_nesting is the minimum depth the BLR is reached at, so a call ever reached
    architecturally (min == 0) keeps a valid target and is never corrupted. (value, alts, spec_nesting)."""
    spec = None
    if cer and s.branch_inst is not None:
        branch_off, code_base = layout.instruction_address[s.branch_inst], cer[0].cpu.pc
        for ite in cer:
            if ite.cpu.pc - code_base != branch_off:
                continue
            nest = ite.metadata.speculation_nesting
            if spec is None or nest < spec:
                spec = int(nest)
    return None, list(s.pool), spec


# ==================================================================================================
# SealedTestCase — holds the unresolved sealings; resolve(input) orchestrates value computation
# ==================================================================================================
class SealedTestCase:
    """Owns the sealings for one test case end-to-end. During construction it PLACES each value
    sealing's slot in the order this concern requires (the sandbox walk supplies clamps + per-access
    sites but never decides value-seal order), then `resolve(input)` computes their values. The
    per-concern subclass owns both the placement order and the resolution."""

    def __init__(self, sealed_tc: TestCase, assemble, sandbox_sealings: List[SandboxSealing],
                 data_sites: List, trace_bytes_fn) -> None:
        self._tc = sealed_tc
        self._trace_bytes_fn = trace_bytes_fn              # (machine code, input, emul_table) -> cer
        self._assemble = assemble                          # tc -> machine code, for the skeleton
        self._sandbox = sandbox_sealings
        self._salt = random.randrange(1 << 64)             # per-test-case; seeds deterministic decoys
        self._resolved: Dict = {}                          # input identity -> its ResolvedSealingTestCase
        self._gst: GlobalSealTable = VectorGlobalSealTable(_GST_CAPACITY_BITS)   # value seals allocate here
        self._insert_slots(data_sites)                     # subclass inserts its value sealings, in order
        _record_positions(self._tc, self._sealings())      # AFTER every slot is inserted
        self._layout = Aarch64ASMLayout(self._tc)

    @property
    def salt(self) -> int:
        return self._salt

    @property
    def object_code(self) -> bytes:
        """The assembled test case (fixed GST-load slots, no per-input values); input-independent."""
        return self._assemble(self._tc)

    def _insert_slots(self, data_sites: List) -> None:
        """Insert this concern's value sealings around each data-access site."""
        raise NotImplementedError

    def _sealings(self) -> List[Sealing]:
        """Every sealing whose slot positions must be recorded: the sandbox clamps plus this concern's
        value seals."""
        return self._sandbox + self._value_sealings()

    def _value_sealings(self) -> List[Sealing]:
        """The non-sandbox value seals this concern resolves (their slots carry the decoy-able values).
        Empty for sandbox-only (PTE)."""
        return []

    def _resolve_entries(self, inp, cer) -> List[_Resolved]:
        """Resolve this concern's value seals from the single resolution trace `cer`. Default: none."""
        return []

    def resolve(self, inp) -> ResolvedSealingTestCase:
        """One CE trace per input (memoized by input identity), reused as the genuine contract trace.
        The object code is input-independent; variants differ only in the GST image."""
        key = inp.identity()
        cached = self._resolved.get(key)
        if cached is not None:
            return cached
        object_code, cer = self._resolve_trace(inp)
        entries = self._resolve_entries(inp, cer)   # value seals only; the sandbox clamp carries no value
        resolved = ResolvedSealingTestCase(entries, object_code, self._gst, self._salt, cer)
        self._resolved[key] = resolved
        return resolved

    def _resolve_trace(self, inp) -> Tuple[bytes, "ContractExecutionResult"]:
        """One trace over the fixed code, shipping the genuine GST. For canon/branch that image is
        all-zero (no flip), so this is the canonical execution and serves as the genuine contract trace."""
        object_code = self._assemble(self._tc)
        simd = self._gst.blank_image().to_section().content
        cer = self._trace_bytes_fn(object_code, inp, emul_table=b"", simd=simd)
        return object_code, cer


class PacSealedTestCase(SealedTestCase):
    """Sandbox clamp + a PAC auth per data access, plus a PAC auth per standalone AUT*."""

    def __init__(self, sealed_tc, assemble, sandbox_sealings, data_sites, signer, enc, auth_specs,
                 trace_bytes_fn) -> None:
        self._signer = signer
        self._enc = enc
        self._auth_specs = auth_specs
        self._pac: List[PacSealing] = []
        super().__init__(sealed_tc, assemble, sandbox_sealings, data_sites, trace_bytes_fn)

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

    def _value_sealings(self) -> List[Sealing]:
        return self._pac

    def _resolve_entries(self, inp, cer) -> List[_Resolved]:
        keys = _input_pac_keys(inp)
        return [_Resolved(s, *_resolve_pac(s, cer, self._layout, self._signer, keys, self._salt))
                for s in self._pac]


class MteSealedTestCase(SealedTestCase):
    """Sandbox clamp + an MTE retag per data access — the retag is the last op before the access."""

    def __init__(self, sealed_tc, assemble, sandbox_sealings, data_sites, trace_bytes_fn) -> None:
        self._mte: List[MteSealing] = []
        super().__init__(sealed_tc, assemble, sandbox_sealings, data_sites, trace_bytes_fn)

    def _insert_slots(self, data_sites) -> None:
        for inst, bb, mem_reg, offset_subs, base_preserved in data_sites:
            s = MteSealing(mem_reg, inst, base_preserved)
            self._mte.append(s)
            _insert(bb, inst, offset_subs, s.slot_insts[:s.n_before])   # cancel offset, then retag
            _place_after(bb, inst, s.slot_insts[s.n_before:])           # ADDG the complement back to the cell tag

    def _value_sealings(self) -> List[Sealing]:
        return self._mte

    def _resolve_entries(self, inp, cer) -> List[_Resolved]:
        return [_Resolved(s, *_resolve_mte(s, cer, self._layout)) for s in self._mte]


class SandboxSealedTestCase(SealedTestCase):
    """Only the sandbox clamp -- no value seal, so genuine code == decoy code (no code decoy).
    Used when the genuine/decoy difference lives entirely in the run environment (PTE fuzzing): the
    test case still needs its accesses clamped and its object code assembled, but nothing is sealed
    with a per-input value. resolve() has no values to compute, but still traces the placeholder once
    so the genuine contract trace is available (the environment axis fuzzes the run, not the code)."""

    def _insert_slots(self, data_sites) -> None:
        for inst, bb, _mem_reg, offset_subs, _base_preserved in data_sites:
            _insert(bb, inst, offset_subs)   # pin each access to its clamped base; no value seal


class CanonSealedTestCase(SealedTestCase):
    """Sandbox clamp + a canonicality flip per data access. The flip EOR is the last op before the
    access (after the offset-cancel SUBs), so nothing re-canonicalizes the pointer before it is used;
    a matching EOR right AFTER the access restores the base (when the access preserves it), so the
    decoy re-converges with the baseline and the flipped bits cannot cascade downstream. No new
    instructions, no CE plugin, no per-input REIF section: the CE traces the canonical placeholder
    (NOP slots) and only supplies speculation depth; the decoy is ordinary EOR words spliced through
    the existing code-relocation path."""

    def __init__(self, sealed_tc, assemble, sandbox_sealings, data_sites, trace_bytes_fn) -> None:
        self._canon: List[CanonSealing] = []
        super().__init__(sealed_tc, assemble, sandbox_sealings, data_sites, trace_bytes_fn)

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
            s = CanonSealing(mem_reg, inst, pool, base_preserved, self._gst)
            self._canon.append(s)
            _insert(bb, inst, offset_subs, s.slot_insts[:s.n_before])   # cancel offset, then flip (before)
            _place_after(bb, inst, s.slot_insts[s.n_before:])           # restore the base (re-converge)

    def _value_sealings(self) -> List[Sealing]:
        return self._canon

    def _resolve_entries(self, inp, cer) -> List[_Resolved]:
        return [_Resolved(s, *_resolve_canon(s, cer, self._layout)) for s in self._canon]


class BranchTargetSealedTestCase(SealedTestCase):
    """Sandbox clamp per data access (no value seal) + a canonicality/alignment flip on each
    SPECULATIVE indirect-call target. The flip EOR lands right before the BLR (after the ADR that
    materialized the target); no after-revert -- the target register is a dead scratch. The CE traces
    the valid placeholder (NOP slots) and supplies speculation depth; the decoy is EOR words spliced in
    through the code-relocation path (no CE plugin, no per-input REIF section)."""

    def __init__(self, sealed_tc, assemble, sandbox_sealings, data_sites, trace_bytes_fn) -> None:
        self._bt: List[BranchTargetSealing] = []
        super().__init__(sealed_tc, assemble, sandbox_sealings, data_sites, trace_bytes_fn)

    def _insert_slots(self, data_sites) -> None:
        # data accesses keep their offset cancellation (sandbox safety); they are not value-sealed here.
        for inst, bb, mem_reg, offset_subs, base_preserved in data_sites:
            _insert(bb, inst, offset_subs)
        # each indirect-call BLR is sealed with probability branch_target_seal_prob; the flip lands
        # right before the BLR (after its target-materializing ADR).
        pool = _branch_target_mask_pool(self._salt)
        rng = random.Random(self._salt)
        for func in self._tc.functions:
            for bb in func:
                for inst in list(bb):
                    if inst.name != "blr" or rng.random() >= CONF.branch_target_seal_prob:
                        continue
                    reg = next(o.value for o in inst.operands if isinstance(o, RegisterOperand))
                    s = BranchTargetSealing(reg, inst, pool, self._gst)
                    self._bt.append(s)
                    _insert(bb, inst, s.slot_insts[:s.n_before])   # flip right before the BLR
                    _place_after(bb, inst, s.slot_insts[s.n_before:])   # re-converge x28 after

    def _value_sealings(self) -> List[Sealing]:
        return self._bt

    def _resolve_entries(self, inp, cer) -> List[_Resolved]:
        return [_Resolved(s, *_resolve_branch_target(s, cer, self._layout)) for s in self._bt]


class MtePacSealedTestCase(SealedTestCase):
    """Sandbox clamp + PAC auth + MTE retag per data access, plus a PAC auth per standalone AUT*.
    The CE tags each accessed pointer from its simulated tag memory (the after-access tag correction),
    so one resolution trace — the placeholder with the committed AUT* at each auth position (resolve-mode
    canonicalizes it) — yields the MTE tag deltas, the PAC signatures, AND the genuine contract trace."""

    def __init__(self, sealed_tc, assemble, sandbox_sealings, data_sites, signer, enc,
                 auth_specs, trace_bytes_fn) -> None:
        self._signer = signer
        self._enc = enc
        self._auth_specs = auth_specs
        self._pac: List[PacSealing] = []
        self._mte: List[MteSealing] = []
        super().__init__(sealed_tc, assemble, sandbox_sealings, data_sites, trace_bytes_fn)

    def _insert_slots(self, data_sites) -> None:
        for inst, bb, mem_reg, offset_subs, base_preserved in data_sites:
            p = PacSealing(mem_reg, self._enc._pick_mem_auth(mem_reg), self._enc, base_preserved)
            m = MteSealing(mem_reg, inst, base_preserved)
            self._pac.append(p)
            self._mte.append(m)
            # before: auth the base, cancel the offset, retag onto the cell tag (retag closest to the access)
            _insert(bb, inst, p.slot_insts[:p.n_before], offset_subs, m.slot_insts[:m.n_before])
            # after: MTE ADDG re-converges to the cell tag, then PAC XPAC strips — both re-converge the decoy.
            _place_after(bb, inst, m.slot_insts[m.n_before:] + p.slot_insts[p.n_before:])
        _seal_auths(self._tc, self._enc, self._auth_specs, self._pac)

    def _value_sealings(self) -> List[Sealing]:
        return self._pac + self._mte

    def _resolve_entries(self, inp, cer) -> List[_Resolved]:
        # One trace. The CE tags every accessed pointer from its simulated tag memory
        # (simulation_hook.c), so the genuine MTE retags need no splice here. The committed AUT* at each
        # auth position (resolve-mode: canonicalizes instead of verify-or-abort) keeps the trace's ctrace
        # exact. MTE tag deltas and PAC signatures both resolve off this single trace.
        keys = _input_pac_keys(inp)
        pac = [_Resolved(s, *_resolve_pac(s, cer, self._layout, self._signer, keys, self._salt))
               for s in self._pac]
        mte = [_Resolved(s, *_resolve_mte(s, cer, self._layout)) for s in self._mte]
        return pac + mte


class PacSignExtSealedTestCase(SealedTestCase):
    """Sandbox clamp + PAC auth + a sign-extension flip per data access, plus a PAC auth per standalone
    AUT*. Every access is PAC-authenticated (the baked signature keeps it arch-safe, as in
    MtePacSealedTestCase); the flip (SignExtSealing) is a flip seal — no per-input value, no CE emulator —
    added with probability `signext_seal_prob`, so only the PAC auth is materialized by the emulation
    table. The single resolution trace (the committed AUT* at each auth position, the canonical
    placeholder at each flip) yields the PAC signatures and the genuine contract trace; the decoy splices
    the flip EOR in."""

    def __init__(self, sealed_tc, assemble, sandbox_sealings, data_sites, signer, enc,
                 auth_specs, trace_bytes_fn) -> None:
        self._signer = signer
        self._enc = enc
        self._auth_specs = auth_specs
        self._pac: List[PacSealing] = []
        self._signext: List[SignExtSealing] = []
        super().__init__(sealed_tc, assemble, sandbox_sealings, data_sites, trace_bytes_fn)

    def _insert_slots(self, data_sites) -> None:
        # Every access must be PAC-authenticated for arch-safety, so pac_seal_prob (the fraction of
        # accesses to seal) is incompatible with this composed primitive — fail loud rather than ignore it.
        if CONF.pac_seal_prob != 1.0:
            raise GeneratorException("pac+signext seals every access; set pac_seal_prob = 1.0")
        pool = _signext_mask_pool()   # shared by every slot in this test case
        rng = random.Random(self._salt)
        for inst, bb, mem_reg, offset_subs, base_preserved in data_sites:
            p = PacSealing(mem_reg, self._enc._pick_mem_auth(mem_reg), self._enc, base_preserved)
            self._pac.append(p)
            if rng.random() >= CONF.signext_seal_prob:
                _insert(bb, inst, p.slot_insts[:p.n_before], offset_subs)
                _place_after(bb, inst, p.slot_insts[p.n_before:])
                continue
            s = SignExtSealing(mem_reg, inst, pool, base_preserved)
            self._signext.append(s)
            # before: auth the base, cancel the offset, break sign-extension (flip closest to the access)
            _insert(bb, inst, p.slot_insts[:p.n_before], offset_subs, s.slot_insts[:1])
            # after: restore the base, then PAC XPAC strips — both re-converge the decoy.
            _place_after(bb, inst, s.slot_insts[1:] + p.slot_insts[p.n_before:])
        _seal_auths(self._tc, self._enc, self._auth_specs, self._pac)

    def _value_sealings(self) -> List[Sealing]:
        return self._pac + self._signext

    def _resolve_entries(self, inp, cer) -> List[_Resolved]:
        keys = _input_pac_keys(inp)
        pac = [_Resolved(s, *_resolve_pac(s, cer, self._layout, self._signer, keys, self._salt))
               for s in self._pac]
        signext = [_Resolved(s, *_resolve_canon(s, cer, self._layout)) for s in self._signext]
        return pac + signext


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


def _pac_encoder(generator, field_span):
    """The PacSign slot encoder + the AUT* spec table."""
    _, auth_specs, xpac_specs = build_pac_specs(generator)
    return PacSign(generator, auth_specs, xpac_specs, field_span), auth_specs


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

    def __init__(self, generator, assemble, primitives: Set[str], signer, trace_bytes_fn) -> None:
        self._walk = SandboxWalk(generator)
        self._trace_bytes_fn = trace_bytes_fn
        self._assemble = assemble
        self._primitives = frozenset(primitives)
        self._signer = signer
        if "pac" in self._primitives:
            self._enc, self._auth_specs = _pac_encoder(generator, signer.field_span())
        else:
            self._enc, self._auth_specs = None, None

    def seal(self, test_case: TestCase) -> SealedTestCase:
        tc = copy.deepcopy(test_case)
        sandbox, data_sites = self._walk.sandbox(tc)
        tr = self._trace_bytes_fn
        if not self._primitives:
            # No code seal: only the sandbox clamp. The genuine/decoy difference (if any) comes
            # the environment axis (PTE fuzzing), not by code.
            return SandboxSealedTestCase(tc, self._assemble, sandbox, data_sites, tr)
        if self._primitives == frozenset({"canon"}):
            return CanonSealedTestCase(tc, self._assemble, sandbox, data_sites, tr)
        if self._primitives == frozenset({"branch_target"}):
            return BranchTargetSealedTestCase(tc, self._assemble, sandbox, data_sites, tr)
        if self._primitives == frozenset({"mte"}):
            return MteSealedTestCase(tc, self._assemble, sandbox, data_sites, tr)
        if self._primitives == frozenset({"pac"}):
            return PacSealedTestCase(tc, self._assemble, sandbox, data_sites,
                                     self._signer, self._enc, self._auth_specs, tr)
        if self._primitives == frozenset({"pac", "mte"}):
            return MtePacSealedTestCase(tc, self._assemble, sandbox, data_sites,
                                        self._signer, self._enc, self._auth_specs, tr)
        if self._primitives == frozenset({"pac", "signext"}):
            return PacSignExtSealedTestCase(tc, self._assemble, sandbox, data_sites,
                                            self._signer, self._enc, self._auth_specs, tr)
        raise GeneratorException(f"no sealed test case for primitives {set(self._primitives)}")


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
        granule_align = any("MTE-TAGMEM-STORE" in s.tags
                            for s in generator.instruction_set.instructions)
        # Floor every access at one MTE granule when a tag store can make granules non-uniform (so a
        # multi-byte access can't straddle two differently-tagged granules); otherwise no floor. Each
        # access is additionally aligned to its own requirement (mem_op.alignment) in _addr_mask, so a
        # single-copy-atomic access (acquire/release, exclusive, LSE atomic) is naturally aligned no
        # matter which config generates it.
        self._granule_floor = 16 if granule_align else 1
        # value-seal clamp (a sealed pointer written into a register): granule-aligned under MTE.
        value_mask = f"#0x{_SANDBOX_MASK & ~0xF:x}" if granule_align else f"#0x{_SANDBOX_MASK:x}"
        self._addr = _Addressing(generator.target_desc.reg_normalized,
                                 value_mask, SANDBOX_BASE_REGISTER)

    def _addr_mask(self, inst) -> str:
        """The sandbox clamp mask for one memory access, aligned to the wider of the MTE granule floor
        and the access's own natural-alignment requirement (mem_op.alignment bytes; 0 = none)."""
        align = max(self._granule_floor, inst.get_mem_operands()[0].alignment)
        return f"#0x{_SANDBOX_MASK & ~(align - 1):x}"

    def _realign_reused_base(self, inst, reg: str) -> List[Instruction]:
        """Instructions to re-align an ALREADY in-region base to this access's natural-alignment
        requirement, for the case where the base was clamped by a prior, weaker-aligned access and the
        clamp is not re-emitted (re-clamping would re-add the sandbox base). Masking only the low bits
        keeps the base in-region (x29 is page-aligned). Empty when the access needs no alignment (a plain
        load/store); atomics take no offset, so aligning the base aligns the effective address."""
        align = inst.get_mem_operands()[0].alignment
        if align <= 1:
            return []
        mask = f"#0x{(-align) & ((1 << 64) - 1):x}"        # ~(align - 1), a 64-bit bitmask immediate
        return [Instruction("and", True, "", False, template=f"AND {reg}, {reg}, {mask}")]

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
                            if addr._is_tag_store(inst):             # STG: granule-aligned clamp, no value
                                prefixes.append((inst, bb, clamp(mem_reg, self._addr_mask(inst)) + offset_subs))
                                curr = curr - frozenset([norm_mem])
                            elif addr._is_tag_load(inst):            # LDG: clamp, no value
                                if norm_mem in curr:
                                    prefixes.append((inst, bb, list(offset_subs)))
                                    if modifies_base:
                                        curr = curr - frozenset([norm_mem])
                                else:
                                    prefixes.append((inst, bb, clamp(mem_reg, self._addr_mask(inst)) + offset_subs))
                                    if not modifies_base:
                                        curr = curr | frozenset([norm_mem])
                            else:                                    # data access: clamp now, value later
                                if norm_mem not in curr:
                                    prefixes.append((inst, bb, clamp(mem_reg, self._addr_mask(inst))))
                                    if not modifies_base:
                                        curr = curr | frozenset([norm_mem])
                                else:                                # base already in-region: re-align it
                                    realign = self._realign_reused_base(inst, mem_reg)
                                    if realign:
                                        prefixes.append((inst, bb, realign))
                                    if modifies_base:
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
                    # A call transfers to a generated callee that may clobber any caller register (only
                    # the sandbox base x29 and the frame-spilled x30 are preserved), so no clamped base
                    # survives it: drop every in-region fact and force a re-clamp of any base used after
                    # the call. Without this a base clamped before the call and clobbered by the callee
                    # is reused unclamped -> an architectural out-of-sandbox access (CE fail-loud).
                    if inst.is_call:
                        curr = frozenset()
                taint_out[bb] = curr
            for anchor, bb, insts in prefixes:   # clamps (and STG/LDG cancellation) go in before value
                for i in insts:
                    bb.insert_before(anchor, i)
        return sandbox, data_sites


def make_sealer(generator, assemble, primitives, signer, trace_bytes_fn) -> Sealer:
    """The Sealer for the active primitives. `assemble(tc) -> bytes` assembles the object code;
    `trace_bytes_fn(code, input, emul_table) -> cer` runs the single per-input resolution trace of
    already-assembled machine code; `signer` is the PAC signer used by resolve when 'pac' is active
    (None otherwise)."""
    prims = frozenset(primitives)
    if prims not in (frozenset(), frozenset({"canon"}), frozenset({"branch_target"}),
                     frozenset({"mte"}), frozenset({"pac"}), frozenset({"pac", "mte"}),
                     frozenset({"pac", "signext"})):
        raise ValueError(f"unsupported seal primitives: {primitives!r}")
    return Sealer(generator, assemble, prims, signer, trace_bytes_fn)
