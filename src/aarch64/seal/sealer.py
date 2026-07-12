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
from ..aarch64_mte import MteTagState, mte_tag_store_effect, MTE_INITIAL_DEFAULT_TAG
from ..aarch64_target_desc import SANDBOX_BASE_REGISTER
from ..aarch64_printer import Aarch64ASMLayout
from ..aarch64_relocations import (NOP_WORD, xpac_word, addg_word, movk_word, aut_word, Relocation)


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

    def __init__(self, value_reg: str, committed_inst: Instruction, encoder: PacSign) -> None:
        super().__init__()
        self.value_reg = value_reg
        self.committed_inst = committed_inst
        self._enc = encoder
        self.slot_insts = self.seal(None, None)   # seal itself with its placeholder = the slot to fill

    def seal(self, value: Optional[int], rng: Optional[random.Random]) -> List[Instruction]:
        auth_mn = self.committed_inst.name.lower()
        ctx_reg = self.committed_inst.operands[1].value if len(self.committed_inst.operands) > 1 else None
        xpac = self._enc.make_xpac_inst(_AUTH_TO_XPAC[auth_mn], self.value_reg)
        if value is None:
            return [make_nop(), xpac]
        movk = self._enc.make_movk(self.value_reg, value, 48)
        if rng.random() < CONF.pac_strip_prob:
            return [movk, xpac]
        return [movk, self._enc.make_auth_inst(auth_mn, self.value_reg, ctx_reg)]


class MteSealing(Sealing):
    """Retag a pointer register by a 4-bit delta: `seal(delta)` is the retag, or [NOP] when 0/None.
    `access_inst` is exposed for the resolver (its accessed cell's tag is the genuine tag)."""

    def __init__(self, value_reg: str, access_inst: Instruction) -> None:
        super().__init__()
        self.value_reg = value_reg
        self.access_inst = access_inst
        self.slot_insts = self.seal(None, None)   # seal itself with its placeholder = the slot to fill

    def seal(self, value: Optional[int], rng: Optional[random.Random]) -> List[Instruction]:
        delta = 0 if value is None else value % 16
        if delta == 0:
            return [make_nop()]
        return [(Instruction("addg", True, "", False,
                             template=f"ADDG {self.value_reg}, {self.value_reg}, #0, #{delta}")
                 .add_op(RegisterOperand(self.value_reg, 64, False, True))
                 .add_op(RegisterOperand(self.value_reg, 64, True, False))
                 .add_op(ImmediateOperand("0", 6))
                 .add_op(ImmediateOperand(str(delta), 4)))]


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

    def _solve_relocations(self, offsets: Dict[int, Tuple[int, ...]],
                           rng: random.Random, decoy: bool) -> Tuple[Relocation, ...]:
        eligible = [r for r in self._entries if r.speculative and r.alts]
        perturb = self._decoy_subset(eligible, rng) if decoy else set()
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
    of wrong signatures that fail AUTH. (value, alts, spec_nesting)."""
    correct_sig, alts, spec = None, [], None
    if not cer or s.committed_inst is None:
        return correct_sig, alts, spec
    xpac = next(i for i in s.slot_insts if i.name.lower() in ("xpaci", "xpacd"))
    xpac_off, code_base = layout.instruction_address[xpac], cer[0].cpu.pc
    pac_mn = _AUTH_TO_PAC[s.committed_inst.name.lower()]
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
        correct_sig = signer.sign16(ptr, cval, pac_mn)
        alts = _wrong_sigs(correct_sig, signer.field_mask16(pac_mn), salt)
    return correct_sig, alts, spec


def _wrong_sigs(correct_sig: int, mask16: int, salt: int) -> List[int]:
    """A deterministic pool of wrong signatures: the correct signature with only its PAC field bits
    perturbed (so each is a genuine AUTH failure). Seeded by (correct_sig, mask, salt), so every member
    of a sealing class forges identically."""
    rng = random.Random(hash((correct_sig, mask16, salt)))
    pool: List[int] = []
    for _ in range(_FORGERY_TRIES):
        sig = (correct_sig & ~mask16) | (rng.randrange(1 << 64) & mask16)
        if sig != correct_sig and sig not in pool:
            pool.append(sig)
            if len(pool) >= _FORGERY_POOL_SIZE:
                break
    if not pool:
        raise GeneratorException(f"no PAC forgery for field mask 0x{mask16:04x}")
    return pool


def _resolve_mte(s: MteSealing, cer, layout) -> Tuple[Optional[int], List[int], Optional[int]]:
    """An MteSealing's tag delta from a trace: classify the accessed cell's allocation tag and the
    pointer's own tag at the guarded access; the genuine delta brings the pointer to the cell tag,
    alternatives are every other tag (a mismatch). (value, alts, spec_nesting)."""
    correct_tag, ptr_tag, spec = None, None, None
    if cer and s.access_inst is not None:
        access_off, code_base = layout.instruction_address[s.access_inst], cer[0].cpu.pc
        tags = MteTagState(MTE_INITIAL_DEFAULT_TAG)
        for ite in cer:
            nest = ite.metadata.speculation_nesting
            tags.to_depth(nest)
            store = mte_tag_store_effect(ite)
            if store is not None:
                tags.set(*store)
            if not ite.metadata.has_memory_access or ite.cpu.pc - code_base != access_off:
                continue
            ea = ite.metadata.memory_access.effective_address
            if spec is None or nest < spec:
                spec = int(nest)
            if nest == 0 or correct_tag is None:
                correct_tag, ptr_tag = tags.tag_at(ea), (ea >> 56) & 0xF
    if correct_tag is None or ptr_tag is None:
        return None, [], spec
    delta = (correct_tag - ptr_tag) % 16
    alts = [d for d in range(16) if d != delta]   # any other tag -> a tag mismatch on the access
    return delta, alts, spec


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
                 data_sites: List) -> None:
        self._tc = sealed_tc
        self._trace_fn = trace_fn
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
        for inst, bb, mem_reg, offset_subs in data_sites:
            if rng.random() >= CONF.pac_seal_prob:
                _insert(bb, inst, offset_subs)
                continue
            s = PacSealing(mem_reg, self._enc._pick_mem_auth(mem_reg), self._enc)
            self._pac.append(s)
            _insert(bb, inst, s.slot_insts, offset_subs)   # auth the base, then cancel the offset
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
        for inst, bb, mem_reg, offset_subs in data_sites:
            s = MteSealing(mem_reg, inst)
            self._mte.append(s)
            _insert(bb, inst, offset_subs, s.slot_insts)   # cancel the offset, then retag last

    def _sealings(self) -> List[Sealing]:
        return self._sandbox + self._mte

    def resolve(self, inp) -> ResolvedSealingTestCase:
        cer = self._trace_fn(self._tc, inp)
        mte = [_Resolved(s, *_resolve_mte(s, cer, self._layout)) for s in self._mte]
        entries = self._clamp_entries(self._sandbox) + mte
        object_code = self._assemble(self._tc)
        return ResolvedSealingTestCase(entries, object_code,
                                       _slot_offsets(self._tc, self._layout, self._mte), self._salt)


class MtePacSealedTestCase(SealedTestCase):
    """Sandbox clamp + PAC auth + MTE retag per data access, plus a PAC auth per standalone AUT*.
    The MTE retag (ADDG) is placed LAST — after the offset-cancel SUBs, immediately before the access
    — so no address op runs between the retag and the access. The CE's after-access tag correction is
    then positionally equivalent to the genuine retag, so the placeholder trace already carries the
    genuine tag at every later AUT*; PAC resolves over that single trace, no genuine-tag re-trace."""

    def __init__(self, sealed_tc, trace_fn, assemble, sandbox_sealings, data_sites, signer, enc, auth_specs) -> None:
        self._signer = signer
        self._enc = enc
        self._auth_specs = auth_specs
        self._pac: List[PacSealing] = []
        self._mte: List[MteSealing] = []
        super().__init__(sealed_tc, trace_fn, assemble, sandbox_sealings, data_sites)

    def _insert_slots(self, data_sites) -> None:
        for inst, bb, mem_reg, offset_subs in data_sites:
            p = PacSealing(mem_reg, self._enc._pick_mem_auth(mem_reg), self._enc)
            m = MteSealing(mem_reg, inst)
            self._pac.append(p)
            self._mte.append(m)
            _insert(bb, inst, p.slot_insts, offset_subs, m.slot_insts)   # auth, cancel the offset, retag
        _seal_auths(self._tc, self._enc, self._auth_specs, self._pac)

    def _sealings(self) -> List[Sealing]:
        return self._sandbox + self._pac + self._mte

    def resolve(self, inp) -> ResolvedSealingTestCase:
        cer = self._trace_fn(self._tc, inp)
        mte = [_Resolved(s, *_resolve_mte(s, cer, self._layout)) for s in self._mte]
        pac = [_Resolved(s, *_resolve_pac(s, cer, self._layout, self._signer, self._salt)) for s in self._pac]
        entries = self._clamp_entries(self._sandbox) + pac + mte
        object_code = self._assemble(self._tc)
        return ResolvedSealingTestCase(entries, object_code,
                                       _slot_offsets(self._tc, self._layout, self._pac + self._mte), self._salt)


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


def _pac_encoder(generator):
    """The PacSign slot encoder + the AUT* spec table. The kernel signer is NOT built here — it wraps
    `local_executor.pac_sign` and is passed to make_sealer (a PacSigner over the seal would be wrong)."""
    _, auth_specs, xpac_specs = build_pac_specs(generator)
    return PacSign(generator, auth_specs, xpac_specs), auth_specs


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

    def __init__(self, generator, trace_fn, assemble, primitives: Set[str], signer) -> None:
        self._walk = SandboxWalk(generator)
        self._trace_fn = trace_fn
        self._assemble = assemble
        self._primitives = frozenset(primitives)
        self._signer = signer
        self._enc, self._auth_specs = _pac_encoder(generator) if "pac" in self._primitives \
            else (None, None)

    def seal(self, test_case: TestCase) -> SealedTestCase:
        tc = copy.deepcopy(test_case)
        sandbox, data_sites = self._walk.sandbox(tc)
        if self._primitives == frozenset({"mte"}):
            return MteSealedTestCase(tc, self._trace_fn, self._assemble, sandbox, data_sites)
        if self._primitives == frozenset({"pac"}):
            return PacSealedTestCase(tc, self._trace_fn, self._assemble, sandbox, data_sites,
                                     self._signer, self._enc, self._auth_specs)
        return MtePacSealedTestCase(tc, self._trace_fn, self._assemble, sandbox, data_sites,
                                    self._signer, self._enc, self._auth_specs)


def _insert(bb, anchor, *groups) -> None:
    """Insert each group of instructions before `anchor`, in order — so they land in group order
    immediately before it (clamp, then value sealings, then the index/displacement cancellation)."""
    for group in groups:
        for inst in group:
            bb.insert_before(anchor, inst)


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
        self._addr = _Addressing(generator.target_desc.reg_normalized,
                                 f"#0x{_SANDBOX_MASK:x}", SANDBOX_BASE_REGISTER)
        self._mask = f"#0x{_SANDBOX_MASK:x}"
        self._stg_mask = f"#0x{_SANDBOX_MASK & ~0xF:x}"

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
                                data_sites.append((inst, bb, mem_reg, offset_subs))
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


def make_sealer(generator, trace_fn, assemble, primitives, signer) -> Sealer:
    """The Sealer for the active primitives. `trace_fn(tc, input) -> cer` runs a CE trace; `assemble(tc)
    -> bytes` assembles the object code; `signer` is the kernel PAC signer (a PacSigner over
    local_executor.pac_sign), used by resolve when 'pac' is active (None otherwise)."""
    prims = frozenset(primitives)
    if prims not in (frozenset({"mte"}), frozenset({"pac"}), frozenset({"pac", "mte"})):
        raise ValueError(f"unsupported seal primitives: {primitives!r}")
    return Sealer(generator, trace_fn, assemble, prims, signer)
