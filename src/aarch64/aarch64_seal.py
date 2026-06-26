"""Generic non-interference *seal* framework (AArch64).

A *seal* commits to a value's identity, not to where the value lives. The engine mints an
all-genuine baseline test case and decoy variants that diverge only on speculative paths (the NI
invariant). This module knows the AArch64 NOP / sandbox-clamp encodings and the sandbox base
register, but nothing about PAC or MTE — concrete seals live in aarch64_pac.py / aarch64_mte.py.
"""
import abc
import copy
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional, Set, Tuple

from ..interfaces import (TestCase, Instruction, BasicBlock, Function, GeneratorException,
                          MemoryOperand, OT, MAIN_AREA_SIZE, FAULTY_AREA_SIZE)
from .aarch64_target_desc import SANDBOX_BASE_REGISTER, AArch64MemRole

_SANDBOX_MASK_BITS = int(math.log(MAIN_AREA_SIZE + FAULTY_AREA_SIZE, 2))
_SANDBOX_MASK = (1 << _SANDBOX_MASK_BITS) - 1


# A slot is a short, fixed-width run of instructions, addressed by position — (func, bb,
# instruction) index within a TestCase. Filling a slot replaces each instruction one-for-one, so a
# position recorded against the sealed TC stays valid in every structural copy.
SlotLoc = Tuple[int, int, int]  # (func_idx, bb_idx, inst_idx)


def make_nop() -> Instruction:
    return Instruction("nop", True, "", False, template="NOP")


def index_instructions(tc: TestCase) -> Dict[int, SlotLoc]:
    """id(inst) -> position, to translate the sealed TC's slot instructions into positions (once)."""
    return {id(inst): (fi, bi, ii)
            for fi, func in enumerate(tc.functions)
            for bi, bb in enumerate(func)
            for ii, inst in enumerate(bb)}


def inst_at(tc: TestCase, loc: SlotLoc) -> Tuple[Instruction, BasicBlock]:
    fi, bi, ii = loc
    bb = tc.functions[fi][bi]
    for k, inst in enumerate(bb):
        if k == ii:
            return inst, bb
    raise GeneratorException(f"slot location {loc} out of range for the test case")


def fill_slot_at(tc: TestCase, slot_locs: List[SlotLoc], new_insts: List[Instruction]) -> None:
    """Replace the instructions at slot_locs (in order) with new_insts, padding short fills with
    NOPs. One-for-one, so all other positions stay valid. A fill longer than the slot is a seal bug
    (it would silently drop the overflow), so reject it rather than truncate."""
    assert len(new_insts) <= len(slot_locs), \
        f"fill of {len(new_insts)} insts exceeds slot size {len(slot_locs)}"
    for pos, loc in enumerate(slot_locs):
        old_inst, bb = inst_at(tc, loc)
        new_inst = new_insts[pos] if pos < len(new_insts) else make_nop()
        bb.insert_before(old_inst, new_inst)
        bb.delete(old_inst)


@dataclass
class FixPoint:
    """A sealing point: the slot the engine can fill plus the data it reads. Generic across seals —
    seal-specific values live on subclasses."""
    slot_id: int
    slot_insts: List[Instruction] = field(default_factory=list)  # placeholder insts in the sealed TC
    slot_locs: List[SlotLoc] = field(default_factory=list)       # their positions, for structural copies
    spec_nesting: Optional[int] = None                           # min speculation depth the slot ran at
    value_reg: str = ""                                          # register holding the committed value here
    seal: Optional["Seal"] = None                                # this slot's seal (set by the sealing pass)

    def reset(self) -> None:
        self.spec_nesting = None


class Seal(abc.ABC):
    """One composable protection the engine can stack onto a committed value.

    A seal commits to a *value*, not to where it lives: the value may move between registers and
    memory and be recomputed — only its identity is sealed. A seal owns just the encoding of its
    slot: the arch-safe placeholder at the sealing point, and the genuine / decoy fills at the use
    site (acting on whatever register holds the value there). It walks no CFG and knows nothing
    about speculation, how its values were resolved, hardware, or other seals — the engine drives
    it, and any notion of *which kind* of decoy a seal emits stays private to the subclass.
    `slot_size` is the slot's fixed instruction count.
    """
    name: str
    slot_size: int

    @abc.abstractmethod
    def placeholder(self, fp) -> List[Instruction]:
        """The arch-safe slot inserted at the sealing point — no behaviour change yet."""

    @abc.abstractmethod
    def genuine(self, fp, rng: random.Random) -> List[Instruction]:
        """The use site resolved to the genuine (committed) value; architecturally safe even when
        the genuine value is unavailable (falls back to the placeholder). `rng` lets a seal vary
        between equivalent genuine encodings (e.g. PAC occasionally stripping instead of
        re-authenticating); seals with a single genuine form ignore it."""

    @abc.abstractmethod
    def decoy(self, fp, rng: random.Random) -> List[Instruction]:
        """The use site resolved to an alternative (decoy) value. Called once per instance, so a
        seal able to mint fresh values returns a different one on each call."""

    def fill(self, fp, rng: random.Random, should_decoy: "DecoyPolicy") -> List[Instruction]:
        """Genuine or decoy for this fix point, per should_decoy(fp, self). Composites override this
        to decoy a subset of their members."""
        return self.decoy(fp, rng) if should_decoy(fp, self) else self.genuine(fp, rng)


class _SandboxInstrumentationBase:
    """Shared helpers for sandbox-taint-based instrumentation passes."""

    _norm: Dict[str, str]
    _sandbox_mask: str
    _sandbox_base_reg: str

    def _norm_reg(self, reg: str) -> str:
        return self._norm.get(reg, reg)

    def _dest_regs(self, inst: Instruction) -> frozenset:
        result = {self._norm_reg(op.value) for op in inst.operands + inst.implicit_operands
                  if op.dest and op.type == OT.REG and op.value in self._norm}
        # writeback (pre/post-index) updates the base register; the extractor marks that on the inner
        # base component (component.dest), so read it directly instead of guessing from operand names.
        for mem_op in inst.get_mem_operands():
            for c in mem_op.inner:
                if c.dest and c.type == OT.REG and c.value in self._norm:
                    result.add(self._norm_reg(c.value))
        return frozenset(result)

    @staticmethod
    def _base_reg(mem_op: MemoryOperand) -> str:
        for c in mem_op.inner:
            if c.mem_role is AArch64MemRole.BASE:
                return c.value
        raise GeneratorException(f"memory operand {mem_op.value!r} has no base register")

    def _get_mem_base_reg(self, inst: Instruction) -> Optional[str]:
        mem_ops = inst.get_mem_operands()
        return self._base_reg(mem_ops[0]) if mem_ops else None

    def _make_offset_sub_insts(self, mem_op: MemoryOperand) -> List[Instruction]:
        """SUBs that remove every non-base contribution to the address (an index register with its
        optional shift/extend, or a standalone displacement) from the base, so the effective address
        lands exactly at the base's already-sandboxed value. Components carry their roles."""
        base = self._base_reg(mem_op)
        comp = {c.mem_role: c for c in mem_op.inner}
        index = comp.get(AArch64MemRole.INDEX)
        offset = comp.get(AArch64MemRole.OFFSET)
        extend = comp.get(AArch64MemRole.EXTEND)
        if index is not None:
            # EA = base + <shift/extend>(index); replicate the modifier in the SUB so it cancels exactly
            if extend is not None:                          # extended register: e.g. `, sxtw #2`
                modifier = f", {extend.value}" + (f" #{offset.value}" if offset is not None else "")
            elif offset is not None:                        # shifted register: `, lsl #amount`
                modifier = f", lsl #{offset.value}"
            else:
                modifier = ""
            return [Instruction("sub", True, "", False,
                                template=f"SUB {base}, {base}, {index.value}{modifier}")]
        if offset is not None:                              # standalone displacement
            return self._make_disp_sub_insts(base, int(offset.value))
        return []

    def _make_disp_sub_insts(self, base: str, disp: int) -> List[Instruction]:
        """Cancel a signed displacement from base (chunked to the 12-bit immediate), so [base+disp]
        resolves to base. A negative displacement is cancelled with ADD."""
        mnemonic = "SUB" if disp >= 0 else "ADD"
        remaining, result = abs(disp), []
        while remaining > 0:
            chunk = min(4095, remaining)
            result.append(Instruction(mnemonic.lower(), True, "", False,
                                       template=f"{mnemonic} {base}, {base}, #{chunk}"))
            remaining -= chunk
        return result

    def _make_sandbox_insts(self, reg: str, align16: bool = False) -> List[Instruction]:
        """[AND reg, reg, #mask; ADD reg, reg, x29] sandboxing reg into the input region.
        align16 clears the low 4 bits too (STG-family tag stores require a 16-byte-aligned address)."""
        mask = self._sandbox_mask
        if align16:
            mask = f"#0x{(_SANDBOX_MASK & ~0xF):x}"
        and_inst = Instruction("and", True, "", False, template=f"AND {reg}, {reg}, {mask}")
        add_inst = Instruction("add", True, "", False, template=f"ADD {reg}, {reg}, {self._sandbox_base_reg}")
        return [and_inst, add_inst]

    @staticmethod
    def _is_tag_store(inst: Instruction) -> bool:
        """STG-family memory-tag stores (write a tag, not data; address must be 16-byte aligned)."""
        return inst.name.lower() in MTE_TAG_STORE_NAMES

    @staticmethod
    def _is_tag_load(inst: Instruction) -> bool:
        """LDG: reads a granule's allocation tag into a register. Touches memory (so the base must be
        clamped) but is not tag-checked and needs no alignment."""
        return inst.name.lower() in MTE_TAG_LOAD_NAMES

    @staticmethod
    def _topo_sort(func: Function) -> Tuple[Dict[BasicBlock, List[BasicBlock]], List[BasicBlock]]:
        """Return (predecessors, topo_order) for func's CFG."""
        predecessors: Dict[BasicBlock, List[BasicBlock]] = {}
        for bb in func:
            predecessors.setdefault(bb, [])
            for succ in bb.successors:
                predecessors.setdefault(succ, []).append(bb)
        topo: List[BasicBlock] = []
        seen: Set[BasicBlock] = set()
        on_stack: Set[BasicBlock] = set()
        def _dfs(bb: BasicBlock) -> None:
            if bb in seen:
                return
            if bb in on_stack:
                raise GeneratorException("CFG contains a cycle; MTE taint dataflow assumes a DAG")
            on_stack.add(bb)
            for succ in bb.successors:
                _dfs(succ)
            on_stack.discard(bb)
            seen.add(bb)
            topo.append(bb)
        _dfs(func.get_first_bb())
        topo.reverse()
        return predecessors, topo


# Decides, per (fix point, seal), whether that seal may be decoyed there — how the engine and
# CompositeSeal choose where decoys land (and let callers protect a seal, e.g. the sandbox).
DecoyPolicy = Callable[[FixPoint, Seal], bool]


def is_speculative(fp: FixPoint) -> bool:
    """True for a slot reached only speculatively, or never reached architecturally (None)."""
    return fp.spec_nesting != 0 if fp.spec_nesting is not None else True


def _default_decoy_policy(fp: FixPoint, seal: Seal) -> bool:
    return is_speculative(fp)


class SealedNIInstrumentation:
    """Non-interference engine over per-fix-point seals: seal once, then baseline() (all genuine)
    and decoys(rng) (an unbounded stream decoying the slots `should_decoy(fp, seal)` allows;
    default: the speculative ones). Decoys landing only on speculative slots is the NI invariant.
    Every fix point carries its own seal (`fp.seal`), so one test case can mix seals (PAC, MTE,
    composite) across slots — the engine is seal-agnostic, with no engine-wide default seal."""

    def __init__(self, should_decoy: Optional[DecoyPolicy] = None):
        self._should_decoy: DecoyPolicy = should_decoy or _default_decoy_policy
        self._sealed_tc: Optional[TestCase] = None
        self._fix_points: List[FixPoint] = []

    def set_sealed(self, sealed_tc: TestCase, fix_points: List[FixPoint]) -> None:
        # Re-derive each slot's positions over the final sealed TC, so fix points produced by
        # several passes (PAC AUT* sites + MTE accesses) all resolve correctly after every pass's
        # structural edits. Idempotent for a single pass.
        locs = index_instructions(sealed_tc)
        for fp in fix_points:
            fp.slot_locs = [locs[id(si)] for si in fp.slot_insts]
        self._sealed_tc = sealed_tc
        self._fix_points = fix_points

    def _fill(self, slot_fn) -> TestCase:
        assert self._sealed_tc is not None, "set_sealed() must be called before filling"
        tc = copy.deepcopy(self._sealed_tc)
        for fp in self._fix_points:
            fill_slot_at(tc, fp.slot_locs, slot_fn(fp))
        return tc

    @staticmethod
    def _seal_for(fp) -> Seal:
        """The seal to apply at this fix point — its own `fp.seal`, set by the instrumentation pass
        that created it (PAC slots carry PacSign; MTE slots carry MteTag or CompositeSeal[Sandbox,
        MteTag]). There is no engine-wide default, so a single engine can fill a mixed slot list."""
        assert fp.seal is not None, f"fix point {getattr(fp, 'slot_id', '?')} has no seal"
        return fp.seal

    def baseline(self, rng: random.Random) -> TestCase:
        return self._fill(lambda fp: self._seal_for(fp).genuine(fp, rng))

    @staticmethod
    def _decoyable_names(fp) -> List[str]:
        """The non-sandbox seal names this fix point can decoy (composite members or the lone seal)."""
        return [s.name for s in getattr(fp.seal, "seals", [fp.seal]) if s.name != "sandbox"]

    def decoys(self, rng: random.Random) -> Iterator[TestCase]:
        names = sorted({n for fp in self._fix_points for n in self._decoyable_names(fp)})
        while True:
            # Each decoy instance picks which primitives mismatch — PAC and MTE are orthogonal (any
            # subset, incl. neither, applied consistently across all slots so a leak is attributable);
            # with a single primitive present it is always the one decoyed.
            enabled = {n for n in names if rng.random() < 0.5} if len(names) > 1 else set(names)
            policy = lambda fp, s: self._should_decoy(fp, s) and s.name in enabled
            yield self._fill(lambda fp: self._seal_for(fp).fill(fp, rng, policy))


class Sandbox(Seal):
    """Sandbox seal: clamps the pointer in-bounds ([AND #mask, ADD x29]). The clamp always runs and
    is never decoyed — run with a policy that never decoys it; decoy() raises. `mask` chooses the
    clamp: the default region mask, or e.g. region mask & ~0xF for the 16-byte-aligned clamp that
    STG-family tag stores require."""
    name = "sandbox"
    slot_size = 2
    _base = SANDBOX_BASE_REGISTER

    def __init__(self, mask: int):
        self._mask = f"#0x{mask:x}"

    def _and(self, reg: str) -> Instruction:
        return Instruction("and", True, "", False, template=f"AND {reg}, {reg}, {self._mask}")

    def _add(self, reg: str) -> Instruction:
        return Instruction("add", True, "", False, template=f"ADD {reg}, {reg}, {self._base}")

    # ---- Seal protocol (stateless) ----
    def placeholder(self, fp) -> List[Instruction]:
        return [self._and(fp.value_reg), self._add(fp.value_reg)]

    def genuine(self, fp, rng: random.Random) -> List[Instruction]:
        return [self._and(fp.value_reg), self._add(fp.value_reg)]

    def decoy(self, fp, rng: random.Random) -> List[Instruction]:
        raise NotImplementedError("the sandbox clamp is never decoyed")


class CompositeSeal(Seal):
    """Stacks several seals onto one pointer: the slot is their slots concatenated. A decoy decoys a
    random non-empty subset of the eligible members, keeping the rest genuine. Members must accept
    the same fix point."""

    def __init__(self, seals):
        self.seals = list(seals)
        assert self.seals, "CompositeSeal needs at least one seal"
        self.name = "+".join(s.name for s in self.seals)
        self.slot_size = sum(s.slot_size for s in self.seals)

    def placeholder(self, fp) -> List[Instruction]:
        return [i for s in self.seals for i in s.placeholder(fp)]

    def genuine(self, fp, rng: random.Random) -> List[Instruction]:
        return [i for s in self.seals for i in s.genuine(fp, rng)]

    def decoy(self, fp, rng: random.Random) -> List[Instruction]:
        return self._fill_subset(fp, rng, list(range(len(self.seals))))

    def fill(self, fp, rng, should_decoy) -> List[Instruction]:
        eligible = [i for i, s in enumerate(self.seals) if should_decoy(fp, s)]
        return self._fill_subset(fp, rng, eligible)

    def _fill_subset(self, fp, rng, eligible) -> List[Instruction]:
        chosen = set()
        if eligible:
            subset = rng.randrange(1, 1 << len(eligible))
            chosen = {eligible[k] for k in range(len(eligible)) if subset & (1 << k)}
        out: List[Instruction] = []
        for i, s in enumerate(self.seals):
            out += s.decoy(fp, rng) if i in chosen else s.genuine(fp, rng)
        return out


# ===========================================================================
# MTE non-interference instrumentation
# ===========================================================================


# STG-family memory-tag stores: write a memory tag (not data), address must be 16-byte aligned.
MTE_TAG_STORE_NAMES = frozenset({"stg", "st2g", "stzg", "stz2g"})

# LDG reads a granule's allocation tag into a register: it touches memory (clamp the base) but is
# not tag-checked, so it gets no tag fix point and needs no 16-byte alignment.
MTE_TAG_LOAD_NAMES = frozenset({"ldg"})


class SealInstrumentation(_SandboxInstrumentationBase):
    """General memory-sealing pass over a taint walk. Seals each memory access with one slot:
    CompositeSeal([Sandbox] + value_seals) where the base is not yet clamped, the value_seals alone
    where it is. value_seals is the ordered list to compose; fixpoint_cls holds their per-input data.
    """

    def __init__(self, generator, value_seals: List[Seal], fixpoint_cls):
        self.generator = generator
        self._norm = generator.target_desc.reg_normalized
        self._sandbox_mask = f"#0x{_SANDBOX_MASK:x}"
        self._sandbox_base_reg = SANDBOX_BASE_REGISTER
        self._fixpoint_cls = fixpoint_cls
        self._sandbox = Sandbox(_SANDBOX_MASK)
        self._value_composite: Seal = value_seals[0] if len(value_seals) == 1 \
            else CompositeSeal(value_seals)
        self._composite = CompositeSeal([self._sandbox] + value_seals)
        # STG-family tag stores get a 16-byte-aligned clamp (no tag slot, never decoyed).
        self._stg_seal = Sandbox(_SANDBOX_MASK & ~0xF)
        self.last_taint_log: List[str] = []

    def make_engine(self, should_decoy=None) -> "SealedNIInstrumentation":
        """A non-interference engine; each fix point carries its own seal. Seal the test case with
        seal_test_case(), then feed the result to engine.set_sealed()."""
        return SealedNIInstrumentation(should_decoy)

    # ------------------------------------------------------------------
    # Taint helpers
    # ------------------------------------------------------------------

    def _address_preserving_tag_op(self, inst: Instruction) -> Optional[Tuple[str, str]]:
        """If inst derives its destination from its source register while leaving the address (the
        low 56 bits) unchanged — only the tag may differ — return (dest_reg, src_reg); else None.

        Such an op keeps an in-region pointer in-region, so the sandbox taint propagates across it
        (a changed tag is corrected later, at the access). Covers IRG (new random tag, same address)
        and ADDG/SUBG whose address offset is zero (imm6 == 0). A nonzero imm6 moves the address by
        up to 1008 bytes, which can leave the region, so taint must NOT propagate — the next access
        re-clamps. (imm4, the tag offset, never affects whether the address stays in-region.)"""
        name = inst.name.lower()
        if name == "irg" and len(inst.operands) >= 2:
            return inst.operands[0].value, inst.operands[1].value
        if name in ("addg", "subg"):
            if len(inst.operands) < 4:
                return None
            try:
                imm6 = int(inst.operands[2].value)   # address offset (scaled x16)
            except (ValueError, IndexError):
                return None
            if imm6 != 0:                            # address moved -> may leave the region
                return None
            return inst.operands[0].value, inst.operands[1].value
        return None

    # ------------------------------------------------------------------
    # Stage-1 dataflow pass
    # ------------------------------------------------------------------

    def _build_slots(
        self,
        func: Function,
        slot_counter: int,
        fix_points: List,
        insertions: List,
        taint_log: List,
    ) -> int:
        """Topological taint pass: build NOP-placeholder fix_points for every memory access.

        taint (curr) = frozenset of normalized register names that hold a
        correctly-sandbox-tagged address (via AND+ADD with x29).
        Cleared on any register write; intersection at CFG join nodes.
        Address-preserving tag ops (IRG, ADDG/SUBG with imm6==0) propagate src taint to dest.

        For each memory access:
          tainted base  → offset_subs + NOP placeholder only
          untainted base → AND+ADD + offset_subs + NOP placeholder; reg added to taint
        """
        predecessors, topo = self._topo_sort(func)
        taint_out: Dict[BasicBlock, frozenset] = {}

        for bb in topo:
            processed = [p for p in predecessors.get(bb, []) if p in taint_out]
            if not processed:
                curr: frozenset = frozenset()
            elif len(processed) == 1:
                curr = taint_out[processed[0]]
            else:
                curr = taint_out[processed[0]]
                for p in processed[1:]:
                    curr = curr & taint_out[p]

            for inst in bb:
                if inst.has_memory_access:
                    if len(inst.get_mem_operands()) > 1:
                        raise GeneratorException(
                            "the memory seal models one memory access per instruction; "
                            f"{inst.name!r} has several")
                    mem_reg = self._get_mem_base_reg(inst)
                    if mem_reg is not None:
                        norm_mem = self._norm_reg(mem_reg)
                        offset_subs = self._make_offset_sub_insts(inst.get_mem_operands()[0])
                        # offset_subs permanently rewrites the base to (sandboxed - offset), so the
                        # base is no longer at its sandboxed value and must not stay tainted.
                        modifies_base = bool(offset_subs)
                        if self._is_tag_store(inst):              # STG-family: 16B-aligned clamp only
                            clamp = self._stg_seal.genuine(FixPoint(slot_id=-1, value_reg=mem_reg),
                                                            random.Random(0))  # Sandbox ignores rng
                            insertions.append((inst, bb, clamp, offset_subs, []))
                            curr = curr - frozenset([norm_mem])   # the clamp rewrote the base
                            decision = "STG-CLAMP"
                        elif self._is_tag_load(inst):             # LDG: clamp the base, no tag seal
                            # LDG is not tag-checked, so a tag decoy could not leak — clamp only, no
                            # fix point. Taint bookkeeping mirrors a data access (minus the tag slot).
                            if norm_mem in curr:                  # already in-region -> no clamp
                                insertions.append((inst, bb, [], offset_subs, []))
                                if modifies_base:
                                    curr = curr - frozenset([norm_mem])
                            else:
                                clamp = self._sandbox.genuine(FixPoint(slot_id=-1, value_reg=mem_reg),
                                                               random.Random(0))  # Sandbox ignores rng
                                insertions.append((inst, bb, clamp, offset_subs, []))
                                if not modifies_base:
                                    curr = curr | frozenset([norm_mem])
                            decision = "LDG-CLAMP"
                        else:
                            sid = slot_counter
                            slot_counter += 1
                            fp = self._fixpoint_cls(slot_id=sid, value_reg=mem_reg)
                            if hasattr(fp, "access_inst"):        # fps that classify by the access record it
                                fp.access_inst = inst
                            if norm_mem in curr:                  # base already clamped -> tag only
                                fp.seal = self._value_composite
                                fp.slot_insts = fp.seal.placeholder(fp)        # [tag]
                                insertions.append((inst, bb, [], offset_subs, fp.slot_insts))
                                decision = "TAG-ONLY"
                                if modifies_base:
                                    curr = curr - frozenset([norm_mem])
                            else:                                 # first use -> sandbox + tag
                                fp.seal = self._composite
                                fp.slot_insts = fp.seal.placeholder(fp)        # [AND, ADD, tag]
                                k = self._sandbox.slot_size
                                insertions.append((inst, bb, fp.slot_insts[:k], offset_subs,
                                                   fp.slot_insts[k:]))
                                decision = "SANDBOX+TAG"
                                if not modifies_base:
                                    curr = curr | frozenset([norm_mem])
                            fix_points.append(fp)
                        taint_log.append(
                            f"  MEM-ACCESS   inst={inst.name:12s}  base={mem_reg}"
                            f"  decision={decision}  taint={sorted(curr)}")
                    else:
                        taint_log.append(
                            f"  MEM-ACCESS   inst={inst.name:12s}  base=None(implicit?)"
                            f"  decision=SKIP")

                prop = self._address_preserving_tag_op(inst)
                for dreg in self._dest_regs(inst):
                    if prop is not None and dreg == self._norm_reg(prop[0]):
                        norm_src = self._norm_reg(prop[1])
                        if norm_src in curr:
                            curr = curr | frozenset([dreg])
                            taint_log.append(
                                f"  TAG-PRESERVE inst={inst.name:12s}  {norm_src}->{dreg}")
                        else:
                            if dreg in curr:
                                taint_log.append(
                                    f"  ARITH-CLEAR  inst={inst.name:12s}  clears={dreg}")
                            curr = curr - frozenset([dreg])
                    else:
                        if dreg in curr:
                            taint_log.append(
                                f"  ARITH-CLEAR  inst={inst.name:12s}  clears={dreg}")
                        curr = curr - frozenset([dreg])

            taint_out[bb] = curr

        return slot_counter

    # ------------------------------------------------------------------
    # Sealing pass
    # ------------------------------------------------------------------

    def seal_test_case(self, tc: TestCase) -> Tuple[TestCase, List[FixPoint]]:
        """Seal tc IN PLACE (the caller owns copying): insert a slot before every memory access
        (sandboxing untainted base registers first). Returns (tc, fix_points)."""
        fix_points: List[FixPoint] = []
        slot_counter = 0
        self.last_taint_log = []

        for func in tc.functions:
            insertions: List = []
            func_log: List[str] = []
            slot_counter = self._build_slots(
                func, slot_counter, fix_points, insertions, func_log)
            self.last_taint_log.extend(func_log)

            # Per access: clamp the base into the region, then the value-seals act on that canonical
            # base, then the offset subtraction (cancelling the index/displacement so the access lands
            # back at the clamped base) is emitted last — the two address adjustments bookend the
            # value-seals. The subtraction must come after the seals: applied first it would leave a
            # non-canonical base (clamped - offset) for a seal to act on, and the effective address
            # could then fall outside the region.
            for mem_inst, bb, clamp_insts, offset_subs, value_seal_insts in insertions:
                for s in clamp_insts:
                    bb.insert_before(mem_inst, s)
                for s in value_seal_insts:
                    bb.insert_before(mem_inst, s)
                for s in offset_subs:
                    bb.insert_before(mem_inst, s)

        # Record each slot's position so the fills can locate it in any structural copy.
        locs = index_instructions(tc)
        for fp in fix_points:
            fp.slot_locs = [locs[id(si)] for si in fp.slot_insts]

        return tc, fix_points
