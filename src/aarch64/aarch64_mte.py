"""MTE allocation-tag modelling used while parsing the contract-executor trace.

MteTagState tracks the allocation tag of each memory granule across speculation. It is a stack of
layers indexed by speculation depth: layer 0 is architectural; entering deeper speculation copies
the current top, and unwinding pops it — that pop is the revert of the speculative tag stores. It
starts uniform (the region's initial tag); per-cell initial tags can be pre-seeded for dynamic
tagging.
"""
from typing import Dict, List, Optional, Tuple

from .aarch64_disasm import disassemble_instruction
import copy
import random
from dataclasses import dataclass

from ..interfaces import TestCase, Instruction, BasicBlock, Function, GeneratorException
from .aarch64_target_desc import SANDBOX_BASE_REGISTER
from .aarch64_seal import (Seal, FixPoint, SealedNIInstrumentation, Sandbox, CompositeSeal,
                           _SandboxInstrumentationBase, make_nop, index_instructions, _SANDBOX_MASK)
from .aarch64_generator import Aarch64Generator

MTE_GRANULE = 16  # bytes covered by one allocation tag

# STG-family memory-tag stores -> number of granules each tags.
_MTE_TAG_STORES = {"stg": 1, "stzg": 1, "st2g": 2, "stz2g": 2}


class MteTagState:
    def __init__(self, default_tag: int):
        self._default = default_tag & 0xF
        self._stack: List[Dict[int, int]] = [{}]  # one tag-override layer per live speculation depth

    @staticmethod
    def granule(addr: int) -> int:
        return (addr & ((1 << 56) - 1)) & ~(MTE_GRANULE - 1)  # drop the tag byte and in-granule offset

    def to_depth(self, nesting: int) -> None:
        """Track the current speculation depth: grow by copying the top (speculation inherits the
        live state), shrink by popping (reverting the deeper levels' speculative stores)."""
        while len(self._stack) <= nesting:
            self._stack.append(dict(self._stack[-1]))
        del self._stack[nesting + 1:]

    def set(self, addr: int, tag: int, n_granules: int = 1) -> None:
        """Tag granules in the current (deepest live) layer."""
        layer, g = self._stack[-1], self.granule(addr)
        for i in range(n_granules):
            layer[g + i * MTE_GRANULE] = tag & 0xF

    def tag_at(self, addr: int) -> int:
        """The tag visible at the current speculation depth (architectural + live speculative stores)."""
        return self._stack[-1].get(self.granule(addr), self._default)


def _reg_value(cpu, name: str) -> int:
    name = name.lower()
    if name == "sp":
        return cpu.sp
    if name.startswith("x") and name[1:].isdigit():
        return cpu.gpr[int(name[1:])]
    return 0


def mte_tag_store_effect(ite) -> Optional[Tuple[int, int, int]]:
    """If ite is an STG-family tag store, return (addr, tag, n_granules); else None. The tag written
    is the logical tag of the first operand Xt (STG Xt, [Xn] tags [Xn..] with Xt's tag)."""
    if not ite.metadata.has_memory_access:
        return None
    parts = (disassemble_instruction(ite.cpu.encoding, ite.cpu.pc) or "").split()
    n = _MTE_TAG_STORES.get(parts[0].lower()) if parts else None
    if n is None:
        return None
    ea = ite.metadata.memory_access.effective_address
    tag = (_reg_value(ite.cpu, parts[1].rstrip(",")) >> 56) & 0xF if len(parts) > 1 else (ea >> 56) & 0xF
    return ea, tag, n


# ===========================================================================
# MTE non-interference seal + sealing pass
# ===========================================================================

MTE_SLOT_SIZE = 1  # one slot instruction per memory access

class MteTag(Seal):
    """MTE allocation-tag seal.  Slot = a single instruction.

    Genuine is a NOP — the sandboxed pointer already carries the region's (correct) tag. A decoy
    retags it: a hardware-random tag (IRG) or a hardcoded one (EOR a tag-field mask). Both touch
    only the tag bits [59:56], leaving the address unchanged, and assume no fixed region tag.
    """
    name = "mte_tag"
    slot_size = MTE_SLOT_SIZE

    # EOR masks confined to the tag field [59:56] (each a valid AArch64 logical immediate); a
    # nonzero one yields a different tag while leaving the address bits untouched.
    _TAG_FLIP_MASKS = (0x1 << 56, 0x2 << 56, 0x4 << 56, 0x8 << 56, 0xF << 56)

    def _irg(self, reg: str) -> Instruction:
        return Instruction("irg", True, "", False, template=f"IRG {reg}, {reg}")

    def _retag(self, reg: str, mask: int) -> Instruction:
        return Instruction("eor", True, "", False, template=f"EOR {reg}, {reg}, #0x{mask:016x}")

    def _addg(self, reg: str, tag_delta: int) -> Instruction:
        # add a 4-bit offset to the pointer's tag, address offset 0 (a clean mod-16 add when
        # GCR_EL1.Exclude == 0); touches only the tag field.
        return Instruction("addg", True, "", False, template=f"ADDG {reg}, {reg}, #0, #{tag_delta}")

    # ---- Seal protocol (stateless; everything it needs is on the fix point) ----
    def placeholder(self, fp) -> List[Instruction]:
        return [make_nop()]

    def genuine(self, fp, rng: random.Random) -> List[Instruction]:
        # The arch path must carry the cell's tag; fix the pointer's tag only when it mismatches.
        if fp.correct_tag is None or fp.ptr_tag is None:
            return [make_nop()]
        delta = (fp.correct_tag - fp.ptr_tag) % 16
        return [make_nop()] if delta == 0 else [self._addg(fp.value_reg, delta)]

    def decoy(self, fp, rng: random.Random) -> List[Instruction]:
        if rng.random() < 0.5:
            return [self._irg(fp.value_reg)]
        return [self._retag(fp.value_reg, rng.choice(self._TAG_FLIP_MASKS))]


@dataclass
class MTEFixPoint(FixPoint):
    # Per-input, from the sealing trace (reset between inputs):
    correct_tag: Optional[int] = None  # the accessed cell's allocation tag (from MteTagState)
    ptr_tag: Optional[int] = None       # the tag the pointer itself carries (top byte of the EA)

    def reset(self) -> None:
        super().reset()
        self.correct_tag = None
        self.ptr_tag = None


class SealInstrumentation(_SandboxInstrumentationBase):
    """General memory-sealing pass over a taint walk. Seals each memory access with one slot:
    CompositeSeal([Sandbox] + value_seals) where the base is not yet clamped, the value_seals alone
    where it is. value_seals is the ordered list to compose; fixpoint_cls holds their per-input data.
    """

    def __init__(self, generator: Aarch64Generator, value_seals: List[Seal], fixpoint_cls):
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

    def _build_mte_slots(
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
                            "MTE instrumentation models one memory access per instruction; "
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

    def seal_test_case(self, tc: TestCase) -> Tuple[TestCase, List[MTEFixPoint]]:
        """Seal tc IN PLACE (the caller owns copying): insert a slot before every memory access
        (sandboxing untainted base registers first). Returns (tc, fix_points)."""
        fix_points: List[MTEFixPoint] = []
        slot_counter = 0
        self.last_taint_log = []

        for func in tc.functions:
            insertions: List = []
            func_log: List[str] = []
            slot_counter = self._build_mte_slots(
                func, slot_counter, fix_points, insertions, func_log)
            self.last_taint_log.extend(func_log)

            for mem_inst, bb, sandbox_insts, offset_subs, nop_insts in insertions:
                for s in sandbox_insts:
                    bb.insert_before(mem_inst, s)
                for s in offset_subs:
                    bb.insert_before(mem_inst, s)
                for s in nop_insts:
                    bb.insert_before(mem_inst, s)

        # Record each slot's position so the fills can locate it in any structural copy.
        locs = index_instructions(tc)
        for fp in fix_points:
            fp.slot_locs = [locs[id(si)] for si in fp.slot_insts]

        return tc, fix_points
