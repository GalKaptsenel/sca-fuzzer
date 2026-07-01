"""Shared AArch64 sealing primitives: slot helpers + the sandbox-clamp instrumentation base.

Knows the AArch64 NOP / sandbox-clamp encodings, slot addressing (a slot is a fixed-width run of
instructions filled by position), and the offset-cancelling SUBs that pin an effective address to
its already-clamped base. The concrete sealings + per-input resolution live in sealer.py.
"""
import copy
import math
from typing import Dict, List, Optional, Set, Tuple

from ...interfaces import (TestCase, Instruction, BasicBlock, Function, GeneratorException,
                          MemoryOperand, OT, MAIN_AREA_SIZE, FAULTY_AREA_SIZE)
from ..aarch64_target_desc import AArch64MemRole

_SANDBOX_MASK_BITS = int(math.log(MAIN_AREA_SIZE + FAULTY_AREA_SIZE, 2))
_SANDBOX_MASK = (1 << _SANDBOX_MASK_BITS) - 1


# A slot is a short, fixed-width run of instructions, addressed by position — (func, bb,
# instruction) index within a TestCase. Filling a slot replaces each instruction one-for-one, so a
# position recorded against the sealed TC stays valid in every structural copy.
SlotLoc = Tuple[int, int, int]  # (func_idx, bb_idx, inst_idx)

# STG-family memory-tag stores: write a memory tag (not data), address must be 16-byte aligned.
MTE_TAG_STORE_NAMES = frozenset({"stg", "st2g", "stzg", "stz2g"})
# LDG reads a granule's allocation tag into a register: it touches memory, so its base is clamped.
MTE_TAG_LOAD_NAMES = frozenset({"ldg"})


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
