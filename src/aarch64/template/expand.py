"""Fill template holes with random instructions under their constraints.

Reuses the generator's own pools and `_pick_random_instruction_spec` / `generate_instruction` -- the
same primitives the legacy `.random_instructions` expansion uses -- and adds only the constraint layer
(pool by `Kind`, register whitelist).
"""
from __future__ import annotations

import random
from typing import List

from ...config import CONF
from ...interfaces import Instruction, TestCase, RegisterOperand
from .objects import Hole, Kind, Reg
from .builder import HoleInstruction, resolve_count


def _default_mem_ratio() -> float:
    """The generator's normal memory-access density, so an ANY hole matches ordinary random generation."""
    return CONF.avg_mem_accesses / CONF.program_size if CONF.program_size else 0.0


def _pool_for(gen, kind: Kind) -> List:
    if kind == Kind.LOAD:
        return gen.load_instruction
    if kind == Kind.STORE:
        return gen.store_instructions
    if kind == Kind.ALU:
        return gen.non_memory_access_instructions
    return []  # ANY -> the generator's mixed picker


def _pick(gen, hole: Hole) -> Instruction:
    if hole.kind == Kind.ANY:
        spec = gen._pick_random_instruction_spec(
            gen.non_memory_access_instructions, gen.store_instructions, gen.load_instruction,
            _default_mem_ratio())
    else:
        pool = _pool_for(gen, hole.kind)
        if not pool:
            raise ValueError(f"template: no instructions available for hole kind {hole.kind.value}")
        spec = random.choice(pool)
    return gen.generate_instruction(spec)


def _constrain_regs(gen, inst: Instruction, regs: List[Reg]) -> None:
    # Retarget top-level GPR operands to the whitelisted register numbers, in each operand's own width.
    # A memory base is left as-is (it is the sandbox base); sp/zr/SIMD are not GPRs and stay.
    allowed = {w: set(gen.target_desc.registers.get(w, ())) for w in (32, 64)}
    numbers = [r.number for r in regs if r.number is not None]
    candidates = {w: [n for n in (f"{'x' if w == 64 else 'w'}{k}" for k in numbers) if n in allowed[w]]
                  for w in (32, 64)}
    if not candidates[32] and not candidates[64]:
        raise ValueError("Hole.regs: no whitelisted register is allowed by the target")

    for op in inst.operands:
        w = op.get_width()
        if isinstance(op, RegisterOperand) and op.value in allowed.get(w, ()) and candidates[w]:
            op.value = random.choice(candidates[w])


def expand_holes(gen, test_case: TestCase) -> None:
    holes = [(inst, bb) for func in test_case.functions for bb in func for inst in bb
             if isinstance(inst, HoleInstruction)]

    for marker, bb in holes:
        hole = marker.hole
        predecessor = marker.previous
        for _ in range(resolve_count(hole.n)):
            inst = _pick(gen, hole)
            if hole.regs:
                _constrain_regs(gen, inst, hole.regs)
            if predecessor:
                bb.insert_after(predecessor, inst)
            else:
                bb.insert_before(bb.get_first(), inst)
            predecessor = inst
        bb.delete(marker)
