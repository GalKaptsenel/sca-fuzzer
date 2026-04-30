"""
Converter from parser-internal types to Revizor's native RT types.

Usage
-----
    from arm_isa_parser.converter import to_rt_instructions
    from arm_isa_parser import Downloader

    instructions = Downloader(["general"], "base.json").run()
    rt_instructions = to_rt_instructions(instructions)
"""
from __future__ import annotations

from typing import List

from ...interfaces import InstructionSpec as RT_InstructionSpec
from ...interfaces import OperandSpec as RT_OperandSpec
from ...interfaces import OT

from .models import InstructionSpec, OperandSpec


# ---------------------------------------------------------------------------
# Operand type mapping
# ---------------------------------------------------------------------------

_TYPE_MAP: dict[str, OT] = {
    "REG":   OT.REG,
    "MEM":   OT.MEM,
    "IMM":   OT.IMM,
    "LABEL": OT.LABEL,
    "FLAGS": OT.FLAGS,
    "COND":  OT.COND,
}


def _convert_operand(op: OperandSpec) -> RT_OperandSpec:
    ot = _TYPE_MAP.get(op.type_, OT.IMM)
    return RT_OperandSpec(
        type=ot,
        width=op.width,
        signed=op.signed,
        src=op.src,
        dest=op.dest,
        values=tuple(op.values),
        name=op.name,
    )


def _convert_instruction(inst: InstructionSpec) -> RT_InstructionSpec:
    return RT_InstructionSpec(
        name=inst.name,
        category=inst.category,
        control_flow=inst.control_flow,
        datatype=inst.datatype,
        template=inst.template,
        operands=tuple(_convert_operand(op) for op in inst.operands),
        implicit_operands=tuple(_convert_operand(op) for op in inst.implicit_operands),
        tags=tuple(inst.tags),
    )


def to_rt_instructions(instructions: List[InstructionSpec]) -> List[RT_InstructionSpec]:
    """Convert a list of parser InstructionSpec to Revizor RT_InstructionSpec."""
    return [_convert_instruction(inst) for inst in instructions]
