"""
Data models for ARM ISA instruction and operand specifications.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List


@dataclass
class OperandSpec:
    name: str
    type_: str       # REG | IMM | MEM | COND | FLAGS | LABEL
    values: List[str]
    src: bool
    dest: bool
    width: int
    signed: bool

    def to_dict(self) -> dict:
        if len(self.values) < 5 and self.type_ == "FLAGS":
            import pdb; pdb.set_trace()

        return {
            "name": self.name,
            "type_": self.type_,
            "values": [v.lower() for v in self.values],
            "src": self.src,
            "dest": self.dest,
            "width": self.width,
            "signed": self.signed,
        }

    def structural_key(self) -> tuple:
        """Identity key that ignores name and values — used for merging."""
        return (self.type_, self.src, self.dest, self.width, self.signed)


@dataclass
class InstructionSpec:
    name: str = ""
    category: str = ""
    control_flow: bool = False
    datatype: str = ""
    template: str = ""
    operands: List[OperandSpec] = field(default_factory=list)
    implicit_operands: List[OperandSpec] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"{self.name} cf={self.control_flow} cat={self.category} "
            f"ops={len(self.operands)} implicit={len(self.implicit_operands)}"
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name.lower(),
            "category": self.category,
            "control_flow": self.control_flow,
            "operands": [o.to_dict() for o in self.operands],
            "implicit_operands": [o.to_dict() for o in self.implicit_operands],
            "template": self.template,
            "tags": self.tags,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def merge_key(self) -> tuple:
        """
        Two specs with the same merge_key are considered the same instruction
        and will have their operand value sets unioned.
        Operand position + structural shape must match; names and values may differ.
        """
        return (
            self.name.lower(),
            self.category,
            self.control_flow,
            tuple(op.structural_key() for op in self.operands),
            tuple(op.structural_key() for op in self.implicit_operands),
        )
