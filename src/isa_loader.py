"""
File:

Copyright (C) Microsoft Corporation
SPDX-License-Identifier: MIT
"""
import json
from typing import Dict, List
from copy import deepcopy
from .interfaces import OT, InstructionSetAbstract, OperandSpec, MemorySpec, InstructionSpec, MemoryRole
from .config import CONF


class InstructionSet(InstructionSetAbstract):
    ot_str_to_enum = {
        "REG": OT.REG,
        "MEM": OT.MEM,
        "IMM": OT.IMM,
        "LABEL": OT.LABEL,
        "AGEN": OT.AGEN,
        "FLAGS": OT.FLAGS,
        "COND": OT.COND,
    }
    instructions: List[InstructionSpec]
    instruction_unfiltered: List[InstructionSpec]

    def __init__(self, filename: str, include_categories=None):
        self.instructions: List[InstructionSpec] = []
        self.init_from_file(filename)
        self.instruction_unfiltered = deepcopy(self.instructions)
        self.reduce(include_categories)
        self.dedup()

    def init_from_file(self, filename: str):
        with open(filename, "r") as f:
            root = json.load(f)
        for instruction_node in root:
            instruction = InstructionSpec()
            instruction.name = instruction_node["name"]
            instruction.category = instruction_node["category"]
            instruction.control_flow = instruction_node["control_flow"]
            instruction.template = instruction_node.get("template", None)
            # tags default to the single category (x86 stores its tag there); optional alias constraints
            instruction.tags = tuple(instruction_node.get("tags") or [instruction.category])
            if "constraints" in instruction_node:
                instruction.constraints = tuple(tuple(c) for c in instruction_node["constraints"])

            for op_node in instruction_node["operands"]:
                op = self.parse_operand(op_node, instruction)
                instruction.operands.append(op)
                if op.magic_value:
                    instruction.has_magic_value = True

            for op_node in instruction_node["implicit_operands"]:
                op = self.parse_operand(op_node, instruction)
                instruction.implicit_operands.append(op)

            self.instructions.append(instruction)

    def parse_operand(self, op: Dict, parent: InstructionSpec) -> OperandSpec:
        op_type = self.ot_str_to_enum[op["type_"]]
        op_name = op.get("name", "")

        if op_type == OT.MEM:
            if "inner" in op:                         # decomposed addressing (AArch64): wrap components
                inner = [self.parse_operand(c, parent) for c in op["inner"]]
                spec = MemorySpec(op["width"], op.get("signed", True), op["src"], op["dest"], inner,
                                  op_name)
            else:                                     # combined address string (x86)
                spec = OperandSpec(op_type, op["width"], op.get("signed", True), op["src"], op["dest"],
                                   op.get("values", []), op_name)
            parent.has_mem_operand = True
            if spec.dest:
                parent.has_write = True
            return spec

        op_values = op.get("values", [])
        if op_type == OT.REG:
            op_values = sorted(op_values)
        spec = OperandSpec(op_type, op["width"], op.get("signed", True), op["src"], op["dest"], op_values, op_name)
        if "mem_role" in op:              # only address components carry a role; others stay None
            spec.mem_role = self._mem_role(op["mem_role"])
        return spec

    @staticmethod
    def _mem_role(value) -> MemoryRole:
        """Resolve a serialized role to this architecture's MemoryRole subtype."""
        if "aarch64" in CONF.instruction_set:
            from .aarch64.aarch64_target_desc import AArch64MemRole
            return AArch64MemRole(value)
        raise NotImplementedError(f"no MemoryRole subtype for instruction set {CONF.instruction_set!r}")

    @staticmethod
    def _value_specs(specs):
        """Yield the value-bearing operand specs, descending into a memory access's address components
        (the base/index registers live inside the MemorySpec, not as top-level operands)."""
        for op in specs:
            if isinstance(op, MemorySpec):
                yield from op.inner
            else:
                yield op

    @classmethod
    def _operands_equal(cls, op1: OperandSpec, op2: OperandSpec) -> bool:
        """Structural equality for dedup: a memory access also matches on its components, so two
        different addressing forms (e.g. `[Xn]` vs `[Xn, Xm]`) are not collapsed."""
        if op1.type != op2.type:
            return False
        if isinstance(op1, MemorySpec) or isinstance(op2, MemorySpec):
            # both must be memory accesses with matching components. Components keep a fixed template
            # order (base, index, offset, ...), so a positional comparison is well-defined.
            return (isinstance(op1, MemorySpec) and isinstance(op2, MemorySpec)
                    and len(op1.inner) == len(op2.inner)
                    and all(cls._operands_equal(a, b) for a, b in zip(op1.inner, op2.inner)))
        # value choices are a set, not an ordered sequence -> compare order-independently
        if sorted(op1.values) != sorted(op2.values):
            return False
        # an immediate is the same operand regardless of its encoded width; registers must match width
        return op1.width == op2.width or op1.type == OT.IMM

    def reduce(self, include_categories):
        """ Remove unsupported instructions and operand values """

        def is_supported(spec: InstructionSpec):

            # supported_instructions is an AArch64-only allow-list; absent on other arches.
            supported = getattr(CONF, "supported_instructions", None)
            if supported and spec.name not in supported:
                return False

            # optionally drop every extended-register form (UXTW/SXTW/SXTX/UXTX/…), whether the
            # extend modifies a memory-address index or an arithmetic source register.
            if getattr(CONF, "avoid_extended_memory_operands", False):
                _EXTENDS = {"uxtb", "uxth", "uxtw", "uxtx", "sxtb", "sxth", "sxtw", "sxtx"}

                def _is_extend(op):
                    return op.type == OT.IMM and op.values and all(v in _EXTENDS for v in op.values)

                for operand in spec.operands:
                    components = operand.inner if isinstance(operand, MemorySpec) else [operand]
                    if any(_is_extend(c) for c in components):
                        return False
            if "aarch64" in CONF.instruction_set and spec.category != "general":
                return False  # aarch64 currently supports only the "general" category

            if CONF._no_generation:
                # if we use an existing test case, then instruction filtering is irrelevant
                return True

            # allowlist has priority over blocklists
            if spec.name in CONF.instruction_allowlist:
                return True

            if spec.name in CONF.instruction_blocklist:
                return False

            if include_categories and any(category in include_categories for category in spec.tags):
                return True

            for operand in spec.operands:
                if operand.type == OT.MEM and operand.values \
                        and operand.values[0] in register_blocklist:
                    return False

            for implicit_operand in spec.implicit_operands:
                assert implicit_operand.type != OT.LABEL  # I know no such instructions
                if implicit_operand.type == OT.MEM and implicit_operand.values \
                        and implicit_operand.values[0] in register_blocklist:
                    return False

                if implicit_operand.type == OT.REG \
                        and implicit_operand.values[0] in register_blocklist:
                    assert len(implicit_operand.values) == 1
                    return False
            return include_categories is None

        skip_list = []
        register_blocklist = set(CONF.register_blocklist) - set(CONF.register_allowlist)

        for s in self.instructions:
            # Unsupported instructions
            if not is_supported(s):
                skip_list.append(s)
                continue

            skip_pending = False
            for op in self._value_specs(s.operands):
                if op.type == OT.REG:
                    choices = sorted(list(set(op.values) - register_blocklist))
                    if not choices:
                        skip_pending = True
                        break
                    op.values = choices

                    # x86-only: temporarily disable generation of higher reg. bytes (ah->al, ...).
                    if "x86" in CONF.instruction_set:
                        for i, reg in enumerate(op.values):
                            if reg[-1] == 'h':
                                op.values[i] = reg.replace('h', 'l')

            if skip_pending:
                skip_list.append(s)

        # remove the unsupported
        for s in skip_list:
            self.instructions.remove(s)

        # set parameters
        for inst in self.instructions:
            if inst.control_flow:
                if "BASE-BRANCH-COND" in inst.tags:
                    self.has_conditional_branch = True
                else:
                    self.has_unconditional_branch = True

            elif inst.has_mem_operand:
                if inst.has_write:
                    self.has_writes = True
                else:
                    self.has_reads = True

    def dedup(self):
        """
        Instruction set spec may contain several copies of the same instruction.
        Remove them.
        """
        skip_list = set()
        for i in range(len(self.instructions)):
            for j in range(i + 1, len(self.instructions)):
                inst1 = self.instructions[i]
                inst2 = self.instructions[j]
                if inst1.name == inst2.name and len(inst1.operands) == len(inst2.operands):
                    match = all(self._operands_equal(op1, op2)
                                for op1, op2 in zip(inst1.operands, inst2.operands))
                    if match:
                        skip_list.add(inst1)

        for s in skip_list:
            self.instructions.remove(s)
