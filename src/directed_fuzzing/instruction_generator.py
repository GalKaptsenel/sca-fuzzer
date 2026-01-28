from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional, Iterable, Tuple, Dict, Any, Union
from itertools import chain

import random
import math
import copy

from ..config import CONF
from ..interfaces import Operand, OperandSpec, Instruction, InstructionSpec, TargetDesc, OT, InstructionSetAbstract, RegisterOperand, ImmediateOperand, FlagsOperand, MemoryOperand, CondOperand


# TODO: Temoprary workaround:
from ..aarch64.aarch64_target_desc import Aarch64TargetDesc


class InstructionTransform(ABC):
    """
    A semantic instruction transform.
    May expand ONE instruction into MANY.
    """

    @abstractmethod
    def applies(self, inst: Instruction) -> bool:
        pass

    @abstractmethod
    def transform(self, inst: Instruction) -> List[Instruction]:
        pass


class InstructionGenerator(ABC):

    @abstractmethod
    def sample(
            self,
            num_samples: int = 1,
            *,
            allowed_regs: Optional[Iterable[Operand]] = None
        ) -> List[Instruction]:
        pass

    @abstractmethod
    def generate_unconditional_branch(self) -> Instruction:
        pass

    @abstractmethod
    def generate_conditional_branch(
            self,
            *,
            allowed_regs: Optional[Iterable[Operand]] = None
        ) -> Tuple[
                Instruction,
                Operand,
                Any,
                Any
        ]:
        pass


class InstructionGeneratorDecorator(InstructionGenerator):
    def __init__(self, inner: InstructionGenerator):
        self._inner = inner

    def sample(self, *args, **kwargs):
        return self._inner.sample(*args, **kwargs)

    def generate_unconditional_branch(self):
        return self._inner.generate_unconditional_branch()

    def generate_conditional_branch(self, *args, **kwargs):
        return self._inner.generate_conditional_branch(*args, **kwargs)


class TransformingInstructionGenerator(InstructionGeneratorDecorator):
    def __init__(self, inner: InstructionGenerator, transforms: List[InstructionTransform]):
        super().__init__(inner)
        self._transforms = transforms

    def _apply_transforms(self, inst):
        insts = [inst]

        for t in self._transforms:
            next_insts = []
            for i in insts:
                if t.applies(i):
                    transformed = t.transform(i)
                    next_insts.extend(transformed)
                else:
                    next_insts.append(i)

            insts = next_insts

        return insts

    def sample(self, *args, **kwargs) -> List[Instruction]:
        logical = self._inner.sample(*args, **kwargs)

        out = []
        for inst in logical:
            out.extend(self._apply_transforms(inst))

        return out

    def generate_unconditional_branch(self) -> Instruction:
        inst = self._inner.generate_unconditional_branch()
        insts = self._apply_transforms(inst)

        if len(insts) != 1:
            raise RuntimeError("Unconditional branches must not expand")

        return insts[0]

    def generate_conditional_branch(self, *args, **kwargs) -> Tuple[
            Instruction,
            Operand,
            Any,
            Any,
        ]:
        inst, ctrl, t, nt = self._inner.generate_conditional_branch(*args, **kwargs)
        insts = self._apply_transforms(inst)

        if len(insts) != 1:
            raise RuntimeError("Conditional branches must not expand")

        return insts[0], ctrl, t, nt


class SimpleAArch64InstructionGenerator(InstructionGenerator):
    def __init__(
            self,
            instruction_set: InstructionSetAbstract,
            target_description: TargetDesc,
            rnd: Optional[random.Random] = None,
            seed: Optional[int] = None,
        ):

        self._rnd = rnd if rnd is not None else random.Random(seed)
        self._instruction_set = instruction_set
        self._target_description = target_description

        self._all_regs = list(chain.from_iterable(self._target_description.registers.values()))

        self._control_flow_instructions = \
            [i for i in self._instruction_set.instructions if i.control_flow]
        assert self._control_flow_instructions or CONF.max_successors_per_bb <= 1, \
               "The instruction set is insufficient to generate a test case"

        self._unconditional_control_flow = []
        self._conditional_control_flow = []
        if self._control_flow_instructions:
            self._unconditional_control_flow = [i for i in self._control_flow_instructions if self._target_description.is_unconditional_branch(i)]
            self._conditional_control_flow = [i for i in self._control_flow_instructions if not self._target_description.is_unconditional_branch(i)]

        self._non_control_flow_instructions = \
            [i for i in self._instruction_set.instructions if not i.control_flow]
        assert self._non_control_flow_instructions, \
            "The instruction set is insufficient to generate a test case"

        self._load_instruction = []
        self._store_instructions = []
        if CONF.avg_mem_accesses != 0:
            memory_access_instructions = \
                [i for i in self._non_control_flow_instructions if i.has_mem_operand]
            self._load_instruction = [i for i in memory_access_instructions if not i.has_write]
            self._store_instructions = [i for i in memory_access_instructions if i.has_write]
            assert self._load_instruction or self._store_instructions, \
                "The instruction set does not have memory accesses while `avg_mem_accesses > 0`"

    def _retrieve_allowed_regs(
            self,
            allowed_regs: Optional[Iterable[Operand]]
        ) -> List[Operand]:

        if allowed_regs is None:
            allowed_regs = self._all_regs
        else:
            allowed_regs = list(filter(lambda x: x in self._all_regs, allowed_regs))

        return allowed_regs

    def _retrieve_satisfiable_instructions(
            self,
            instructions: List[InstructionSpec],
            allowed_regs: Iterable[Operand]
        ) -> List[InstructionSpec]:

        allowed_regs = set(allowed_regs)

        def satisfiable(spec: InstructionSpec) -> bool:
            for ops in chain(spec.operands, spec.implicit_operands):
                if ops.type == OT.REG and not (set(ops.values) & allowed_regs):
                    return False
            return spec.has_mem_operand
            return True

        return list(filter(lambda x: satisfiable(x), instructions))

    def _make_operand(self, spec: OperandSpec, inst: Instruction) -> Operand:

        def _make_register(spec: OperandSpec, inst: Instruction) -> RegisterOperand:
            bad_operands = set( name 
                            for reg in chain(inst.get_mem_operands(), inst.get_implicit_mem_operands())
                            for name in self._target_description.reg_denormalized[
                                self._target_description.reg_normalized[reg.value]
                                ].values()
            )
            filtered_options = list(filter(lambda o: o not in bad_operands and o in self._all_regs, spec.values))
            ro = RegisterOperand(value=self._rnd.choice(filtered_options), width=spec.width, src=spec.src, dest=spec.dest)
            ro.name = spec.name
            return ro
    
        def _make_imm(spec: OperandSpec, _: Instruction) -> ImmediateOperand:
            imm = ImmediateOperand(value=self._rnd.choice(spec.values), width=spec.width)
            imm.name = spec.name
            return imm
    
        def _make_cond(spec: OperandSpec, _: Instruction) -> CondOperand:
            cond = CondOperand(value=self._rnd.choice(spec.values))
            cond.name = spec.name
            return cond 
    
        def _make_flags(spec: OperandSpec, _: Instruction) -> FlagsOperand:
            flags = FlagsOperand(spec.values)
            flags.name = spec.name
            return flags
    
        def _make_memory(spec: OperandSpec, inst: Instruction) -> MemoryOperand:
            bad_operands = set( name 
                            for reg in chain(inst.get_reg_operands(), inst.get_mem_operands(), inst.get_implicit_mem_operands())
                            for name in self._target_description.reg_denormalized[
                                self._target_description.reg_normalized[reg.value]
                                ].values()
            )
            # In Aarch64, a memory operand may contain registers and also an immidiate.
            # We filter out because for instructions that use as destination\source a register that is part of the memory operand, the compiler raises warnings, for example LDR x3, [x0, x3]
            # Additionally, due to our current common sandboxing technique, we have to avoid memory operands with the same reigster twice, for example LDR x1, [x3, x3]
            #TODO: Temporarily, a workaround, to identify if an operand is a memory operand of type immidiate, we use the Aarch64TargetDesc
            filtered_options = list(filter(lambda o: o not in chain.from_iterable(Aarch64TargetDesc.registers.values()) or (o not in bad_operands and o in self._all_regs), spec.values))
            mo = MemoryOperand(address=self._rnd.choice(filtered_options), width=spec.width, src=spec.src, dest=spec.dest)
            mo.name = spec.name
            return mo
    
        _dispatch_map = {
            OT.REG: _make_register,
            OT.IMM: _make_imm,
            OT.COND: _make_cond,
            OT.MEM: _make_memory,
            OT.FLAGS: _make_flags,
        }
    
        cls = _dispatch_map.get(spec.type)
        if cls is None:
            raise ValueError(f"Unknown operand type: {spec.type}")
        return cls(spec, inst)

    def _generate_taken_nontaken(
            self, inst: Instruction
        ) -> Tuple[
                Operand,
                Optional[Union[int, Dict[str, int]]],
                Optional[Union[int, Dict[str, int]]]
        ]:
        """
        Compute concrete controlling operand values that force a conditional control-flow
        instruction to be taken and non-taken.
        
        This helper is used internally when generating conditional branches. Given an
        already-instantiated Instruction, it determines which operand controls the
        branch decision and returns values that deterministically exercise both
        outcomes.
        
        Supported instruction classes:
        - CBZ / CBNZ:
            * The controlling operand is the single non-label register operand.
            * Returned values are integers:
                - CBZ:    taken=0, non_taken!=0
                - CBNZ:   taken!=0, non_taken=0
        
        - Conditional branches (B.<cond>):
            * The controlling operand is the implicit FLAGS (NZCV) operand.
            * Returned values are dictionaries mapping flag names {"N","Z","C","V"} to {0,1}.
            * For each condition, the minimal set of flags required to satisfy the
              architectural condition is fixed.
            * All remaining flags are filled with randomized values.
            * The non-taken case is constructed to violate the condition (by flipping
              at least one controlling relation, e.g., Z=1 -> Z=0, N==V -> N!=V).
        
        Special cases:
        - AL (always):
            * Branch is always taken.
            * Returns (flags=random, non_taken=None).
        
        - NV (never):
            * Branch is never taken.
            * Returns (taken=None, flags=random).
        
        Assumptions / invariants:
        - If a conditional (OT.COND) operand exists, an implicit FLAGS operand also exists.
        - CBZ/CBNZ templates contain exactly one non-label controlling operand.
        - Returned flag dictionaries always fully specify NZCV after random completion.
        
        Return value:
            (controlling_operand, value_for_taken, value_for_non_taken)
        
        Where:
        - controlling_operand: Operand
            The register or FLAGS operand that determines branch direction.
        - value_for_taken:
            * int (for CBZ/CBNZ), or
            * Dict[str, int] for NZCV flags, or
            * None if the branch can never be taken.
        - value_for_non_taken:
            * Same type as value_for_taken, or
            * None if the branch is always taken.
        """

        def _update_if_missing(target: dict, source: dict):
            for k, v in source.items():
                target.setdefault(k, v)

        mn = inst.name.lower()
        if mn == "cbz":
            taken = 0
            non_taken = self._rnd.randint(1, 1024)
            controlling_operands = [t for t in inst.operands if t.type != OT.LABEL]
            assert len(controlling_operands) == 1, f"Assumed that there is exactly a single non-label controlling operand in '{mn}' template"
            controlling_operand = controlling_operands[0]
        elif mn == "cbnz":
            taken = self._rnd.randint(1, 1024)
            non_taken = 0
            controlling_operands = [t for t in inst.operands if t.type != OT.LABEL]
            assert len(controlling_operands) == 1, f"Assumed that there is exactly a single non-label controlling operand in '{mn}' template"
            controlling_operand = controlling_operands[0]
        elif inst.get_cond_operand() is not None:

            # Assumed that if OT.COND exists in the template, then also OT.FLAGS exists as an implicit operand

            cond_operand = inst.get_cond_operand()
            cond = cond_operand.value.lower()

            flags = ["N", "Z", "C", "V"]
            rand_flags = lambda: {f: self._rnd.randint(0, 1) for f in flags}

            flags_operand = inst.get_flags_operand()
            assert flags_operand is not None, f"Assumed that FLAGS operand exists for '{mn}' template"
            controlling_operand = flags_operand

            if cond == "eq":
                taken = {"Z": 1};
                non_taken = {"Z": 0};
            elif cond == "ne":
                taken = {"Z": 0}
                non_taken = {"Z": 1}
            elif cond == "cs":
                taken = {"C": 1}
                non_taken = {"C": 0}
            elif cond == "cc":
                taken = {"C": 0}
                non_taken = {"C": 1}
            elif cond == "mi":
                taken = {"N": 1}
                non_taken = {"N": 0}
            elif cond == "pl":
                taken = {"N": 0}
                non_taken = {"N": 1}
            elif cond == "vs":
                taken = {"V": 1}
                non_taken = {"V": 0}
            elif cond == "vc":
                taken = {"V": 0}
                non_taken = {"V": 1}
            elif cond == "hi":
                # C == 1 AND Z == 0
                taken = {"C": 1, "Z": 0}

                if self._rnd.choice([True, False]):
                    non_taken = {"C": 0}
                else:
                    non_taken = {"Z": 1}

            elif cond == "ls":
                # C == 0 OR Z == 1
                if self._rnd.choice([True, False]):
                    taken = {"C": 0}
                else:
                    taken = {"Z": 1}

                non_taken = {"C": 1, "Z": 0}

            elif cond == "ge":
                # N == V
                val = self._rnd.randint(0, 1)

                taken = {"N": val, "V": val}

                if self._rnd.choice([True, False]):
                    non_taken = {"N": val, "V": 1 - val}
                else:
                    non_taken = {"N": 1 - val, "V": val}

            elif cond == "lt":
                # N != V
                val = self._rnd.randint(0, 1)

                non_taken = {"N": val, "V": val}

                if self._rnd.choice([True, False]):
                    taken = {"N": val, "V": 1 - val}
                else:
                    taken = {"N": 1 - val, "V": val}


            elif cond == "gt":
                # Z == 0 AND N == V
                val = self._rnd.randint(0, 1)

                taken = {"Z": 0, "N": val, "V": val}

                if self._rnd.choice([True, False]):
                    if self._rnd.choice([True, False]):
                        non_taken = {"N": val, "V": 1 - val}
                    else:
                        non_taken = {"N": 1 - val, "V": val}

                else:
                    non_taken = {"Z": 1}

            elif cond == "le":
                # Z == 1 OR N != V

                val = self._rnd.randint(0, 1)

                if self._rnd.choice([True, False]):
                    if self._rnd.choice([True, False]):
                        taken = {"N": val, "V": 1 - val}
                    else:
                        taken = {"N": 1 - val, "V": val}

                else:
                    taken = {"Z": 1}

                non_taken = {"Z": 0, "N": val, "V": val}

            elif cond == "al":
                return controlling_operand, rand_flags(), None
            elif cond == "nv":
                return controlling_operand, None, rand_flags()

            _update_if_missing(taken, rand_flags())
            _update_if_missing(non_taken, rand_flags())

        else:
            raise ValueError(f"Unsupported instruction mnemonic: {mn}")

        return controlling_operand, taken, non_taken

    def sample(
            self,
            num_samples: int = 1,
            *,
            allowed_regs: Optional[Iterable[Operand]] = None
        ) -> List[Instruction]:

        allowed_regs = self._retrieve_allowed_regs(allowed_regs)
        samples = []

        for _ in range(num_samples):
            relevant_instructions = self._retrieve_satisfiable_instructions(self._non_control_flow_instructions, allowed_regs)
            spec = self._rnd.choice(relevant_instructions)
            if not spec:
                raise RuntimeError(f'Unable to obtain a satisfiable instruction with the allowed registers: {allowed_regs}')
    
            inst = Instruction.from_spec(spec)
    
            for o_spec in spec.operands:
                if o_spec.type != OT.LABEL:
                    assert o_spec.values, "Assumed that any operand type that is not LABEL will have a list of all possible concrete operands in the arch configuration"
                    op = self._make_operand(o_spec, inst)
                    inst.add_op(op)

            for o_spec in spec.implicit_operands:
                if o_spec.type != OT.LABEL:
                    assert o_spec.values, "Assumed that any operand type that is not LABEL will have a list of all possible concrete operands in the arch configuration"
                    op = self._make_operand(o_spec, inst)
                    inst.add_op(op, implicit=True)
    
            samples.append(inst)

        return samples

    def generate_unconditional_branch(self) -> Instruction:
        spec = self._rnd.choice(self._unconditional_control_flow)
        inst = Instruction.from_spec(spec)
        return inst

    def generate_conditional_branch(
            self,
            *,
            allowed_regs: Optional[Iterable[Operand]] = None
        ) -> Tuple[
                Instruction,
                Operand,
                Any,
                Any,
            ]:
        allowed_regs = self._retrieve_allowed_regs(allowed_regs)
        relevant_instructions = self._retrieve_satisfiable_instructions(self._conditional_control_flow, allowed_regs)
        spec = self._rnd.choice(relevant_instructions)
        if not spec:
            raise RuntimeError(f'Unable to obtain a satisfiable conditional instruction with the allowed registers: {allowed_regs}')

        inst = Instruction.from_spec(spec)
    
        for o_spec in spec.operands:
            if o_spec.type != OT.LABEL:
                assert o_spec.values, "Assumed that any operand type that is not LABEL will have a list of all possible concrete operands in the arch configuration"
                op = self._make_operand(o_spec, inst)
                inst.add_op(op)

        for o_spec in spec.implicit_operands:
            op = self._make_operand(o_spec, inst)
            inst.add_op(op, implicit=True)

        controlling_operand, value_for_taken, value_for_non_taken = self._generate_taken_nontaken(inst)
    
        return inst, controlling_operand, value_for_taken, value_for_non_taken


class SandboxMemoryTransform(InstructionTransform):
    """
    Enforces: addr = sandbox_base + (addr & sandbox_mask)
    """

    def __init__(self, sandbox_base_reg: str, sandbox_size: int):
        assert isinstance(sandbox_base_reg, str)
        assert isinstance(sandbox_size, int)
        self._base_reg = RegisterOperand(sandbox_base_reg, 64, False, False) # TODO: This is a workaround -  the sandbox base register is not set as a src, in order for the later checks won't treat it as an input for the program that could be changed
        self._base_reg.name = "sandbox_base_reg"
        self._mask_size = int(math.log(sandbox_size, 2))
        self._mask = "0b" + "1" * self._mask_size

    def applies(self, inst: Instruction):
        return inst.has_mem_operand and not inst.control_flow

    def transform(self, inst: Instruction):
        assert self.applies(inst)
        if inst.get_implicit_mem_operands():
            raise RuntimeError("Implicit memory accesses are not supported")

        mem_operands = inst.get_mem_operands()
        assert mem_operands

        result_seq = []

        base_operand = mem_operands[0]
        apply_mask_inst = self._apply_mask(base_operand)
        add_base_inst   = self._add_base(base_operand)

        sub_instructions = []
        for i, next_op in enumerate(mem_operands[1:]):
            sub_op = copy.deepcopy(next_op)
            sub_op.name += f"sub_op_{i}"
            sub_op.src = True
            sub_op.dest = False
            op_src = copy.deepcopy(base_operand)
            op_src.name += "_src"
            op_src.src = True
            op_src.dest = False
            op_dst = copy.deepcopy(base_operand)
            op_dst.name += "_dst"
            op_dst.src = False
            op_dst.dest = True
            
            sub_inst = Instruction("SUB", True).add_op(op_dst).add_op(op_src).add_op(sub_op)
            sub_inst.template = f"SUB {{{op_dst.name}}}, {{{op_src.name}}}, {{{sub_op.name}}}"

            sub_instructions.append(sub_inst)

        return [
                apply_mask_inst,
                add_base_inst,
                *sub_instructions,
                inst,
        ]

    def _apply_mask(self, op) -> Instruction:
        mask_size_pow2 = 1 << (self._mask_size - 1).bit_length()
        imm_op = ImmediateOperand(self._mask, max(op.width, mask_size_pow2))
        imm_op.name = "imm_op"
        op_src = copy.deepcopy(op)
        op_src.name += "_src"
        op_src.src = True
        op_src.dest = False
        op_dst = copy.deepcopy(op)
        op_dst.name += "_dst"
        op_dst.src = False
        op_dst.dest = True
        apply_mask = Instruction("AND", True).add_op(op_dst).add_op(op_src).add_op(imm_op)
        apply_mask.template = f"AND {{{op_dst.name}}}, {{{op_src.name}}}, {{{imm_op.name}}}"
        return apply_mask

    def _add_base(self, op) -> Instruction:
        # Assumes the base register is 64 bits
        op_src = copy.deepcopy(op)
        op_src.name += "_src"
        op_src.src = True
        op_src.dest = False
        op_dst = copy.deepcopy(op)
        op_dst.name += "_dst"
        op_dst.src = False
        op_dst.dest = True
        add_instruction = Instruction("ADD", True).add_op(op_dst).add_op(op_src).add_op(self._base_reg)
        add_instruction.template = f"ADD {{{op_dst.name}}}, {{{op_src.name}}}, {{{self._base_reg.name}}}"
        return add_instruction


