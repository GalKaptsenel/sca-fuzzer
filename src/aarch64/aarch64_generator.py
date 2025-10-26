"""
File: x86 implementation of the test case generator

Copyright (C) Microsoft Corporation
SPDX-License-Identifier: MIT
"""
import abc
import math
import random
import copy
import struct
from itertools import chain
from typing import List, Tuple, Optional, Type
from dataclasses import dataclass, field

from ..isa_loader import InstructionSet
from ..interfaces import TestCase, Operand, Instruction, BasicBlock, Function, InstructionSpec, \
    GeneratorException, RegisterOperand, ImmediateOperand, MAIN_AREA_SIZE, FAULTY_AREA_SIZE, \
    MemoryOperand, AgenOperand, OT, OperandSpec, CondOperand
from ..generator import ConfigurableGenerator, RandomGenerator, Pass, Printer
from ..config import CONF
from .aarch64_target_desc import Aarch64TargetDesc
from .aarch64_elf_parser import Aarch64ElfParser
from .aarch64_connection import ExecutorAuxBuffer, AuxBufferType, register_aux_buffer


class FaultFilter:

    def __init__(self) -> None:
        self.div_by_zero: bool = 'div-by-zero' in CONF.generator_faults_allowlist
        self.div_overflow: bool = 'div-overflow' in CONF.generator_faults_allowlist
        self.non_canonical_access: bool = 'non-canonical-access' in CONF.generator_faults_allowlist
        self.u2k_access: bool = 'user-to-kernel-access' in CONF.generator_faults_allowlist


class Aarch64Generator(ConfigurableGenerator, abc.ABC):
    faults: FaultFilter
    target_desc: Aarch64TargetDesc

    def __init__(self, instruction_set: InstructionSet, seed: int):
        super(Aarch64Generator, self).__init__(instruction_set, seed)
        self.target_desc = Aarch64TargetDesc()
        self.elf_parser = Aarch64ElfParser(self.target_desc)
        self.faults = FaultFilter()

        # configure instrumentation passes
        self.passes = [
            Aarch64PatchUndefinedLoadsPass(self.target_desc),
#            Aarch64SandboxPass(),
#            Aarch64DsbSyPass(),
        ]

        self.printer = Aarch64Printer(self.target_desc)

    def get_return_instruction(self) -> Instruction:
        return Instruction("ret", False, "", True, template="RET")

    def get_unconditional_jump_instruction(self) -> Instruction:
        return Instruction("b", False, "UNCOND_BR", True, template="B {label}")

    def get_elf_data(self, test_case: TestCase, obj_file: str) -> None:
        self.elf_parser.parse(test_case, obj_file)


class Aarch64DsbSyPass(Pass):

    def run_on_test_case(self, test_case: TestCase) -> None:
        for func in test_case.functions:
            for bb in func:
                insertion_points = []
                for instr in bb:
                    # make a copy to avoid infinite insertions
                    insertion_points.append(instr)

                for instr in insertion_points:
                    bb.insert_after(instr, Instruction("DSB SY", True, template="DSB SY"))


class Aarch64PatchUndefinedLoadsPass(Pass):
    def __init__(self, target_desc) -> None:
        self.target_desc: Aarch64TargetDesc = target_desc
        super().__init__()

    def run_on_test_case(self, test_case: TestCase) -> None:
        for func in test_case.functions:
            for bb in func:

                to_patch: List[Instruction] = []
                for inst in bb:
                    # check if it's a load with post-index
                    if "ldr" in inst.name and inst.get_imm_operands():
                        ops = inst.operands
                        assert isinstance(ops[0], RegisterOperand)
                        assert isinstance(ops[1], MemoryOperand)
                        normalized_dest = self.target_desc.reg_normalized[ops[0].value]
                        if normalized_dest in ops[1].value:
                            to_patch.append(inst)

                # fix operands
                for inst in to_patch:
                    org_dest = inst.operands[0]
                    options = self.target_desc.registers[org_dest.width]
                    options = [i for i in options if i != org_dest.value]
                    new_value = random.choice(options)
                    inst.operands[0].value = new_value


class Aarch64NonCanonicalAddressPass(Pass):

    def run_on_test_case(self, test_case: TestCase) -> None:
        pass

class Aarch64TagMemoryAccesses(Pass):
    def __init__(self, memory_accesses_to_guess_tag: Optional[List[int]] = None):
        super().__init__()
        if memory_accesses_to_guess_tag is None:
            memory_accesses_to_guess_tag = []

        self.memory_accesses_to_guess_tag = memory_accesses_to_guess_tag

    def run_on_test_case(self, test_case: TestCase) -> None:
        for func in test_case.functions:
            for bb in func:

                memory_instructions = []

                for inst in bb:
                    if inst.has_memory_access:
                        memory_instructions.append(inst)

                for inst in memory_instructions:
                    to_subtract = []
                    mem_operands = inst.get_mem_operands()
                    base_operand: Optional[MemoryOperand] = None
                    for operand in mem_operands:
                        if operand.value in chain.from_iterable(Aarch64TargetDesc.registers.values()) and base_operand is None:
                            base_operand = operand
                            if inst.memory_access_id not in self.memory_accesses_to_guess_tag:
                                mte_tag = base_operand.mte_memory_tag
                            else:
                                lst = list(range(0, 15))
                                lst.remove(base_operand.mte_memory_tag)
                                mte_tag = random.choice(lst)

                            x7_register = RegisterOperand("x7", 64, True, True)
                            x7_register.name = "x7_reg"
                            imm_width = 4
                            imm_op = ImmediateOperand(f'0b{mte_tag:0{imm_width}b}', imm_width)
                            imm_op.name = "imm_op"
                            tag_register_instruction = Instruction("MOV", True).add_op(x7_register).add_op(imm_op)
                            tag_register_instruction.template = f"MOV {{{x7_register.name}}}, {{{imm_op.name}}}"
                            bb.insert_before(position=inst ,inst=tag_register_instruction)


                            ubfx_instruction = Instruction("UBFX", True).add_op(base_operand).add_op(base_operand)
                            ubfx_instruction.template = f"UBFX {{{base_operand.name}}}, {{{base_operand.name}}}, #0, #56"
                            bb.insert_before(position=inst ,inst=ubfx_instruction)

                            set_tag_register_instruction = Instruction("ORR", True).add_op(base_operand).add_op(base_operand).add_op(x7_register)
                            set_tag_register_instruction.template = f"ORR {{{base_operand.name}}}, {{{base_operand.name}}}, {{{x7_register.name}}}, LSL 56"
                            bb.insert_before(position=inst ,inst=set_tag_register_instruction)
                        else:
                            to_subtract.append(operand)

                    if base_operand is not None:
                        for operand in to_subtract:
                            base_operand_cpy = copy.deepcopy(base_operand)
                            other_operand_cpy = copy.deepcopy(operand)
                            base_operand_cpy.name += '0'
                            other_operand_cpy.name += '1'
                            other_operand_cpy.src = True
                            other_operand_cpy.dest = False
                            base_operand_cpy.src = True
                            base_operand_cpy.dest = True
                            sub_inst = Instruction("SUB", True).add_op(base_operand_cpy).add_op(base_operand_cpy).add_op(other_operand_cpy)

                            sub_inst.template = f'sub {{{base_operand_cpy.name}}}, {{{base_operand_cpy.name}}}, {{{other_operand_cpy.name}}}'  # TODO: this should be done in the constructor
                            bb.insert_before(inst, sub_inst)


@dataclass
class InstructionLog:
	pc: int = 0
	flags: int = 0
	regs: List[int] = field(default_factory=lambda: [0]*31)
	effective_address: int = 0
	mem_before: int = 0
	mem_after: int = 0
	encoding: int = 0

	def __repr__(self) -> str:
		regs_preview = ", ".join(f"x{i}=0x{r:x}" for i, r in enumerate(self.regs[:4]))
		if len(self.regs) > 4:
			regs_preview += ", ..."
		return (
				f"InstructionLog(pc=0x{self.pc:x}, flags=0x{self.flags:x}, "
				f"{regs_preview}, ea=0x{self.effective_address:x}, "
				f"mem_before=0x{self.mem_before:x}, mem_after=0x{self.mem_after:x}), "
                f"encoding=0x{self.encoding:08x})"
			)

	def __str__(self) -> str:
		regs_str = "\n      ".join(
					f"x{i:02d}: 0x{val:016x}" for i, val in enumerate(self.regs)
				) if self.regs else "<no registers>"
		return (
				f"InstructionLog:\n"
				f"  PC:             0x{self.pc:016x}\n"
				f"  FLAGS:          0x{self.flags:016x}\n"
				f"  EFFECTIVE_ADDR: 0x{self.effective_address:016x}\n"
				f"  MEM_BEFORE:     0x{self.mem_before:016x}\n"
				f"  MEM_AFTER:      0x{self.mem_after:016x}\n"
                f"  ENCODING:       0x{self.encoding:08x}\n"
				f"  REGS:\n      {regs_str}"
			)
    
@register_aux_buffer(AuxBufferType.FULL_TRACE)
@dataclass
class FullTraceAuxBuffer(ExecutorAuxBuffer):
	instruction_log_array_offset: int = 0
	instruction_log_entry_count: int = 0
	instruction_log_max_count: int = 0
	instruction_logs: List[InstructionLog] = field(default_factory=list)

	def __post_init__(self):
		super().__init__(AuxBufferType.FULL_TRACE)

	@classmethod
	def from_bytes(cls: Type["FullTraceAuxBuffer"], data: bytes):
		"""
		Parse binary data into a FullTraceAuxBuffer.

		Expected binary layout:
			3 consecutive 8-byte unsigned integers (little-endian):
                instruction_log_array_offset
                instruction_log_entry_count
                instruction_log_max_count
            and then, from offset 'instruction_log_array_offset' an array of 'instruction_log_entry_count' entries of the format:
                pc
                flags
                regs[31]
                affective_address
                memory_before
                memory_after
                encoding (lower 32 bits are valid encoding)
		"""
		header_size = 3 * 8
		if len(data) < header_size:
			raise ValueError(f"Invalid data size ({len(data)} bytes), expected >= {header_size} bytes")

		# Little-endian unsigned 64-bit integers
		instruction_log_array_offset, entry_count, max_count = struct.unpack("<3Q", data[:header_size])

		logs = []
		offset = instruction_log_array_offset
		entry_size = 8 + 8 + 31*8 + 8 + 8 + 8 + 8 # pc + flags + regs[31] + effective_address + mem_before + mem_after + encoding

		for i in range(entry_count):
			entry_data = data[offset:offset+entry_size]
			if len(entry_data) < entry_size:
				raise ValueError(f"Incomplete instruction log entry {i} out of {entry_count} logged instructions")
			pc, flags = struct.unpack("<2Q", entry_data[:16])
			regs = list(struct.unpack("<31Q", entry_data[16:16+31*8]))
			effective_address, mem_before, mem_after, raw_encoding = struct.unpack("<4Q", entry_data[16+31*8:])
			encoding = raw_encoding & 0xFFFFFFFF
			logs.append(InstructionLog(pc, flags, regs, effective_address, mem_before, mem_after, encoding))
			offset += entry_size

		return cls(
				instruction_log_array_offset=instruction_log_array_offset,
				instruction_log_entry_count=entry_count,
				instruction_log_max_count=max_count,
				instruction_logs=logs
			)

	def to_bytes(self) -> bytes:
		"""Serialize the buffer back into bytes."""
		header = struct.pack(
				"<3Q",
				self.instruction_log_array_offset,
				self.instruction_log_entry_count,
				self.instruction_log_max_count
			)
		logs_bytes = b""
		for log in self.instruction_logs:
			logs_bytes += struct.pack(
					"<2Q31Q3QQ",
					log.pc,
					log.flags,
					*log.regs,
					log.effective_address,
					log.mem_before,
					log.mem_after,
                    log.encoding & 0xFFFFFFFF
				)

		return header + logs_bytes


	def __repr__(self) -> str:
		return (
				f"{self.__class__.__name__}("
				f"offset=0x{self.instruction_log_array_offset:x}, "
				f"entry_count={self.instruction_log_entry_count}, "
				f"max_count={self.instruction_log_max_count}, "
				f"instruction_logs=[{len(self.instruction_logs)} entries])"
			)

	def __str__(self) -> str:
		lines = [
				f"FullTraceAuxBuffer:",
				f"  instruction_log_array_offset = 0x{self.instruction_log_array_offset:x}",
				f"  instruction_log_entry_count  = {self.instruction_log_entry_count}",
				f"  instruction_log_max_count    = {self.instruction_log_max_count}",
				f"  instruction_logs ({len(self.instruction_logs)} entries):",
			]

		for i, log in enumerate(self.instruction_logs):
			lines.append(
					f"    [{i}] {str(log)}"
				)

		return "\n".join(lines)



class Aarch64FullTrace(Pass):
	"""
	Pass that inserts full tracing instrumentation for each instruction.
	Logs PC, operands, flags, and memory accesses into the auxiliary buffer.
	"""

	TEMPLATE_TRACE_INSTRUCTION = "TRACE_INSTRUCTION {base_reg}, {t0}, {t1}, {t2}, {mem_type}, {addr_reg}, {val_reg}"
	TEMPLATE_TRACE_INIT = "TRACE_INIT {base_reg}, {t0}, {t1}"

	def __init__(self, base_reg="x7", temp_regs=None):
		default_registers = ["x9","x10", "x11", "x12"]
		if temp_regs is None:
			temp_regs = default_registers
		if len(temp_regs) < 4:
			raise ValueError("Need at least {len(default_registers)} temporary registers")
		self.base_reg = base_reg
		self.temp_regs = temp_regs

	@staticmethod
	def _reg_name_to_index(reg_name: str) -> int:
		if not isinstance(reg_name, str):
			return -1

		rn = reg_name.strip().lower()
		if rn.startswith('x') or rn.startswith('w'):
			try:
				num = int(rn[1:])
				if 0 <= num <= 30:
					return num
			except ValueError:
				return -1

		return -1



	def instrument_inst(self, bb: BasicBlock, inst: Instruction) -> None:
		instrs_to_insert = []

		value_reg = "-"
		addr_reg = "-"
		mem_type = 0

		if inst.has_memory_access:
			mem_operands = inst.get_mem_operands()

			base_template = f"MOV {self.temp_regs[0]}, {mem_operands[0].value}"
			instrs_to_insert.append(Instruction(f"MOV", True, template=base_template))
			for op in mem_operands[1:]:

				add_template = f"ADD {self.temp_regs[0]}, {self.temp_regs[0]}, {op.value}"
				instrs_to_insert.append(Instruction(f"ADD", True, template=add_template))

			addr_reg = self.temp_regs[0]
			mem_type = 1 if mem_operands[0].src else 2

			if mem_type == 2:
				value_operand = [op for op in inst.operands if op.src and op.type == OT.REG]
				assert len(value_operand) == 1
				value_reg = value_operand[0].value

		asm_trace = self.TEMPLATE_TRACE_INSTRUCTION.format(
				base_reg=self.base_reg, t0=self.temp_regs[1], t1=self.temp_regs[2], t2=self.temp_regs[3],
				mem_type=mem_type, addr_reg=addr_reg, val_reg=value_reg
			)
		instrs_to_insert.append(Instruction(f"TRACE_INSTRUCTION_{inst.name}", True, template=asm_trace))

		# Insert all
		for i in instrs_to_insert:
			bb.insert_before(inst, i)

	def run_on_test_case(self, test_case: TestCase):
		initialized = False
		for func in test_case.functions:
			for bb in func:
				instructions = []
				for inst in bb:
					if not initialized:
						trace_init_template = self.TEMPLATE_TRACE_INIT.format(
								base_reg=self.base_reg, t0=self.temp_regs[1], t1=self.temp_regs[2]
							)
						trace_init_inst = Instruction(f"TRACE_INIT", True, template=trace_init_template)
						bb.insert_before(inst, trace_init_inst)
						initialized = True
					try:
						self.instrument_inst(bb, inst)
					except Exception as e:
						print(f"[Aarch64FullTrace] failed on {inst}: {e}")
				else:
					trace_bb_exit_template =  self.TEMPLATE_TRACE_INSTRUCTION.format(
							base_reg=self.base_reg, t0=self.temp_regs[1], t1=self.temp_regs[2], t2=self.temp_regs[3],
							mem_type=0, addr_reg='-', val_reg='-'
						)
					trace_bb_exit_inst = Instruction(f"TRACE_INSTRUCTION_EXIT_{bb.name}", True, template=trace_bb_exit_template)
#					bb.insert_after(bb.end, trace_bb_exit_inst)
					bb.terminators = [elem for pair in zip([trace_bb_exit_inst]*len(bb.terminators), bb.terminators) for elem in pair]

class Aarch64MarkRegisterTaints(Pass):
	"""
	Pass that inserts taint instrumentation for GPR register reads/writes.
	It emits macro invocations that take the auxiliary buffer base register and temporary regs.
	
	Constructor args:
	  base_reg: str, e.g. "x7"  -- the register containing base address of auxiliary buffer
	  temp_regs: List[str] -- list of available temporary registers, e.g. ["x9","x10","x11"]
	    - For TAINT_REG_WRITE we use temp_regs[0], temp_regs[1]
	    - For TAINT_REG_PROPAGATE_OR_USED we use temp_regs[0], temp_regs[1]
	"""
	TEMPLATE_TAINT_REG_WRITE = "TAINT_REG_WRITE {dst}, {base}, {t0}, {t1}"
	TEMPLATE_TAINT_REG_READ = "TAINT_REG_READ {src}, {base}, {t0}, {t1}"

	def __init__(self, base_reg: str = "x7", temp_regs: List[str] = None):
		super().__init__()
		self.base_reg = base_reg
		if temp_regs is None:
			temp_regs = ["x9", "x10"]
		if len(temp_regs) < 2:
			raise ValueError(f"temp_regs must contain at least 2 registers other then base_reg ({base_reg}), now it containts {temp_regs}")
		self.temp_regs = temp_regs

	@staticmethod
	def _reg_name_to_index(reg_name: str) -> int:
		if not isinstance(reg_name, str):
			return -1
	
		rn = reg_name.strip().lower()
		if rn.startswith('x') or rn.startswith('w'):
			try:
				num = int(rn[1:])
				if 0 <= num <= 30:
					return num
			except ValueError:
				return -1
	
		return -1

	@classmethod
	def _get_regs_from_inst(cls, inst: Instruction) -> List[Operand]:
		regs = []
		for op in inst.operands:
			if op.type == OT.REG:
				regs.append(op)


		for op in inst.implicit_operands:
			if op.type == OT.REG:
				regs.append(op)

		return regs

	def _collect_src_dst_indices(self, inst: Instruction) ->Tuple[List[int], List[int]]:
		register_names = [x for lst in Aarch64TargetDesc.registers.values() for x in lst]
		mem_src_regs = [op for op in inst.get_mem_operands() if op.value in register_names]

		regs = self._get_regs_from_inst(inst)

		src_idxs = [self._reg_name_to_index(r.value) for r in regs if r.src]
		src_idxs += [self._reg_name_to_index(r.value) for r in mem_src_regs]
		src_idxs = [idx for idx in src_idxs if idx >= 0]

		dest_idxs = [self._reg_name_to_index(r.value) for r in regs if r.dest]
		dest_idxs = [idx for idx in dest_idxs if idx >= 0]

		return src_idxs, dest_idxs

	def mark_register_taints(self, bb: BasicBlock, inst: Instruction) -> None:
		src_idxs, dst_idxs = self._collect_src_dst_indices(inst)
		if not src_idxs and not dst_idxs:
			return

		instrs_to_insert = []

        # Create macro invocations. First MUST do src operands, and only later dst operands.
		# create macro invocations for each src
		for src in src_idxs:
			asm_text = self.TEMPLATE_TAINT_REG_READ.format(
					src=src, base=self.base_reg,
                    t0=self.temp_regs[0], t1=self.temp_regs[1]
			)
			iobj = Instruction(f"TAINT_REG_READ_{src}", True, template=asm_text)
			instrs_to_insert.append(iobj)


		# create macro invocations for each dst 
		for dst in dst_idxs:
			asm_text = self.TEMPLATE_TAINT_REG_WRITE.format(
                    dst=dst, base=self.base_reg,
                    t0=self.temp_regs[0], t1=self.temp_regs[1]
            )
			iobj = Instruction(f"TAINT_REG_WRITE_{dst}", True, template=asm_text)
			instrs_to_insert.append(iobj)


		# insert after instruction (reverse so first ends up closest to inst if insert_after prepends)
		for i in instrs_to_insert:
			bb.insert_before(position=inst, inst=i)

	def run_on_test_case(self, test_case: TestCase) -> None:
		for func in test_case.functions:
			for bb in func:
				for inst in list(bb):
					try:
						self.mark_register_taints(bb, inst)
					except Exception as e:
						print(f"[Aarch64MarkRegisterTaints] warn: failed to instrument {inst}: {e}")



class Aarch64MarkMemoryAccessesNEON(Pass):

    @staticmethod
    def mark_memory_access(bb: BasicBlock, inst: Instruction):

        access_id = inst.memory_access_id

        if not (0 <= access_id <= 127):
            raise ValueError("NEON bit must be between 0 and 127, inclusive")


        neon_register_bitmap = 'v0'
        neon_register_temporary = 'v1'
        scalar_register_temporary = 'w7'

        byte_index = access_id // 8
        bit_shift = access_id % 8
        bit_mask = 1 << bit_shift

        mov_scalar_template = f"mov {scalar_register_temporary}, #{bit_mask}"
        movi_template = f"movi {neon_register_temporary}.16b, #0"
        ins_template  = f"ins {neon_register_temporary}.b[{byte_index}], {scalar_register_temporary}"
        orr_template = f"orr {neon_register_bitmap}.16b, {neon_register_bitmap}.16b, {neon_register_temporary}.16b"

        mov_instruction = Instruction("MOV", True, template=mov_scalar_template)
        movi_instruction = Instruction("MOVI", True, template=movi_template)
        ins_instruction  = Instruction("INS", True, template=ins_template)
        orr_instruction  = Instruction("ORR", True, template=orr_template)

        bb.insert_after(position=inst, inst=orr_instruction)
        bb.insert_after(position=inst, inst=ins_instruction)
        bb.insert_after(position=inst, inst=movi_instruction)
        bb.insert_after(position=inst, inst=mov_instruction)

    def run_on_test_case(self, test_case: TestCase) -> None:
        for func in test_case.functions:
            for bb in func:
                memory_instructions = [inst for inst in bb if inst.has_memory_access]

                for inst in memory_instructions:
                    self.mark_memory_access(bb, inst)



class Aarch64MarkMemoryAccessesSVE(Pass):

    @staticmethod
    def mark_memory_access(bb: BasicBlock, inst: Instruction):

        access_id = inst.memory_access_id

        if not (0 <= access_id <= 127):
            raise ValueError("SVE bit must be between 0 and 127, inclusive")


        sve_register_bitmap = 'z0'
        sve_register_temporary_1 = 'z1'
        sve_register_temporary_2 = 'z2'
        predicate_register = 'p1'

        byte_index = access_id // 8
        bit_shift = access_id % 8

        ptrue_template = f"ptrue {predicate_register}.B, ALL"
        index_template = f"index {sve_register_temporary_2}.B, #0, #1"
        compeq_template = f"cmpeq {predicate_register}.B, {predicate_register}/z, {sve_register_temporary_2}.B, #{byte_index}"
        mov_template = f"mov {sve_register_temporary_1}.B, #0b{1 << bit_shift:08b}"
        orr_template = f"orr {sve_register_bitmap}.B, {predicate_register}/M, {sve_register_bitmap}.B, {sve_register_temporary_1}.B"

        ptrue_instruction = Instruction("PTRUE", True, template=ptrue_template)
        index_instruction = Instruction("INDEX", True, template=index_template)
        cmpeq_instruction = Instruction("CMPEQ", True, template=compeq_template)
        mov_instruction = Instruction("MOV", True, template=mov_template)
        orr_instruction = Instruction("ORR", True, template=orr_template)

        bb.insert_after(position=inst, inst=orr_instruction)
        bb.insert_after(position=inst, inst=mov_instruction)
        bb.insert_after(position=inst, inst=cmpeq_instruction)
        bb.insert_after(position=inst, inst=index_instruction)
        bb.insert_after(position=inst, inst=ptrue_instruction)

    def run_on_test_case(self, test_case: TestCase) -> None:
        for func in test_case.functions:
            for bb in func:

                memory_instructions = []

                for inst in bb:
                    if inst.has_memory_access:
                        memory_instructions.append(inst)

                for inst in memory_instructions:
                    Aarch64MarkMemoryAccesses.mark_memory_access(bb, inst)


class Aarch64MarkMemoryTaints(Pass):
	"""
	Pass that inserts taint instrumentation for memory accesses.
	It emits macro invocations that take the auxiliary buffer base register and temporary regs.
	
	Constructor args:
	  base_reg: str, e.g. "x7"  -- register holding base address of auxiliary buffer
	  temp_regs: List[str] -- list of temp registers, e.g. ["x9","x10"]
	    - temp_regs[0], temp_regs[1] are used for address computation and bit masking
	"""
	
	# Templates for memory taint macros (you can create these in taint_instrument.S later)
	MEM_TEMPLATE_LOAD = "TAINT_LOAD {addr}, {base}, {t0}, {t1}"
	MEM_TEMPLATE_STORE = "TAINT_STORE {addr}, {base}, {t0}, {t1}"
	
	def __init__(self, base_reg: str = "x7", temp_regs: List[str] = None):
		super().__init__()
		self.base_reg = base_reg
		if temp_regs is None:
			temp_regs = ["x9", "x10", "x11"]
		if len(temp_regs) < 3:
			raise ValueError("temp_regs must contain at least 3 registers other than base_reg")
		self.temp_regs = temp_regs
	
	@staticmethod
	def _reg_name_to_index(reg_name: str) -> int:
		if not isinstance(reg_name, str):
			return -1
	
		rn = reg_name.strip().lower()
		if rn.startswith('x') or rn.startswith('w'):
			try:
				num = int(rn[1:])
				if 0 <= num <= 30:
					return num
			except ValueError:
				return -1

		return -1

	def mark_memory_taint(self, bb: BasicBlock, inst: Instruction) -> None:
		instrs_to_insert = []

		mem_operands = inst.get_mem_operands()

		base_template = f"MOV {self.temp_regs[0]}, {mem_operands[0].value}"
		instrs_to_insert.append(Instruction(f"MOV", True, template=base_template))
		for op in mem_operands[1:]:

			add_template = f"ADD {self.temp_regs[0]}, {self.temp_regs[0]}, {op.value}"
			instrs_to_insert.append(Instruction(f"ADD", True, template=add_template))

		addr_reg = self.temp_regs[0]

		addr_idx = self._reg_name_to_index(addr_reg)
		assert 0 <= addr_idx <= 30, "Register name in Aarch64 is expected to have id between 0 to 30"

		left_operand = next((op for op in inst.operands if op.type == OT.REG))
		
        # write macro (for loads)
		if left_operand.src:
			asm_text = self.MEM_TEMPLATE_LOAD.format(addr=addr_idx, base=self.base_reg,
											 t0=self.temp_regs[1], t1=self.temp_regs[2])
			instrs_to_insert.append(Instruction(f"TAINT_LOAD_{addr_idx}", True, template=asm_text))


		# write macro (for stores)
		if left_operand.dest:
			asm_text = self.MEM_TEMPLATE_STORE.format(addr=addr_idx, base=self.base_reg,
											 t0=self.temp_regs[1], t1=self.temp_regs[2])
			instrs_to_insert.append(Instruction(f"TAINT_STORE_{addr_idx}", True, template=asm_text))

		# Insert all
		for i in instrs_to_insert:
			bb.insert_before(inst, i)

	def run_on_test_case(self, test_case: TestCase):
		for func in test_case.functions:
			for bb in func:
				for inst in bb:
					if inst.has_memory_access:
						try:
							self.mark_memory_taint(bb, inst)
						except Exception as e:
							print(f"[Aarch64MarkMemoryTaints] failed on {inst}: {e}")



class BitmapAccessor:
	def __init__(self, parent, attr_name, size_in_bits):
		self.parent = parent
		self.attr_name = attr_name
		self.size_in_bits = size_in_bits
	
	def _get_value(self):
		return getattr(self.parent, self.attr_name)

	def _set_value(self, value):
		setattr(self.parent, self.attr_name, value & ((1 << self.size_in_bits) - 1))  # mask to size

	def __getitem__(self, index):
		value = self._get_value()

		if isinstance(index, slice):
			start, stop, step = index.indices(self.size_in_bits)
			if step != 1:
				raise ValueError("BitmapAccessor does not support stepped slices")
			width = stop - start
			if width <= 0:
				return 0
			mask = (1 << width) - 1
			return (value >> start) & mask

		if index < 0 or index >= self.size_in_bits:
				raise IndexError(f"Bit index out of range: {index=} not in [0, {self.size_in_bits}]")
		return bool((value >> index) & 1)

	def __setitem__(self, index, val):
		value = self._get_value()

		if isinstance(index, slice):
			start, stop, step = index.indices(self.size_in_bits)
			if step != 1:
				raise ValueError("BitmapAccessor does not support stepped slices")
			width = stop - start
			if width <= 0:
				return
			mask = ((1 << width) - 1) << start
			value = (value & ~mask) | ((int(val) << start) & mask)
			self._set_value(value)
			return

		if index < 0 or index >= self.size_in_bits:
			raise IndexError(f"Bit index out of range: {index=} not in [0, {self.size_in_bits}]")

		if val:
			value |= (1 << index)
		else:
			value &= ~(1 << index)
		self._set_value(value)

	def __int__(self):
		return self._get_value()

	def __index__(self):
		return self._get_value()

	def __repr__(self):
		return f"{self._get_value():0{self.size_in_bits}b}"

	def __str__(self):
		return f"0b{self._get_value():0{self.size_in_bits}b}"

	def __call__(self, new_val: int):
		self._set_value(new_val)

	def __invert__(self): return ~self._get_value()
	def __and__(self, other): return int(self) & int(other)
	def __or__(self, other):  return int(self) | int(other)
	def __xor__(self, other): return int(self) ^ int(other)
	def __rand__(self, other): return int(other) & int(self)
	def __ror__(self, other):  return int(other) | int(self)
	def __rxor__(self, other): return int(other) ^ int(self)
	
	def __iand__(self, other): self._set_value(int(self) & int(other)); return self
	def __ior__(self, other): self._set_value(int(self) | int(other)); return self
	def __ixor__(self, other): self._set_value(int(self) ^ int(other)); return self
	
	def __lshift__(self, other): return int(self) << int(other)
	def __rshift__(self, other): return int(self) >> int(other)
	def __ilshift__(self, other): self._set_value(int(self) << int(other)); return self
	def __irshift__(self, other): self._set_value(int(self) >> int(other)); return self

	def __add__(self, other): return int(self) + int(other)
	def __sub__(self, other): return int(self) - int(other)
	def __mul__(self, other): return int(self) * int(other)
	def __floordiv__(self, other): return int(self) // int(other)
	def __mod__(self, other): return int(self) % int(other)
	def __pow__(self, other, modulo=None): return pow(int(self), int(other), modulo)
	def __neg__(self): return -int(self)
	def __pos__(self): return +int(self)

	def __iadd__(self, other): self._set_value(int(self) + int(other)); return self
	def __isub__(self, other): self._set_value(int(self) - int(other)); return self
	def __imul__(self, other): self._set_value(int(self) * int(other)); return self
	def __ifloordiv__(self, other): self._set_value(int(self) // int(other)); return self
	def __imod__(self, other): self._set_value(int(self) % int(other)); return self


	def __eq__(self, other): return int(self) == int(other)
	def __ne__(self, other): return int(self) != int(other)
	def __lt__(self, other): return int(self) < int(other)
	def __le__(self, other): return int(self) <= int(other)
	def __gt__(self, other): return int(self) > int(other)
	def __ge__(self, other): return int(self) >= int(other)


@register_aux_buffer(AuxBufferType.BITMAP_TAINTS)
@dataclass
class BitmapTaintsAuxBuffer(ExecutorAuxBuffer):
	_size_in_bits: int = field(default=64, init=False)
	_regs_write_bits: int = field(default=0, repr=False)
	_regs_read_bits: int = field(default=0, repr=False)
	_regs_input_read_bits: int = field(default=0, repr=False)
	_mem_write_bits: int = field(default=0, repr=False)
	_mem_read_bits: int = field(default=0, repr=False)
	_mem_input_read_bits: int = field(default=0, repr=False)

	def __post_init__(self):
		super().__init__(AuxBufferType.BITMAP_TAINTS)
		self.regs_write_bits = BitmapAccessor(self, "_regs_write_bits", self._size_in_bits)
		self.regs_read_bits = BitmapAccessor(self, "_regs_read_bits", self._size_in_bits)
		self.regs_input_read_bits = BitmapAccessor(self, "_regs_input_read_bits", self._size_in_bits)
		self.mem_write_bits = BitmapAccessor(self, "_mem_write_bits", self._size_in_bits)
		self.mem_read_bits = BitmapAccessor(self, "_mem_read_bits", self._size_in_bits)
		self.mem_input_read_bits = BitmapAccessor(self, "_mem_input_read_bits", self._size_in_bits)

	@classmethod
	def from_bytes(cls: Type["BitmapTaintsAuxBuffer"], data: bytes):
		"""
		Parse binary data into a BitmapTaintsAuxBuffer.

		Expected binary layout:
			6 consecutive 8-byte unsigned integers (little-endian):
				regs_write_bits
				regs_read_bits
				regs_input_read_bits
				mem_write_bits
				mem_read_bits
				mem_input_read_bits
		"""
		expected_size = 6 * 8
		if len(data) < expected_size:
			raise ValueError(f"Invalid data size ({len(data)} bytes), expected >= {expected_size} bytes")
		# Little-endian unsigned 64-bit integers
		fields = struct.unpack("<6Q", data[:expected_size])
		return cls(*fields)

	def to_bytes(self) -> bytes:
		"""Serialize the buffer back into bytes."""
		return struct.pack(
				"<6Q",
				self.regs_write_bits,
				self.regs_read_bits,
				self.regs_input_read_bits,
				self.mem_write_bits,
				self.mem_read_bits,
				self.mem_input_read_bits,
			)

	def __repr__(self):
		return (
			f"<BitmapTaintsAuxBuffer "
			f"regs_write_bits={self.regs_write_bits}, "
			f"regs_read_bits={self.regs_read_bits}, "
			f"regs_input_read_bits={self.regs_input_read_bits}, "
			f"mem_write_bits={self.mem_write_bits}, "
			f"mem_read_bits={self.mem_read_bits}, "
			f"mem_input_read_bits={self.mem_input_read_bits}>"
		)


class Aarch64SandboxPass(Pass):
	def __init__(self):
		super().__init__()
		input_memory_size = MAIN_AREA_SIZE + FAULTY_AREA_SIZE
		mask_size = int(math.log(input_memory_size, 2))
		self.sandbox_address_mask = "0b" + "1" * mask_size

	def run_on_test_case(self, test_case: TestCase) -> None:
		for func in test_case.functions:
			for bb in func:

				memory_instructions = []

				for inst in bb:
					if inst.has_mem_operand(True):
						memory_instructions.append(inst)

				for inst in memory_instructions:
					self.sandbox_memory_access(inst, bb)

	def sandbox_memory_access(self, instr: Instruction, parent: BasicBlock):
		""" Force the memory accesses into the page starting from R14 """

		def generate_template(mnemonic: str, op0: Operand, op1: Operand, op2: Operand) -> Tuple[
			str, Operand, Operand, Operand]:
			op0_cpy = copy.deepcopy(op0)
			op1_cpy = copy.deepcopy(op1)
			op2_cpy = copy.deepcopy(op2)
			op0_cpy.name += "0"
			op1_cpy.name += "1"
			op2_cpy.name += "2"
			template = f"{mnemonic} {{{op0_cpy.name}}}, {{{op1_cpy.name}}}, {{{op2_cpy.name}}}"
			return template, op0_cpy, op1_cpy, op2_cpy

		mem_operands = instr.get_mem_operands()
		implicit_mem_operands = instr.get_implicit_mem_operands()
		if mem_operands and not implicit_mem_operands:
			#            assert len(mem_operands) == 1, f"Unexpected instruction format {instr.name}"
			base_operand: Operand = mem_operands[0]
			base_operand_copy = RegisterOperand(base_operand.value, base_operand.width, True, True)
			base_operand_copy.name = base_operand.name
			
			# TODO: Very bad implemented! Must fix
			
			imm_width = min(base_operand_copy.width, 32)
			imm_op = ImmediateOperand(self.sandbox_address_mask, imm_width)
			imm_op.name = "imm_op"
			template, op0, op1, op2 = generate_template("AND", base_operand_copy,
			                                            base_operand_copy, imm_op)
			apply_mask = Instruction("AND", True).add_op(op0).add_op(op1).add_op(op2)
			apply_mask.template = template
			parent.insert_before(instr, apply_mask)
			
			x30_register = RegisterOperand("x30", 64, True, False)
			x30_register.name = "x30_reg"
			template, op0, op1, op2 = generate_template("ADD", base_operand_copy, base_operand_copy,
			                                            x30_register)
			add_base = Instruction("ADD", True).add_op(op0).add_op(op1).add_op(op2)
			add_base.template = template
			parent.insert_before(instr, add_base)
			
			for op in mem_operands[1:]:
            
				template, op0, op1, op2 = generate_template("SUB", base_operand_copy, base_operand_copy, op)
				op2.dest = False
				op2.src = True
				sub_inst = Instruction("SUB", True).add_op(op0).add_op(op1).add_op(op2)
				sub_inst.template = template  # TODO: this should be done in the constructor
				parent.insert_before(instr, sub_inst)

			return

		if implicit_mem_operands:
			raise GeneratorException("Implicit memory accesses are not supported")

		raise GeneratorException("Attempt to sandbox an instruction without memory operands")


class Aarch64Printer(Printer):
    prologue_template = [
        ".test_case_enter:\n",
    ]

    epilogue_template = [
        ".section .data.main\n",
        ".test_case_exit:\n",
    ]

    def __init__(self, _: Aarch64TargetDesc) -> None:
        super().__init__()

    def print(self, test_case: TestCase, outfile: str) -> None:
        with open(outfile, "w") as f:
            # print prologue
            for line in self.prologue_template:
                f.write(line)

            # print the test case
            for func in test_case.functions:
                self.print_function(func, f)

            # print epilogue
            for line in self.epilogue_template:
                f.write(line)

    def print_function(self, func: Function, file):
        file.write(f".section .data.{func.owner.name}\n")
        file.write(f"{func.name}:\n")
        for bb in func:
            self.print_basic_block(bb, file)

        self.print_basic_block(func.exit, file)

    def print_basic_block(self, bb: BasicBlock, file):
        file.write(f"{bb.name.lower()}:\n")
        for inst in bb:
            file.write(self.instruction_to_str(inst) + "\n")
        for inst in bb.terminators:
            file.write(self.instruction_to_str(inst) + "\n")

    def instruction_to_str(self, inst: Instruction):
        if inst.name == "macro":
            return self.macro_to_str(inst)

        values = {}
        for op in inst.operands:
            values[op.name] = op.value

        instruction = inst.template.format(**values)

        if inst.is_instrumentation:
            comment = "// instrumentation"
        elif inst.is_noremove:
            comment = "// noremove"
        else:
            comment = ""
        return f"{instruction} {comment}"

    def operand_to_str(self, op: Operand) -> str:
        if isinstance(op, MemoryOperand) or isinstance(op, AgenOperand):
            return f"[{op.value}]"

        if isinstance(op, ImmediateOperand) or isinstance(op, AgenOperand):
            return f"#{op.value}"

        return op.value

    def macro_to_str(self, inst: Instruction):
        macro_placeholder = "NOP"
        if inst.operands[1].value.lower() == ".noarg":
            return f".macro{inst.operands[0].value}: {macro_placeholder}"
        else:
            return f".macro{inst.operands[0].value}{inst.operands[1].value}: {macro_placeholder}"


class Aarch64RandomGenerator(Aarch64Generator, RandomGenerator):

    def __init__(self, instruction_set: InstructionSet, seed: int):
        super().__init__(instruction_set, seed)

    def _filter_invalid_operands(self, spec: OperandSpec, inst: Instruction) -> List[str]:
        result: List[str] = []
        register_prefixes = ("x", "w", "q", "v", "d", "s", "h", "b", "sp")

        for op in spec.values:
            if 'pc' == op:
                result.append(op)
            elif not op.startswith(register_prefixes):
                result.append(op)
            elif op in chain.from_iterable(self.target_desc.registers.values()):
                # We omit situations where the same physical register is in memory operand and outside the memory operand.
                # in causes warning of the assembler and unrecognized instructions
                cond = lambda o: o.type == OT.MEM and o.value in chain.from_iterable(self.target_desc.registers.values())
                if spec.type == OT.MEM:
                    cond = lambda _: True

                memory_registers = [o for o in inst.operands if cond(o)]
                if all(Aarch64TargetDesc.reg_normalized[op] != Aarch64TargetDesc.reg_normalized[o.value] for o in memory_registers):
                    result.append(op)
        return result

    def generate_reg_operand(self, spec: OperandSpec, inst: Instruction) -> Operand:
        choices = self._filter_invalid_operands(spec, inst)
        reg = random.choice(choices)
        return RegisterOperand(reg, spec.width, spec.src, spec.dest)

    def generate_cond_operand(self, spec: OperandSpec, _: Instruction) -> Operand:
        cond = random.choice(spec.values)
        return CondOperand(cond)

    def generate_mem_operand(self, spec: OperandSpec, inst: Instruction) -> Operand:
        choices = self._filter_invalid_operands(spec, inst)
        address_reg = random.choice(choices)
        return MemoryOperand(address_reg, spec.width, spec.src, spec.dest)
