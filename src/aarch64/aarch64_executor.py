"""
File: Implementation of executor for x86 architecture
  - Interfacing with the kernel module
  - Aggregation of the results

Copyright (C) Microsoft Corporation
SPDX-License-Identifier: MIT
"""
from __future__ import annotations
import copy
import subprocess
import os.path
import os
import json
import warnings
import base64
import shlex

import numpy as np
import time
from typing import List, Tuple, Set, Generator, Optional, Any, Dict, Iterable, Callable, Protocol, runtime_checkable, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import defaultdict, OrderedDict, deque
from enum import Enum, auto
from contextlib import contextmanager
from pathlib import Path


from .aarch64_generator import Aarch64TagMemoryAccesses, Aarch64Printer, Aarch64MarkMemoryAccessesNEON, Aarch64SandboxPass, Aarch64MarkRegisterTaints, Aarch64MarkMemoryTaints, Aarch64FullTrace, FullTraceAuxBuffer, BitmapTaintsAuxBuffer
from .. import ConfigurableGenerator
from ..interfaces import HTrace, Input, TestCase, Executor, HardwareTracingError, Analyser, CTrace, InputTaint, TargetDesc
from ..config import CONF
from ..util import Logger, STAT
from .aarch64_target_desc import Aarch64TargetDesc
from .aarch64_connection import Connection, UserlandExecutorImp, LocalExecutorImp, TestCaseRegion, InputRegion, \
    HWMeasurement, ExecutorBatch, aux_buffer_from_bytes, AuxBufferType, ExecutorAuxBuffer
from .aarch64_generator import Pass
from .aarch64_inputgen import solve_for_inputs

from .aarch64_connection import profile_op, ExecutorMemory

# ==================================================================================================
# Helper functions
# ==================================================================================================
def km_write(value, path: str) -> None:
    subprocess.run(f"echo -n {value} > {path}", shell=True, check=True)


def km_write_bytes(value: bytes, path: str) -> None:
    with open(path, "wb") as f:
        f.write(value)


def can_set_reserved() -> bool:
    """
    Check if setting reserved bits is possible on the current CPU.
    :return: True if it's possible, False otherwise
    """
    reserved_requested = \
        any(CONF._actors[a]['data_properties']['reserved_bit'] for a in CONF._actors) or \
        any(CONF._actors[a]['data_ept_properties']['reserved_bit'] for a in CONF._actors)
    if reserved_requested:
        physical_bits = int(
            subprocess.run(
                "lscpu | grep 'Address sizes' | awk '{print $3}'",
                shell=True,
                check=True,
                capture_output=True).stdout.decode().strip())
        if physical_bits > 51:
            return False
    return True


def is_kernel_module_installed() -> bool:
    return os.path.isfile("/dev/executor")


def configure_kernel_module() -> None:
    km_write(CONF.executor_warmups, '/sys/x86_executor/warmups')
    km_write("1" if getattr(CONF, 'x86_executor_enable_ssbp_patch') else "0",
             "/sys/x86_executor/enable_ssbp_patch")
    km_write("1" if getattr(CONF, 'x86_executor_enable_prefetcher') else "0",
             "/sys/x86_executor/enable_prefetcher")
    km_write("1" if CONF.enable_pre_run_flush else "0", "/sys/x86_executor/enable_pre_run_flush")
    km_write(CONF.executor_mode, "/sys/x86_executor/measurement_mode")
    km_write("1" if getattr(CONF, 'x86_enable_hpa_gpa_collisions') else "0",
             "/sys/x86_executor/enable_hpa_gpa_collisions")


def read_trace(
    n_reps: int,
    n_inputs: int,
    enable_warnings: bool = True) -> Generator[Tuple[int, int, int, List[int]], None, None]:
    """
    Generator function that reads the traces from the kernel module.
    The generator handles the batched output of the kernel module and yields the traces one by one.
    The traces are read in reverse order.

    Example:
    Assume the kernel module output for n_reps=2 and n_inputs=2 is:
    ```
    htrace1, pfc1..5
    htrace0, pfc1..5
    done
    htrace1, pfc1..5
    htrace0, pfc1..5
    done
    ```
    then the generator will yield the following tuples:
    ```
    (0, 1, htrace1, [pfc1..5])
    (0, 0, htrace0, [pfc1..5])
    (1, 1, htrace1, [pfc1..5])
    (1, 0, htrace0, [pfc1..5])
    ```

    :param n_reps: number of repetitions of the measurements
    :param n_inputs: number of inputs
    :param enable_warnings: if True, the function will print warnings if the kernel module output is
           malformed or if it returns an error
    :return: a generator that yields a tuple (repetition, input_id, htrace, [pfc1, ..., pfc5])
    :raises HardwareTracingError: if the kernel module output is malformed
    """
    if n_inputs <= 0:
        return
    LOG = Logger()

    rep_id = 0
    last_input_id = n_inputs - 1
    while rep_id < n_reps:
        input_id: int = last_input_id
        reading_finished: bool = False
        while not reading_finished:
            # read the next batch of traces from the kernel module
            output = subprocess.check_output(
                f"taskset -c {CONF.executor_taskset} cat /sys/x86_executor/trace", shell=True)
            lines = output.decode().split("\n")

            # parse the output
            for line in lines:
                # print(rep_id, input_id, line)
                # skip empty lines
                if not line:
                    continue

                # we reached the end of the batch? read the next batch
                if 'done' in line:
                    reading_finished = True
                    break

                # transform the line into a sequence of ints
                line_words = line.split(",")
                line_ints = [int(x) for x in line_words]

                # if the line width is unexpected, it's an error
                if len(line_words) != 6:
                    if enable_warnings:
                        LOG.warning("executor", f"Unexpected line width: {len(line_words)}")
                    rewind_km_output_to_end()
                    raise HardwareTracingError()

                # if the hardware trace is zero, it's an error
                if line_ints[0] == 0:
                    if enable_warnings:
                        LOG.warning("executor", "Kernel module error; see dmesg for details")
                    rewind_km_output_to_end()
                    raise HardwareTracingError()

                # yield the trace
                yield rep_id, input_id, line_ints[0], line_ints[1:]

                # move to next input
                input_id -= 1
                if input_id < 0:
                    # if we reached the end of a repetition, restart the input counter
                    input_id = last_input_id
                    rep_id += 1
        assert input_id == last_input_id, f"input_id: {input_id}, rep_id: {rep_id}"
    return


def rewind_km_output_to_end():
    """
    Read to the end of the kernel module output, until the 'done' line.
    """
    while True:
        output = subprocess.check_output(
            f"taskset -c {CONF.executor_taskset} cat /sys/x86_executor/trace", shell=True)
        if 'done' in output.decode():
            break

# Helpers

def pass_on_test_case(test_case: TestCase, passes: List[Pass]):
	for p in passes:
		p.run_on_test_case(test_case)

def write_test_case_remotely(test_case: TestCase, connection: Connection, workdir: str, local_filename: str, target_desc: TargetDesc) -> str:

	def upload_test(test_case: TestCase, connection: Connection, workdir: str) -> str:
		remote_filename = f'{workdir}/remote_{os.path.basename(test_case.bin_path)}'
		connection.push(test_case.bin_path, remote_filename)
		return remote_filename

	def assemble_local_test_case(test_case: TestCase, base_filename: str):
		printer = Aarch64Printer(target_desc)
		test_case.bin_path, test_case.asm_path, test_case.obj_path = (f'{base_filename}.{suffix}' for suffix in ('bin', 'asm', 'o'))
	
		printer.print(test_case, test_case.asm_path)
	
		ConfigurableGenerator.assemble(test_case.asm_path, test_case.obj_path, test_case.bin_path)

	assemble_local_test_case(test_case, local_filename)

	remote_filename = upload_test(test_case, connection, workdir)

	os.remove(test_case.bin_path)
	#os.remove(test_case.asm_path)
	os.remove(test_case.obj_path)
	return remote_filename


def create_scenario_batch(remote_inputs: List[str],
						  remote_test_filename: str,
						  repeats: int,
						  remote_batch_output_filename: str) -> ExecutorBatch:
	batch = ExecutorBatch()
	batch.repeats = repeats

	for input_fname in remote_inputs:
		batch.add_input(input_fname)

	batch.add_test(remote_test_filename)

	batch.output = remote_batch_output_filename or 'remote_batch_output'

	return batch


def extract_json_objects(blob: str) -> List[dict]:
	"""Extract JSON objects from a raw blob string returned by the executor."""
	objs = []
	brace_level = 0
	start_idx = None

	for i, char in enumerate(blob):
		if char == '{':
			if brace_level == 0:
				start_idx = i
			brace_level += 1
		elif char == '}':
			brace_level -= 1
			if brace_level == 0 and start_idx is not None:
				obj_str = blob[start_idx:i + 1]
				try:
					objs.append(json.loads(obj_str))
				except json.JSONDecodeError as e:
					warnings.warn(f"Skipping malformed JSON object: {obj_str} ({e}).")
				start_idx = None  # reset

	return objs

def flatten_ordered_dict(d: OrderedDict) -> OrderedDict:
	flat = OrderedDict()
	for outer_key, inner_dict in d.items():
		for inner_key, value in inner_dict.items():
			flat[inner_key] = value
	return flat


# Blocks
class StageType(Enum):
	LOAD = auto()
	TRANSFORM = auto()
	EXECUTE = auto()
	ANALYZE = auto()

	@classmethod
	def order(cls) -> List[StageType]:
		return list(cls)

	@classmethod
	def index(cls, stage_type: StageType) -> int:
		return cls.order().index(stage_type)

	@classmethod
	def before(cls, a: StageType, b: StageType) -> bool:
		return cls.index(a) < cls.index(b)

	def __repr__(self):
		return f"<StageType.{self.name}>"



@runtime_checkable
class BlockProtocol(Protocol):
	required_keys: Set[str]
	provided_keys: Set[str]
	name: str
	stage_type: StageType

	def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
		...

class Block(ABC, BlockProtocol):
	"""Abstract base for pipeline blocks."""
	required_keys: Set[str] = set()
	provided_keys: Set[str] = set()
	name: str = "uninitialized"
	stage_type: StageType


	def __init__(self, stage_type: StageType = None):
		if stage_type is None:
			raise ValueError(f"{self.__class__.__name__} must have a stage_type explicitly set.")
		self.stage_type = stage_type

		if self.name == "uninitialized":
			self.name = self.__class__.__name__


	@abstractmethod
	def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute this block with a given context,
		producing new key-value pairs for the pipeline context."""
		pass

	# Chain multiple blocks into a Stage using &
	def __and__(self, other: Block) -> StageBuilder:
		if not isinstance(other, BlockProtocol):
			raise ValueError(f"other is of type {type(other)}, Should be of type Block.")
		if self.stage_type != other.stage_type:
			raise ValueError(f"other block is for stage {other.stage_type} while self block is for stage {self.stage_type}.")

		return StageBuilder([self, other], stage_type=self.stage_type)

	# Support chaining blocks into PipelineBuilder using |
	def __or__(self, other: Union[Block, StageBuilder, PipelineBuilder]) -> PipelineBuilder:
		builder = PipelineBuilder()
		builder.add_stagebuilder(StageBuilder([self], stage_type=self.stage_type))
		if isinstance(other, BlockProtocol):
			builder.add_stagebuilder(StageBuilder([other], stage_type=other.stage_type))
		elif isinstance(other, StageBuilder):
			builder.add_stagebuilder(other)
		elif isinstance(other, PipelineBuilder):
			builder.merge(other)
		else:
			raise TypeError(f"Unsupported type {type(other)} for | operator with Block")
		return builder

	def __repr__(self):
		return (
			f"<{self.__class__.__name__}:Block"
			f"name={self.name} "
			f"stage={self.stage_type} "
			f"requires={list(self.required_keys)} "
			f"provides={list(self.provided_keys)}>"
		)


# =====================================================
# Stage management
# =====================================================
class StageBuilder:
	def __init__(self, blocks: List[Block] = None, stage_type: StageType = None):
		self.blocks: List[Block] = blocks or []
		if stage_type is None:
			raise ValueError(f"{self.__class__.__name__} must have a stage_type explicitly set. ")
		self.stage_type: StageType = stage_type

	# Add more blocks via &
	def __and__(self, other: Union[Block, StageBuilder]) -> StageBuilder:

		if not isinstance(other, (Block, StageBuilder)):
			raise TypeError(f"Cannot combine StageBuilder with object of type: {type(other)}")

		if self.stage_type != other.stage_type:
			raise ValueError(f"other is for stage {other.stage_type} while self stage builder is for stage {self.stage_type}.")

		new_blocks = self.blocks.copy()

		if isinstance(other, BlockProtocol):
			new_blocks.append(other)
		else:
			new_blocks.extend(other.blocks)

		return StageBuilder(new_blocks, stage_type=self.stage_type)

	# Merge into pipeline builder via |
	def __or__(self, other: Union[Block, StageBuilder, PipelineBuilder]) -> PipelineBuilder:
		builder = PipelineBuilder()
		builder.add_stagebuilder(self)
		if isinstance(other, BlockProtocol):
			builder.add_stagebuilder(StageBuilder([other], stage_type=other.stage_type))
		elif isinstance(other, StageBuilder):
			builder.add_stagebuilder(other)
		elif isinstance(other, PipelineBuilder):
			builder.merge(other)
		else:
			raise TypeError(f"Unsupported type {type(other)} for | operator with StageBuilder")
		return builder

	def __repr__(self):
		block_repr = ", ".join(repr(b) for b in self.blocks)
		return f"<StageBuilder stage={self.stage_type} blocks=[{block_repr}]>"

class Stage:
	def __init__(self, stage_type: StageType):
		self.stage_type = stage_type
		self.blocks: List[Block] = []

	def add_block(self, block: Block):
		if self.stage_type != block.stage_type:
			raise ValueError(f"Block is for stage {block.stage_type} while self stage is for stage {self.stage_type}.")
		self.blocks.append(block)

	def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
		ordered = self._topo_sort(ctx)

		for block in ordered:
        
			before_keys = set(ctx.keys())

#			print(f"Executing block of type: {block.__class__.__name__}")
			provided = block.run(ctx)
			if not isinstance(provided, dict):
				raise TypeError(f"Block {block.name} must return a dict, got {type(provided)}")

			after_keys = set(provided.keys())

			added_keys = after_keys - before_keys

			if added_keys != block.provided_keys:
				missing_keys = block.provided_keys - added_keys
				unexpected_keys =  added_keys - block.provided_keys
				raise RuntimeError(
							f"Stage {self.__class__.__name__} of type {self.stage_type} violated provided_keys contract of Block {block.name}. "
							f"Missing: {missing_keys}, Extra: {unexpected_keys}"
						)

			self._merge_into(ctx, provided)

		return ctx

	def _merge_into(self, ctx: dict, updates: dict) -> None:
		ctx.update(updates)

	def _topo_sort(self, ctx: Dict[str, Any]) -> List[Block]:
		producers: Dict[str, Block] = {}
		for b in self.blocks:
			for key in b.provided_keys:
				if key in ctx:
					raise RuntimeError(f"key {key} already satisfied from ctx context {ctx}.")
				if key in producers:
					raise RuntimeError(f"key {key} defined by multiple blocks: {producers[key].name} and {b.name}.")

				producers[key] = b

		deps: Dict[Block, Set[Block]] = {b: set() for b in self.blocks}
		for b in self.blocks:
			for req in b.required_keys:
				if req in producers:
					deps[b].add(producers[req])
				elif req not in ctx:
					raise RuntimeError(f"required key {req} for block {b.name}  is not provided by any block in the stage or by the given context ({ctx=}, {deps=}).")

			if b in deps[b]:
				warnings.warn(f"Block {b.name} is dependent on its own keys, ignoring self-dependency.")
				deps[b].discard(b)

		indeg = {b: len(deps[b]) for b in self.blocks}
		q = deque([b for b in self.blocks if indeg[b] == 0])
		ordered = []
		while q:
			b = q.popleft()
			ordered.append(b)
			for other in self.blocks:
				if b in deps[other]:
					indeg[other] -= 1
					if indeg[other] == 0:
						q.append(other)

		if len(ordered) != len(self.blocks):
			raise RuntimeError(f"Cycle detected in stage {self.stage_type}, cannot topologically sort block within stage {self.stage_type}.")

		return ordered

	def __repr__(self):
		block_repr = ", ".join(repr(b) for b in self.blocks)
		return f"<Stage stage={self.stage_type} blocks=[{block_repr}]>"

# =====================================================
# Pipeline builder
# =====================================================
class PipelineBuilder:

	def __init__(self):
		self._stages: Dict[StageType, Stage] = {}

	def add_block(self, block: Block):
		self.add_stagebuilder(StageBuilder([block], stage_type=block.stage_type))

	def add_stagebuilder(self, stagebuilder: StageBuilder):
		if stagebuilder.stage_type not in self._stages:
			self._stages[stagebuilder.stage_type] = Stage(stagebuilder.stage_type)
		for b in stagebuilder.blocks:
			self._stages[stagebuilder.stage_type].add_block(b)

	def merge(self, other: PipelineBuilder):
		for stype, stage in other._stages.items():
			self._stages.setdefault(stype, Stage(stype)).blocks.extend(stage.blocks)

	def __or__(self, other: Union[Block, StageBuilder, PipelineBuilder]):
		if isinstance(other, BlockProtocol):
			self.add_block(other)
		elif isinstance(other, StageBuilder):
			self.add_stagebuilder(other)
		elif isinstance(other, PipelineBuilder):
			self.merge(other)
		else:
			raise TypeError(f"Unsupported type {type(other)} for | operator")
		return self

	def build(self) -> Pipeline:
		return Pipeline(self._stages)

	def __repr__(self):
		repr_str = "\n".join(f"{stype}: {repr(stage)}" for stype, stage in self._stages.items())
		return f"<PipelineBuilder stages:\n{repr_str}>"

class Pipeline:
	def __init__(self, stages: Dict[StageType, Stage]):
		self._stages = stages

	def run(self, ctx: Dict[str, Any] = None) -> Dict[str, Any]:
		ctx = ctx or {}
		for stype in StageType.order():
			if stype in self._stages:
				ctx = self._stages[stype].run(ctx)
		return ctx

	def __repr__(self):
		repr_str = "\n".join(f"{stype}: {repr(stage)}" for stype, stage in self._stages.items())
		return f"<Pipeline stages:\n{repr_str}>"



class LoadInputsBlock(Block):
	required_keys = {"inputs", "connection", "workdir"}
	provided_keys = {"input_to_filename_dict"}

	def __init__(self, cleanable: Cleanable):
		super().__init__(StageType.LOAD)
		self.cleanable = cleanable

	def _upload_inputs(self, inputs: List[Input], connection: Connection, workdir: str) -> OrderedDict[Input, str]:
		remote_filenames = OrderedDict()
		for idx, inp in enumerate(inputs):
			input_name = f"input{idx}.bin"
			remote_filename = f'{workdir}/{input_name}'
			inp.save(input_name)
			connection.push(input_name, remote_filename)
			os.remove(input_name)
			remote_filenames[inp] = remote_filename
			self.cleanable.register_temp_file(remote_filename)
		return remote_filenames

	def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
		inputs: List[Input] = ctx["inputs"]
		connection: Connection = ctx["connection"]
		workdir: str = ctx["workdir"]

		ctx.update({"input_to_filename_dict": self._upload_inputs(inputs, connection, workdir)})

		return ctx

class LoadSandboxedTestCaseBlock(Block):
	required_keys = {"test_case", "connection", "workdir"}
	provided_keys = {"remote_test_filename", "sandboxed_test_case"}

	def __init__(self, executor: Aarch64RemoteExecutor):
		super().__init__(StageType.TRANSFORM)
		self.executor = executor

	def _upload_sandboxed_test(self, test_case: TestCase, connection: Connection, workdir: str) -> Tuple[str, TestCase]:
		sandboxed_test_case = copy.deepcopy(test_case)
		Aarch64SandboxPass().run_on_test_case(sandboxed_test_case)
		return write_test_case_remotely(sandboxed_test_case, connection, workdir, f'sandboxed_test_case', self.executor.target_desc), sandboxed_test_case

	def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
		test_case: TestCase = ctx["test_case"]
		connection: Connection = ctx["connection"]
		workdir: str= ctx["workdir"]

		remote_test_filename, sandboxed_test_case = self._upload_sandboxed_test(test_case, connection, workdir)
		self.executor.register_temp_file(remote_test_filename)
		ctx.update(
				{
					"remote_test_filename": remote_test_filename,
					"sandboxed_test_case": sandboxed_test_case,
				}
		)
		return ctx

class LoadRawTestCaseBlock(Block):
	required_keys = {"test_case", "connection", "workdir"}
	provided_keys = {"remote_test_filename"}

	def __init__(self, executor: Aarch64RemoteExecutor):
		super().__init__(StageType.LOAD)
		self.executor = executor

	def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
		test_case: TestCase = ctx["test_case"]
		connection: Connection = ctx["connection"]
		workdir: str= ctx["workdir"]

		base_name = os.path.basename(test_case.bin_path)
		remote_test_filename = write_test_case_remotely(test_case, connection, workdir, base_name, self.executor.target_desc)
		self.executor.register_temp_file(remote_test_filename)

		ctx.update(
				{
					"remote_test_filename": remote_test_filename,
				}
		)

		return ctx

class GenerateTaintsAndCTracesBlock(Block):
	required_keys = {"input_to_bitmap_taints_buffer", "input_to_full_trace_buffer"}
	provided_keys = {"input_to_taint", "input_to_ctrace"}

	def __init__(self):
		super().__init__(StageType.TRANSFORM)

	def _process_input(self, input_to_process: Input, bitmap_aux_buffer: BitmapTaintsAuxBuffer, full_trace_buffer: FullTraceAuxBuff) -> Tuple[InputTaint, CTrace]:
		from capstone import Cs, CS_ARCH_ARM64, CS_MODE_ARM
		from capstone.arm64 import ARM64_OP_REG

		def memory_access_size(insnt):
			ops = insnt.operands
			reg_op = ops[0]
			assert len(ops) >= 2, "Unexpected number of operands"
			if reg_op.type == ARM64_OP_REG and dis.reg_name(reg_op.reg).lower().startswith('w'):
				access_size = 4
			else:
				access_size = 8
			return access_size

		num_of_gprs = input_to_process['gpr'].shape[1]

		input_taint = InputTaint()

		gpr_region = input_taint[0]['gpr']
		for i in range(num_of_gprs):
			mask: int = 1 << i
			if bitmap_aux_buffer.regs_input_read_bits & mask:
				gpr_region[i] = True

		# currently, simd region used to store x8 and x9 initial values which corrently used to initialize flags and sp respectively (sp is being overriden with executor appropriate value)
		simd_region = input_taint[0]['simd']

		FLAGS_REG_INDEX = 8
		if bitmap_aux_buffer.regs_input_read_bits & (1 << FLAGS_REG_INDEX):
			simd_region[0] = True

		SP_REG_INDEX = 9
		if bitmap_aux_buffer.regs_input_read_bits & (1 << SP_REG_INDEX):
			simd_region[1] = True

		# Sandbox base is stored in x30 while executing the test case. It should not change while executing the test case. Take the first one.
		sandbox_base = full_trace_buffer.instruction_logs[0].regs[30]
		faulty_region_u8 = input_taint[0]["faulty"].view(np.uint8)
		main_region_u8 = input_taint[0]["main"].view(np.uint8)
		sandbox_size = main_region_u8.size + faulty_region_u8.size

		NOT_ACCESSED, READ, WRITE = 0, 1, 2
		access_table = np.zeros(sandbox_size, dtype=np.uint8)

		dis = Cs(CS_ARCH_ARM64, CS_MODE_ARM)
		dis.detail = True

		for inst_log in full_trace_buffer.instruction_logs:
			encoding_bytes = inst_log.encoding.to_bytes(4, byteorder='little')
			insns = list(dis.disasm(encoding_bytes, 0))
			assert insns, f"Unable to decode Aarch64 instruction: {encoding_bytes.hex()}"

			if inst_log.effective_address == 0xFFFFFFFFFFFFFFFF:
				continue
			offset = inst_log.effective_address - sandbox_base
			mnem = insns[0].mnemonic.lower()
			decoded = f"{mnem} {insns[0].op_str}"
			msg = f"Unexpected offset into sandbox: {offset} = {inst_log.effective_address:#08x} - {sandbox_base:#08x} = inst_log.effective_address - sandbox_base: {decoded}"
			assert 0 <= offset and offset < sandbox_size, msg

			access_size = memory_access_size(insns[0])

			is_write = "str" in mnem
			is_read = "ldr" in mnem
			for i in range(access_size):
				idx = offset + i
				if idx >= sandbox_size:
					break

				if is_read and access_table[idx] == NOT_ACCESSED:
					access_table[idx] = READ
				elif is_write and access_table[idx] == NOT_ACCESSED:
					access_table[idx] = WRITE

		accessed_addresses: List[int] = []

		for offset in range(sandbox_size):
			access = access_table[offset]
			if access != NOT_ACCESSED:
				accessed_addresses.append(offset)

			if access == READ:
				to_change_region = main_region_u8
				to_change_offset = offset

				if offset >= main_region_u8.size:
					to_change_region = faulty_region_u8
					to_change_offset = offset - main_region_u8.size

				to_change_region[to_change_offset] = True

		line_size = 64
		num_sets = 64
		accessed_sets: List[int] = []
		cache_sets = sorted(set(map(lambda addr: (addr // line_size) % num_sets, accessed_addresses)))
		ctrace = CTrace(raw_trace=accessed_addresses)

		return input_taint, ctrace


	def _process_inputs(self, input_to_bitmap_taints_buffer: Dict[str, BitmapTaintsAuxBuffer], input_to_full_trace_buffer: Dict[str, FullTraceAuxBuffer]) -> Tuple[OrderedDict[Input, InputTaint], OrderedDict[Input, CTrace]]:

		assert set(input_to_bitmap_taints_buffer) == set(input_to_full_trace_buffer)
		inputs_to_process = list(input_to_bitmap_taints_buffer.keys())
		if not inputs_to_process:
			return {}

		taints: OrderedDict[Input, InputTaint] = OrderedDict()
		ctraces: OrderedDict[Input, CTrace] = OrderedDict()

		for input_to_process, bitmap_aux_buffer in input_to_bitmap_taints_buffer.items():
			full_trace_buffer = input_to_full_trace_buffer[input_to_process]
			taints[input_to_process], ctraces[input_to_process] = self._process_input(input_to_process, bitmap_aux_buffer, full_trace_buffer)

		return taints, ctraces



	def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
		input_to_bitmap_taints_buffer = ctx['input_to_bitmap_taints_buffer']
		input_to_full_trace_buffer = ctx['input_to_full_trace_buffer']

		input_to_taint, input_to_ctrace = self._process_inputs(input_to_bitmap_taints_buffer, input_to_full_trace_buffer)

		ctx.update({
			"input_to_taint": input_to_taint,
			"input_to_ctrace": input_to_ctrace
		})

		return ctx





class GenerateInputArchFlowBlock(Block):
	required_keys = {"test_case", "input_to_filename_dict", "connection", "workdir", "userland_executor"}
	provided_keys = {"input_to_bitmap_taints_buffer", "input_to_full_trace_buffer"}

	def __init__(self, executor: Aarch64Executor):
		super().__init__(StageType.LOAD)
		self.executor = executor

	def _write_test_with_taint_tracker(self, test_case: TestCase, connection: Connection, workdir: str) -> str:
		patched_test_case = copy.deepcopy(test_case)
		register_taints_pass = Aarch64MarkRegisterTaints()
		sandbox_pass = Aarch64SandboxPass()
		memory_taints_pass = Aarch64MarkMemoryTaints()
		pass_on_test_case(patched_test_case, [register_taints_pass, sandbox_pass, memory_taints_pass])
		return write_test_case_remotely(patched_test_case, connection, workdir, 'generated_taint_tracker', self.executor.target_desc)

	def _write_test_with_full_trace_tracker(self, test_case: TestCase, connection: Connection, workdir: str) -> str:
		patched_test_case = copy.deepcopy(test_case)
		sandbox_pass = Aarch64SandboxPass()
		full_trace_pass = Aarch64FullTrace()
		pass_on_test_case(patched_test_case, [sandbox_pass, full_trace_pass])
		return write_test_case_remotely(patched_test_case, connection, workdir, 'generated_full_trace_tracker', self.executor.target_desc)


	def _measure(cls, test_case: TestCase, input_to_filename: Dict[Input, str],  workdir: str, userland_executor: UserlandExecutor, connection: Connection) -> Tuple[OrderedDict[Input, BitmapTaintsAuxBuffer], OrderedDict[Input, FullTraceAuxBuffer]]:

		def _load_trace_parse(write_test_method, buffer_layout, input_to_iid) -> OrderedDict[Input, ExecutorAuxBuffer]:
			remote_test_case_filename = write_test_method(test_case, connection, workdir)

			userland_executor.checkout_region(TestCaseRegion())
			userland_executor.write_file(remote_test_case_filename)
			userland_executor.trace()

			result: OrderedDict[Input, ExecutorAuxBuffer] = OrderedDict()

			for inp, iid in input_to_iid.items():
				userland_executor.checkout_region(InputRegion(iid))
				result[inp] = aux_buffer_from_bytes(buffer_layout, userland_executor.aux_buffer)

			connection.shell(f'rm {remote_test_case_filename}')

			return result

		input_to_iids = {}
		for inp, remote_input_filename in input_to_filename.items():
			iid = userland_executor.allocate_iid()
			input_to_iids[inp] = iid
			userland_executor.checkout_region(InputRegion(iid))
			userland_executor.write_file(remote_input_filename)

		input_to_bitmap_taints_buffer: OrderedDict[Input, BitmapTaintsAuxBuffer] = _load_trace_parse(cls._write_test_with_taint_tracker, AuxBufferType.BITMAP_TAINTS, input_to_iids)
		input_to_full_trace_buffer: OrderedDict[Input, FullTraceAuxBuffer] = _load_trace_parse(cls._write_test_with_full_trace_tracker, AuxBufferType.FULL_TRACE, input_to_iids)

		userland_executor.discard_all_inputs()
		return input_to_bitmap_taints_buffer, input_to_full_trace_buffer

	def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
		test_case: TestCase = ctx["test_case"]
		input_to_filename_dict: OrderedDict[Input, str] = ctx["input_to_filename_dict"]
		connection: Connection = ctx["connection"]
		workdir: str = ctx["workdir"]
		userland_executor: UserlandExecutor = ctx["userland_executor"]

		input_to_bitmap_taints_buffer, input_to_full_trace_buffer = self._measure(test_case, input_to_filename_dict, workdir, userland_executor, connection)

		ctx.update({
		    "input_to_bitmap_taints_buffer": input_to_bitmap_taints_buffer,
			"input_to_full_trace_buffer": input_to_full_trace_buffer
		})

		return ctx


class LoadInputVariantsBlock(Block):
	required_keys = {"input_to_filename_dict", "input_to_equivalence_class", "connection", "workdir"}
	provided_keys = {"input_equivalence_classes_filenames"}

	def __init__(self):
		super().__init__(StageType.LOAD)

	@classmethod
	def _upload_inputs(cls, input_to_filename_dict: OrderedDict[Input, str], input_to_equivalence_class: OrderedDict[Input, List[Input]], 
					connection: Connection, workdir: str) -> OrderedDict[Input, OrderedDict[Input, str]]:
		remote_filenames = OrderedDict()
		for inp, remote_filename in input_to_filename_dict.items():
			remote_filenames[inp] = OrderedDict()
			remote_filenames[inp][inp] = remote_filename # Add original input to it's own equivalence class of remote filenames

			for idx, new_input in enumerate(input_to_equivalence_class[inp]):
				new_input_name = f"{os.path.basename(remote_filename)}.{idx}"
				new_remote_filename = f'{workdir}/{new_input_name}'
				new_input.save(new_input_name)
				connection.push(new_input_name, new_remote_filename)
				os.remove(new_input_name)
				remote_filenames[inp][new_input] = new_remote_filename
		return remote_filenames

	def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
		input_to_equivalence_class: OrderedDict[Input, List[Input]] = ctx["input_to_equivalence_class"]
		input_to_filename_dict: OrderedDict[Input, str] = ctx["input_to_filename_dict"]
		connection: Connection = ctx["connection"]
		workdir: str = ctx["workdir"]

		input_equivalence_classes_filenames = self._upload_inputs(input_to_filename_dict, input_to_equivalence_class, connection, workdir)

		ctx.update(
				{
					"input_equivalence_classes_filenames": input_equivalence_classes_filenames,
				}
			)

		return ctx


class GenerateInputVariantsBlock(Block):
	required_keys = {"input_to_bitmap_taints_buffer", "input_to_full_trace_buffer", "variants_per_input"}
	provided_keys = {"input_to_equivalence_class"}

	def __init__(self, seed: Optional[int] = None):
		super().__init__(StageType.LOAD)
		if seed is None:
			seed = int(time.time() * 1000) & ((1 << 63) - 1)
		self.seed = seed
		self._seed_seq = np.random.SeedSequence(seed)
		self.rng = np.random.default_rng(self._seed_seq)


	def _generate_single_input_mutation(self, input_to_mutate: Input, bitmap_aux_buffer: BitmapTaintsAuxBuffer, full_trace_buffer: FullTraceAux, p_high: float, p_low: float) -> Input:
		from capstone import Cs, CS_ARCH_ARM64, CS_MODE_ARM
		from capstone.arm64 import ARM64_OP_REG

		def _classify_resources(W: int, R: int, IR: int, number_of_relevant_bits: int = 64) -> Tuple[int, int, int, int]:
			if not isinstance(number_of_relevant_bits, int) or not(64 >= number_of_relevant_bits >= 0):
				raise ValueError(
						f"number_of_relevant_bits must be between 0 and 64, inclusice."
						f"Got {number_of_relevant_bits}."
				)
	
			mask = (1 << number_of_relevant_bits) - 1
	
			modifiable = (~IR) & mask
			high_priority = modifiable & (R | W)
			low_priority = modifiable & ~(R | W)
			must_keep = IR & mask
	
			return modifiable, high_priority, low_priority, must_keep


		def _mutate_region(region: np.ndarray, region_name: str, offset: int, prob: float):
			if self.rng.random() < prob:
				prev = region[offset]
				region[offset] = self.rng.integers(0, 256, dtype=np.uint8)
				print(f"{self.__class__.__name__}: {region_name}[{offset}] {prev} ===> {region[offset]}")

		def memory_access_size(insnt):
			ops = insnt.operands
			reg_op = ops[0]
			assert len(ops) >= 2, "Unexpected number of operands"
			if reg_op.type == ARM64_OP_REG and dis.reg_name(reg_op.reg).lower().startswith('w'):
				access_size = 4
			else:
				access_size = 8
			return access_size


		num_of_gprs = input_to_mutate['gpr'].shape[1]

		_, high_priority_registers, low_priority_registers, _ = _classify_resources(
				bitmap_aux_buffer.regs_write_bits,
				bitmap_aux_buffer.regs_read_bits,
				bitmap_aux_buffer.regs_input_read_bits,
				num_of_gprs
			)

		new_input = input_to_mutate.copy()

		gpr_region = new_input[0]['gpr']
		for i in range(num_of_gprs):
			mask: int = 1 << i
   
			if high_priority_registers & mask:
				if self.rng.random() < p_high:
					prev = gpr_region[i]
					gpr_region[i] = self.rng.integers(0, np.iinfo(np.uint64).max + 1, dtype=np.uint64)
					print(f"{self.__class__.__name__}: GPR[{i}] {prev} ===> {gpr_region[i]}")

			elif low_priority_registers & mask:
				if self.rng.random() < p_low:
					prev = gpr_region[i]
					gpr_region[i] = self.rng.integers(0, np.iinfo(np.uint64).max + 1, dtype=np.uint64)
					print(f"{self.__class__.__name__}: GPR[{i}] {prev} ===> {gpr_region[i]}")

		# Sandbox base is stored in x30 while executing the test case. It should not change while executing the test case. Take the first one.
		sandbox_base = full_trace_buffer.instruction_logs[0].regs[30]
		faulty_region_u8 = new_input[0]["faulty"].view(np.uint8)
		main_region_u8 = new_input[0]["main"].view(np.uint8)
		sandbox_size = main_region_u8.size + faulty_region_u8.size

		NOT_ACCESSED, READ, WRITE = 0, 1, 2
		access_table = np.zeros(sandbox_size, dtype=np.uint8)

		dis = Cs(CS_ARCH_ARM64, CS_MODE_ARM)
		dis.detail = True

		for inst_log in full_trace_buffer.instruction_logs:
			encoding_bytes = inst_log.encoding.to_bytes(4, byteorder='little')
			insns = list(dis.disasm(encoding_bytes, 0))
			assert insns, f"Unable to decode Aarch64 instruction: {encoding_bytes.hex()}"

			if inst_log.effective_address == 0xFFFFFFFFFFFFFFFF:
				continue
			offset = inst_log.effective_address - sandbox_base
			mnem = insns[0].mnemonic.lower()
			decoded = f"{mnem} {insns[0].op_str}"
			msg = f"Unexpected offset into sandbox: {offset} = {inst_log.effective_address:#08x} - {sandbox_base:#08x} = inst_log.effective_address - sandbox_base: {decoded}"
			assert 0 <= offset and offset < sandbox_size, msg

			access_size = memory_access_size(insns[0])

			is_write = "str" in mnem
			is_read = "ldr" in mnem
			for i in range(access_size):
				idx = offset + i
				if idx >= sandbox_size:
					break

				if is_read and access_table[idx] == NOT_ACCESSED:
					access_table[idx] = READ
				elif is_write and access_table[idx] == NOT_ACCESSED:
					access_table[idx] = WRITE


		for offset in range(sandbox_size):
			access = access_table[offset]
			if access == READ:
				continue

			if offset < main_region_u8.size:
				_mutate_region(main_region_u8, "main", offset, p_high)
			else:
				_mutate_region(faulty_region_u8, "faulty", offset - main_region_u8.size, p_high)

		return new_input


	def _mutate_input(self, input_to_mutate: Input, variants: int, bitmap_aux_buffer: BitmapTaintsAuxBuffer, full_trace_buffer: FullTraceAuxBuff, p_high: float , p_low: float) -> List[Input]:

		results: List[Input] = []

		for _ in range(variants):
			new_input_variation = self._generate_single_input_mutation(input_to_mutate, bitmap_aux_buffer, full_trace_buffer, p_high, p_low)
			results.append(new_input_variation)

		return results


	def _mutate_inputs(self, input_to_bitmap_taints_buffer: Dict[str, BitmapTaintsAuxBuffer], input_to_full_trace_buffer: Dict[str, FullTraceAuxBuffer], variants: int, p_high: float = 0.8, p_low: float = 0.2) -> OrderedDict[Input, List[Input]]:

		assert set(input_to_bitmap_taints_buffer) == set(input_to_full_trace_buffer)
		inputs_to_mutate = list(input_to_bitmap_taints_buffer.keys())
		if not inputs_to_mutate:
			return {} 

		results: OrderedDict[Input, List[Input]] = OrderedDict()

		for input_to_mutate, bitmap_aux_buffer in input_to_bitmap_taints_buffer.items():
			full_trace_buffer = input_to_full_trace_buffer[input_to_mutate]
			results[input_to_mutate] = self._mutate_input(input_to_mutate, variants, bitmap_aux_buffer, full_trace_buffer, p_high, p_low)
		
		return results

	def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
		variants_per_input: int = ctx["variants_per_input"]
		input_to_bitmap_taints_buffer: OrderedDict[Input, BitmapTaintsAuxBuffer] = ctx["input_to_bitmap_taints_buffer"]
		input_to_full_trace_buffer: OrderedDict[Input, FullTraceAuxBuffer] = ctx["input_to_full_trace_buffer"]

		input_to_equivalence_class: OrderedDict[Input, List[Input]] = self._mutate_inputs(input_to_bitmap_taints_buffer, input_to_full_trace_buffer, variants_per_input, p_high = 1, p_low = 0.8);

		ctx.update({
			"input_to_equivalence_class": input_to_equivalence_class
		})

		return ctx

class GenerateMTETestVariants(Block):
	required_keys = {"test_case", "input_to_filename_dict", "connection", "workdir", "userland_executor"}
	provided_keys = {"remote_test_filename_correct_tags", "remote_input_to_test_case_filename_incorrect_tags"}

	def __init__(self, executor: Aarch64Executor):
		super().__init__(StageType.TRANSFORM)
		self.executor = executor

	@classmethod
	def _write_test_with_correct_tags(cls, test_case: TestCase, connection: Connection, workdir: str) -> str:
		patched_test_case = copy.deepcopy(test_case)
		tagging_pass = Aarch64TagMemoryAccesses(memory_accesses_to_guess_tag=None)
		pass_on_test_case(patched_test_case, [tagging_pass])
		return write_test_case_remotely(patched_test_case, connection, workdir, 'generated_correct_tags', self.executor.target_desc)

	@classmethod
	def _write_test_with_incorrect_tags(cls, test_case: TestCase, connection: Connection, workdir: str, filename_suffix: str, tag_ids_to_guess: List[int]) -> str:
		patched_test_case = copy.deepcopy(test_case)
		tagging_pass = Aarch64TagMemoryAccesses(memory_accesses_to_guess_tag=tag_ids_to_guess)
		pass_on_test_case(patched_test_case, [tagging_pass])
		return write_test_case_remotely(patched_test_case, connection, workdir, f'generated_patched_{filename_suffix}', self.executor.target_desc)

	@classmethod
	def _write_test_case_with_bitmap_trace(cls, test_case: TestCase, connection: Connection, workdir: str) -> str:
		patched_test_case = copy.deepcopy(test_case)
		tagging_pass = Aarch64TagMemoryAccesses(memory_accesses_to_guess_tag=None)
		marking_pass = Aarch64MarkMemoryAccessesNEON()
		pass_on_test_case(patched_test_case, [tagging_pass, marking_pass])
		return write_test_case_remotely(patched_test_case, connection, workdir, 'generated_retrieve_bitmap', self.executor.target_desc)

	@classmethod
	def _measure_architecturally_not_accessed_memory_addresses(cls, test_case: TestCase, input_to_filename: Dict[Input, str],  workdir: str, userland_executor: UserlandExecutor, connection: Connection) -> Dict[Input, List[int]]:

		def parse_bitmap(n: str, bit: int) -> List[int]:
			result = []
			for index, b in enumerate(n):
				if bit == int(b):
					result.append(len(n) - (index + 1))
			return result

		inp_to_iids = {}
		for inp, remote_input_filename in input_to_filename.items():
			iid = userland_executor.allocate_iid()
			inp_to_iids[inp] = iid
			userland_executor.checkout_region(InputRegion(iid))
			userland_executor.write_file(remote_input_filename)


		remote_test_case_filename = cls._write_test_case_with_bitmap_trace(test_case, connection, workdir)

		userland_executor.checkout_region(TestCaseRegion())
		userland_executor.write_file(remote_test_case_filename)
		userland_executor.trace()

		not_architectural_memory_accesses: Dict[str, List[int]] = {}

		for inp, iid in inp_to_iids.items():
			userland_executor.checkout_region(InputRegion(iid))
			measurement = userland_executor.hardware_measurement().memory_ids
			not_architectural_memory_accesses[inp] = parse_bitmap(measurement, bit=0)

		userland_executor.discard_all_inputs()
		connection.shell(f'rm {remote_test_case_filename}')

		return not_architectural_memory_accesses

	@classmethod
	def _create_tests_with_incorrect_tags(cls, test_case: TestCase, input_to_filename: Dict[Input, str], workdir: str, userland_executor: UserlandExecutor, connection: Connection) -> Dict[Input, str]:

		input_to_test_case_filename: Dict[str, str] = {}

		not_architectural_memory_accesses: Dict[Input, List[int]] = cls._measure_architecturally_not_accessed_memory_addresses(test_case, input_to_filename, workdir, userland_executor, connection)

		for inp, measurement in not_architectural_memory_accesses.items() :
			remote_input_filename = input_to_filename[inp]
			filename_suffix = os.path.basename(remote_input_filename.rstrip('/'))
			input_to_test_case_filename[inp] = cls._write_test_with_incorrect_tags(
					test_case, connection, workdir, filename_suffix, measurement)

		return input_to_test_case_filename

	def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
		test_case: TestCase = ctx["test_case"]
		input_to_filename_dict: OrderedDict[Input, str] = ctx["input_to_filename_dict"]
		connection: Connection = ctx["connection"]
		workdir: str = ctx["workdir"]
		userland_executor: UserlandExecutor = ctx["userland_executor"]

		# Correct tags
		remote_filename_correct_tags = self._write_test_with_correct_tags(test_case, connection, workdir)

		# Incorrect tags
		input_to_test_case_filename_incorrect_tags = self._create_tests_with_incorrect_tags(test_case, input_to_filename_dict, workdir, userland_executor, connection)

		ctx.update({
		    "remote_test_filename_correct_tags": remote_filename_correct_tags,
		    "remote_input_to_test_case_filename_incorrect_tags": input_to_test_case_filename_incorrect_tags
		})

		return ctx

class TraceScenarioBatch(Block):
	required_keys = {"input_to_filename_dict", "remote_test_filename", "repeats", "workdir", "userland_executor"}
	provided_keys = {"filename_to_jsons"}  # keyed by remote input filename, values = list of HTrace objects

	def __init__(self, cleanable: Cleanable):
		super().__init__(StageType.EXECUTE)
		self.cleanable = cleanable


	def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
		input_to_filename_dict: OrderedDict[Input, str] = ctx["input_to_filename_dict"]
		remote_test_filename: str = ctx["remote_test_filename"]
		repeats: List[int] = ctx["repeats"]
		workdir: str = ctx["workdir"]
		userland_executor: UserlandExecutor = ctx["userland_executor"]

		remote_inputs = input_to_filename_dict.values()
		remote_batch_output = f'{workdir}/remote_batch_output_r{repeats}'
		batch = create_scenario_batch(remote_inputs, remote_test_filename, repeats, remote_batch_output)
		self.cleanable.register_temp_file(batch.output)

		trace_output = userland_executor.trace(batch)
		with profile_op('extract_jsons'):
			json_objs = extract_json_objects(trace_output)

		# Aggregate traces per remote input
		filename_to_jsons: Dict[str, OrderedDict[str, object]] = {input_name: [] for input_name in remote_inputs}

		for js in json_objs:
			input_name = js.get('input_name')
			if input_name not in remote_inputs:
				warnings.warn(f"Unexpected input_name returned: {input_name}")
				continue

			filename_to_jsons[input_name].append(js)

		ctx.update({
			"filename_to_jsons": filename_to_jsons,
		})

		return ctx


class ArchiteturalFlowTraceBlock(Block):
	required_keys = {"test_case", "workdir", "connection", "userland_executor", "input_to_filename_dict"}
	provided_keys = {"architectural_flows"}

	def __init__(self, executor: Aarch64Executor):
		super().__init__(StageType.LOAD)
		self.executor = executor

	@classmethod
	def _write_test_case_with_bitmap_trace(cls, test_case: TestCase, connection: Connection, workdir: str) -> str:
		patched_test_case = copy.deepcopy(test_case)
		marking_pass = Aarch64MarkMemoryAccessesNEON()
		sandbox = Aarch64SandboxPass()
		pass_on_test_case(patched_test_case, [sandbox, marking_pass])
		return write_test_case_remotely(patched_test_case, connection, workdir, 'generated_retrieve_bitmap', self.executor.target_desc)

	@classmethod
	def _measure_architecturally_not_accessed_memory_addresses(cls, test_case: TestCase, input_to_filename: Dict[Input, str],  workdir: str, userland_executor: UserlandExecutor, connection: Connection) -> OrderedDict[Input, List[int]]:

		def parse_bitmap(n: str, bit: int) -> List[int]:
			result = []
			for index, b in enumerate(n):
				if bit == int(b):
					result.append(len(n) - (index + 1))
			return result

		inp_to_iids = {}
		for inp, remote_input_filename in input_to_filename.items():
			iid = userland_executor.allocate_iid()
			inp_to_iids[inp] = iid
			userland_executor.checkout_region(InputRegion(iid))
			userland_executor.write_file(remote_input_filename)


		remote_test_case_filename = cls._write_test_case_with_bitmap_trace(test_case, connection, workdir)

		userland_executor.checkout_region(TestCaseRegion())
		userland_executor.write_file(remote_test_case_filename)
		userland_executor.trace()

		not_architectural_memory_accesses: OrderedDict[str, List[int]] = OrderedDict()

		for inp, iid in inp_to_iids.items():
			userland_executor.checkout_region(InputRegion(iid))
			measurement = userland_executor.hardware_measurement().memory_ids
			not_architectural_memory_accesses[inp] = parse_bitmap(measurement, bit=1)

		userland_executor.discard_all_inputs()
		connection.shell(f'rm {remote_test_case_filename}')

		return not_architectural_memory_accesses

	def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
		connection: Connection = ctx["connection"]
		test_case: TestCase = ctx["test_case"]
		workdir: str = ctx["workdir"]
		userland_executor: UserlandExecutor = ctx["userland_executor"]
		input_to_filename_dict: OrderedDict[Input, List[int]] = ctx["input_to_filename_dict"]

		architectural_flows = self._measure_architecturally_not_accessed_memory_addresses(test_case, input_to_filename_dict, workdir, userland_executor, connection)

		ctx.update({
            "architectural_flows": architectural_flows
		})

		return ctx

class CleanupRemoteFiles(Block):
	required_keys = {"input_to_filename_dict", "input_equivalence_classes_filenames", "remote_sandboxed_test_filename", "connection", "scenario_batch_filenames"}
	provided_keys = set()

	def __init__(self):
		super().__init__(StageType.EXECUTE)

	def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
		connection: Connection = ctx["connection"]
		scenario_batch: ExecutorBatch = ctx["scenario_batch"]
		remote_sandboxed_test_filename: str = ctx["remote_sandboxed_test_filename"]
		input_to_filename_dict: OrderedDict[Input, str] = ctx["input_to_filename_dict"]
		input_equivalence_classes_filenames: OrderedDict[Input, OrderedDict[Input, str]] = ctx["input_equivalence_classes_filenames"]

		for remote_input in input_to_filename_dict.values():
			try:
				connection.shell(f"rm {remote_input}", privileged=True)
			except Exception as e:
				warnings.warn(f"Failed to remove main input: {remote_input} ({e}).")


		for remote_equivalence_class_filenames in input_equivalence_classes_filenames.values():
			for remote_input in remote_equivalence_class_filenames.values():
				try:
					connection.shell(f"rm {remote_input}", privileged=True)
				except Exception as e:
					warnings.warn(f"Failed to remove input of equivalence class: {remote_input} ({e}).")

		try:
			connection.shell(f"rm {remote_sandboxed_test_filename}", privileged=True)
		except Exception as e:
			warnings.warn(f"Failed to remove test file: {remote_sandboxed_test_filename} ({e}).")

		for filename in scenario_batch_filenames:
			try:
				connection.shell(f"rm {filename}", privileged=True)
			except Exception as e:
				warnings.warn(f"Failed to remove batch output file: {filename} ({e}).")

		return ctx

class AnalyserBlock(Block):
	required_keys = {"filename_to_htraces_list", "input_equivalence_classes_filenames", "sandboxed_test_case"}
	provided_keys = {"violations"}

	def __init__(self, analyser: Analyser):
		super().__init__(StageType.ANALYZE)
		self.analyser = analyser

	def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
		filename_to_htraces_list: Dict[str, List[HTrace]] = ctx["filename_to_htraces_list"]
		input_equivalence_classes_filenames: OrderedDict[Input, OrderedDict[Input, str]] = ctx["input_equivalence_classes_filenames"]
		test_case: TestCase = ctx["sandboxed_test_case"]

		violations = []
		for eq_class_filenames in input_equivalence_classes_filenames.values():

			ctraces = [CTrace.get_null()] * len(eq_class_filenames)
			test_cases = [test_case] * len(eq_class_filenames)
			inputs = list(eq_class_filenames.keys())
			htraces = []
			for inp_filename in eq_class_filenames.values():
				inp_htrace = filename_to_htraces_list[inp_filename]
				htraces.append(HTrace(trace_list=inp_htrace))
	
			violations.extend(self.analyser.filter_violations(inputs, ctraces, htraces, test_cases, stats=True))
	
		ctx.update({"violations": violations})

		return ctx


class Cleanable:
	def __init__(self,
			  check_file_exists: Callable[str, bool] = lambda p: os.path.exists(p),
			  remove_file: Callable[str, None] = lambda p: os.remove(p),
			  *args, **kwargs):
		super().__init__(*args, **kwargs) # For cooperative MTO support
		self._temp_files = []
		self._check_file_exists = check_file_exists
		self._remove_file = remove_file

	def register_temp_file(self, path):
		if self._check_file_exists(path) and path not in self._temp_files:
			self._temp_files.append(path)

	def _cleanup_temp_files(self):
		import pdb; pdb.set_trace()
		for path in self._temp_files:
			print(f"Cleaning up {path}")
			self._remove_file(path)
		self._temp_files.clear()

	# Context manager support
	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_value, traceback):
		self._cleanup_temp_files()

	# Destructor as a fallback
	def __del__(self):
		try:
			self._cleanup_temp_files()
		except Exception:
			# Avoid raising exceptions in __del__
			pass


# ==================================================================================================
# Main executor class
# ==================================================================================================
class Aarch64Executor(Executor):
    """
    The executor for aarch64 architecture. The executor interfaces with the kernel module to collect
    measurements.

    The high-level workflow is as follows:
    1. Load a test case into the kernel module (see _write_test_case).
    2. Load a set of inputs into the kernel module (see __write_inputs).
    3. Run the measurements by calling the kernel module (see _get_raw_measurements). Each
       measurement is repeated `n_reps` times.
    4. Aggregate the measurements into sets of traces (see _aggregate_measurements).
    """

    previous_num_inputs: int = 0
    curr_test_case: TestCase
    ignore_list: Set[int]

    def __init__(self, enable_mismatch_check_mode: bool = False):
        super().__init__(enable_mismatch_check_mode)
        self.LOG = Logger()
        self.target_desc = Aarch64TargetDesc()
        self.ignore_list = set()

        # Check the execution environment:
        if self._is_smt_enabled() and not enable_mismatch_check_mode:
            self.LOG.warning("executor", "SMT is on! You may experience false positives.")
        if not can_set_reserved():
            self.LOG.error("executor: Cannot set reserved bits on this CPU")

    def _is_smt_enabled(self) -> bool:
        """
        Check if SMT is enabled on the current CPU.

        :return: True if SMT is enabled, False otherwise
        """
        pass

    def set_vendor_specific_features(self):
        pass  # override in vendor-specific executors

    # ==============================================================================================
    # Interface: Quick and Dirty Mode
    def set_quick_and_dirty(self, state: bool):
        """
        Enable or disable the quick and dirty mode in the executor. In this mode, the executor
        will skip some of the stabilization phases, which will make the measurements faster but
        less reliable.

        :param state: True to enable the quick and dirty mode, False to disable it
        """
        pass

    # ==============================================================================================
    # Interface: Ignore List
    def set_ignore_list(self, ignore_list: List[int]):
        """
        Sets a list of inputs IDs that should be ignored by the executor.
        The executor will executed the inputs with these IDs as normal (in case they are
        necessary for priming the uarch state), but their htraces will be set to zero

        :param ignore_list: a list of input IDs to ignore
        """
        self.ignore_list = set(ignore_list)

    def extend_ignore_list(self, ignore_list: List[int]):
        """
        Add a list of new inputs IDs to the current ignore list.

        :param ignore_list: a list of input IDs to add to the ignore list
        """
        self.ignore_list.update(ignore_list)

    # ==============================================================================================
    # Interface: Base Addresses
    def read_base_addresses(self):
        """
        Read the base addresses of the code and the sandbox from the kernel module.
        This function is used to synchronize the memory layout between the executor and the model
        :return: a tuple (sandbox_base, code_base)
        """
        raise NotImplemented()

    # ==============================================================================================
    # Interface: Test Case Loading
    def load_test_case(self, test_case: TestCase):
        """
        Load a test case into the executor.
        This function must be called before calling `trace_test_case`.

        This function also sets the mismatch check mode in the kernel module if requested.
        The flag has to be set before loading the test case because the kernel module links
        the test case code with different measurement functions based on this flag.

        :param test_case: the test case object to load
        """
        # write the test case to the kernel module
        written_tc = self._write_test_case(test_case)
        self.curr_test_case = test_case

        # reset the ignore list; as we are testing a new program now, the old ignore list is not
        # relevant anymore
        self.ignore_list = set()
        return written_tc

    def _write_test_case(self, test_case: TestCase):
        raise NotImplemented()

    # ==============================================================================================
    # Interface: Test Case Tracing
    def trace_test_case(self, inputs: List[Input], n_reps: int) -> List[HTrace]:
        """
        Call the executor kernel module to collect the hardware traces for
        the test case (previously loaded with `load_test_case`) and the given inputs.

        :param inputs: list of inputs to be used for the test case
        :param n_reps: number of times to repeat each measurement
        :return: a list of HTrace objects, one for each input
        :raises HardwareTracingError: if the kernel module output is malformed
        """
        raise NotImplemented()


# ==================================================================================================
# Vendor-specific executors
# ==================================================================================================
class Aarch64RemoteExecutor(Aarch64Executor, Cleanable):

    def __init__(self, connection: Connection, workdir: str, *args, **kwargs):
        self.connection = connection

        Aarch64Executor.__init__(self, *args, **kwargs)

        Cleanable.__init__(
                self,
                check_file_exists=lambda path: connection.is_file_present(path),
                remove_file=lambda path: connection.shell(f"rm {shlex.quote(path)}", privileged=True),
            )
        self.test_case: Optional[TestCase] = None
        self.workdir = workdir
        self.userland_executor = UserlandExecutorImp(
				connection,
				f'{self.workdir}/executor_userland',
				'/dev/executor',
				'/sys/executor',
				f'{self.workdir}/revizor-executor.ko',
			)

        if self.target_desc.cpu_desc.vendor.lower() != "aarch64":  # Technically ARM currently does not produce ARM processors, and other vendors do produce ARM processors
            self.LOG.error(
                "Attempting to run ARM aarch64 remote executor on a non-ARM CPUs!\n"
                "Change the `executor` configuration option to the appropriate vendor value.")


    def _is_smt_enabled(self):
        smt_file = '/sys/devices/system/cpu/smt/control'
        if self.connection.is_file_present(smt_file):
            result = self.connection.shell(f'cat {smt_file}').lower().split()
            return 'on' in result or '1' in result

        return False

    def set_vendor_specific_features(self):
        pass

    def read_base_addresses(self):
        """
        Read the base addresses of the code and the sandbox from the kernel module.
        This function is used to synchronize the memory layout between the executor and the model
        :return: a tuple (sandbox_base, code_base)
        """
        return self.userland_executor.sandbox_base, self.userland_executor.code_base

    def _write_test_case(self, test_case: TestCase) -> None:
        self.test_case = test_case

    def trace_test_case(self, inputs: List[Input], n_reps: int) -> Tuple[List[HTrace], List[TestCase]]:
        """
        Call the executor kernel module to collect the hardware traces for
        the test case (previously loaded with `load_test_case`) and the given inputs.

        :param inputs: list of inputs to be used for the test case
        :param n_reps: number of times to repeat each measurement
        :return: a list of HTrace objects, one for each input
        :raises HardwareTracingError: if the kernel module output is malformed
        """
        # Skip if it's a dummy call
        if not inputs or self.test_case is None:
            return []



        # Store statistics
        n_inputs = len(inputs)
        STAT.executor_reruns += n_reps * n_inputs
        pipeline = (
                    LoadInputsBlock(self) | 
                    LoadSandboxedTestCaseBlock(self) |
					TraceScenarioBatch(self)
                ).build()

        initial_context = {
            "inputs": inputs,
            "test_case": self.test_case,
            "repeats": n_reps,
            "workdir": self.workdir,
            "userland_executor": self.userland_executor,
            "connection": self.connection,
        }

        ctx = pipeline.run(initial_context)

        filename_to_jsons = ctx["filename_to_jsons"]
        input_to_filename_dict = ctx["input_to_filename_dict"]
        sandboxed_testcase = ctx["sandboxed_test_case"]

		# Aggregate traces per remote input
        filenames = input_to_filename_dict.values()

        results = []
        for i in inputs:
            js_list = filename_to_jsons[input_to_filename_dict[i]]
            trace_list = []

            for js in js_list:

                input_name = js.get('input_name')
                if input_name not in filenames:
                    warnings.warn(f"Unexpected input_name returned: {input_name}")
                    continue


                trace_bin = js['htraces'][0]
                trace_int = int(trace_bin, 2)

                pfcs = tuple(js['pfcs'])
                trace_list.append(trace_int)

            htrace = HTrace(trace_list=trace_list)
            results.append(htrace)

        return results, [sandboxed_testcase] * len(results)

    def trace_test_case_with_taints(self, inputs: List[Input], expected_ctraces = None) -> Tuple[List[CTrace], List[InputTaint], List[FullTraceAuxBuffer], List[BitmapTaintsAuxBuffer]]:
        """
        Call the executor kernel module to collect the hardware traces for
        the test case (previously loaded with `load_test_case`) and the given inputs.

        :param inputs: list of inputs to be used for the test case
        :return: a tuple of CTrace
        """
        # Skip if it's a dummy call
        if not inputs or self.test_case is None:
            return []

        pipeline = (
                    LoadInputsBlock(self) |
                    GenerateInputArchFlowBlock(self) |
                    GenerateTaintsAndCTracesBlock()
                ).build()

        initial_context = {
            "inputs": inputs,
            "test_case": self.test_case,
            "workdir": self.workdir,
            "userland_executor": self.userland_executor,
            "repeats": 1,
            "connection": self.connection,
            "expected_ctraces": expected_ctraces
        }

        ctx = pipeline.run(initial_context)


        input_to_taint = ctx["input_to_taint"]
        input_to_ctrace = ctx["input_to_ctrace"]
        input_to_bitmap_taints_buffer = ctx['input_to_bitmap_taints_buffer']
        input_to_full_trace_buffer = ctx['input_to_full_trace_buffer']


        taints = []
        ctraces = []
        full_trace = []
        bitmaps = []

        for i in inputs:
            taints.append(input_to_taint[i])
            ctraces.append(input_to_ctrace[i])
            full_trace.append(input_to_full_trace_buffer[i])
            bitmaps.append(input_to_bitmap_taints_buffer[i])

        return ctraces, taints, full_trace, bitmaps


class Aarch64LocalExecutor(Aarch64Executor):

    def __init__(self, workdir: str, *args, **kwargs):
        Aarch64Executor.__init__(self, *args, **kwargs)

        self.test_case: Optional[TestCase] = None
        self.workdir = workdir
        self.local_executor = LocalExecutorImp(
				'/dev/executor',
				'/sys/executor',
				f'{self.workdir}/revizor-executor.ko',
			)

        if self.target_desc.cpu_desc.vendor.lower() != "aarch64":  # Technically ARM currently does not produce ARM processors, and other vendors do produce ARM processors
            self.LOG.error(
                "Attempting to run ARM aarch64 remote executor on a non-ARM CPUs!\n"
                "Change the `executor` configuration option to the appropriate vendor value.")


    def _is_smt_enabled(self):
        smt_file = Path('/sys/devices/system/cpu/smt/control')
        if smt_file.is_file():
            result = smt_file.read_text().strip()
            return 'on' in result or '1' in result

        return False

    def set_vendor_specific_features(self):
        pass

    def read_base_addresses(self):
        """
        Read the base addresses of the code and the sandbox from the kernel module.
        This function is used to synchronize the memory layout between the executor and the model
        :return: a tuple (sandbox_base, code_base)
        """
        return self.local_executor.sandbox_base, self.local_executor.code_base

    def _write_test_case(self, test_case: TestCase) -> None:
        self.test_case = test_case

    def _write_mod_test_case_to_local_executor(self, local_name: str, passes: List[Pass]):
        patched = copy.deepcopy(self.test_case)
        pass_on_test_case(patched, passes)

        printer = Aarch64Printer(self.target_desc)
        patched.bin_path, patched.asm_path, patched.obj_path = (f'{local_name}.{suffix}' for suffix in ('bin', 'asm', 'o'))
        printer.print(patched, patched.asm_path)
        ConfigurableGenerator.assemble(patched.asm_path, patched.obj_path, patched.bin_path)

        self.local_executor.checkout_region(TestCaseRegion())
        with open(patched.bin_path, "rb") as f:
            self.local_executor.write(f.read())

        os.remove(patched.bin_path)
        os.remove(patched.obj_path)
        #os.remove(patched.asm_path)
        return patched

    def trace_test_case(self, inputs: List[Input], n_reps: int) -> List[HTrace]:
        """
        Call the executor kernel module to collect the hardware traces for
        the test case (previously loaded with `load_test_case`) and the given inputs.

        :param inputs: list of inputs to be used for the test case
        :param n_reps: number of times to repeat each measurement
        :return: a list of HTrace objects, one for each input
        :raises HardwareTracingError: if the kernel module output is malformed
        """
        # Skip if it's a dummy call
        if not inputs or self.test_case is None:
            return []

        # Store statistics
        n_inputs = len(inputs)
        STAT.executor_reruns += n_reps * n_inputs

        self.local_executor.discard_all_inputs()

        input_to_iid = {}
        for idx, i in enumerate(inputs):
            input_to_iid[idx] = self.local_executor.allocate_iid()
            self.local_executor.checkout_region(InputRegion(input_to_iid[idx]))
            self.local_executor.write(ExecutorMemory(i.tobytes()))

        sandboxed_test_case = self._write_mod_test_case_to_local_executor("sandboxed_test_case", [Aarch64SandboxPass()])

        input_to_trace_list = defaultdict(list)

        for _ in range(n_reps):
            self.local_executor.trace()
            for idx, i in enumerate(inputs):
                self.local_executor.checkout_region(InputRegion(input_to_iid[idx]))
                hwm = self.local_executor.hardware_measurement()
                input_to_trace_list[idx].append(hwm.htrace)

        results = []
        for idx, i in enumerate(inputs):
            htrace = HTrace(trace_list=input_to_trace_list[idx])
            assert len(input_to_trace_list[idx]) == n_reps
            results.append(htrace)

        assert len(inputs) == len(results)
        return results, len(inputs) * [sandboxed_test_case]

    def _process_input(self, input_to_process: Input, bitmap_aux_buffer: BitmapTaintsAuxBuffer, full_trace_buffer: FullTraceAuxBuff) -> Tuple[InputTaint, CTrace]:
        from capstone import Cs, CS_ARCH_ARM64, CS_MODE_ARM
        from capstone.arm64 import ARM64_OP_REG

        def memory_access_size(insnt):
            ops = insnt.operands
            reg_op = ops[0]
            assert len(ops) >= 2, "Unexpected number of operands"
            if reg_op.type == ARM64_OP_REG and dis.reg_name(reg_op.reg).lower().startswith('w'):
                access_size = 4
            else:
                access_size = 8
            return access_size

        num_of_gprs = input_to_process['gpr'].shape[1]

        input_taint = InputTaint()

        gpr_region = input_taint[0]['gpr']
        for i in range(num_of_gprs):
            mask: int = 1 << i
            if bitmap_aux_buffer.regs_input_read_bits & mask:
                gpr_region[i] = True

		# currently, simd region used to store x8 and x9 initial values which corrently used to initialize flags and sp respectively (sp is being overriden with executor appropriate value)
        simd_region = input_taint[0]['simd']

        FLAGS_REG_INDEX = 8
        if bitmap_aux_buffer.regs_input_read_bits & (1 << FLAGS_REG_INDEX):
            simd_region[0] = True

        SP_REG_INDEX = 9
        if bitmap_aux_buffer.regs_input_read_bits & (1 << SP_REG_INDEX):
            simd_region[1] = True

        # Sandbox base is stored in x30 while executing the test case. It should not change while executing the test case. Take the first one.
        sandbox_base = full_trace_buffer.instruction_logs[0].regs[30]
        faulty_region_u8 = input_taint[0]["faulty"].view(np.uint8)
        main_region_u8 = input_taint[0]["main"].view(np.uint8)
        sandbox_size = main_region_u8.size + faulty_region_u8.size

        NOT_ACCESSED, READ, WRITE = 0, 1, 2
        access_table = np.zeros(sandbox_size, dtype=np.uint8)

        dis = Cs(CS_ARCH_ARM64, CS_MODE_ARM)
        dis.detail = True

        for inst_log in full_trace_buffer.instruction_logs:
            encoding_bytes = inst_log.encoding.to_bytes(4, byteorder='little')
            insns = list(dis.disasm(encoding_bytes, 0))
            assert insns, f"Unable to decode Aarch64 instruction: {encoding_bytes.hex()}"

            if inst_log.effective_address == 0xFFFFFFFFFFFFFFFF:
                continue

            offset = inst_log.effective_address - sandbox_base
            mnem = insns[0].mnemonic.lower()
            decoded = f"{mnem} {insns[0].op_str}"
            msg = f"Unexpected offset into sandbox: {offset} = {inst_log.effective_address:#08x} - {sandbox_base:#08x} = inst_log.effective_address - sandbox_base: {decoded}"
            assert 0 <= offset and offset < sandbox_size, msg

            access_size = memory_access_size(insns[0])

            is_write = "str" in mnem
            is_read = "ldr" in mnem
            for i in range(access_size):
                idx = offset + i
                if idx >= sandbox_size:
                    break

                if is_read and access_table[idx] == NOT_ACCESSED:
                    access_table[idx] = READ
                elif is_write and access_table[idx] == NOT_ACCESSED:
                    access_table[idx] = WRITE

        accessed_addresses: List[int] = []

        for offset in range(sandbox_size):
            access = access_table[offset]
            if access != NOT_ACCESSED:
                accessed_addresses.append(offset)

            if access == READ:
                to_change_region = main_region_u8
                to_change_offset = offset

                if offset >= main_region_u8.size:
                    to_change_region = faulty_region_u8
                    to_change_offset = offset - main_region_u8.size
            
                to_change_region[to_change_offset] = True

        line_size = 64
        num_sets = 64
        accessed_sets: List[int] = []
        cache_sets = sorted(set(map(lambda addr: (addr // line_size) % num_sets, accessed_addresses)))
        ctrace = CTrace(raw_trace=accessed_addresses)

        return input_taint, ctrace


    def trace_test_case_with_taints(self, inputs: List[Input], expected_ctraces = None) -> Tuple[List[CTrace], List[InputTaint], List[FullTraceAuxBuffer], List[BitmapTaintsAuxBuffer]]:
        """
        Call the executor kernel module to collect the hardware traces for
        the test case (previously loaded with `load_test_case`) and the given inputs.

        :param inputs: list of inputs to be used for the test case
        :return: a tuple of CTrace
        """
        # Skip if it's a dummy call
        if not inputs or self.test_case is None:
            return []

        # Store statistics
        n_inputs = len(inputs)

        self.local_executor.discard_all_inputs()

        input_to_iid = {}
        for i in inputs:
            input_to_iid[i] = self.local_executor.allocate_iid()
            self.local_executor.checkout_region(InputRegion(input_to_iid[i]))
            self.local_executor.write(ExecutorMemory(i.tobytes()))


        self._write_mod_test_case_to_local_executor("generated_taint_tracker", [Aarch64MarkRegisterTaints(), Aarch64SandboxPass(), Aarch64MarkMemoryTaints()])

        input_to_taints_buffer: OrderedDict[Input, BitmapTaintsAuxBuffer] = OrderedDict()

        self.local_executor.trace()
        for i in inputs:
            self.local_executor.checkout_region(InputRegion(input_to_iid[i]))
            input_to_taints_buffer[i] = aux_buffer_from_bytes(AuxBufferType.BITMAP_TAINTS, self.local_executor.aux_buffer)

        self._write_mod_test_case_to_local_executor("generated_full_trace_tracker", [Aarch64SandboxPass(), Aarch64FullTrace()])


        input_to_full_trace_buffer: OrderedDict[Input, FullTraceAuxBuffer] = OrderedDict()

        self.local_executor.trace()
        for i in inputs:
            self.local_executor.checkout_region(InputRegion(input_to_iid[i]))
            input_to_full_trace_buffer[i] = aux_buffer_from_bytes(AuxBufferType.FULL_TRACE, self.local_executor.aux_buffer)

        taints, ctraces, full_traces, bitmaps = [], [], [], []

        for i in inputs:
            taint, ctrace = self._process_input(i, input_to_taints_buffer[i], input_to_full_trace_buffer[i])
            taints.append(taint)
            ctraces.append(ctrace)
            full_traces.append(input_to_full_trace_buffer[i])
            bitmaps.append(input_to_taints_buffer[i])

        return ctraces, taints, full_traces, bitmaps


def disassemble_instruction(encoding: int, pc: int):
    from capstone import Cs, CS_ARCH_ARM64, CS_MODE_ARM
    import json
    
    # Capstone disassembler instance
    capstone_arm64 = Cs(CS_ARCH_ARM64, CS_MODE_ARM)
    capstone_arm64.detail = True

    try:
        code_bytes = encoding.to_bytes(4, byteorder="little")
        insns = list(capstone_arm64.disasm(code_bytes, pc))
        if insns:
            insn = insns[0]
            return f"{insn.mnemonic} {insn.op_str}".strip()
        else:
            return "<unknown>"
    except Exception as e:
        return f"<decode error: {e}>"


def compare_traces(trace_ref, trace_new):
    """Return (index, ref_inst, new_inst) if traces diverge, else (None, None, None)."""
    min_len = min(trace_ref.instruction_log_entry_count, trace_new.instruction_log_entry_count)
    for i in range(min_len):
        ref = trace_ref.instruction_logs[i]
        new = trace_new.instruction_logs[i]
        if (ref.pc != new.pc or ref.flags != new.flags or ref.regs != new.regs):
            return i, ref, new
    if trace_ref.instruction_log_entry_count != trace_new.instruction_log_entry_count:
        return min_len, None, None  # diverged by length
    return None, None, None


def show_context(trace, idx, window=-1):
    if(window < 0):
        window = trace.instruction_log_entry_count
    start = max(0, idx - window)
    end = min(trace.instruction_log_entry_count, idx + window + 1)
    for j in range(start, end):
        insn = trace.instruction_logs[j]
        disas = disassemble_instruction(insn.encoding, insn.pc)
        marker = "→" if j == idx else " "
        print(f"{marker} [{j:03d}] 0x{insn.pc:016x}: {disas}")


def print_taint(title, taint):
    print(f"  {title}:")
    print(f"    regs_write_bits       : {taint.regs_write_bits}")
    print(f"    regs_read_bits        : {taint.regs_read_bits}")
    print(f"    regs_input_read_bits  : {taint.regs_input_read_bits}")
    print(f"    mem_write_bits        : {taint.mem_write_bits}")
    print(f"    mem_read_bits         : {taint.mem_read_bits}")
    print(f"    mem_input_read_bits   : {taint.mem_input_read_bits}")


def dump_debug_to_file(prefix, input_ref, input_new, trace_ref, trace_new, taint_ref, taint_new):
    with open(f"{prefix}_trace_ref.json", "w") as f:
        json.dump([vars(x) for x in trace_ref.instruction_logs], f, indent=2)
    with open(f"{prefix}_trace_new.json", "w") as f:
        json.dump([vars(x) for x in trace_new.instruction_logs], f, indent=2)
    with open(f"{prefix}_taint_ref.txt", "w") as f:
        f.write(repr(taint_ref))
    with open(f"{prefix}_taint_new.txt", "w") as f:
        f.write(repr(taint_new))
    with open(f"{prefix}_inputs.txt", "w") as f:
        f.write(f"REF:\n{input_ref}\n\nNEW:\n{input_new}")

def compare_and_debug_trace_pair(
    input_ref,
    input_new,
    trace_ref,
    trace_new,
    taint_ref,
    taint_new,
    prefix="debug_trace",
    dump_files=True
):
    """
    Compare two FullTraceAuxBuffers and print a detailed divergence report.
    If dump_files=True, saves traces and taint to disk.
    """
    idx, ref_inst, new_inst = compare_traces(trace_ref, trace_new)

    if idx is None:
        print(f"[✓] Traces are architecturally equivalent ({trace_ref.instruction_log_entry_count} instructions).")
        return

    print("=" * 100)
    print(f"❗ Divergence detected at instruction #{idx}")

    if ref_inst and new_inst:
        ref_disas = disassemble_instruction(ref_inst.encoding, ref_inst.pc)
        new_disas = disassemble_instruction(new_inst.encoding, new_inst.pc)
        print(f"  Ref PC:  0x{ref_inst.pc:X}   ({ref_disas} [Encoding: 0x{ref_inst.encoding:08X}])")
        print(f"  New PC:  0x{new_inst.pc:X}   ({new_disas} [Encoding: 0x{new_inst.encoding:08X}])")
    else:
        print("  Diverged by trace length")

    print("\n--- Registers ---")
    if ref_inst and new_inst:
        same_regs = []
        diff_regs = []
        for i, (r_ref, r_new) in enumerate(zip(ref_inst.regs, new_inst.regs)):
            if r_ref == r_new:
                same_regs.append((i, r_ref))
            else:
                diff_regs.append((i, r_ref, r_new))

        if same_regs:
            print("  Common registers:")
            for i, val in same_regs:
                print(f"    X{i:02d}: {val:#018x}")
        else:
            print("  No common registers.")

        if diff_regs:
            print("\n  Differing registers:")
            for i, r_ref, r_new in diff_regs:
                print(f"    X{i:02d}: {r_ref:#018x} -> {r_new:#018x}")
        else:
            print("\n  No differing registers.")

    print("\n--- Flags ---")
    if ref_inst and new_inst and ref_inst.flags != new_inst.flags:
        print(f"  Flags changed: {ref_inst.flags:#x} -> {new_inst.flags:#x}")
    elif ref_inst and new_inst:
        print(f"  Flags identical: {ref_inst.flags:#x}")

    print("\n--- Memory ---")
    if ref_inst and new_inst:
        if ref_inst.effective_address != new_inst.effective_address:
            print(f"  EA: {ref_inst.effective_address:#x} -> {new_inst.effective_address:#x}")
        if ref_inst.mem_before != new_inst.mem_before:
            print(f"  Mem before: {ref_inst.mem_before:#x} -> {new_inst.mem_before:#x}")
        if ref_inst.mem_after != new_inst.mem_after:
            print(f"  Mem after:  {ref_inst.mem_after:#x} -> {new_inst.mem_after:#x}")
        if (
            ref_inst.effective_address == new_inst.effective_address
            and ref_inst.mem_before == new_inst.mem_before
            and ref_inst.mem_after == new_inst.mem_after
        ):
            print("  Memory identical.")

    print("\n--- Reference trace context ---")
    show_context(trace_ref, idx)
    print("\n--- New trace context ---")
    show_context(trace_new, idx)

    print("\n--- Taint Bitmaps ---")
    print_taint("Reference Taint", taint_ref)
    print_taint("New Taint", taint_new)

    print("\n--- Inputs ---")
    print("  Reference input:")
    print(input_ref)
    print("  New input:")
    print(input_new)

    print("=" * 100)

    if dump_files:
        dump_debug_to_file(prefix, input_ref, input_new, trace_ref, trace_new, taint_ref, taint_new)
        print(f"[+] Debug data dumped to: {prefix}_*.json/txt")

