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

import numpy as np
import time
from typing import List, Tuple, Set, Generator, Optional, Any, Dict, Iterable, Callable, Protocol, runtime_checkable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import defaultdict, OrderedDict, deque
from enum import Enum, auto

from .aarch64_generator import Aarch64TagMemoryAccesses, Aarch64Printer, Aarch64MarkMemoryAccesses
from .. import ConfigurableGenerator
from ..interfaces import HTrace, Input, TestCase, Executor, HardwareTracingError, Analyser, CTrace
from ..config import CONF
from ..util import Logger, STAT
from .aarch64_target_desc import Aarch64TargetDesc
from .aarch64_connection import Connection, UserlandExecutorImp, TestCaseRegion, InputRegion, \
    HWMeasurement, ExecutorBatch
from .aarch64_generator import Pass



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



class LoadInputs(Block):
	required_keys = {"inputs", "connection", "workdir", "userland_executor", "repeats"}
#	provided_keys = {"input_to_iids_dict", "input_to_filename_dict"}
	provided_keys = {"input_to_filename_dict"}

	def __init__(self):
		super().__init__(StageType.LOAD)

	@classmethod
	def _upload_inputs(cls, inputs: List[Input], connection: Connection, workdir: str) -> Dict[Input, str]:
		remote_filenames = {}
		for idx, inp in enumerate(inputs):
			input_name = f"input{idx}.bin"
			remote_filename = f'{workdir}/{input_name}'
			inp.save(input_name)
			connection.push(input_name, remote_filename)
			os.remove(input_name)
			remote_filenames[inp] = remote_filename
		return remote_filenames


	@classmethod
	def _write_inputs_to_connection(cls, inputs: List[Input], connection: Connection, workdir: str, userland_executor: UserlandExecutor, n_reps: int) -> Tuple[OrderedDict[Input, List[int]], Dict[Input, str]]:

#		input_to_iids_dict: OrderedDict[Input, List[int]] = OrderedDict()

		input_to_filename_dict: OrderedDict[Input, str] = cls._upload_inputs(inputs, connection, workdir)

#		for inp in input_to_filename_dict:
#			input_to_iids_dict[inp] = []
#
#		# Notice the order: Repeat the same input n_reps times, and then go to the next input
#		for inp, remote_filename in input_to_filename_dict.items():
#			for _ in range(n_reps):
#				iid = userland_executor.allocate_iid()
#				userland_executor.checkout_region(InputRegion(iid))
#				userland_executor.write_file(remote_filename)
#				input_to_iids_dict[inp].append(iid)

		return input_to_filename_dict #input_to_iids_dict, input_to_filename_dict 



	def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
		inputs: List[Input] = ctx["inputs"]
		connection: Connection = ctx["connection"]
		workdir: str = ctx["workdir"]
		userland_executor: UserlandExecutor= ctx["userland_executor"]
		repeats: int = ctx["repeats"]

#		input_to_iids_dict, input_to_filename_dict = self._write_inputs_to_connection(inputs, connection, workdir, userland_executor, repeats)
		input_to_filename_dict = self._write_inputs_to_connection(inputs, connection, workdir, userland_executor, repeats)

#		ctx.update({"input_to_iids_dict": input_to_iids_dict,
#                    "input_to_filename_dict": input_to_filename_dict})

		ctx.update({"input_to_filename_dict": input_to_filename_dict})


		return ctx

class LoadTest(Block):
	required_keys = {"test_case", "connection", "workdir"}
	provided_keys = {"remote_test_filename"}

	def __init__(self):
		super().__init__(StageType.LOAD)

	@classmethod
	def _upload_test(cls, test_case: TestCase, connection: Connection, workdir: str) -> str:
		remote_filename = f'{workdir}/remote_{os.path.basename(test_case.bin_path)}'
		connection.push(test_case.bin_path, remote_filename)
		return remote_filename

	def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
		test_case: TestCase = ctx["test_case"]
		connection: Connection = ctx["connection"]
		workdir: str= ctx["workdir"]
		ctx.update({"remote_test_filename": self._upload_test(test_case, connection, workdir)})
		return ctx


class GenerateMTETestVariants(Block):
#	required_keys = {"test_case", "input_to_iids_dict", "input_to_filename_dict", "connection", "workdir", "userland_executor"}
	required_keys = {"test_case", "input_to_filename_dict", "connection", "workdir", "userland_executor"}
	provided_keys = {"remote_test_filename_correct_tags", "remote_input_to_test_case_filename_incorrect_tags"}

	def __init__(self):
		super().__init__(StageType.TRANSFORM)

	@classmethod
	def _upload_test(cls, test_case: TestCase, connection: Connection, workdir: str) -> str:
		remote_filename = f'{workdir}/remote_{os.path.basename(test_case.bin_path)}'
		connection.push(test_case.bin_path, remote_filename)
		return remote_filename

	@staticmethod
	def _assemble_local_test_case(test_case: TestCase, base_filename: str):
		printer = Aarch64Printer(Aarch64TargetDesc())
		test_case.bin_path, test_case.asm_path, test_case.obj_path = (f'{base_filename}.{suffix}' for suffix in ('bin', 'asm', 'o'))

		printer.print(test_case, test_case.asm_path)

		ConfigurableGenerator.assemble(test_case.asm_path, test_case.obj_path, test_case.bin_path)



	@classmethod
	def _write_test_case_remotely(cls, test_case: TestCase, connection: Connection, workdir: str, local_filename: str) -> str:

		cls._assemble_local_test_case(test_case, local_filename)

		remote_filename = cls._upload_test(test_case, connection, workdir)

		os.remove(test_case.bin_path)
		#os.remove(test_case.asm_path)
		os.remove(test_case.obj_path)
		return remote_filename

	@classmethod
	def _pass_on_test_case(cls, test_case: TestCase, passes: List[Pass]):
		for p in passes:
			p.run_on_test_case(test_case)


	@classmethod
	def _write_test_with_correct_tags(cls, test_case: TestCase, connection: Connection, workdir: str) -> str:
		patched_test_case = copy.deepcopy(test_case)
		tagging_pass = Aarch64TagMemoryAccesses(memory_accesses_to_guess_tag=None)
		cls._pass_on_test_case(patched_test_case, [tagging_pass])
		return cls._write_test_case_remotely(patched_test_case, connection, workdir, 'generated_correct_tags')

	@classmethod
	def _write_test_with_incorrect_tags(cls, test_case: TestCase, connection: Connection, workdir: str, filename_suffix: str, tag_ids_to_guess: List[int]) -> str:
		patched_test_case = copy.deepcopy(test_case)
		tagging_pass = Aarch64TagMemoryAccesses(memory_accesses_to_guess_tag=tag_ids_to_guess)
		cls._pass_on_test_case(patched_test_case, [tagging_pass])
		return cls._write_test_case_remotely(patched_test_case, connection, workdir, f'generated_patched_{filename_suffix}')

	@classmethod
	def _write_test_case_with_bitmap_trace(cls, test_case: TestCase, connection: Connection, workdir: str) -> str:
		patched_test_case = copy.deepcopy(test_case)
		tagging_pass = Aarch64TagMemoryAccesses(memory_accesses_to_guess_tag=None)
		marking_pass = Aarch64MarkMemoryAccesses()
		cls._pass_on_test_case(patched_test_case, [tagging_pass, marking_pass])
		return cls._write_test_case_remotely(patched_test_case, connection, workdir, 'generated_retrieve_bitmap')

#	@classmethod
#	def _measure_architecturally_not_accessed_memory_addresses(cls, test_case: TestCase, input_to_iids_dict: OrderedDict[Input, List[int]], input_to_filename: Dict[Input, str],  workdir: str, userland_executor: UserlandExecutor, connection: Connection) -> Dict[Input, List[int]]:
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

#		for inp, iids in input_to_filename.items():
#			remote_input_filename = input_to_filename[inp]

		for inp, iid in inp_to_iids.items():
#			if not iids:
#				warnings.warn(f"Skipping remote input filename {remote_input_filename}: iid list is empty ({iids}).")
#				continue

			userland_executor.checkout_region(InputRegion(iid))
			measurement = userland_executor.hardware_measurement().memory_ids
			not_architectural_memory_accesses[inp] = parse_bitmap(measurement, bit=0)

		userland_executor.discard_all_inputs()
		connection.shell(f'rm {remote_test_case_filename}')

		return not_architectural_memory_accesses

#	@classmethod
#	def _create_tests_with_incorrect_tags(cls, test_case: TestCase, input_to_iids_dict: OrderedDict[Input, List[int]], input_to_filename: Dict[Input, str], workdir: str, userland_executor: UserlandExecutor, connection: Connection) -> Dict[Input, str]:
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
#		input_to_iids_dict: OrderedDict[Input, List[int]] = ctx["input_to_iids_dict"]
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

class PrepareScenarioBatch(Block):
	required_keys = {"input_to_filename_dict", "remote_test_filename", "repeats", "workdir"}
	provided_keys = {"scenario_batch"}

	def __init__(self):
		super().__init__(StageType.EXECUTE)

	@staticmethod
	def _create_scenario_batch(remote_inputs: List[str],
					remote_test_filename: str,
					repeats: int,
					workdir: str,
					output: str = None) -> ExecutorBatch:
		batch = ExecutorBatch()
		batch.repeats = repeats

		for input_fname in remote_inputs:
			batch.add_input(input_fname)

		# Add the single test case
		batch.add_test(remote_test_filename)

		# Optional output filename
		batch.output = output or f"{workdir}/remote_batch_output"

		return batch


	def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
		input_to_filename_dict: OrderedDict[Input, str] = ctx["input_to_filename_dict"]
		remote_test_filename: str = ctx["remote_test_filename"]
		repeats: int = ctx["repeats"]
		workdir: str = ctx["workdir"]

		batch = self._create_scenario_batch(input_to_filename_dict.values(), remote_test_filename, repeats, workdir)

		ctx.update({
			"scenario_batch": batch,
		})

		return ctx


class TraceScenarioBatch(Block):
	required_keys = {"scenario_batch", "userland_executor", "input_to_filename_dict"}
	provided_keys = {"filename_to_htraces_list"}  # keyed by remote input filename, values = list of HTrace objects

	def __init__(self):
		super().__init__(StageType.EXECUTE)

	@staticmethod
	def _extract_json_objects(blob: str) -> List[dict]:
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

	def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
		batch: ExecutorBatch = ctx["scenario_batch"]
		userland_executor: UserlandExecutor = ctx["userland_executor"]
		input_to_filename_dict: OrderedDict[Input, List[int]] = ctx["input_to_filename_dict"]

		# Temporary remote batch file
		import subprocess
		subprocess.run("cp ~/revizor/a ~/revizor/remote_generated; cp ~/revizor/spectrev1_arch.bin ~/revizor/input0.bin; cp ~/revizor/spectrev1_spec.bin ~/revizor/input1.bin", shell=True, check=True)
		raw_output = userland_executor.trace(batch)

		json_objs = self._extract_json_objects(raw_output)

		# Aggregate traces per remote input
		filename_to_htraces_list: Dict[str, List[int]] = {input_name: [] for input_name in input_to_filename_dict.values()}

		for js in json_objs:
			input_name = js.get('input_name')
			if input_name not in filename_to_htraces_list:
				warnings.warn(f"Unexpected input_name returned: {input_name}")
				continue

			trace_bin = js['htraces'][0]
			trace_int = int(trace_bin, 2)

			pfcs = tuple(js['pfcs'])

#			htraces[input_name].append(HTrace(trace_list=[trace_int], perf_counters=pfcs))
			filename_to_htraces_list[input_name].append(trace_int)

		ctx.update({
			"filename_to_htraces_list": filename_to_htraces_list,
		})

		return ctx


class CleanupRemoteFiles(Block):
	required_keys = {"input_to_filename_dict", "remote_test_filename", "connection", "scenario_batch"}
	provided_keys = set()

	def __init__(self):
		super().__init__(StageType.EXECUTE)

	def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
		connection: Connection = ctx["connection"]
		scenario_batch: ExecutorBatch = ctx["scenario_batch"]
		remote_test_filename: str = ctx["remote_test_filename"]
		input_to_filename_dict: OrderedDict[Input, str] = ctx["input_to_filename_dict"]

		for remote_input in input_to_filename_dict.values():
			try:
				connection.shell(f"rm {remote_input}", privileged=True)
			except Exception as e:
				warnings.warn(f"Failed to remove input: {remote_input} ({e}).")

		try:
			connection.shell(f"rm {remote_test_filename}", privileged=True)
		except Exception as e:
			warnings.warn(f"Failed to remove test file: {remote_test_filename} ({e}).")

		try:
			connection.shell(f"rm {scenario_batch.output}", privileged=True)
		except Exception as e:
			warnings.warn(f"Failed to remove batch output file: {scenario_batch.output} ({e}).")

		return ctx

class AnalyserBlock(Block):
	required_keys = {"filename_to_htraces_list", "input_to_filename_dict", "test_case"}
	provided_keys = {"violations"}

	def __init__(self, analyser: Analyser):
		super().__init__(StageType.ANALYZE)
		self.analyser = analyser

	def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
		#filename_to_htraces: Dict[str, List[HTrace]] = ctx["htraces"]
		filename_to_htraces_list: Dict[str, List[HTrace]] = ctx["filename_to_htraces_list"]
		input_to_filename_dict: Dict[Input, str] = ctx["input_to_filename_dict"]
		test_case: TestCase = ctx["test_case"]
	
		ctraces = [CTrace.get_null()] * len(input_to_filename_dict)
		test_cases = [test_case] * len(input_to_filename_dict)
		htraces = []
		inputs = list(input_to_filename_dict.keys())
	
		for inp, remote_input_filename in input_to_filename_dict.items():
#			number_of_added_htraces = len(filename_to_htraces[remote_input_filename])
			htraces.append(HTrace(trace_list=filename_to_htraces_list[remote_input_filename]))
#			inputs += [inp] * number_of_added_htraces
#			ctraces += [CTrace.get_null()] * number_of_added_htraces
#			test_cases += [test_case] * number_of_added_htraces
	
		violations = self.analyser.filter_violations(inputs, ctraces, htraces, test_cases, stats=True)
	
		ctx.update({"violations": violations})
	
		#import pdb; pdb.set_trace()
	
		return ctx


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
class Aarch64RemoteExecutor(Aarch64Executor):

    def __init__(self, connection: Connection, workdir: str, *args):
        self.connection = connection
        super().__init__(*args)
        self.test_case: Optional[TestCase] = None
        self.workdir = workdir
        self.userland_executor = UserlandExecutorImp(connection, f'{self.workdir}/executor_userland',
                                                     '/dev/executor', '/sys/executor',
                                                     f'{self.workdir}/revizor-executor.ko',
                                                     )
        if self.target_desc.cpu_desc.vendor.lower() != "arm":  # Technically ARM currently does not produce ARM processors, and other vendors do produce ARM processors
            self.LOG.error(
                "Attempting to run ARM executor on a non-ARM CPUs!\n"
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

    def trace_test_case(self, inputs: List[Input], n_reps: int, analyser: Analyser) -> Dict[Input, List[HTrace]]:
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
                    (LoadInputs() & LoadTest()) | 
                    (PrepareScenarioBatch() & TraceScenarioBatch() & CleanupRemoteFiles()) |
                    AnalyserBlock(analyser)
                ).build()
        
        initial_context = {
            "inputs": inputs,
            "test_case": self.test_case,
            "repeats": n_reps,
            "workdir": self.workdir,
            "userland_executor": self.userland_executor,
            "connection": self.connection
        }
        
        ctx = pipeline.run(initial_context)
        
        violations = ctx["violations"]
        
        return violations


class Aarch64RemoteExecutorMTE(Aarch64Executor):

    def __init__(self, connection: Connection, *args):
        self.connection = connection
        super().__init__(*args)
        self.test_case: Optional[TestCase] = None
        self.workdir = workdir
        self.userland_executor = UserlandExecutorImp(connection, f'{self.workdir}/executor_userland',
                                                     '/dev/executor', '/sys/executor',
                                                     f'{self.workdir}/revizor-executor.ko',
                                                     )
        if self.target_desc.cpu_desc.vendor.lower() != "arm":  # Technically ARM currently does not produce ARM processors, and other vendors do produce ARM processors
            self.LOG.error(
                "Attempting to run ARM executor on a non-ARM CPUs!\n"
                "Change the `executor` configuration option to the appropriate vendor value.")

    def _is_smt_enabled(self):
        result = self.connection.shell('cat /sys/devices/system/cpu/smt/control')
        return 'on' in result.lower().split()

    def set_vendor_specific_features(self):
        pass

    def read_base_addresses(self):
        """
        Read the base addresses of the code and the sandbox from the kernel module.
        This function is used to synchronize the memory layout between the executor and the model
        :return: a tuple (sandbox_base, code_base)
        """
        return self.userland_executor.sandbox_base, self.userland_executor.code_base

    def _write_test_case_with_bitmap_trace(self, test_case: TestCase) -> None:
        patched_test_case = copy.deepcopy(test_case)
        tagging_pass = Aarch64TagMemoryAccesses(memory_accesses_to_guess_tag=None)
        marking_pass = Aarch64MarkMemoryAccesses()
        Aarch64RemoteExecutor._pass_on_test_case(patched_test_case, [tagging_pass, marking_pass])

        Aarch64RemoteExecutor._assemble_local_test_case(patched_test_case, 'generated_retrieve_bitmap')

        remote_testcase_name = self._write_test_case_remotely(patched_test_case)
        self._load__and_remove_remote_test_case(remote_testcase_name)

        os.remove(patched_test_case.bin_path)
        os.remove(patched_test_case.asm_path)
        os.remove(patched_test_case.obj_path)

    @staticmethod
    def _pass_on_test_case(test_case: TestCase, passes: List[Pass]):
        for p in passes:
            p.run_on_test_case(test_case)

    def _write_test_case_remotely(self, test_case: TestCase):
        remote_filename = f'{self.workdir}/remote_{test_case.bin_path}'
        self.connection.push(test_case.bin_path, remote_filename)
        return remote_filename


    def _load__and_remove_remote_test_case(self, remote_filename: str):
        self.userland_executor.checkout_region(TestCaseRegion())
        self.userland_executor.write_file(remote_filename)
        self.connection.shell(f'rm {remote_filename}')

    @staticmethod
    def _assemble_local_test_case(test_case: TestCase, base_filename: str):
        printer = Aarch64Printer(Aarch64TargetDesc())
        test_case.bin_path, test_case.asm_path, test_case.obj_path = \
            (f'{base_filename}.{suffix}' for suffix in ('bin', 'asm', 'o'))

        printer.print(test_case, test_case.asm_path)

        ConfigurableGenerator.assemble(test_case.asm_path, test_case.obj_path, test_case.bin_path)

    def _write_test_with_correct_tags(self) -> Tuple[str, TestCase]:
        patched_test_case = copy.deepcopy(self.test_case)
        #os.remove(patched_test_case.bin_path)
        #os.remove(patched_test_case.asm_path)
        #os.remove(patched_test_case.obj_path)
        tagging_pass = Aarch64TagMemoryAccesses(memory_accesses_to_guess_tag=None)
        Aarch64RemoteExecutor._pass_on_test_case(patched_test_case, [tagging_pass])

        local_filename = 'generated_correct_tags'
        Aarch64RemoteExecutor._assemble_local_test_case(patched_test_case, local_filename)
        remote_filename = self._write_test_case_remotely(patched_test_case)
        os.remove(patched_test_case.bin_path)
        #os.remove(patched_test_case.asm_path)
        os.remove(patched_test_case.obj_path)
        return remote_filename, patched_test_case

    def _write_test_with_incorrect_tags(self, filename_suffix: str, memory_accesses_to_guess_tag: List[int]) -> Tuple[str, TestCase]:
        patched_test_case = copy.deepcopy(self.test_case)
        tagging_pass = Aarch64TagMemoryAccesses(
            memory_accesses_to_guess_tag=memory_accesses_to_guess_tag)
        Aarch64RemoteExecutor._pass_on_test_case(patched_test_case, [tagging_pass])

        local_filename = f'generated_patched_{filename_suffix}'
        Aarch64RemoteExecutor._assemble_local_test_case(patched_test_case, local_filename)
        remote_filename = self._write_test_case_remotely(patched_test_case)
        os.remove(patched_test_case.bin_path)
        #os.remove(patched_test_case.asm_path)
        os.remove(patched_test_case.obj_path)
        return remote_filename, patched_test_case

    def _write_test_case(self, test_case: TestCase) -> None:
        self.test_case = test_case

    def _upload_inputs(self, inputs: List[Input]):
        remote_filenames = []
        for idx, inp in enumerate(inputs):
            inpname = f"input{idx}.bin"
            remote_fname = f'{self.workdir}/{inpname}'
            inp.save(inpname)
            self.connection.push(inpname, remote_fname)
            os.remove(inpname)
            remote_filenames.append(remote_fname)
        return remote_filenames

    def _write_inputs_to_connection(self, inputs: List[Input], n_reps: int) -> Tuple[
        np.ndarray[int, int], List[str]]:
        array: np.ndarray[int, int] = np.zeros((len(inputs), n_reps), dtype=np.uint64)
        remote_filenames = self._upload_inputs(inputs)

        for col in range(n_reps):
            for row, fname in enumerate(remote_filenames):
                array[row, col] = self.userland_executor.allocate_iid()
                self.userland_executor.checkout_region(InputRegion(array[row, col]))
                self.userland_executor.write_file(fname)

        return array, remote_filenames

    def _measure_architecturaly_accessed_memory_addresses(self, remote_input_filenames: List[str]) -> Dict[str, str]:

        all_architectural_memory_accesses = {}
        iids: np.ndarray[int] = np.zeros((len(remote_input_filenames)), dtype=np.uint64)

        self._write_test_case_with_bitmap_trace(self.test_case)

        for idx, remote_filename in enumerate(remote_input_filenames):
            iids[idx] = self.userland_executor.allocate_iid()
            self.userland_executor.checkout_region(InputRegion(iids[idx]))
            self.userland_executor.write_file(remote_filename)

        self.userland_executor.trace()

        for iid, remote_input_filename in zip(iids, remote_input_filenames):
            self.userland_executor.checkout_region(InputRegion(iid))
            all_architectural_memory_accesses[remote_input_filename] = self.userland_executor.hardware_measurement().memory_ids

        self.userland_executor.discard_all_inputs()

        return all_architectural_memory_accesses
 
    def _measure_architecturaly_not_accessed_memory_addresses(self, remote_input_filenames: List[str]) -> Dict[str, List[int]]:

        def parse_bitmap(n: str, bit: int) -> List[int]:
            result = []
            for index, b in enumerate(n):
                if bit == int(b):
                    result.append(len(n) - (index + 1))
            return result

        all_not_architectural_memory_accesses: Dict[int, List[int]] = {}
        for remote_filename, measurement in self._measure_architecturaly_accessed_memory_addresses(remote_input_filenames).items():
            all_not_architectural_memory_accesses[remote_filename]: List[int] = \
                parse_bitmap(measurement, bit=0)

        return all_not_architectural_memory_accesses

    def _create_tests_with_incorrect_tags(self, remote_input_filenames: List[str]) -> Dict[str, Tuple[str, TestCase]]:

        pair_filename_tc_incorrect_tags: Dict[str, Tuple[str, TestCase]] = {}

        all_not_architectural_memory_accesses = self._measure_architecturaly_not_accessed_memory_addresses(remote_input_filenames)

        for remote_input_filename, measurement in all_not_architectural_memory_accesses.items() :
            pair_filename_tc_incorrect_tags[remote_input_filename] = self._write_test_with_incorrect_tags(
                remote_input_filename.rstrip('/').split('/')[-1], measurement)

        return pair_filename_tc_incorrect_tags

    def _create_scenario_batch(self, remote_input_filenames: List[str], test_cases: List[str], repeats: int, output: Optional[str] = None) -> ExecutorBatch:
        executor_batch = ExecutorBatch()

        executor_batch.repeats = repeats

        for remote_filename in remote_input_filenames:
            executor_batch.add_input(remote_filename)
        
        for tc in test_cases:
            executor_batch.add_test(tc)

        if output is not None:
            executor_batch.output = output

        return executor_batch
 

    def trace_test_case(self, inputs: List[Input], n_reps: int) -> List[Tuple[
        Tuple[TestCase,HTrace], Tuple[TestCase,HTrace]]]:
        """
        Call the executor kernel module to collect the hardware traces for
        the test case (previously loaded with `load_test_case`) and the given inputs.

        :param inputs: list of inputs to be used for the test case
        :param n_reps: number of times to repeat each measurement
        :return: a list of HTrace objects, one for each input
        :raises HardwareTracingError: if the kernel module output is malformed
        """
        def _measure_input(iid: int, iids: np.ndarray[int, int]) -> np.ndarray[Any, np.dtype[HWMeasurement]]:

            def checkout_and_measure(cid: int) -> HWMeasurement:
                self.userland_executor.checkout_region(InputRegion(iids[iid][cid]))
                return self.userland_executor.hardware_measurement()

            hwmeasurements = list(map(checkout_and_measure, range(iids.shape[1])))
            return np.array(hwmeasurements, dtype=object)

        # Skip if it's a dummy call
        if not inputs or self.test_case is None:
            return []

        # Store statistics
        n_inputs = len(inputs)
        STAT.executor_reruns += n_reps * n_inputs

        
        def extract_json_objects(blob):
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
                        objs.append(json.loads(obj_str))
                        start_idx = None  # reset
        
            return objs

        remote_output_filename = f"{self.workdir}/remote_tmp_output"
        remote_batch_filename = f'{self.workdir}/executor_batch'

        remote_input_filenames = self._upload_inputs(inputs)
        remote_filename_correct_tags, tc_correct_tags  = self._write_test_with_correct_tags()
        remote_pair_filenames_tcs_incorrect_tags = self._create_tests_with_incorrect_tags(remote_input_filenames)
        remote_filenames_incorrect_tags = { k: v[0] for k,v in remote_pair_filenames_tcs_incorrect_tags.items() }
        tcs_incorrect_tags = { k: v[1] for k,v in remote_pair_filenames_tcs_incorrect_tags.items() }

        scenario_batch: ExecutorBatch = self._create_scenario_batch(remote_input_filenames, [remote_filename_correct_tags] + list(remote_filenames_incorrect_tags.values()), n_reps, remote_output_filename)
        output = self.userland_executor.trace(scenario_batch, remote_batch_filename)

        jsons = extract_json_objects(output)
        
        correct_traces_by_input = defaultdict(list)
        incorrect_traces_by_input = defaultdict(list)
        correct_pfcs_by_input = defaultdict(list)
        incorrect_pfcs_by_input = defaultdict(list)

        for js in jsons:
            input_name = js['input_name']
            test_name = js['test_name']

            trace = int(js['htraces'][0], 2)
            pfcs = tuple(js['pfcs'])

            if test_name == remote_filename_correct_tags:
                correct_traces_by_input[input_name].append(trace)
                correct_pfcs_by_input[input_name].append(pfcs)
            elif test_name == remote_filenames_incorrect_tags.get(input_name, None):
                incorrect_traces_by_input[input_name].append(trace)
                incorrect_pfcs_by_input[input_name].append(pfcs)

        remote_filenames = list(remote_filenames_incorrect_tags.values()) + remote_input_filenames + [remote_output_filename, remote_filename_correct_tags, remote_batch_filename] 
        for filename in remote_filenames:
            self.connection.shell(f'rm {filename}', privileged=True)

        self.LOG.dbg_executor_raw_traces(correct_traces_by_input, correct_pfcs_by_input)
        self.LOG.dbg_executor_raw_traces(incorrect_traces_by_input, incorrect_pfcs_by_input)

#        # Post-process the results and check for errors
#        if not self.mismatch_check_mode:  # no need to post-process in mismatch check mode
#            mask = np.uint64(0x0FFFFFFFFFFFFFF0)
#            for input_id in range(n_inputs):
#                for rep_id in range(n_reps):
#                    # Zero-out traces for ignored inputs
#                    if input_id in self.ignore_list:
#                        all_correct_tags_traces[input_id][rep_id] = 0
#                        all_incorrect_tags_traces[input_id][rep_id] = 0
#                        continue
#
#                    # When using TSC mode, we need to mask the lower 4 bits of the trace
#                    if CONF.executor_mode == 'TSC':
#                        all_correct_tags_traces[input_id][rep_id] &= mask
#                        all_incorrect_tags_traces[input_id][rep_id] &= mask

        # Aggregate measurements into HTrace objects
        traces = []
        for remote_input_filename in remote_input_filenames:

            trace_correct = correct_traces_by_input[remote_input_filename]
            pfcs_correct = correct_pfcs_by_input[remote_input_filename]
            trace_incorrect = incorrect_traces_by_input[remote_input_filename]
            pfcs_incorrect = incorrect_pfcs_by_input[remote_input_filename]

            traces.append(
                    (
                        (tc_correct_tags, HTrace(trace_list=trace_correct, perf_counters=pfcs_correct)),
                        (tcs_incorrect_tags[remote_input_filename], HTrace(trace_list=trace_incorrect, perf_counters=pfcs_incorrect))
                    )
                )
                
        return traces

#class Aarch64RemoteExecutor(Aarch64Executor):
#
#    def __init__(self, connection: Connection, workdir: str, *args):
#        self.connection = connection
#        super().__init__(*args)
#        self.test_case: Optional[TestCase] = None
#        self.workdir = workdir
#        self.userland_executor = UserlandExecutorImp(connection, f'{self.workdir}/executor_userland',
#                                                     '/dev/executor', '/sys/executor',
#                                                     f'{self.workdir}/revizor-executor.ko',
#                                                     )
#        if self.target_desc.cpu_desc.vendor.lower() != "arm":  # Technically ARM currently does not produce ARM processors, and other vendors do produce ARM processors
#            self.LOG.error(
#                "Attempting to run ARM executor on a non-ARM CPUs!\n"
#                "Change the `executor` configuration option to the appropriate vendor value.")
#
#    def _is_smt_enabled(self):
#        smt_file = '/sys/devices/system/cpu/smt/control'
#        if self.connection.is_file_present(smt_file):
#            result = self.connection.shell(f'cat {smt_file}').lower().split()
#            return 'on' in result or '1' in result
#
#        return False
#
#    def set_vendor_specific_features(self):
#        pass
#
#    def read_base_addresses(self):
#        """
#        Read the base addresses of the code and the sandbox from the kernel module.
#        This function is used to synchronize the memory layout between the executor and the model
#        :return: a tuple (sandbox_base, code_base)
#        """
#        return self.userland_executor.sandbox_base, self.userland_executor.code_base
#
#    def _write_test_case_remotely(self, test_case: TestCase):
#        remote_filename = f'{self.workdir}/remote_{test_case.bin_path}'
#        self.connection.push(test_case.bin_path, remote_filename)
#        return remote_filename
#
#    def _load__and_remove_remote_test_case(self, remote_filename: str):
#        self.userland_executor.checkout_region(TestCaseRegion())
#        self.userland_executor.write_file(remote_filename)
#        self.connection.shell(f'rm {remote_filename}')
#
#    def _write_test_case(self, test_case: TestCase) -> None:
#        self.test_case = test_case
#
#    def _upload_inputs(self, inputs: List[Input]):
#        remote_filenames = []
#        for idx, inp in enumerate(inputs):
#            inpname = f"input{idx}.bin"
#            remote_fname = f'{self.workdir}/{inpname}'
#            inp.save(inpname)
#            self.connection.push(inpname, remote_fname)
#            os.remove(inpname)
#            remote_filenames.append(remote_fname)
#        return remote_filenames
#
#    def _write_inputs_to_connection(self, inputs: List[Input], n_reps: int) -> Tuple[
#        np.ndarray[int, int], List[str]]:
#        array: np.ndarray[int, int] = np.zeros((len(inputs), n_reps), dtype=np.uint64)
#        remote_filenames = self._upload_inputs(inputs)
#
#        for col in range(n_reps):
#            for row, fname in enumerate(remote_filenames):
#                array[row, col] = self.userland_executor.allocate_iid()
#                self.userland_executor.checkout_region(InputRegion(array[row, col]))
#                self.userland_executor.write_file(fname)
#
#        return array, remote_filenames
#
#    def _create_scenario_batch(self, remote_input_filenames: List[str], test_cases: List[str], repeats: int, output: Optional[str] = None) -> ExecutorBatch:
#        executor_batch = ExecutorBatch()
#
#        executor_batch.repeats = repeats
#
#        for remote_filename in remote_input_filenames:
#            executor_batch.add_input(remote_filename)
#        
#        for tc in test_cases:
#            executor_batch.add_test(tc)
#
#        if output is not None:
#            executor_batch.output = output
#
#        return executor_batch
# 
#
#    def trace_test_case(self, inputs: List[Input], n_reps: int) -> List[Tuple[
#        Tuple[TestCase,HTrace], Tuple[TestCase,HTrace]]]:
#        """
#        Call the executor kernel module to collect the hardware traces for
#        the test case (previously loaded with `load_test_case`) and the given inputs.
#
#        :param inputs: list of inputs to be used for the test case
#        :param n_reps: number of times to repeat each measurement
#        :return: a list of HTrace objects, one for each input
#        :raises HardwareTracingError: if the kernel module output is malformed
#        """
#        def _measure_input(iid: int, iids: np.ndarray[int, int]) -> np.ndarray[Any, np.dtype[HWMeasurement]]:
#
#            def checkout_and_measure(cid: int) -> HWMeasurement:
#                self.userland_executor.checkout_region(InputRegion(iids[iid][cid]))
#                return self.userland_executor.hardware_measurement()
#
#            hwmeasurements = list(map(checkout_and_measure, range(iids.shape[1])))
#            return np.array(hwmeasurements, dtype=object)
#
#        # Skip if it's a dummy call
#        if not inputs or self.test_case is None:
#            return []
#
#        # Store statistics
#        n_inputs = len(inputs)
#        STAT.executor_reruns += n_reps * n_inputs
#
#        
#        def extract_json_objects(blob):
#            objs = []
#            brace_level = 0
#            start_idx = None
#        
#            for i, char in enumerate(blob):
#                if char == '{':
#                    if brace_level == 0:
#                        start_idx = i
#                    brace_level += 1
#                elif char == '}':
#                    brace_level -= 1
#                    if brace_level == 0 and start_idx is not None:
#                        obj_str = blob[start_idx:i + 1]
#                        objs.append(json.loads(obj_str))
#                        start_idx = None  # reset
#        
#            return objs
#
#        remote_output_filename = f"{self.workdir}/remote_tmp_output"
#        remote_batch_filename = f'{self.workdir}/executor_batch'
#
#        remote_input_filenames = self._upload_inputs(inputs)
#        remote_filename = self._write_test_case_remotely(self.test_case)
#
#        scenario_batch: ExecutorBatch = self._create_scenario_batch(remote_input_filenames, [remote_filename], n_reps, remote_output_filename)
#        output = self.userland_executor.trace(scenario_batch, remote_batch_filename)
#
#        jsons = extract_json_objects(output)
#        
#        traces_by_input = defaultdict(list)
#        pfcs_by_input = defaultdict(list)
#
#        for js in jsons:
#            input_name = js['input_name']
#            test_name = js['test_name']
#
#            trace = int(js['htraces'][0], 2)
#            pfcs = tuple(js['pfcs'])
#            traces_by_input[input_name].append(trace)
#            pfcs_by_input[input_name].append(pfcs)
#
#        import pdb; pdb.set_trace()
#        remote_filenames = remote_input_filenames + [remote_output_filename, remote_filename, remote_batch_filename] 
#        for filename in remote_filenames:
#            self.connection.shell(f'rm {filename}', privileged=True)
#
#        self.LOG.dbg_executor_raw_traces(traces_by_input, pfcs_by_input)
#
#        # Aggregate measurements into HTrace objects
#        traces = []
#        for remote_input_filename in remote_input_filenames:
#
#            trace = correct_traces_by_input[remote_input_filename]
#            pfcs = correct_pfcs_by_input[remote_input_filename]
#
#            traces.append(
#                    (
#                        (tc_correct_tags, HTrace(trace_list=trace_correct, perf_counters=pfcs_correct)),
#                        (tcs_incorrect_tags[remote_input_filename], HTrace(trace_list=trace_incorrect, perf_counters=pfcs_incorrect))
#                    )
#                )
#                
#        return traces
#
