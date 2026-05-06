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
from typing import List, Union, Tuple, Set, Generator, Optional, Any, Dict, Iterable, Callable, Protocol, runtime_checkable, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import defaultdict, OrderedDict, deque
from enum import Enum, auto
from contextlib import contextmanager
from pathlib import Path
from itertools import chain


from capstone import Cs, CS_ARCH_ARM64, CS_MODE_ARM, CS_AC_READ, CS_AC_WRITE
from capstone.arm64 import ARM64_OP_REG, ARM64_OP_MEM

from .aarch64_generator import Aarch64Printer, Aarch64ASMLayout, Aarch64Generator
from .. import ConfigurableGenerator
from ..interfaces import HTrace, Input, TestCase, Executor, HardwareTracingError, Analyser, CTrace, InputTaint, TargetDesc
from ..config import CONF
from ..util import Logger, STAT
from .aarch64_target_desc import Aarch64TargetDesc
from .aarch64_connection import LocalExecutorImp, TestCaseRegion, InputRegion
from .aarch64_generator import Pass, Aarch64SandboxPass, PACInstrumentation, FixPoint

from .aarch64_connection import profile_op, ExecutorMemory

from .aarch64_contract_executor import *
# ==================================================================================================
# Helper functions
# ==================================================================================================
def km_write(value, path: str) -> None:
    with open(path, 'w') as f:
        f.write(str(value))


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


def pass_on_test_case(test_case: TestCase, passes: List[Pass]):
	for p in passes:
		p.run_on_test_case(test_case)


# ==================================================================================================
# NON-INTERFERENCE: per-input TC variant result containers
# Used by Aarch64NonInterferenceExecutor; regular-Revizor mode never produces these.
# ==================================================================================================
@dataclass
class TCVariants:
    """The three TC variant test cases generated for a single input (stage-2 output).
    HW execution happens separately — see trace_tc_variants_on_hw()."""
    tc1: TestCase
    tc2: TestCase
    tc3: TestCase

    def items(self):
        """Iterate (name, test_case) pairs in a stable order."""
        return [
            ('tc1', self.tc1),
            ('tc2', self.tc2),
            ('tc3', self.tc3),
        ]


@dataclass
class TCVariantCETraces:
    """CE traces for a single input across all three TC variants (mirrors TCVariants)."""
    tc1: ContractExecutionResult
    tc2: ContractExecutionResult
    tc3: ContractExecutionResult

    def items(self):
        return [
            ('tc1', self.tc1),
            ('tc2', self.tc2),
            ('tc3', self.tc3),
        ]


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
        raise NotImplementedError()

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
        raise NotImplementedError()

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
        raise NotImplementedError()


# ==================================================================================================
# Vendor-specific executors
# ==================================================================================================

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

        self._contract_executor = ContractExecutorService("/home/gal_k_1_1998/revizor/sca-fuzzer/src/aarch64/contract_executor/contract_executor")

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
        layout: Aarch64ASMLayout = Aarch64ASMLayout(patched)
        assembly = Aarch64Printer(self.target_desc).print_layout(layout)
        tc_bytes = ConfigurableGenerator.in_memory_assemble(assembly)

        patched.asm_path, patched.obj_path, patched.bin_path = local_name, local_name, local_name

        self.local_executor.checkout_region(TestCaseRegion())
        self.local_executor.write(tc_bytes)

        return patched

    def _tc_to_bytes(self, tc: TestCase) -> bytes:
        """Assemble a TestCase to binary without writing it anywhere."""
        layout = Aarch64ASMLayout(tc)
        assembly = Aarch64Printer(self.target_desc).print_layout(layout)
        return ConfigurableGenerator.in_memory_assemble(assembly)

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
            return [], []

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

        counter_log = defaultdict(list)
        for _ in range(n_reps):
            self.local_executor.trace()
            for idx, i in enumerate(inputs):
                self.local_executor.checkout_region(InputRegion(input_to_iid[idx]))
                hwm = self.local_executor.hardware_measurement()
                input_to_trace_list[idx].append(hwm.htrace)
                counter_log[idx].append(hwm.pfcs[2])
                    
        for idx, vals in counter_log.items():
            pass
#            print(f"[LOG Gal] input {idx} miss-rate: {sum(vals)} / {n_reps} = {100 * ((1.0 * sum(vals)) / n_reps)}%: {vals}")

        results = []
        for idx, i in enumerate(inputs):
            htrace = HTrace(trace_list=input_to_trace_list[idx])
            assert len(input_to_trace_list[idx]) == n_reps
            results.append(htrace)


        assert len(inputs) == len(results)
        return results, len(inputs) * [sandboxed_test_case]

    def trace_test_case_with_taints(self, inputs: List[Input], nesting: int) -> Tuple[List[CTrace], List[InputTaint], List[ContractExecutionResult], List]:
        """
        Run the Contract Executor on the sandboxed raw test case and compute CTraces/taints.

        This is the base (regular Revizor) implementation.  It applies the sandbox pass,
        runs CE for each input, and returns cache-set CTraces and input taints.
        No PAC instrumentation or TC variants are produced here.

        Non-interference mode (PAC/MTE) overrides this in Aarch64NonInterferenceExecutor.

        :return: (ctraces, taints, ce_traces, [])  — tc_variants is always empty here
        """
        if not inputs or self.test_case is None:
            return [], [], [], []

        patched = copy.deepcopy(self.test_case)
        pass_on_test_case(patched, [Aarch64SandboxPass()])
        layout: Aarch64ASMLayout = Aarch64ASMLayout(patched)
        assembly = Aarch64Printer(self.target_desc).print_layout(layout)
        tc_bytes = ConfigurableGenerator.in_memory_assemble(assembly)

        sandbox_base, _ = self.read_base_addresses()
        traces: List[ContractExecutionResult] = []

        for inp in inputs:
            data = inp.tobytes()
            tc_memory = data[:0x2000]
            tc_regs = bytearray(data[0x2000:])
            view = memoryview(tc_regs).cast('Q')
            view[6] = (view[6] << 28) & ((1 << 64) - 1)
            tc_regs = bytes(tc_regs)
            execution = ContractExecution(tc_bytes, tc_memory, tc_regs, SimArch.RVZR_ARCH_AARCH64, 5, 10,
                                          req_mem_base_virt=sandbox_base)
            cer = self._contract_executor.run(execution)
            traces.append(cer)

        taints, ctraces = [], []
        for cer in traces:
            overriden_gprs = set()
            overriden_memory = set()
            input_taint = InputTaint()
            accessed_memory = []
            sandbox_u8 = input_taint.view(np.uint8)
            actual_sandbox_memory_size = input_taint[0]["main"].view(np.uint8).size + input_taint[0]["faulty"].view(np.uint8).size
            gpr_region_u8 = input_taint[0]["gpr"].view(np.uint8)
            for ite in cer:
                if ite.metadata.has_memory_access:
                    for byte_idx in range(ite.metadata.memory_access.element_size):
                        byte_offset = ite.metadata.memory_access.effective_address + byte_idx - ite.cpu.gpr[29]
                        accessed_memory.append(byte_offset)
                        if not ite.metadata.memory_access.is_write:
                            if 0 <= byte_offset < actual_sandbox_memory_size:
                                sandbox_u8[byte_offset] = True
                        else:
                            if not sandbox_u8[byte_offset]:
                                overriden_memory.add(byte_offset)
                srcs, dests = get_srcs_dests_operands(ite.cpu.encoding, ite.cpu.pc)
                input_overriden_dests = set(filter(lambda op: op not in srcs and all(not gpr_region_u8[offset] for offset in map_operand_to_input_offsets(op)), dests))
                overriden_gprs |= input_overriden_dests
                for offset in chain.from_iterable(map(map_operand_to_input_offsets, srcs)):
                    gpr_region_u8[offset] = True
            for offset in overriden_memory:
                sandbox_u8[offset] = False
            for offset in chain.from_iterable(map(map_operand_to_input_offsets, overriden_gprs)):
                gpr_region_u8[offset] = False
            taints.append(input_taint)

            line_size = 64
            num_sets = 64
            cache_sets = sorted(set(map(lambda addr: (addr // line_size) % num_sets, accessed_memory)))
            ctraces.append(CTrace(raw_trace=cache_sets))

        return ctraces, taints, traces, []


# ==================================================================================================
# Non-interference executor (PAC / MTE)
# ==================================================================================================
class Aarch64NonInterferenceExecutor(Aarch64LocalExecutor):
    """
    Executor for non-interference testing (PAC, and later MTE).

    Extends Aarch64LocalExecutor with PAC stage-1 instrumentation that is
    cached once per test case inside load_test_case().  This makes
    trace_test_case_with_taints() deterministic across repeated calls for the
    same TC — critical for the two-call pattern in fuzzer._boost_inputs /
    fuzzer._collect_traces.

    Lifecycle
    ─────────
    1. fuzzer calls set_generator(generator) once after initialize_modules().
    2. fuzzer calls executor.load_test_case(tc)  ← stage-1 runs here, cached.
    3. fuzzer calls trace_test_case_with_taints() one or more times  ← uses cache.
       Each call is deterministic because stage-1 results are fixed for this TC.
    4. fuzzer calls load_test_case(new_tc)  ← cache is replaced.

    Regular Revizor mode uses Aarch64LocalExecutor directly (no stage-1, no variants).
    """

    def __init__(self, workdir: str, generator: Aarch64Generator, *args, **kwargs):
        super().__init__(workdir, *args, **kwargs)
        self._generator: Aarch64Generator = generator
        # Stage-1 cache — populated by load_test_case(), consumed by trace_test_case_with_taints()
        self._stage1_pac = None
        self._stage1_tmp = None
        self._stage1_fix_points = None
        self._stage1_tc_bytes = None
        self._stage1_pac_offset_to_fps: Optional[Dict[int, List[FixPoint]]] = None

    def load_test_case(self, test_case: TestCase):
        result = super().load_test_case(test_case)
        self._run_stage1()
        return result

    def _run_stage1(self) -> None:
        """Run PAC stage-1 instrumentation on self.test_case and cache the results."""
        patched = copy.deepcopy(self.test_case)
        pac = PACInstrumentation(self._generator, 0.2, 0.2, 0.2)
        tmp, fix_points = pac.instrument_stage1(patched)
        layout: Aarch64ASMLayout = Aarch64ASMLayout(tmp)

        sandbox_base, _ = self.read_base_addresses()
        pac_offset_to_fps: Dict[int, List[FixPoint]] = {}
        for fp in fix_points:
            pac_inst = fp.info.pac_inst
            if pac_inst is not None and pac_inst in layout.instruction_address:
                offset = layout.instruction_address[pac_inst] + 4  # capture state after PACIA
                pac_offset_to_fps.setdefault(offset, []).append(fp)

        assembly = Aarch64Printer(self.target_desc).print_layout(layout)
        tc_bytes = ConfigurableGenerator.in_memory_assemble(assembly)

        self._stage1_pac = pac
        self._stage1_tmp = tmp
        self._stage1_fix_points = fix_points
        self._stage1_tc_bytes = tc_bytes
        self._stage1_pac_offset_to_fps = pac_offset_to_fps

    # ── Helpers only needed in non-interference mode ───────────────────────────

    def _write_prebuilt_tc_to_executor(self, tc: TestCase) -> None:
        """Write an already-instrumented TestCase to the hardware executor, no passes applied."""
        layout = Aarch64ASMLayout(tc)
        assembly = Aarch64Printer(self.target_desc).print_layout(layout)
        tc_bytes = ConfigurableGenerator.in_memory_assemble(assembly)
        self.local_executor.checkout_region(TestCaseRegion())
        self.local_executor.write(tc_bytes)

    def _hw_trace_single_input(self, tc: TestCase, inp: Input, n_reps: int = 1) -> HTrace:
        """Load tc + inp into the hardware executor, trace n_reps times, return HTrace."""
        self.local_executor.discard_all_inputs()
        iid = self.local_executor.allocate_iid()
        self.local_executor.checkout_region(InputRegion(iid))
        self.local_executor.write(ExecutorMemory(inp.tobytes()))
        self._write_prebuilt_tc_to_executor(tc)
        trace_list = []
        for _ in range(n_reps):
            self.local_executor.trace()
            self.local_executor.checkout_region(InputRegion(iid))
            hwm = self.local_executor.hardware_measurement()
            trace_list.append(hwm.htrace)
        return HTrace(trace_list=trace_list)

    def trace_tc_variants_on_ce(
        self,
        inputs: List[Input],
        tc_variants_per_input: List[TCVariants],
    ) -> List[TCVariantCETraces]:
        """DEBUG: run every TC variant for every input through the Contract Executor."""
        sandbox_base, _ = self.read_base_addresses()
        results: List[TCVariantCETraces] = []
        for inp, variants in zip(inputs, tc_variants_per_input):
            data = inp.tobytes()
            tc_memory = data[:0x2000]
            tc_regs = bytearray(data[0x2000:])
            view = memoryview(tc_regs).cast('Q')
            view[6] = (view[6] << 28) & ((1 << 64) - 1)
            tc_regs = bytes(tc_regs)
            ce_traces: Dict[str, ContractExecutionResult] = {}
            for name, tc in variants.items():
                tc_bytes = self._tc_to_bytes(tc)
                execution = ContractExecution(
                    tc_bytes, tc_memory, tc_regs,
                    SimArch.RVZR_ARCH_AARCH64, 5, 10,
                    req_mem_base_virt=sandbox_base,
                )
                ce_traces[name] = self._contract_executor.run(execution)
            results.append(TCVariantCETraces(
                tc1=ce_traces['tc1'],
                tc2=ce_traces['tc2'],
                tc3=ce_traces['tc3'],
            ))
        return results

    @staticmethod
    def _ctrace_from_ce(cer: ContractExecutionResult) -> CTrace:
        """Compute a cache-set CTrace from a single CE result."""
        line_size = 64
        num_sets = 64
        accessed_memory = []
        for ite in cer:
            if ite.metadata.has_memory_access:
                for byte_idx in range(ite.metadata.memory_access.element_size):
                    byte_offset = (ite.metadata.memory_access.effective_address
                                   + byte_idx - ite.cpu.gpr[29])
                    accessed_memory.append(byte_offset)
        cache_sets = sorted(set((addr // line_size) % num_sets for addr in accessed_memory))
        return CTrace(raw_trace=cache_sets)

    def debug_compare_variants_on_ce(
        self,
        inputs: List[Input],
        tc_variants_per_input: List[TCVariants],
    ) -> None:
        """
        Debug: run TC1/TC2/TC3 through the CE for every input and compare their CTraces.

        TC1 (strip-only) and TC2 (full restore + auth) should produce identical CTraces
        — they are architecturally equivalent.  TC3 (auth with wrong pointer, spec path)
        may produce a different CTrace, which would indicate a speculative cache side-effect.
        """
        try:
            ce_variant_traces = self.trace_tc_variants_on_ce(inputs, tc_variants_per_input)
        except RuntimeError as e:
            print(f"[CE_VARIANT_COMPARE] CE crashed while running variants: {e}")
            return

        for i, variant_ce in enumerate(ce_variant_traces):
            tc1_ct = self._ctrace_from_ce(variant_ce.tc1)
            tc2_ct = self._ctrace_from_ce(variant_ce.tc2)
            tc3_ct = self._ctrace_from_ce(variant_ce.tc3)

            mismatch_1_2 = tc1_ct != tc2_ct
            mismatch_3_1 = tc3_ct != tc1_ct

            if mismatch_1_2:
                import pdb; pdb.set_trace()
                print(f"[CE_VARIANT_COMPARE] input {i}: TC1 != TC2  (UNEXPECTED — should be arch-equivalent)")
                print(f"  TC1 ctrace: {tc1_ct.raw}")
                print(f"  TC2 ctrace: {tc2_ct.raw}")
            if mismatch_3_1:
                import pdb; pdb.set_trace()
                print(f"[CE_VARIANT_COMPARE] input {i}: TC3 != TC1  (potential speculation side-effect)")
                print(f"  TC1 ctrace: {tc1_ct.raw}")
                print(f"  TC3 ctrace: {tc3_ct.raw}")
            if not mismatch_1_2 and not mismatch_3_1:
                print(f"[CE_VARIANT_COMPARE] input {i}: TC1 == TC2 == TC3  {tc1_ct.raw}")

    def trace_test_case_from_variants(
        self,
        inp: Input,
        tc_variants: List[TCVariants],
        n_reps: int = 1,
    ) -> List[List[HTrace]]:
        """
        Run a single input through TC1/TC2/TC3 for each variant set on real hardware.

        Returns a list of [tc1_htrace, tc2_htrace, tc3_htrace] per entry in tc_variants.
        TC1 and TC2 should produce identical HTraces (arch-equivalent).
        TC3 may differ from TC1 if speculation through a wrong AUTH leaks cache state.
        """
        results = []
        for variants in tc_variants:
            traces = []
            for tc in [variants.tc1, variants.tc2, variants.tc3]:
                traces.append(self._hw_trace_single_input(tc, inp, n_reps))
            results.append(traces)
        return results

    # ── Core non-interference tracing ─────────────────────────────────────────

    def trace_test_case_with_taints(self, inputs: List[Input], nesting: int) -> Tuple[List[CTrace], List[InputTaint], List[ContractExecutionResult], List[TCVariants]]:
        """
        Non-interference mode: CE on the stage-1 PAC-instrumented TC, with stage-2
        variant generation.

        Uses the stage-1 cache built by load_test_case() — every call for the same
        TC is deterministic (no random re-instrumentation).

        :return: (ctraces, taints, ce_traces, tc_variants_per_input)
        """
        if not inputs or self.test_case is None:
            return [], [], [], []

        assert self._stage1_pac is not None, \
            "stage-1 cache is empty: load_test_case() was not called"

        pac = self._stage1_pac
        tmp = self._stage1_tmp
        fix_points = self._stage1_fix_points
        tc_bytes = self._stage1_tc_bytes
        pac_offset_to_fps = self._stage1_pac_offset_to_fps

        sandbox_base, _ = self.read_base_addresses()
        traces: List[ContractExecutionResult] = []
        tc_variants_per_input: List[TCVariants] = []
        last_cer: Optional[ContractExecutionResult] = None

        for inp in inputs:
            # Reset per-input fix-point state (signed_value, ctx_value, spec_nesting).
            for fp in fix_points:
                fp.reset()

            data = inp.tobytes()
            tc_memory = data[:0x2000]
            tc_regs = bytearray(data[0x2000:])
            view = memoryview(tc_regs).cast('Q')
            # The hardware executor shifts the flags register left 28 bits; mirror that here.
            view[6] = (view[6] << 28) & ((1 << 64) - 1)
            tc_regs = bytes(tc_regs)

            execution = ContractExecution(tc_bytes, tc_memory, tc_regs, SimArch.RVZR_ARCH_AARCH64, 5, 10,
                                          req_mem_base_virt=sandbox_base)
            try:
                cer = self._contract_executor.run(execution)
            except RuntimeError as e:
                import datetime
                inp_idx = inputs.index(inp)
                regs_u64 = list(memoryview(data[0x2000:]).cast('Q'))

                # Disassemble the stage-1 machine code for the crash dump
                cs = Cs(CS_ARCH_ARM64, CS_MODE_ARM)
                cs.detail = False
                disasm_lines = []
                for i, insn in enumerate(cs.disasm(tc_bytes, 0)):
                    disasm_lines.append(f"  +{insn.address:04x} ({i:3d}): {insn.mnemonic} {insn.op_str}")

                print(f"[CE CRASH] Stage-1 CE crash for input {inp_idx}: {e}")
                print(f"  machine_code ({len(tc_bytes)} bytes): {tc_bytes.hex()}")
                print(f"  regs: {[hex(v) for v in regs_u64[:10]]}")
                print("  disassembly:")
                for line in disasm_lines:
                    print(line)
                if last_cer is not None:
                    print(f"  Last successful CE trace (input {inp_idx - 1}):")
                    last_cer.pretty_print()

                # Save full crash info to file for later offline inspection
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                crash_path = f"/tmp/ce_crash_{ts}.txt"
                try:
                    with open(crash_path, "w") as f:
                        f.write(f"[CE CRASH] input index: {inp_idx}\n")
                        f.write(f"error: {e}\n\n")
                        f.write(f"machine_code ({len(tc_bytes)} bytes):\n{tc_bytes.hex()}\n\n")
                        f.write("disassembly:\n")
                        f.write("\n".join(disasm_lines) + "\n\n")
                        f.write(f"regs: {[hex(v) for v in regs_u64[:10]]}\n\n")
                        if last_cer is not None:
                            import io, sys
                            buf = io.StringIO()
                            old_stdout = sys.stdout
                            sys.stdout = buf
                            last_cer.pretty_print()
                            sys.stdout = old_stdout
                            f.write(f"last successful CE trace (input {inp_idx - 1}):\n")
                            f.write(buf.getvalue())
                    print(f"  crash info saved to {crash_path}")
                except Exception as save_err:
                    print(f"  [warn] could not save crash file: {save_err}")
                raise
            last_cer = cer
            traces.append(cer)

            # Capture signed pointer and context values immediately after PACIA executes.
            # The pac_sign_hook patches PACIA to NOP after the first (architectural) execution,
            # so speculative re-runs at the same PC see NOP and the register holds the original
            # unsigned pointer P instead of P*.  Overwriting fp.signed_value with P would make
            # TC2's AUTIA(P, K, C) fail authentication → corrupted pointer → wrong CTrace.
            # Fix: lock in the architectural (nesting=0) capture; don't overwrite it with
            # speculative re-executions.
            if pac_offset_to_fps and len(cer) > 0:
                actual_code_base = cer[0].cpu.pc
                pac_pc_to_fps = {actual_code_base + off: fps
                                 for off, fps in pac_offset_to_fps.items()}
                arch_captured: Set[int] = set()  # slot_ids captured at nesting==0
                for ite in cer:
                    fps = pac_pc_to_fps.get(ite.cpu.pc)
                    if fps is not None:
                        for fp in fps:
                            nesting = ite.metadata.speculation_nesting
                            # Accept this entry if: architectural (nesting=0), or not yet captured.
                            if nesting == 0 or fp.slot_id not in arch_captured:
                                reg = fp.info.reg
                                if reg.startswith('x') and reg[1:].isdigit():
                                    fp.signed_value = ite.cpu.gpr[int(reg[1:])]
                                ctx = fp.info.ctx_reg
                                if ctx is not None and ctx.startswith('x') and ctx[1:].isdigit():
                                    fp.ctx_value = ite.cpu.gpr[int(ctx[1:])]
                                fp.spec_nesting = nesting
                                if nesting == 0:
                                    arch_captured.add(fp.slot_id)

            # Stage 2: generate TC variants (TC1/TC2/TC3) from the signed values captured above.
            tc1, tc2, tc3 = pac.instrument_stage2(tmp, fix_points)
            tc_variants_per_input.append(TCVariants(tc1=tc1, tc2=tc2, tc3=tc3))

        # Taint tracking and CTrace computation (same logic as base class).
        taints, ctraces = [], []
        for cer in traces:
            overriden_gprs = set()
            overriden_memory = set()
            input_taint = InputTaint()
            accessed_memory = []
            sandbox_u8 = input_taint.view(np.uint8)
            actual_sandbox_memory_size = input_taint[0]["main"].view(np.uint8).size + input_taint[0]["faulty"].view(np.uint8).size
            gpr_region_u8 = input_taint[0]["gpr"].view(np.uint8)
            for ite in cer:
                if ite.metadata.has_memory_access:
                    for byte_idx in range(ite.metadata.memory_access.element_size):
                        byte_offset = ite.metadata.memory_access.effective_address + byte_idx - ite.cpu.gpr[29]
                        accessed_memory.append(byte_offset)
                        if not ite.metadata.memory_access.is_write:
                            if 0 <= byte_offset < actual_sandbox_memory_size:
                                sandbox_u8[byte_offset] = True
                        else:
                            if not sandbox_u8[byte_offset]:
                                overriden_memory.add(byte_offset)
                srcs, dests = get_srcs_dests_operands(ite.cpu.encoding, ite.cpu.pc)
                input_overriden_dests = set(filter(lambda op: op not in srcs and all(not gpr_region_u8[offset] for offset in map_operand_to_input_offsets(op)), dests))
                overriden_gprs |= input_overriden_dests
                for offset in chain.from_iterable(map(map_operand_to_input_offsets, srcs)):
                    gpr_region_u8[offset] = True
            for offset in overriden_memory:
                sandbox_u8[offset] = False
            for offset in chain.from_iterable(map(map_operand_to_input_offsets, overriden_gprs)):
                gpr_region_u8[offset] = False
            taints.append(input_taint)

            line_size = 64
            num_sets = 64
            cache_sets = sorted(set(map(lambda addr: (addr // line_size) % num_sets, accessed_memory)))
            ctraces.append(CTrace(raw_trace=cache_sets))

        return ctraces, taints, traces, tc_variants_per_input


def map_operand_to_input_offsets(op: str) -> List[int]:
    if op.lower().startswith(("x", "w")):
        n = int(op[1:])
        if n < 0 or n > 5:
            raise ValueError(f"Register index must be 0–5, got {op}")
        size = 8 if op.lower().startswith("x") else 4
    elif op.lower() in ("n", "z", "c", "v"):
        n = 6
        size = 8
    elif op.lower() == "sp":
        n = 7
        size = 8
    elif op.lower() == "fp":
        return []
    else:
        raise RuntimeError(f"Unexpected src register '{op}'")

    return list(map(lambda byte_index: n * 8 + byte_index, range(size)))

def get_srcs_dests_operands(encoding: int, pc: int) -> Union[List, List]:
    FLAG_BITS = {"N", "Z", "C", "V"}

    def cc_to_read_flags(cc: int):
        cond_map = {
                0: {"Z"},
                8: {"Z"},
                1: {"C"},
                9: {"C"},
                2:  {"N"},
                10: {"N"},
                3:  {"V"},
                11: {"V"},
                4:  {"C", "Z"},
                12: {"C", "Z"},
                5:  {"N", "V"},
                13: {"N", "V"},
                6:  {"Z", "N", "V"},
                14: {"Z", "N", "V"},
                7:  set(),
                15: set(),
        }
        assert cc in cond_map
        return cond_map[cc]

    capstone_arm64 = Cs(CS_ARCH_ARM64, CS_MODE_ARM)
    capstone_arm64.detail = True

    dest = set()
    src = set()
    try:
        code_bytes = encoding.to_bytes(4, byteorder="little")
        insns = list(capstone_arm64.disasm(code_bytes, pc))
        assert insns and len(insns) == 1
        insn = insns[0]
        if insn.update_flags:
            dest |= FLAG_BITS
        if insn.cc is not None:
            src |= cc_to_read_flags(insn.cc)

        for op in insn.operands:
            if op.type == ARM64_OP_REG:
                reg = insn.reg_name(op.reg)
                if op.access & CS_AC_WRITE:
                    dest.add(reg)
                if op.access & CS_AC_READ:
                    src.add(reg)

            elif op.type == ARM64_OP_MEM:
                if op.mem.base != 0:
                    src.add(insn.reg_name(op.mem.base))

                if op.mem.index != 0:
                    src.add(insn.reg_name(op.mem.index))


        return sorted(src), sorted(dest)

    except Exception as e:
        return [], []




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


def show_context(trace, idx, window=-1):
    if(window < 0):
        window = len(trace)
    start = max(0, idx - window)
    end = min(len(trace), idx + window + 1)
    for j in range(start, end):
        insn = trace[j]
        disas = disassemble_instruction(insn.cpu.encoding, insn.cpu.pc)
        marker = "→" if j == idx else " "
        print(f"{marker} [{j:03d}] 0x{insn.cpu.pc:016x}: {disas}")


