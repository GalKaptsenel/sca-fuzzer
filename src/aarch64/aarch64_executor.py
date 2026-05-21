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
from capstone.arm64 import (ARM64_OP_REG, ARM64_OP_MEM,
                             ARM64_CC_INVALID, ARM64_CC_EQ, ARM64_CC_NE,
                             ARM64_CC_HS, ARM64_CC_LO, ARM64_CC_MI, ARM64_CC_PL,
                             ARM64_CC_VS, ARM64_CC_VC, ARM64_CC_HI, ARM64_CC_LS,
                             ARM64_CC_GE, ARM64_CC_LT, ARM64_CC_GT, ARM64_CC_LE,
                             ARM64_CC_AL, ARM64_CC_NV)

from .aarch64_generator import Aarch64Printer, Aarch64ASMLayout, Aarch64Generator
from .. import ConfigurableGenerator
from ..interfaces import HTrace, Input, TestCase, Executor, HardwareTracingError, Analyser, CTrace, InputTaint, TargetDesc
from ..config import CONF
from ..util import Logger, STAT
from .aarch64_target_desc import Aarch64TargetDesc, NZCVScheme
from .aarch64_connection import LocalExecutorImp, TestCaseRegion, InputRegion, PacKeys
from .aarch64_generator import Pass, Aarch64SandboxPass, PACInstrumentation, FixPoint, AUTH_SLOT_POS, MTEInstrumentation, MTEFixPoint

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

    def set_branch_mistraining(self, cer) -> None:
        """Train base predictor opposite to CE branch outcomes for one input.

        For each architectural conditional branch in cer, saturates the base predictor
        in the opposite direction so the first HW execution always mispredicts it.
        Must be called once per input, before the corresponding HW trace call.

        :param cer: ContractExecutionResult for the input about to be measured
        """
        if cer is None or len(cer) == 0:
            self.local_executor.clear_branch_training()
            return

        ce_code_base = cer[0].cpu.pc
        entries = []

        for i, ite in enumerate(cer):
            if ite.metadata.speculation_nesting != 0:
                continue
            if not is_conditional_branch(ite.cpu.encoding):
                continue
            if i + 1 >= len(cer):
                continue

            taken = (cer[i + 1].cpu.pc != ite.cpu.pc + 4)
            byte_offset = ite.cpu.pc - ce_code_base
            entries.append((byte_offset, not taken))  # opposite → guaranteed mispredict

        if entries:
            self.local_executor.write_branch_training_config(entries)
        else:
            self.local_executor.clear_branch_training()
        return entries

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
#        for idx, i in enumerate(inputs):
#            input_to_iid[idx] = self.local_executor.allocate_iid()
#            self.local_executor.checkout_region(InputRegion(input_to_iid[idx]))
#            self.local_executor.write(ExecutorMemory(i.tobytes()))
#
#        sandboxed_test_case = self._write_mod_test_case_to_local_executor("sandboxed_test_case", [Aarch64SandboxPass()])
#
#        input_to_trace_list = defaultdict(list)
#
#        counter_log = defaultdict(list)
#        for _ in range(n_reps):
#            self.local_executor.trace()
#            for idx, i in enumerate(inputs):
#                self.local_executor.checkout_region(InputRegion(input_to_iid[idx]))
#                hwm = self.local_executor.hardware_measurement()
#                input_to_trace_list[idx].append(hwm.htrace)
#                counter_log[idx].append(hwm.pfcs[2])

        sandboxed_test_case = self._write_mod_test_case_to_local_executor("sandboxed_test_case", [Aarch64SandboxPass()])
        iid = self.local_executor.allocate_iid()
        self.local_executor.checkout_region(InputRegion(iid))

        input_to_trace_list = defaultdict(list)
        counter_log = defaultdict(list)
        expected_log = {}

        for _ in range(n_reps):
            for idx, i in enumerate(inputs):
                self.local_executor.write(ExecutorMemory(_input_bytes_with_pstate(i)))
                trained_entries = []
                if hasattr(i, "_arch_trace"):
                    trained_entries = self.set_branch_mistraining(i._arch_trace)
                if idx not in expected_log:
                    expected_log[idx] = len(trained_entries)
                assert expected_log[idx] == len(trained_entries)
                self.local_executor.trace()
                hwm = self.local_executor.hardware_measurement()
                input_to_trace_list[idx].append(hwm.htrace)
                counter_log[idx].append(hwm.pfcs[2])
                if hasattr(i, "_arch_trace"):
                    pass
#                    print(f"[LOG mistraining GAL] expected: {len(trained_entries)}, Got {hwm.pfcs[2]} --> {'OK' if len(trained_entries) <= hwm.pfcs[2] else 'FAIL'}")



#        input_to_trace_list = defaultdict(list)
#
#        counter_log = defaultdict(list)
#        for _ in range(n_reps):
#            self.local_executor.trace()
#            for idx, i in enumerate(inputs):
#                self.local_executor.checkout_region(InputRegion(input_to_iid[idx]))
#                hwm = self.local_executor.hardware_measurement()
#                input_to_trace_list[idx].append(hwm.htrace)
#                counter_log[idx].append(hwm.pfcs[2])
# 
                    
        for idx, vals in counter_log.items():
            avg = 100 * (((1.0 * sum(vals)) / len(vals)) / expected_log[idx]) if expected_log[idx] > 0 else "NA"
            print(f"[LOG Gal] input {idx} miss-rate: {sum(vals) / len(vals)} / {expected_log[idx]} = {avg}%: {vals}")

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
            _reconstruct_pstate(view)
            tc_regs = bytes(tc_regs)
            execution = ContractExecution(tc_bytes, tc_memory, tc_regs, SimArch.RVZR_ARCH_AARCH64, 5, 10,
                                          req_mem_base_virt=sandbox_base)
            cer = self._contract_executor.run(execution)
            traces.append(cer)

        taints = [compute_taint(cer) for cer in traces]
        ctraces = [compute_ctrace(cer) for cer in traces]

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
    1. Constructed via factory.get_noninterference_executor(generator).
    2. fuzzer calls executor.load_test_case(tc)  ← stage-1 runs here, cached.
    3. fuzzer calls trace_test_case_with_taints() one or more times  ← uses cache.
       Each call is deterministic because stage-1 results are fixed for this TC.
    4. fuzzer calls load_test_case(new_tc)  ← cache is replaced.

    Regular Revizor mode uses Aarch64LocalExecutor directly (no stage-1, no variants).
    """

    def __init__(self, workdir: str, generator: Aarch64Generator, *args, **kwargs):
        super().__init__(workdir, *args, **kwargs)
        self._generator: Aarch64Generator = generator

        # Snapshot current EL1 PAC keys (this process's keys) and store them in the
        # kernel module.  All subsequent PAC sign/auth ioctls and HW TC executions
        # will save current HW keys, install these keys, run, then restore — ensuring
        # consistent key use regardless of CPU or scheduler context.
        _pac_keys = self.local_executor.get_pac_keys()
        self.local_executor.set_pac_keys(_pac_keys)
        self._init_pac_keys = _pac_keys

        # Create PACInstrumentation once so register_controlled_instructions("BASE-PAC")
        # fires before the generator ever produces a test case.
        self._pac_instrumentation = PACInstrumentation(
            self._generator, CONF.pac_xpac_weight, CONF.pac_auth_weight, CONF.pac_sign_weight)

        # Stage-1 cache — populated by load_test_case(), consumed by trace_test_case_with_taints()
        self._stage1_pac_instrumentation = None
        self._stage1_tc = None
        self._stage1_fix_points = None
        self._stage1_tc_bytes = None
        self._stage1_pac_offset_to_fps: Optional[Dict[int, List[FixPoint]]] = None
        self._stage1_slot_auth_offset_to_fp: Optional[Dict[int, FixPoint]] = None
        self._tc_counter: int = 0

        import datetime, os as _os
        log_dir = _os.path.expanduser("~/revizor/logs")
        _os.makedirs(log_dir, exist_ok=True)
        log_path = _os.path.join(log_dir, f"pac_fuzzer_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        self._pac_log_file = open(log_path, 'w', buffering=1)
        self._pac_log_file_path = log_path
        print(f"[PAC LOG] Writing detailed log to {log_path}")

    def _pac_log(self, msg: str, also_print: bool = False) -> None:
        self._pac_log_file.write(msg + "\n")
        if also_print:
            print(msg)

    def load_test_case(self, test_case: TestCase):
        result = super().load_test_case(test_case)
        self._run_stage1()
        return result

    def _run_stage1(self) -> None:
        """Run PAC stage-1 instrumentation on self.test_case and cache the results."""
        self._tc_counter += 1
        tc_id = self._tc_counter

        patched = copy.deepcopy(self.test_case)
        pac = self._pac_instrumentation
        stage1_tc, fix_points = pac.instrument_stage1(patched)
        layout: Aarch64ASMLayout = Aarch64ASMLayout(stage1_tc)

        sandbox_base, _ = self.read_base_addresses()
        pac_offset_to_fps: Dict[int, List[FixPoint]] = {}
        for fp in fix_points:
            assert fp.pac_insts, f"FixPoint slot {fp.slot_id} has empty pac_insts"
            for pac_inst in fp.pac_insts:
                assert pac_inst in layout.instruction_address, \
                    f"PACIA instruction not found in layout — pac_insts out of sync with stage1_tc"
                offset = layout.instruction_address[pac_inst] + 4  # capture state after PACIA
                pac_offset_to_fps.setdefault(offset, []).append(fp)

        assembly = Aarch64Printer(self.target_desc).print_layout(layout)
        tc_bytes = ConfigurableGenerator.in_memory_assemble(assembly)

        # Map from AUTH slot byte-offset in tc_bytes → FixPoint.
        # Used to detect if a MISSING-capture slot was actually executed at nesting==0 in CE
        # (which would indicate a capture bug, not a harmless untaken branch).
        slot_auth_offset_to_fp: Dict[int, FixPoint] = {}
        for func in stage1_tc.functions:
            for bb in func:
                for inst in bb:
                    if (hasattr(inst, '_pac_slot_id') and
                            hasattr(inst, '_pac_slot_pos') and
                            inst._pac_slot_pos == AUTH_SLOT_POS and
                            inst in layout.instruction_address):
                        slot_id = inst._pac_slot_id
                        for fp in fix_points:
                            if fp.slot_id == slot_id:
                                slot_auth_offset_to_fp[layout.instruction_address[inst]] = fp
                                break

        # Sanity check: every fix_point's slot must be discoverable in stage1_tc.
        _slot_ids_in_tc = set()
        for _func in stage1_tc.functions:
            for _bb in _func:
                for _inst in _bb:
                    if hasattr(_inst, '_pac_slot_id'):
                        _slot_ids_in_tc.add(_inst._pac_slot_id)
        for fp in fix_points:
            assert fp.slot_id in _slot_ids_in_tc, (
                f"stage1 bug: fix_point slot_id={fp.slot_id} not found in stage1_tc "
                f"(tc slot_ids={sorted(_slot_ids_in_tc)}, "
                f"fp slot_ids={sorted(f.slot_id for f in fix_points)})"
            )

        self._stage1_pac_instrumentation = pac
        self._stage1_tc = stage1_tc
        self._stage1_fix_points = fix_points
        self._stage1_tc_bytes = tc_bytes
        self._stage1_pac_offset_to_fps = pac_offset_to_fps
        self._stage1_slot_auth_offset_to_fp = slot_auth_offset_to_fp

        # ── Logging ──────────────────────────────────────────────────────────
        self._pac_log(f"\n{'='*72}")
        self._pac_log(f"[TC {tc_id}] Stage-1 instrumented TC ({len(tc_bytes)} bytes, "
                      f"{len(fix_points)} fix-points, sandbox_base=0x{sandbox_base:016x})")
        cs = Cs(CS_ARCH_ARM64, CS_MODE_ARM)
        cs.detail = False
        self._pac_log("  Stage-1 disassembly:")
        for insn in cs.disasm(tc_bytes, 0):
            self._pac_log(f"    +{insn.address:04x}: {insn.mnemonic} {insn.op_str}")
        self._pac_log(f"  Fix-points ({len(fix_points)}):")
        for fp in fix_points:
            capture_offsets = [off for off, fps in pac_offset_to_fps.items() if fp in fps]
            auth_offset = next((off for off, f in slot_auth_offset_to_fp.items() if f is fp), None)
            self._pac_log(f"    slot={fp.slot_id:3d}  reg={fp.info.reg:4s}  "
                          f"ctx={str(fp.info.ctx_reg):4s}  "
                          f"pac={fp.info.pac_mnemonic:8s}  auth={fp.info.auth_mnemonic:8s}  "
                          f"capture_offsets={[hex(o) for o in capture_offsets]}  "
                          f"auth_slot_offset={hex(auth_offset) if auth_offset is not None else 'None'}")
        self._pac_log(f"  Taint decisions ({len(pac.last_taint_log)} events):")
        for line in pac.last_taint_log:
            self._pac_log(line)

    # ── Helpers only needed in non-interference mode ───────────────────────────

    def _write_prebuilt_tc_to_executor(self, tc: TestCase) -> None:
        """Write an already-instrumented TestCase to the hardware executor, no passes applied."""
        layout = Aarch64ASMLayout(tc)
        assembly = Aarch64Printer(self.target_desc).print_layout(layout)
        tc_bytes = ConfigurableGenerator.in_memory_assemble(assembly)
        self.local_executor.checkout_region(TestCaseRegion())
        self.local_executor.write(tc_bytes)

    def _disasm_tc(self, tc: TestCase) -> List[str]:
        """Disassemble a TestCase to a list of '+offset: mnem ops' strings."""
        layout = Aarch64ASMLayout(tc)
        asm = Aarch64Printer(self.target_desc).print_layout(layout)
        tc_bytes = ConfigurableGenerator.in_memory_assemble(asm)
        cs = Cs(CS_ARCH_ARM64, CS_MODE_ARM)
        cs.detail = False
        return [f"+{insn.address:04x}: {insn.mnemonic} {insn.op_str}"
                for insn in cs.disasm(tc_bytes, 0)]

    def _dump_pre_hw(self, tc_name: str, tc: TestCase, inp: Input,
                     fix_points: List, inp_idx: int, dump_path: str,
                     all_variants: Optional['TCVariants'] = None,
                     stage1_bytes: Optional[bytes] = None) -> None:
        """Write a full pre-execution dump to dump_path, synced to disk.

        The file is overwritten on every call so after a kernel-panic reset it
        always contains exactly the last thing that was about to run on HW.
        Uses os.fsync + os.sync so the data reaches the disk before we hand
        control to the kernel module.

        all_variants: if provided, all three TC variant disassemblies are included.
        stage1_bytes: if provided, the stage-1 (pre-stage-2) disassembly is included.
        """
        import os as _os
        data = inp.tobytes()
        # GPR input area: 8 uint64 slots — x0, x1, x2, x3, x4, x5, nzcv, sp
        regs = list(memoryview(data[0x2000:0x2040]).cast('Q'))
        reg_names = ["x0", "x1", "x2", "x3", "x4", "x5", "nzcv", "sp"]
        disasm = self._disasm_tc(tc)
        lines = [
            f"=== PRE-HW DUMP ===",
            f"tc_counter={self._tc_counter}  variant={tc_name}  input_idx={inp_idx}",
            f"input registers (x0-x5, nzcv, sp):",
        ]
        for name, val in zip(reg_names, regs):
            lines.append(f"  {name:4s} = 0x{val:016x}")
        lines.append(f"input memory (first 256 bytes hex):")
        mem = data[:256]
        for off in range(0, len(mem), 16):
            chunk = mem[off:off + 16].hex(' ')
            lines.append(f"  +{off:04x}: {chunk}")
        lines.append(f"fix-points ({len(fix_points)}):")

        for fp in fix_points:
            sv = hex(fp.signed_value) if fp.signed_value is not None else "None"
            cv = hex(fp.ctx_value) if fp.ctx_value is not None else "None"
            lines.append(
                f"  slot={fp.slot_id:3d}  reg={fp.info.reg:4s}  ctx={str(fp.info.ctx_reg):4s}"
                f"  signed={sv}  ctx={cv}  spec_nesting={fp.spec_nesting}"
            )
        # Verify each captured signed value is authenticatable with the stored keys.
        # Also log the current APDB key so we can compare CE-signing keys vs HW-execution keys.
        try:
            cur_keys = self.local_executor.get_pac_keys()
            init_keys = self._init_pac_keys
            apda_match = (cur_keys.apda_lo == init_keys.apda_lo and cur_keys.apda_hi == init_keys.apda_hi)
            apdb_match = (cur_keys.apdb_lo == init_keys.apdb_lo and cur_keys.apdb_hi == init_keys.apdb_hi)
            apia_match = (cur_keys.apia_lo == init_keys.apia_lo and cur_keys.apia_hi == init_keys.apia_hi)
            apib_match = (cur_keys.apib_lo == init_keys.apib_lo and cur_keys.apib_hi == init_keys.apib_hi)
            apga_match = (cur_keys.apga_lo == init_keys.apga_lo and cur_keys.apga_hi == init_keys.apga_hi)
            lines.append(f"PAC keys (init-time vs current HW registers):")
            lines.append(f"  apia: init=lo={init_keys.apia_lo:016x}/hi={init_keys.apia_hi:016x}"
                         f"  cur=lo={cur_keys.apia_lo:016x}/hi={cur_keys.apia_hi:016x}"
                         f"  {'MATCH' if apia_match else '*** MISMATCH ***'}")
            lines.append(f"  apib: init=lo={init_keys.apib_lo:016x}/hi={init_keys.apib_hi:016x}"
                         f"  cur=lo={cur_keys.apib_lo:016x}/hi={cur_keys.apib_hi:016x}"
                         f"  {'MATCH' if apib_match else '*** MISMATCH ***'}")
            lines.append(f"  apda: init=lo={init_keys.apda_lo:016x}/hi={init_keys.apda_hi:016x}"
                         f"  cur=lo={cur_keys.apda_lo:016x}/hi={cur_keys.apda_hi:016x}"
                         f"  {'MATCH' if apda_match else '*** MISMATCH ***'}")
            lines.append(f"  apdb: init=lo={init_keys.apdb_lo:016x}/hi={init_keys.apdb_hi:016x}"
                         f"  cur=lo={cur_keys.apdb_lo:016x}/hi={cur_keys.apdb_hi:016x}"
                         f"  {'MATCH' if apdb_match else '*** MISMATCH ***'}")
            lines.append(f"  apga: init=lo={init_keys.apga_lo:016x}/hi={init_keys.apga_hi:016x}"
                         f"  cur=lo={cur_keys.apga_lo:016x}/hi={cur_keys.apga_hi:016x}"
                         f"  {'MATCH' if apga_match else '*** MISMATCH ***'}")
        except Exception as e:
            lines.append(f"  [warn] could not read PAC keys: {e}")

        lines.append(f"pre-HW auth verification (using stored executor.config.pac_keys):")
        for fp in fix_points:
            if fp.signed_value is None:
                lines.append(f"  slot={fp.slot_id:3d}  SKIP (no signed value)")
                continue
            try:
                auth_result = self.local_executor.pac_auth(
                    fp.signed_value,
                    fp.ctx_value if fp.ctx_value is not None else 0,
                    fp.info.auth_mnemonic,
                )
                # On ARM64, auth failure sets bit 62, making bits[63:55] a non-canonical
                # pattern. A canonical kernel address has bits[63:55] = all-1s (0x1ff)
                # and a canonical user address has bits[63:55] = all-0s (0x000).
                top9 = (auth_result >> 55) & 0x1ff
                failed = (top9 != 0x1ff) and (top9 != 0x000)
                status = f"FAIL top9=0x{top9:03x}" if failed else "OK"
                lines.append(
                    f"  slot={fp.slot_id:3d}  {fp.info.auth_mnemonic:8s}"
                    f"  signed=0x{fp.signed_value:016x}  ctx=0x{(fp.ctx_value or 0):016x}"
                    f"  auth_result=0x{auth_result:016x}  -> {status}"
                )
            except Exception as e:
                lines.append(f"  slot={fp.slot_id:3d}  [error calling pac_auth: {e}]")

        if stage1_bytes is not None:
            cs = Cs(CS_ARCH_ARM64, CS_MODE_ARM)
            cs.detail = False
            lines.append(f"stage-1 TC disassembly ({len(stage1_bytes)} bytes):")
            for insn in cs.disasm(stage1_bytes, 0):
                lines.append(f"  +{insn.address:04x}: {insn.mnemonic} {insn.op_str}")

        if all_variants is not None:
            cs = Cs(CS_ARCH_ARM64, CS_MODE_ARM)
            cs.detail = False
            for v_name, v_tc in [("TC1", all_variants.tc1),
                                  ("TC2", all_variants.tc2),
                                  ("TC3", all_variants.tc3)]:
                marker = " <-- ABOUT TO RUN" if v_name == tc_name else ""
                v_disasm = self._disasm_tc(v_tc)
                lines.append(f"{v_name} disassembly ({len(v_disasm)} instructions){marker}:")
                lines.extend(f"  {l}" for l in v_disasm)
        else:
            lines.append(f"{tc_name} disassembly ({len(disasm)} instructions):")
            lines.extend(f"  {l}" for l in disasm)

        text = "\n".join(lines) + "\n"

        # Emit to pac log (also flushed below).
        self._pac_log(text)

        # Flush pac log to disk before HW call.
        self._pac_log_file.flush()
        try:
            _os.fsync(self._pac_log_file.fileno())
        except Exception:
            pass

        # Write the crash-ready dump file with explicit sync so it survives a
        # kernel panic + hard reset.  Opened with os.O_SYNC so each write goes
        # straight to the block device without relying on the page cache.
        try:
            fd = _os.open(dump_path,
                          _os.O_WRONLY | _os.O_CREAT | _os.O_TRUNC | _os.O_SYNC,
                          0o644)
            _os.write(fd, text.encode())
            _os.fsync(fd)
            _os.close(fd)
        except Exception as e:
            self._pac_log(f"  [warn] could not write crash dump: {e}")

        # Flush the whole block layer — belt-and-suspenders.
        try:
            _os.sync()
        except Exception:
            pass

    def _hw_trace_single_input(self, tc: TestCase, inp: Input, n_reps: int = 1) -> HTrace:
        """Load tc + inp into the hardware executor, trace n_reps times, return HTrace."""
        self.local_executor.discard_all_inputs()
        iid = self.local_executor.allocate_iid()
        self.local_executor.checkout_region(InputRegion(iid))
        self.local_executor.write(ExecutorMemory(_input_bytes_with_pstate(inp)))
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
            _reconstruct_pstate(view)
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
            #print(f"[CE_VARIANT_COMPARE] CE crashed while running variants: {e}")
            return

        for i, variant_ce in enumerate(ce_variant_traces):
            tc1_ct = self._ctrace_from_ce(variant_ce.tc1)
            tc2_ct = self._ctrace_from_ce(variant_ce.tc2)
            tc3_ct = self._ctrace_from_ce(variant_ce.tc3)

            mismatch_1_2 = tc1_ct != tc2_ct
            mismatch_3_1 = tc3_ct != tc1_ct

            if mismatch_1_2:
                print(f"[CE_VARIANT_COMPARE] input {i}: TC1 != TC2  (UNEXPECTED — should be arch-equivalent)")
                print(f"  TC1 ctrace: {tc1_ct.raw}")
                print(f"  TC2 ctrace: {tc2_ct.raw}")
            if mismatch_3_1:
                print(f"[CE_VARIANT_COMPARE] input {i}: TC3 != TC1  (potential speculation side-effect)")
                print(f"  TC1 ctrace: {tc1_ct.raw}")
                print(f"  TC3 ctrace: {tc3_ct.raw}")
            #if not mismatch_1_2 and not mismatch_3_1:
            #    print(f"[CE_VARIANT_COMPARE] input {i}: TC1 == TC2 == TC3  {tc1_ct.raw}")

    def run_pac_hardware_comparison(
        self,
        inputs: List[Input],
        tc_variants_per_input: List[TCVariants],
        n_reps: int = 1,
    ) -> List[Tuple[HTrace, HTrace, HTrace]]:
        """
        Run each input through its own TC1/TC2/TC3 on real hardware.

        tc_variants_per_input[i] was generated from inputs[i]; this method pairs them
        correctly so every variant runs with the input whose signed values it embeds.

        Returns List[(tc1_ht, tc2_ht, tc3_ht)] — one tuple per input.

        Interpretation:
          TC1 == TC2:  correct — strip-only and full-restore give the same arch trace.
          TC1 != TC2:  instrumentation bug — the restore values are wrong.
          TC3 != TC1:  PAC non-interference candidate — speculative AUTH with wrong
                       pointer/context produced an observable side channel.
        """
        assert len(inputs) == len(tc_variants_per_input), (
            "run_pac_hardware_comparison: inputs and tc_variants_per_input must have the same length"
        )
        import os as _os
        dump_path = _os.path.join(_os.path.dirname(self._pac_log_file_path), "pac_last_hw_attempt.txt")
        results: List[Tuple[HTrace, HTrace, HTrace]] = []
        fix_points = self._stage1_fix_points or []
        for i, (inp, variants) in enumerate(zip(inputs, tc_variants_per_input)):
            for tc_name, tc in [("TC1", variants.tc1), ("TC2", variants.tc2), ("TC3", variants.tc3)]:
                self._dump_pre_hw(tc_name, tc, inp, fix_points, i, dump_path,
                                  all_variants=variants,
                                  stage1_bytes=self._stage1_tc_bytes)
                spec_summary = [(fp.slot_id, fp.spec_nesting) for fp in fix_points
                                if fp.spec_nesting is not None and fp.spec_nesting > 0]
                #print(f"[PAC HW] tc={self._tc_counter} {tc_name} input={i}"
                #      f"  spec_slots={spec_summary}  STARTING", flush=True)
                try:
                    ht = self._hw_trace_single_input(tc, inp, n_reps)
                #    print(f"[PAC HW] tc={self._tc_counter} {tc_name} input={i}  OK", flush=True)
                except Exception as e:
                    self._pac_log(
                        f"\n!!! HW CRASH: {tc_name} input={i} tc_counter={self._tc_counter}: {e}",
                        also_print=True)
                    self._pac_log_file.flush()
                    raise
                if tc_name == "TC1":
                    tc1_ht = ht
                elif tc_name == "TC2":
                    tc2_ht = ht
                else:
                    tc3_ht = ht
            results.append((tc1_ht, tc2_ht, tc3_ht))

            if tc1_ht != tc2_ht:
                self._pac_log(f"\n  [HW MISMATCH] TC {self._tc_counter} input {i}: TC1≠TC2 — logging variants",
                              also_print=True)
                cs = Cs(CS_ARCH_ARM64, CS_MODE_ARM)
                cs.detail = False
                for name, variant_tc in [("TC1", variants.tc1), ("TC2", variants.tc2)]:
                    layout = Aarch64ASMLayout(variant_tc)
                    asm = Aarch64Printer(self.target_desc).print_layout(layout)
                    variant_bytes = ConfigurableGenerator.in_memory_assemble(asm)
                    self._pac_log(f"    {name} disassembly:")
                    for insn in cs.disasm(variant_bytes, 0):
                        self._pac_log(f"      +{insn.address:04x}: {insn.mnemonic} {insn.op_str}")

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

        assert self._stage1_pac_instrumentation is not None, \
            "stage-1 cache is empty: load_test_case() was not called"

        pac = self._stage1_pac_instrumentation
        tc = self._stage1_tc
        fix_points = self._stage1_fix_points
        tc_bytes = self._stage1_tc_bytes
        pac_offset_to_fps = self._stage1_pac_offset_to_fps

        sandbox_base, _ = self.read_base_addresses()
        traces: List[ContractExecutionResult] = []
        tc_variants_per_input: List[TCVariants] = []
        last_cer: Optional[ContractExecutionResult] = None

        # Build the sandboxed TC once — CE on this produces branch offsets that match
        # what trace_test_case() loads into the kernel module, so mistraining is correct.
        sandboxed_patched = copy.deepcopy(self.test_case)
        pass_on_test_case(sandboxed_patched, [Aarch64SandboxPass()])
        sandboxed_layout = Aarch64ASMLayout(sandboxed_patched)
        sandboxed_assembly = Aarch64Printer(self.target_desc).print_layout(sandboxed_layout)
        sandboxed_tc_bytes = ConfigurableGenerator.in_memory_assemble(sandboxed_assembly)
        sandboxed_traces: List[ContractExecutionResult] = []

        for inp in inputs:
            # Reset per-input fix-point state (signed_value, ctx_value, spec_nesting).
            for fp in fix_points:
                fp.reset()

            data = inp.tobytes()
            tc_memory = data[:0x2000]
            tc_regs = bytearray(data[0x2000:])
            view = memoryview(tc_regs).cast('Q')
            # Reconstruct PSTATE from per-flag bytes (N→byte48, Z→49, C→50, V→51 → bits 31:28).
            _reconstruct_pstate(view)
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

            # Run CE on the sandboxed TC so that branch offsets in the returned trace match
            # what trace_test_case() loads into the kernel module for mistraining.
            sandboxed_execution = ContractExecution(
                sandboxed_tc_bytes, tc_memory, tc_regs,
                SimArch.RVZR_ARCH_AARCH64, 5, 10,
                req_mem_base_virt=sandbox_base,
            )
            try:
                sandboxed_cer = self._contract_executor.run(sandboxed_execution)
            except RuntimeError:
                sandboxed_cer = cer  # fall back to stage-1 trace if sandboxed CE crashes
            sandboxed_traces.append(sandboxed_cer)

            # Capture signed pointer and context values immediately after PACIA executes.
            if pac_offset_to_fps and len(cer) > 0:
                actual_code_base = cer[0].cpu.pc
                pac_pc_to_fps = {actual_code_base + off: fps
                                 for off, fps in pac_offset_to_fps.items()}
                # Slots that have received at least one arch (nesting==0) capture.
                # nesting==0 always overwrites (last arch firing = last signing in
                # program order = the correct value to restore).
                # nesting>0 is accepted only before any arch capture; once an arch
                # value is recorded, speculative re-executions cannot overwrite it.
                arch_seen_slots: Set[int] = set()
                for ite in cer:
                    fps = pac_pc_to_fps.get(ite.cpu.pc)
                    if fps is not None:
                        nesting = ite.metadata.speculation_nesting
                        for fp in fps:
                            if nesting != 0 and fp.slot_id in arch_seen_slots:
                                # Spec firing after arch: discard.
                                continue
                            reg = fp.info.reg
                            if reg.startswith('x') and reg[1:].isdigit():
                                fp.signed_value = ite.cpu.gpr[int(reg[1:])]
                            else:
                                raise RuntimeError(
                                    f"PAC capture: unrecognised signed reg format '{reg}' "
                                    f"for slot {fp.slot_id} — expected xN"
                                )
                            ctx = fp.info.ctx_reg
                            if ctx is None:
                                pass  # zero-context variant (PACIZA/PACIZB): no ctx to capture
                            elif ctx == 'sp':
                                fp.ctx_value = ite.cpu.sp
                            elif ctx.startswith('x') and ctx[1:].isdigit():
                                fp.ctx_value = ite.cpu.gpr[int(ctx[1:])]
                            else:
                                raise RuntimeError(
                                    f"PAC capture: unrecognised ctx_reg format '{ctx}' "
                                    f"for slot {fp.slot_id} — expected xN or sp"
                                )
                            fp.spec_nesting = nesting
                            if nesting == 0:
                                arch_seen_slots.add(fp.slot_id)

            # ── Log per-input fixpoint captures + detect capture bugs ────────
            inp_idx = inputs.index(inp)
            regs_u64 = list(memoryview(data[0x2000:]).cast('Q'))
            self._pac_log(f"\n  [TC {self._tc_counter} / input {inp_idx}]"
                          f"  x0-x5={[hex(v) for v in regs_u64[:6]]}")

            # Build a set of all slot-AUTH PCs that were hit at nesting==0 in this CE run.
            slot_auth_pc_to_fp: Dict[int, FixPoint] = {}
            if self._stage1_slot_auth_offset_to_fp and len(cer) > 0:
                base = cer[0].cpu.pc
                slot_auth_pc_to_fp = {base + off: fp
                                      for off, fp in self._stage1_slot_auth_offset_to_fp.items()}
            arch_reached_auth_pcs: Set[int] = set()
            for ite in cer:
                if ite.metadata.speculation_nesting == 0 and ite.cpu.pc in slot_auth_pc_to_fp:
                    arch_reached_auth_pcs.add(ite.cpu.pc)

            # ── CE execution profile ─────────────────────────────────────────
            # For every PAC sign, slot auth, and conditional branch, log which
            # nesting level(s) the CE executed it at.  This lets us verify that
            # spec slots really are spec-only, not arch or both.
            if len(cer) > 0:
                ce_base = cer[0].cpu.pc
                pc_nestings: Dict[int, Set[int]] = {}
                for _ite in cer:
                    pc_nestings.setdefault(_ite.cpu.pc, set()).add(
                        _ite.metadata.speculation_nesting)

                def _nesting_label(ns: Set[int]) -> str:
                    if not ns:
                        return "NEVER-EXECUTED"
                    if ns == {0}:
                        return "ARCH-ONLY"
                    if 0 in ns:
                        return f"ARCH+SPEC nestings={sorted(ns)}"
                    return f"SPEC-ONLY nestings={sorted(ns)}"

                self._pac_log(f"  CE execution profile (sign / slot-auth / cond-branch):")
                for _pc, _fps in sorted(pac_pc_to_fps.items()):
                    _ns = pc_nestings.get(_pc, set())
                    for _fp in _fps:
                        self._pac_log(
                            f"    sign @+0x{_pc - ce_base:04x}  slot={_fp.slot_id}"
                            f"  {_nesting_label(_ns)}")
                for _auth_pc, _fp in sorted(slot_auth_pc_to_fp.items()):
                    _ns = pc_nestings.get(_auth_pc, set())
                    self._pac_log(
                        f"    auth @+0x{_auth_pc - ce_base:04x}  slot={_fp.slot_id}"
                        f"  {_nesting_label(_ns)}")
                _cond_pcs = sorted({_ite.cpu.pc for _ite in cer
                                    if is_conditional_branch(_ite.cpu.encoding)})
                for _pc in _cond_pcs:
                    _ns = pc_nestings.get(_pc, set())
                    self._pac_log(
                        f"    branch@+0x{_pc - ce_base:04x}  {_nesting_label(_ns)}")

            for fp in fix_points:
                is_spec = fp.spec_nesting is not None and fp.spec_nesting > 0
                signed_str = hex(fp.signed_value) if fp.signed_value is not None else "MISSING"
                ctx_str    = hex(fp.ctx_value)    if fp.ctx_value    is not None else (
                    "N/A" if fp.info.ctx_reg is None else "MISSING")
                flag = ""
                if fp.signed_value is None:
                    # Find this fp's auth PC
                    auth_pc = next((pc for pc, f in slot_auth_pc_to_fp.items() if f is fp), None)
                    if auth_pc is not None and auth_pc in arch_reached_auth_pcs:
                        # Auth slot executed at nesting==0 but signing was never captured → BUG
                        flag = " *** BUG: AUTH slot reached arch (nesting=0) but signed_value not captured ***"
                    else:
                        flag = " [branch not taken in CE — poison loaded; slot should not fire in HW arch]"
                self._pac_log(f"    slot={fp.slot_id:3d}  spec={is_spec}  nesting={fp.spec_nesting}  "
                              f"signed={signed_str:22s}  ctx={ctx_str}{flag}")

            # Raise if any auth slot fired architecturally without a captured signed_value.
            bugs = [fp for fp in fix_points
                    if fp.signed_value is None
                    and any(slot_auth_pc_to_fp.get(pc) is fp
                            for pc in arch_reached_auth_pcs)]
            if bugs:
                bug_details = "; ".join(
                    f"slot={fp.slot_id} reg={fp.info.reg} auth={fp.info.auth_mnemonic}"
                    for fp in bugs)
                msg = (f"[CAPTURE BUG] TC {self._tc_counter} input {inp_idx}: "
                       f"AUTH slot(s) executed at arch nesting=0 but signed_value was never captured. "
                       f"Slots: {bug_details}")
                self._pac_log(msg, also_print=True)
                raise RuntimeError(msg)

            # Stage 2: generate TC variants (TC1/TC2/TC3) from the signed values captured above.
            tc1, tc2, tc3 = pac.instrument_stage2(tc, fix_points)
            tc_variants_per_input.append(TCVariants(tc1=tc1, tc2=tc2, tc3=tc3))

            # Log variant disassemblies + fix-point values for this input.
            inp_idx = len(tc_variants_per_input) - 1
            self._pac_log(f"\n--- Stage-2 variants for input {inp_idx} ---")
            self._pac_log(f"  fix-point signed values:")
            for fp in fix_points:
                sv = hex(fp.signed_value) if fp.signed_value is not None else "None"
                cv = hex(fp.ctx_value) if fp.ctx_value is not None else "None"
                self._pac_log(f"    slot={fp.slot_id:3d}  reg={fp.info.reg:4s}"
                              f"  signed={sv}  ctx={cv}  spec_nesting={fp.spec_nesting}")
            for tc_name, variant in [("TC1", tc1), ("TC2", tc2), ("TC3", tc3)]:
                disasm = self._disasm_tc(variant)
                self._pac_log(f"  {tc_name} ({len(disasm)} insts):")
                for l in disasm:
                    self._pac_log(f"    {l}")
            self._pac_log_file.flush()

        # Use sandboxed-TC traces for ctrace/taint/mistraining — their branch offsets
        # match the sandboxed TC that trace_test_case() loads into the kernel module.
        taints = [compute_taint(cer) for cer in sandboxed_traces]
        ctraces = [compute_ctrace(cer) for cer in sandboxed_traces]

        return ctraces, taints, sandboxed_traces, tc_variants_per_input


# ==================================================================================================
# MTE non-interference executor
# ==================================================================================================
class Aarch64MteNonInterferenceExecutor(Aarch64LocalExecutor):
    """
    Executor for MTE non-interference testing.

    Lifecycle:
    1. load_test_case(tc)  — runs MTEInstrumentation.instrument_stage1(), caches result.
    2. trace_test_case_with_taints(inputs, nesting)  — for each input:
         a. Run stage-1 TC through CE (CONTRACT_ALWAYS_MISPREDICT) to discover which
            memory-access slots have spec_nesting > 0 (speculative) vs 0 (architectural).
            Uses arch_seen semantics: once a slot fires at nesting==0 it is permanently arch.
         b. Call MTEInstrumentation.instrument_stage2() to produce TC1/TC2/TC3:
              TC1 — all NOP (correct-tag baseline)
              TC2 — spec slots → IRG Xd,Xd (random tag); arch slots → NOP
              TC3 — spec slots → MOVK Xd,#wrong_upper16,LSL#48; arch slots → NOP
    """

    def __init__(self, workdir: str, generator: Aarch64Generator, *args, **kwargs):
        super().__init__(workdir, *args, **kwargs)
        self._generator: Aarch64Generator = generator
        self._mte = MTEInstrumentation(generator)

        # Stage-1 cache — populated by load_test_case()
        self._stage1_tc: Optional[TestCase] = None
        self._stage1_fix_points: Optional[List[MTEFixPoint]] = None
        self._stage1_tc_bytes: Optional[bytes] = None
        # Maps byte offset of each memory access in the stage-1 binary → MTEFixPoint
        self._stage1_mem_access_offset_to_fp: Optional[Dict[int, MTEFixPoint]] = None
        self._tc_counter: int = 0

    def load_test_case(self, test_case: TestCase):
        result = super().load_test_case(test_case)
        self._run_stage1()
        return result

    def _run_stage1(self) -> None:
        """Instrument test_case with NOP placeholders (stage-1) and cache layout."""
        self._tc_counter += 1

        patched = copy.deepcopy(self.test_case)
        stage1_tc, fix_points = self._mte.instrument_stage1(patched)
        layout = Aarch64ASMLayout(stage1_tc)

        assembly = Aarch64Printer(self.target_desc).print_layout(layout)
        tc_bytes = ConfigurableGenerator.in_memory_assemble(assembly)

        # The NOP placeholder is inserted immediately before its memory access.
        # So: mem_access is at nop_offset + 4.
        mem_access_offset_to_fp: Dict[int, MTEFixPoint] = {}
        for fp in fix_points:
            nop_offset = layout.instruction_address[fp.slot_insts[0]]
            mem_access_offset_to_fp[nop_offset + 4] = fp

        self._stage1_tc = stage1_tc
        self._stage1_fix_points = fix_points
        self._stage1_tc_bytes = tc_bytes
        self._stage1_mem_access_offset_to_fp = mem_access_offset_to_fp

    def trace_test_case_with_taints(
        self,
        inputs: List[Input],
        nesting: int,
    ) -> Tuple[List[CTrace], List[InputTaint], List[ContractExecutionResult], List[TCVariants]]:
        """
        For each input, run CE to capture per-slot spec_nesting, then produce TC variants.
        """
        if not inputs or self.test_case is None:
            return [], [], [], []

        assert self._stage1_tc is not None, \
            "MTE stage-1 cache empty — load_test_case() not called"

        tc = self._stage1_tc
        fix_points = self._stage1_fix_points
        tc_bytes = self._stage1_tc_bytes
        offset_to_fp = self._stage1_mem_access_offset_to_fp

        sandbox_base, _ = self.read_base_addresses()
        traces: List[ContractExecutionResult] = []
        tc_variants_per_input: List[TCVariants] = []

        for inp in inputs:
            for fp in fix_points:
                fp.reset()

            data = inp.tobytes()
            tc_memory = data[:0x2000]
            tc_regs = bytearray(data[0x2000:])
            _reconstruct_pstate(memoryview(tc_regs).cast('Q'))
            tc_regs = bytes(tc_regs)

            execution = ContractExecution(
                tc_bytes, tc_memory, tc_regs,
                SimArch.RVZR_ARCH_AARCH64,
                nesting,   # max_misspred_branch_nesting
                10,        # max_misspred_instructions (unused by CE)
                req_mem_base_virt=sandbox_base,
            )
            # contract_type defaults to ALWAYS_MISPREDICT — reveals speculative paths
            cer = self._contract_executor.run(execution)
            traces.append(cer)

            # Capture spec_nesting per slot with arch_seen semantics:
            # nesting==0 wins and cannot be overwritten by later speculative firings.
            if len(cer) > 0:
                code_base = cer[0].cpu.pc
                arch_seen: Set[int] = set()
                for ite in cer:
                    if not ite.metadata.has_memory_access:
                        continue
                    pc_offset = ite.cpu.pc - code_base
                    fp = offset_to_fp.get(pc_offset)
                    if fp is None:
                        continue
                    nest = ite.metadata.speculation_nesting
                    if nest == 0:
                        fp.spec_nesting = 0
                        arch_seen.add(fp.slot_id)
                    elif fp.slot_id not in arch_seen and fp.spec_nesting is None:
                        fp.spec_nesting = int(nest)

            tc1, tc2, tc3 = self._mte.instrument_stage2(tc, fix_points, sandbox_base)
            tc_variants_per_input.append(TCVariants(tc1=tc1, tc2=tc2, tc3=tc3))

        taints = [compute_taint(cer) for cer in traces]
        ctraces = [compute_ctrace(cer) for cer in traces]

        return ctraces, taints, traces, tc_variants_per_input


def _reconstruct_pstate(view: memoryview) -> None:
    """Convert per-flag NZCV encoding in slot 6 to ARM PSTATE format via NZCVScheme."""
    view[6] = NZCVScheme.to_pstate(int(view[6]))


def _input_bytes_with_pstate(inp) -> bytes:
    """Return inp.tobytes() with slot-6 converted from per-flag to PSTATE format."""
    data = bytearray(inp.tobytes())
    _reconstruct_pstate(memoryview(data[0x2000:]).cast('Q'))
    return bytes(data)


def map_operand_to_input_offsets(op: str) -> List[int]:
    """Input byte offsets read by op (source/read side; w-regs cover only lower 4 bytes)."""
    op_l = op.lower()
    if op_l in ("fp", "lr", "xzr", "wzr"):
        return []
    if op_l.startswith(("x", "w")):
        try:
            n = int(op_l[1:])
        except ValueError:
            return []
        if n < 0 or n > 5:
            return []
        size = 8 if op_l.startswith("x") else 4
    elif op_l in NZCVScheme.flag_names():
        return [NZCVScheme.input_byte(op_l)]
    elif op_l == "sp":
        n, size = 7, 8
    else:
        return []
    return list(range(n * 8, n * 8 + size))


def map_operand_to_dest_offsets(op: str) -> List[int]:
    """Input byte offsets overwritten by op (destination/write side; w-reg writes zero-extend to all 8 bytes)."""
    op_l = op.lower()
    if op_l in ("fp", "lr", "xzr", "wzr"):
        return []
    if op_l.startswith(("x", "w")):
        try:
            n = int(op_l[1:])
        except ValueError:
            return []
        if n < 0 or n > 5:
            return []
    elif op_l in NZCVScheme.flag_names():
        return [NZCVScheme.input_byte(op_l)]
    elif op_l == "sp":
        n = 7
    else:
        return []
    return list(range(n * 8, n * 8 + 8))


class TaintTracker:
    """
    Tracks must-preserve input byte offsets across a DFS-ordered CE execution trace.

    A depth-indexed write stack mirrors speculation nesting: writes at depth D are
    visible to reads at depth <= D and are discarded when nesting decreases (squashed).
    Depth-0 writes are architectural and persist for the whole trace.

    Extensible: instantiate one TaintTracker per input region (GPR, memory, SIMD, …).
    """

    def __init__(self) -> None:
        self._written: List[Set[int]] = [set()]
        self.must_preserve: Set[int] = set()

    def set_depth(self, depth: int) -> None:
        while len(self._written) <= depth:
            self._written.append(set())
        while len(self._written) > depth + 1:
            self._written.pop()

    def on_read(self, offsets: Iterable[int], depth: int) -> None:
        written = set().union(*self._written[:depth + 1])
        self.must_preserve.update(o for o in offsets if o not in written)

    def on_write(self, offsets: Iterable[int], depth: int) -> None:
        self._written[depth].update(offsets)


def compute_taint(cer: ContractExecutionResult) -> InputTaint:
    """
    Derive input taint from a single CE execution result.

    A byte is marked must-preserve iff any execution path (arch or speculative)
    reads it before writing to it.
    """
    input_taint = InputTaint()
    sandbox_u8 = input_taint.view(np.uint8)
    mem_size = (input_taint[0]["main"].view(np.uint8).size
                + input_taint[0]["faulty"].view(np.uint8).size)
    gpr_u8 = input_taint[0]["gpr"].view(np.uint8)

    gpr_tracker = TaintTracker()
    mem_tracker = TaintTracker()

    for ite in cer:
        depth = ite.metadata.speculation_nesting
        gpr_tracker.set_depth(depth)
        mem_tracker.set_depth(depth)

        if ite.metadata.has_memory_access:
            ma = ite.metadata.memory_access
            for byte_idx in range(ma.element_size):
                byte_offset = ma.effective_address + byte_idx - ite.cpu.gpr[29]
                if 0 <= byte_offset < mem_size:
                    if ma.is_write:
                        mem_tracker.on_write((byte_offset,), depth)
                    else:
                        mem_tracker.on_read((byte_offset,), depth)

        srcs, dests = get_srcs_dests_operands(ite.cpu.encoding, ite.cpu.pc)
        gpr_tracker.on_read(chain.from_iterable(map(map_operand_to_input_offsets, srcs)), depth)
        gpr_tracker.on_write(chain.from_iterable(map(map_operand_to_dest_offsets, dests)), depth)

    for offset in gpr_tracker.must_preserve:
        gpr_u8[offset] = True
    for offset in mem_tracker.must_preserve:
        sandbox_u8[offset] = True

    return input_taint


def compute_ctrace(cer: ContractExecutionResult) -> CTrace:
    """Derive CTrace (cache-set footprint) from a single CE execution result."""
    line_size, num_sets = 64, 64
    cache_sets: Set[int] = set()
    for ite in cer:
        if ite.metadata.has_memory_access:
            ma = ite.metadata.memory_access
            for byte_idx in range(ma.element_size):
                addr = ma.effective_address + byte_idx - ite.cpu.gpr[29]
                cache_sets.add((addr // line_size) % num_sets)
    return CTrace(raw_trace=sorted(cache_sets))

def get_srcs_dests_operands(encoding: int, pc: int) -> Union[List, List]:
    FLAG_BITS = {"N", "Z", "C", "V"}

    def cc_to_read_flags(cc: int):
        cond_map = {
            ARM64_CC_EQ: {"Z"},
            ARM64_CC_NE: {"Z"},
            ARM64_CC_HS: {"C"},
            ARM64_CC_LO: {"C"},
            ARM64_CC_MI: {"N"},
            ARM64_CC_PL: {"N"},
            ARM64_CC_VS: {"V"},
            ARM64_CC_VC: {"V"},
            ARM64_CC_HI: {"C", "Z"},
            ARM64_CC_LS: {"C", "Z"},
            ARM64_CC_GE: {"N", "V"},
            ARM64_CC_LT: {"N", "V"},
            ARM64_CC_GT: {"Z", "N", "V"},
            ARM64_CC_LE: {"Z", "N", "V"},
            ARM64_CC_AL: set(),
            ARM64_CC_NV: set(),
        }
        return cond_map.get(cc, {"N", "Z", "C", "V"})

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
        # ARM64_CC_INVALID means unconditional; only add flag reads for real conditions.
        if insn.cc is not None and insn.cc != ARM64_CC_INVALID:
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




def is_conditional_branch(encoding: int) -> bool:
    """Return True if encoding is a conditional branch: B.cond, CBZ/CBNZ (32/64), TBZ/TBNZ."""
    op = (encoding >> 24) & 0xFF
    return op in (0x54,                          # B.cond
                  0x34, 0x35, 0xB4, 0xB5,        # CBZ/CBNZ w/x
                  0x36, 0x37, 0xB6, 0xB7)        # TBZ/TBNZ w/x


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


