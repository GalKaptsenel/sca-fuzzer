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

from .aarch64_generator import Aarch64TagMemoryAccesses, Aarch64Printer, Aarch64MarkMemoryAccessesNEON, Aarch64SandboxPass, Aarch64SpecContractPass, Aarch64MarkRegisterTaints, Aarch64MarkMemoryTaints, Aarch64FullTrace, FullTraceAuxBuffer, BitmapTaintsAuxBuffer
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

from .aarch64_contract_executor import *
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


def pass_on_test_case(test_case: TestCase, passes: List[Pass]):
	for p in passes:
		p.run_on_test_case(test_case)


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
        assembly = Aarch64Printer(self.target_desc).print(patched)
        tc_bytes = ConfigurableGenerator.in_memory_assemble(assembly)

        patched.asm_path, patched.obj_path, patched.bin_path = local_name, local_name, local_name 

        self.local_executor.checkout_region(TestCaseRegion())
        self.local_executor.write(tc_bytes)

        return patched

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

    def trace_test_case_with_taints(self, inputs: List[Input], nesting: int) -> Tuple[List[CTrace], List[InputTaint], List[ContractExecutionResult]]:
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
        patched = copy.deepcopy(self.test_case)
        pass_on_test_case(patched, [Aarch64SandboxPass()])

        assembly = Aarch64Printer(self.target_desc).print(patched)
        tc_bytes = ConfigurableGenerator.in_memory_assemble(assembly)
        sandbox_base, code_base = self.read_base_addresses()

        traces: List[ContractExecutionResult] = []
        for i in inputs:
            data = i.tobytes()
            tc_memory = data[:0x2000]

            tc_regs = bytearray(data[0x2000:])
            view = memoryview(tc_regs).cast('Q')
            view[6] = (view[6] << 28) & ((1 << 64) - 1) # in the executor we shift left 28 bits the value for the flags register
            tc_regs = bytes(tc_regs)

            execution = ContractExecution(tc_bytes, tc_memory, tc_regs, SimArch.RVZR_ARCH_AARCH64, 5, 10, req_code_base_virt=code_base,
                                      req_mem_base_virt=sandbox_base)
            traces.append(self._contract_executor.run(execution))

        taints, ctraces, = [], []

        for cer in traces:
            if 0 == len(cer):
                continue

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
                            if 0 <= byte_offset < actual_sandbox_memory_size: # sometimes overflow may occur, we won't taint the overflows
                                sandbox_u8[byte_offset] = True

                srcs, _ = get_srcs_dests_operands(ite.cpu.encoding, ite.cpu.pc)
                for offset in chain.from_iterable(map(map_operand_to_input_offsets, srcs)):
                    gpr_region_u8[offset] = True
            taints.append(input_taint)

            line_size = 64
            num_sets = 64
            cache_sets = sorted(set(map(lambda addr: (addr // line_size) % num_sets, accessed_memory)))
            ctraces.append(CTrace(raw_trace=cache_sets))

        return ctraces, taints, traces

def map_operand_to_input_offsets(op: str) -> List[int]:
    if op.lower().startswith(("x", "w")):
        n = int(op[1:])
        if n < 0 or n > 5:
            raise ValueError(f"Register index must be 0–7, got {op}")
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
        raise RuntimeError(f"Unexpected src register '{src}'")

    return list(map(lambda byte_index: n * 8 + byte_index, range(size)))

def get_srcs_dests_operands(encoding: int, pc: int) -> Union[List, List]:
    FLAG_BITS = {"N", "Z", "C", "V"}

    def cc_to_read_flags(cc: int):
        cond_map = {
                0: set(),
                1: {"Z"},
                2: {"Z"},
                3: {"C"},
                4: {"C"},
                5: {"N"},
                6: {"N"},
                7: {"V"},
                8: {"V"},
                9: {"C", "Z"},
                10: {"C", "Z"},
                11: {"N", "V"},
                12: {"N", "V"},
                13: {"Z", "N", "V"},
                14: {"Z", "N", "V"},
                15: set(),
                16: set(),
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
        import pdb; pdb.set_trace()
        return f"<decode error: {e}>"




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
    min_len = min(len(trace_ref), len(trace_new))
    for i in range(min_len):
        ref = trace_ref[i]
        new = trace_new[i]
#        if (ref.cpu.pc != new.cpu.pc or ref.cpu.nzcv != new.cpu.nzcv or ref.cpu.gprs != new.cpu.gprs):
        if (ref.cpu.nzcv != new.cpu.nzcv or \
                ref.cpu.sp != new.cpu.sp or \
                ref.cpu.gpr[0] != new.cpu.gpr[0] or \
                ref.cpu.gpr[1] != new.cpu.gpr[1] or \
                ref.cpu.gpr[2] != new.cpu.gpr[2] or \
                ref.cpu.gpr[3] != new.cpu.gpr[3] or \
                ref.cpu.gpr[4] != new.cpu.gpr[4] or \
                ref.cpu.gpr[5] != new.cpu.gpr[5]):
            return i, ref, new
    if len(trace_ref) != len(trace_new):
        return min_len, None, None  # diverged by length
    return None, None, None


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
    input_ref.save(f"{prefix}_input_ref.bin")
    input_new.save(f"{prefix}_input_new.bin")

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
        ref_disas = disassemble_instruction(ref_inst.cpu.encoding, ref_inst.cpu.pc)
        new_disas = disassemble_instruction(new_inst.cpu.encoding, new_inst.cpu.pc)
        print(f"  Ref PC:  0x{ref_inst.cpu.pc:X}   ({ref_disas} [Encoding: 0x{ref_inst.cpu.encoding:08X}])")
        print(f"  New PC:  0x{new_inst.cpu.pc:X}   ({new_disas} [Encoding: 0x{new_inst.cpu.encoding:08X}])")
    else:
        print("  Diverged by trace length")

    print("\n--- Registers ---")
    if ref_inst and new_inst:
        same_regs = []
        diff_regs = []
        for i, (r_ref, r_new) in enumerate(zip(ref_inst.cpu.gpr, new_inst.cpu.gpr)):
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
    if ref_inst and new_inst and ref_inst.cpu.nzcv != new_inst.cpu.nzcv:
        print(f"  Flags changed: {ref_inst.cpu.nzcv:#x} -> {new_inst.cpu.nzcv:#x}")
    elif ref_inst and new_inst:
        print(f"  Flags identical: {ref_inst.cpu.nzcv:#x}")

    print("\n--- Memory ---")
    if ref_inst and new_inst:
        if ref_inst.metadata.memory_access.effective_address != new_inst.metadata.memory_access.effective_address:
            print(f"  EA: {ref_inst.metadata.memory_access.effective_address:#x} -> {new_inst.metadata.memory_access.effective_address:#x}")
        if ref_inst.metadata.memory_access.before != new_inst.metadata.memory_access.before:
            print(f"  Mem before: {ref_inst.metadata.memory_access.before:#x} -> {new_inst.metadata.memory_access.before:#x}")
        if ref_inst.metadata.memory_access.after != new_inst.metadata.memory_access.after:
            print(f"  Mem after:  {ref_inst.metadata.memory_access.after:#x} -> {new_inst.metadata.memory_access.after:#x}")
        if (
            ref_inst.metadata.memory_access.effective_address == new_inst.metadata.memory_access.effective_address
            and ref_inst.metadata.memory_access.before == new_inst.metadata.memory_access.before
            and ref_inst.metadata.memory_access.after == new_inst.metadata.memory_access.after
        ):
            print("  Memory identical.")

    print("\n--- Reference trace context ---")
    show_context(trace_ref, idx)
    print("\n--- New trace context ---")
    show_context(trace_new, idx)

#    print("\n--- Taint Bitmaps ---")
#    print_taint("Reference Taint", taint_ref)
#    print_taint("New Taint", taint_new)

#    print("\n--- Inputs ---")
#    print("  Reference input:")
#    print(input_ref)
#    print("  New input:")
#    print(input_new)

    print("=" * 100)

#    if dump_files:
#        dump_debug_to_file(prefix, input_ref, input_new, trace_ref, trace_new, taint_ref, taint_new)
#        print(f"[+] Debug data dumped to: {prefix}_*.json/txt")

