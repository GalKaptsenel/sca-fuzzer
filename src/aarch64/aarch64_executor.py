"""
File: Implementation of executor for x86 architecture
  - Interfacing with the kernel module
  - Aggregation of the results

Copyright (C) Microsoft Corporation
SPDX-License-Identifier: MIT
"""
from __future__ import annotations
import copy
import random
import subprocess
import os.path
import os
import json
import warnings
import base64
import shlex
import datetime


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
from .aarch64_generator import Pass, Aarch64SandboxPass, PACInstrumentation, FixPoint, AUTH_SLOT_POS, SLOT_SIG_POS, MTEInstrumentation, MTEFixPoint, PACVariant, MTEVariant, _AUTH_TO_PAC

from .aarch64_connection import profile_op, ExecutorMemory

from .aarch64_contract_executor import *

# ANSI color codes for terminal-viewable logs (files get codes too; use `less -R` or `cat`)
_C_RESET  = "\033[0m"
_C_ARCH   = "\033[32m"    # green  — arch execution (nesting 0)
_C_SPEC1  = "\033[33m"    # yellow — speculative (nesting 1)
_C_SPEC2  = "\033[31m"    # red    — deeply speculative (nesting > 1)
_C_TAKEN  = "\033[36m"    # cyan   — branch taken
_C_NTAKEN = "\033[35m"    # magenta — branch not taken
_C_BOLD   = "\033[1m"
_C_DIM    = "\033[2m"

# ==================================================================================================
# Fuzzer logger — one append-only log file per process, written to logs/session_TIMESTAMP.log.
# All writes are line-buffered (flushed immediately) so the log survives a machine crash.
# ==================================================================================================
class _FuzzLogger:
    _instance: Optional['_FuzzLogger'] = None
    # Verbosity levels:
    #   0 — off (no logging)
    #   1 — normal: structured per-component log files under logs/SESSION_DIR/
    #   2 — verbose: level 1 + CE trace comparison table written to comparison.log
    _VERBOSITY: int = 0

    @classmethod
    def get(cls) -> '_FuzzLogger':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        import string
        self._null    = open(os.devnull, "w")
        self._channels: Dict[str, Any] = {}   # name → file handle
        self._active  = "session"
        self._session_dir: Optional[Path] = None

        if self._VERBOSITY == 0:
            return

        ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        rand_id = ''.join(random.choices(string.hexdigits[:16], k=6)).lower()
        self._session_dir = Path("logs") / f"{ts}_{rand_id}"
        self._session_dir.mkdir(parents=True, exist_ok=True)

        # Only session.log by default — everything else is registered on demand.
        self.register("session", "session.log", min_verbosity=1)
        self.w(f"=== PAC fuzzer session {ts}_{rand_id} ===", ch="session")
        self.w(f"log directory: {self._session_dir}\n",      ch="session")

    # ------------------------------------------------------------------
    # Channel registration
    # ------------------------------------------------------------------

    def register(self, name: str, rel_path: str, min_verbosity: int = 1) -> Optional[str]:
        """Open and register a named log channel inside the session directory.

        Returns a channel descriptor (cd) — the channel name — which can be
        passed directly to w(), header(), use(), and broadcast().

        The channel file is only created if _VERBOSITY >= min_verbosity;
        writes to an unregistered channel are silently dropped.
        Safe to call multiple times with the same name (returns the name
        immediately without re-opening).  rel_path may contain subdirectories
        — they are created automatically.
        """
        if name in self._channels:
            return None  # already registered
        if self._session_dir is not None and self._VERBOSITY >= min_verbosity:
            full = self._session_dir / rel_path
            full.parent.mkdir(parents=True, exist_ok=True)
            self._channels[name] = open(full, "w", buffering=1)
        return name

    def use(self, channel: str) -> '_FuzzLogger':
        """Switch the active channel; returns self for chaining."""
        self._active = channel
        return self

    # ------------------------------------------------------------------
    # Write primitives
    # ------------------------------------------------------------------

    def _fh(self, ch: Optional[str] = None) -> Any:
        name = ch if ch is not None else self._active
        return self._channels.get(name, self._null)

    def w(self, msg: str, ch: Optional[str] = None) -> None:
        """Write to the active channel (or ch if specified)."""
        self._fh(ch).write(msg + "\n")

    def wp(self, msg: str) -> None:
        """Write to the session channel (persistent high-level events)."""
        self.w(msg, ch="session")

    def broadcast(self, msg: str, channels: List[str]) -> None:
        """Write the same line to every listed channel."""
        for ch in channels:
            self.w(msg, ch=ch)

    def header(self, title: str, ch: Optional[str] = None) -> None:
        sep = "=" * 72
        self.w(f"\n{sep}\n  {title}\n{sep}", ch=ch)

    def ensure_flushed(self) -> None:
        """fsync all open channels so data survives a machine crash."""
        for fh in self._channels.values():
            try:
                fh.flush()
                os.fsync(fh.fileno())
            except OSError:
                pass


def _fmt_ce_entry(ite, code_base: int) -> str:
    disas = disassemble_instruction(ite.cpu.encoding, ite.cpu.pc) or "<unk>"
    offset = ite.cpu.pc - code_base
    nest = ite.metadata.speculation_nesting
    srcs, _ = get_srcs_dests_operands(ite.cpu.encoding, ite.cpu.pc)
    reg_parts = []
    for r in srcs[:4]:
        rl = r.lower()
        if rl == "sp":
            reg_parts.append(f"sp={ite.cpu.sp:016x}")
        elif rl.startswith("x") and rl[1:].isdigit():
            n = int(rl[1:])
            if 0 <= n <= 30:
                reg_parts.append(f"x{n}={ite.cpu.gpr[n]:016x}")
    ea = (f"  EA=0x{ite.metadata.memory_access.effective_address:016x}"
          if ite.metadata.has_memory_access else "")
    return f"[{nest}]+{offset:04x}  {disas:<28}  {'  '.join(reg_parts)}{ea}"


def log_start_test_case(log: _FuzzLogger, tc_counter: int) -> None:
    sep = "#" * 72
    ts  = datetime.datetime.now().strftime("%H:%M:%S.%f")
    for line in (f"\n{sep}", f"  TEST CASE #{tc_counter}   {ts}", f"{sep}\n"):
        log.w(line, ch="session")
    log.use("session")


def log_input(log: _FuzzLogger, inp_idx: int, inp, ch: Optional[str] = None) -> None:
    log.header(f"INPUT  inp={inp_idx}", ch=ch)
    data     = inp.tobytes()
    reg_blob = data[0x2000:]
    log.w("  Registers (raw input slots 0-7 = x0..x7, slot 6 = NZCV flags):", ch=ch)
    for slot in range(min(8, len(reg_blob) // 8)):
        val  = int.from_bytes(reg_blob[slot * 8: slot * 8 + 8], "little")
        name = ([f"x{i}" for i in range(6)] + ["?", "?"])[slot]
        log.w(f"    slot {slot} ({name}): 0x{val:016x}", ch=ch)
    mem = data[:0x2000]
    log.w("  Memory (first 128 bytes):", ch=ch)
    for row_off in range(0, min(128, len(mem)), 16):
        chunk    = mem[row_off: row_off + 16]
        hex_part = " ".join(f"{b:02x}" for b in chunk)
        log.w(f"    {row_off:04x}:  {hex_part}", ch=ch)
    log.w("", ch=ch)


def log_ce_trace(log: _FuzzLogger, label: str, inp_idx: int, cer_list: List,
                 ch: Optional[str] = None) -> None:
    log.header(f"CE TRACE: {label}  inp={inp_idx}  rows={len(cer_list)}", ch=ch)
    if not cer_list:
        log.w("  <empty>", ch=ch)
        return
    code_base = cer_list[0].cpu.pc
    log.w(f"  code_base=0x{code_base:016x}", ch=ch)
    log.w(f"  {'Tag':<6}  {'Row':>4}  {'Offset':>6}  {'Instruction':<28}  Regs / EA", ch=ch)
    log.w(f"  {'':-<6}  {'':-<4}  {'':-<6}  {'':-<28}  ---------", ch=ch)

    for row, ite in enumerate(cer_list):
        nest = ite.metadata.speculation_nesting
        if nest == 0:
            color, tag = _C_ARCH, "[ARCH]"
        elif nest == 1:
            color, tag = _C_SPEC1, "[SPEC]"
        else:
            color, tag = _C_SPEC2, f"[S{nest}] "

        branch_note = ""
        if row + 1 < len(cer_list) and is_conditional_branch(ite.cpu.encoding):
            nxt = cer_list[row + 1].cpu.pc
            taken = (nxt != ite.cpu.pc + 4)
            if taken:
                branch_note = f"  {_C_TAKEN}→ TAKEN{_C_RESET}"
            else:
                branch_note = f"  {_C_NTAKEN}→ NOT-TAKEN{_C_RESET}"

        entry = _fmt_ce_entry(ite, code_base)
        log.w(f"  {color}{tag}{_C_RESET}  {row:>4}  {color}{entry}{_C_RESET}{branch_note}", ch=ch)
    log.w("", ch=ch)


def log_bb_map(log: _FuzzLogger, variant_label: str, inp_idx: int,
               sorted_pcs: List[int], bb_info: Dict[int, Dict],
               pc_to_id: Dict[int, int], code_base: int, entry_x7: int,
               ch: Optional[str] = None) -> None:
    log.header(f"BB MAP: {variant_label}  inp={inp_idx}", ch=ch)
    log.w(f"  x7 before run = 0x{entry_x7:016x}  "
          f"(if panic shows this value, crash is in BB 1 — the entry block)", ch=ch)
    log.w(f"  {'ID':>3}  {'Offset':>7}  {'Status':<6}  Instruction", ch=ch)
    log.w(f"  {'':-<3}  {'':-<7}  {'':-<6}  -----------", ch=ch)
    for pc in sorted_pcs:
        info   = bb_info[pc]
        bb_id  = pc_to_id[pc]
        a, s   = info["arch"], info["spec"]
        status = "both" if a and s else ("arch" if a else "spec")
        note   = "  [entry — not patched]" if pc == code_base else ""
        log.w(f"  {bb_id:>3}  +{info['offset']:04x}    {status:<6}  {info['disas']}{note}", ch=ch)
    log.w("", ch=ch)


def log_tc_binary(log: _FuzzLogger, label: str, tc_bytes: bytes,
                  ch: Optional[str] = None) -> None:
    log.header(f"TC BINARY: {label}  ({len(tc_bytes)} bytes)", ch=ch)
    for row_off in range(0, len(tc_bytes), 16):
        chunk    = tc_bytes[row_off: row_off + 16]
        hex_part = " ".join(f"{b:02x}" for b in chunk)
        log.w(f"  {row_off:04x}:  {hex_part}", ch=ch)
    log.w("\n  Disassembly:", ch=ch)
    try:
        from capstone import Cs, CS_ARCH_ARM64, CS_MODE_ARM
        cs = Cs(CS_ARCH_ARM64, CS_MODE_ARM)
        for insn in cs.disasm(tc_bytes, 0):
            log.w(f"    +{insn.address:04x}:  {insn.mnemonic} {insn.op_str}", ch=ch)
    except Exception as exc:
        log.w(f"  [disasm error: {exc}]", ch=ch)
    log.w("", ch=ch)


def log_pac_op(log: _FuzzLogger, op: str, mnemonic: str,
               ptr: int, ctx: int, result: int) -> None:
    log.w(f"  PAC {op:<6}  {mnemonic:<10}  ptr=0x{ptr:016x}  "
          f"ctx=0x{ctx:016x}  =>  0x{result:016x}", ch="pac_signing")


def log_slot(log: _FuzzLogger, inp_idx: int, fp) -> None:
    cs  = f"0x{fp.correct_sig:04x}" if fp.correct_sig is not None else "None"
    alt = f"0x{fp.alt_sig:04x}"     if fp.alt_sig     is not None else "None"
    log.w(f"  slot={fp.slot_id:2d}  spec_nesting={str(fp.spec_nesting):<6}  "
          f"correct_sig={cs}  alt_sig={alt}", ch="pac_signing")


def log_mistraining(log: _FuzzLogger, tc_counter: int, inp_idx: int,
                    entries: List[Tuple[int, bool]], cer,
                    ch: Optional[str] = None) -> None:
    """Log branch mistraining config with arch-flow context."""
    log.header(f"MISTRAINING  tc={tc_counter}  inp={inp_idx}  n_branches={len(entries)}", ch=ch)
    if not entries:
        log.w("  <no trainable branches>", ch=ch)
        return

    # Build offset → (actual_taken, disassembly) from the arch path of the CE trace
    arch_map: Dict[int, Tuple[bool, str]] = {}
    if cer and len(cer) > 0:
        code_base = cer[0].cpu.pc
        for i, ite in enumerate(cer):
            if ite.metadata.speculation_nesting != 0:
                continue
            if not is_conditional_branch(ite.cpu.encoding):
                continue
            if i + 1 >= len(cer):
                continue
            taken = (cer[i + 1].cpu.pc != ite.cpu.pc + 4)
            disas = disassemble_instruction(ite.cpu.encoding, ite.cpu.pc) or "<unk>"
            arch_map[ite.cpu.pc - code_base] = (taken, disas)

    log.w(f"  {'Offset':>6}  {'Instruction':<28}  {'Arch dir':<14}  Train dir", ch=ch)
    log.w(f"  {'':-<6}  {'':-<28}  {'':-<14}  ---------", ch=ch)
    for off, train_taken in entries:
        actual = arch_map.get(off)
        disas = actual[1] if actual else "<unknown>"
        actual_text = ("TAKEN    " if actual and actual[0]
                       else "NOT-TAKEN" if actual
                       else "?        ")
        actual_dir = f"{_C_TAKEN if actual and actual[0] else _C_NTAKEN}{actual_text}{_C_RESET}"
        train_text = "TAKEN    " if train_taken else "NOT-TAKEN"
        train_dir  = f"{_C_TAKEN if train_taken else _C_NTAKEN}{train_text}{_C_RESET}"
        log.w(f"  +{off:04x}   {disas:<28}  {actual_dir}  {train_dir}", ch=ch)
    log.w("", ch=ch)


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
# Maps variant enum → TestCase.  Extensible: add more entries in instrument_stage2 as needed.
# ==================================================================================================
from enum import Enum as _Enum
TCVariants = Dict[_Enum, TestCase]


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
        raise NotImplementedError()

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
        self._tc_counter: int = 0
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
            return []

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

    def load_test_case(self, test_case: TestCase):
        self._tc_counter += 1
        log = _FuzzLogger.get()
        log.register("basic_flow", "basic/flow.log", min_verbosity=1)
        log.register("basic_hw",   "basic/hw.log",  min_verbosity=1)
        log.wp(f"[TC #{self._tc_counter}] loading test case")
        written_tc = self._write_test_case(test_case)
        self.curr_test_case = test_case
        self.ignore_list = set()
        return written_tc

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

        sandboxed_test_case = self._write_mod_test_case_to_local_executor("sandboxed_test_case", [Aarch64SandboxPass()])
        iid = self.local_executor.allocate_iid()
        self.local_executor.checkout_region(InputRegion(iid))

        input_to_trace_list = defaultdict(list)
        counter_log = defaultdict(list)
        expected_log = {}
        log = _FuzzLogger.get()

        for rep in range(n_reps):
            for idx, i in enumerate(inputs):
                self.local_executor.write(ExecutorMemory(_input_bytes_with_pstate(i)))
                trained_entries = []
                if hasattr(i, "_arch_trace"):
                    trained_entries = self.set_branch_mistraining(i._arch_trace)
                    if rep == 0 and trained_entries:
                        log_mistraining(log, self._tc_counter, idx, trained_entries,
                                        i._arch_trace, ch="basic_hw")
                if idx not in expected_log:
                    expected_log[idx] = len(trained_entries)
                assert expected_log[idx] == len(trained_entries)
                self.local_executor.trace()
                hwm = self.local_executor.hardware_measurement()
                input_to_trace_list[idx].append(hwm.htrace)
                counter_log[idx].append(hwm.pfcs[2])
        for idx, vals in counter_log.items():
            avg = 100 * (((1.0 * sum(vals)) / len(vals)) / expected_log[idx]) if expected_log[idx] > 0 else "NA"
            log.w(f"  input={idx} miss-rate: {sum(vals)/len(vals):.1f}/{expected_log[idx]} = {avg}%  reps={vals}",
                  ch="basic_hw")

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
    """Executor for PAC non-interference testing.

    Lifecycle:
    1. load_test_case(tc)  — runs stage-1 PAC instrumentation, caches result.
    2. trace_test_case_with_taints(inputs, nesting)  — for each input:
         a. Run CE on the stage-1 TC to capture signed pointer values per fix-point.
         b. Call instrument_stage2() to produce TC1/TC2/TC3.
       Returns ctraces/taints derived from the stage-1 CE traces.
    """

    def __init__(self, workdir: str, generator: Aarch64Generator, *args, **kwargs):
        super().__init__(workdir, *args, **kwargs)
        self._generator: Aarch64Generator = generator
        self._pac_instrumentation = PACInstrumentation(
            self._generator, CONF.pac_xpac_weight, CONF.pac_auth_weight)

        self._stage1_tc: Optional[TestCase] = None
        self._stage1_fix_points: Optional[List[FixPoint]] = None
        self._stage1_tc_bytes: Optional[bytes] = None
        self._stage1_slot_offset_to_fp: Optional[Dict[int, FixPoint]] = None
        self._last_tc_variants: Optional[List[TCVariants]] = None
        self._last_stage1_traces: Optional[List[ContractExecutionResult]] = None

    def load_test_case(self, test_case: TestCase):
        log = _FuzzLogger.get()
        # Register channels on first use; subsequent calls are no-ops.
        log.register("pac_flow",  "pac/flow.log",   min_verbosity=1)
        log.register("pac_stage1","pac/stage1.log", min_verbosity=2)
        log.register("pac_hw",    "pac/hw.log",     min_verbosity=1)
        result = super().load_test_case(test_case)
        log_start_test_case(log, self._tc_counter)
        # Pin the PAC keys to this process's current keys so that pac_sign (called
        # from this Python process via ioctl) and kernel_pac_auth (called from the
        # CE subprocess via ioctl) use the same key material.  Without this, each
        # process uses its own OS-assigned PAC keys and AUTH fails with correct_sig.
        keys = self.local_executor.get_pac_keys()
        self.local_executor.set_pac_keys(keys)
        log.wp(f"  [KEY] PAC keys pinned: DA={keys.apda_lo:#018x}/{keys.apda_hi:#018x}")
        self._run_stage1()
        # Log the stage-1 TC binary immediately after assembly — before any CE or HW.
        if self._stage1_tc_bytes:
            log_tc_binary(log, "STAGE1", self._stage1_tc_bytes)
            log.ensure_flushed()
        return result

    def _run_stage1(self) -> None:
        patched = copy.deepcopy(self.test_case)
        stage1_tc, fix_points = self._pac_instrumentation.instrument_stage1(patched)
        layout = Aarch64ASMLayout(stage1_tc)
        assembly = Aarch64Printer(self.target_desc).print_layout(layout)
        tc_bytes = ConfigurableGenerator.in_memory_assemble(assembly)

        # Map byte offset of each XPAC placeholder → FixPoint (pre-slot state captured here)
        xpac_offset_to_fp: Dict[int, FixPoint] = {}
        for fp in fix_points:
            xpac_inst = fp.slot_insts[AUTH_SLOT_POS]
            xpac_offset_to_fp[layout.instruction_address[xpac_inst]] = fp

        self._stage1_tc = stage1_tc
        self._stage1_fix_points = fix_points
        self._stage1_tc_bytes = tc_bytes
        self._stage1_slot_offset_to_fp = xpac_offset_to_fp
        _FuzzLogger.get().wp(f"  [STAGE1] fix_points={len(fix_points)}")

    def _write_prebuilt_tc_to_executor(self, tc: TestCase) -> None:
        layout = Aarch64ASMLayout(tc)
        assembly = Aarch64Printer(self.target_desc).print_layout(layout)
        tc_bytes = ConfigurableGenerator.in_memory_assemble(assembly)
        self.local_executor.checkout_region(TestCaseRegion())
        self.local_executor.write(tc_bytes)

    def _hw_trace_single_input(self, tc: TestCase, inp: Input, n_reps: int = 1) -> HTrace:
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

    def debug_compare_variants_on_ce(
        self,
        inputs: List[Input],
        tc_variants_per_input: List[TCVariants],
    ) -> None:
        """Run all TC variants through CE for every input and log any CTrace differences."""
        sandbox_base, _ = self.read_base_addresses()
        for i, (inp, variants) in enumerate(zip(inputs, tc_variants_per_input)):
            data = inp.tobytes()
            tc_memory = data[:0x2000]
            tc_regs = bytearray(data[0x2000:])
            _reconstruct_pstate(memoryview(tc_regs).cast('Q'))
            tc_regs = bytes(tc_regs)
            ce_ctraces = {}
            try:
                for variant, tc in variants.items():
                    tc_bytes = self._tc_to_bytes(tc)
                    execution = ContractExecution(
                        tc_bytes, tc_memory, tc_regs,
                        SimArch.RVZR_ARCH_AARCH64, 5, 10,
                        req_mem_base_virt=sandbox_base,
                    )
                    ce_ctraces[variant] = compute_ctrace(self._contract_executor.run(execution))
            except RuntimeError:
                return
            variant_list = list(ce_ctraces.items())
            baseline_variant, baseline_ct = variant_list[0]
            for variant, ct in variant_list[1:]:
                if ct != baseline_ct:
                    _FuzzLogger.get().w(
                        f"  CE ctrace diff: {baseline_variant.name}={baseline_ct.raw}"
                        f"  {variant.name}={ct.raw}", ch="pac_flow")

    def run_pac_hardware_comparison(
        self,
        inputs: List[Input],
        tc_variants_per_input: List[TCVariants],
        n_reps: int = 1,
    ) -> List[Dict[PACVariant, HTrace]]:
        """Run each input through all its TC variants on real hardware.

        STRIP_ONLY == AUTH_CORRECT: correct (strip and correct-auth are arch-equivalent).
        AUTH_WRONG != STRIP_ONLY: PAC non-interference candidate.
        """
        results = []
        for inp, variants in zip(inputs, tc_variants_per_input):
            per_variant = {v: self._hw_trace_single_input(tc, inp, n_reps)
                           for v, tc in variants.items()}
            results.append(per_variant)
        return results

    def trace_test_case_with_taints(
        self,
        inputs: List[Input],
        nesting: int,
    ) -> Tuple[List[CTrace], List[InputTaint], List[ContractExecutionResult], List[TCVariants]]:
        if not inputs or self.test_case is None:
            return [], [], [], []

        assert self._stage1_tc is not None, \
            "stage-1 cache is empty: load_test_case() was not called"

        tc = self._stage1_tc
        fix_points = self._stage1_fix_points
        tc_bytes = self._stage1_tc_bytes
        sandbox_base, _ = self.read_base_addresses()

        stage1_traces: List[ContractExecutionResult] = []
        tc_variants_per_input: List[TCVariants] = []

        xpac_offset_to_fp = self._stage1_slot_offset_to_fp

        def _read_reg(cpu, reg: str) -> Optional[int]:
            if reg is None:
                return None
            return cpu.sp if reg == 'sp' else cpu.gpr[int(reg[1:])]

        log = _FuzzLogger.get()
        log.register("pac_signing",    "pac/signing.log",    min_verbosity=1)
        log.register("pac_comparison", "pac/comparison.log", min_verbosity=2)

        for inp_idx, inp in enumerate(inputs):
            # Log input before CE so it's on disk even if CE crashes.
            log_input(log, inp_idx, inp, ch="pac_flow")
            log.ensure_flushed()

            for fp in fix_points:
                fp.reset()

            data = inp.tobytes()
            tc_memory = data[:0x2000]
            tc_regs = bytearray(data[0x2000:])
            _reconstruct_pstate(memoryview(tc_regs).cast('Q'))
            tc_regs = bytes(tc_regs)

            execution = ContractExecution(
                tc_bytes, tc_memory, tc_regs,
                SimArch.RVZR_ARCH_AARCH64, nesting, 10,
                req_mem_base_virt=sandbox_base,
            )
            cer = self._contract_executor.run(execution)
            stage1_traces.append(cer)

            def _other_gpr_value(cpu, exclude1: str, exclude2: Optional[str]) -> int:
                """Pick a value from a GPR other than exclude1/exclude2 (avoid x29/x30)."""
                excl = {exclude1, exclude2} if exclude2 else {exclude1}
                candidates = [f"x{i}" for i in range(29) if f"x{i}" not in excl]
                return _read_reg(cpu, random.choice(candidates))

            log = _FuzzLogger.get()
            log_ce_trace(log, "STAGE1", inp_idx, list(cer))
            log.header(f"STAGE1 PAC-SIGN OPS  inp={inp_idx}")

            if len(cer) > 0:
                code_base = cer[0].cpu.pc
                for ite in cer:
                    rel_pc = ite.cpu.pc - code_base
                    if rel_pc not in xpac_offset_to_fp:
                        continue
                    fp = xpac_offset_to_fp[rel_pc]
                    depth = ite.metadata.speculation_nesting
                    if fp.spec_nesting is None or depth < fp.spec_nesting:
                        fp.spec_nesting = depth
                    # Prefer arch-path values; fall back to first occurrence
                    if depth == 0 or fp.correct_sig is None:
                        mn      = fp.committed_inst.name.lower()
                        pac_mn  = _AUTH_TO_PAC[mn]
                        ptr_reg = fp.committed_inst.operands[0].value
                        has_ctx = len(fp.committed_inst.operands) > 1
                        ctx_reg = fp.committed_inst.operands[1].value if has_ctx else None

                        ptr = _read_reg(ite.cpu, ptr_reg)
                        ctx = _read_reg(ite.cpu, ctx_reg) if has_ctx else 0

                        signed = self.local_executor.pac_sign(ptr, ctx, pac_mn)
                        fp.correct_sig = (signed >> 48) & 0xFFFF
                        log_pac_op(log, "SIGN", pac_mn, ptr, ctx, signed)
                        log.w(f"    => slot={fp.slot_id}  correct_sig=0x{fp.correct_sig:04x}")

                        # alt_sig: independently randomize ptr and ctx (4 options).
                        # Force bits[63:48]=0xFFFF on alt_ptr so it is a canonical kernel VA:
                        # PACDA then sets bit 55 in the signed pointer, so XPAC can correctly
                        # sign-extend it back to 0xffff... (avoids non-canonical 0x0000... crash).
                        use_wrong_ptr = random.choice([True, False])
                        use_wrong_ctx = has_ctx and random.choice([True, False])
                        alt_ptr = _other_gpr_value(ite.cpu, ptr_reg, ctx_reg) if use_wrong_ptr else ptr
                        alt_ptr = (alt_ptr & 0x0000FFFFFFFFFFFF) | 0xFFFF000000000000
                        alt_ctx = (_other_gpr_value(ite.cpu, ctx_reg, ptr_reg) if use_wrong_ctx else ctx) if has_ctx else 0
                        alt_signed = self.local_executor.pac_sign(alt_ptr, alt_ctx, pac_mn)
                        fp.alt_sig = (alt_signed >> 48) & 0xFFFF
                        log_pac_op(log, "SIGN(alt)", pac_mn, alt_ptr, alt_ctx, alt_signed)
                        log.w(f"    => slot={fp.slot_id}  alt_sig=0x{fp.alt_sig:04x}")

            log.header(f"STAGE1 SLOT CLASSIFICATION  inp={inp_idx}")
            for fp in fix_points:
                log_slot(log, inp_idx, fp)

            # For fixpoints that CE never reached, compute alt_sig from random values.
            for fp in fix_points:
                if fp.alt_sig is None:
                    mn      = fp.committed_inst.name.lower()
                    pac_mn  = _AUTH_TO_PAC[mn]
                    has_ctx = len(fp.committed_inst.operands) > 1
                    alt_ptr = random.randrange(1 << 48) | 0xFFFF000000000000
                    alt_ctx = random.randrange(1 << 64) if has_ctx else 0
                    alt_signed = self.local_executor.pac_sign(alt_ptr, alt_ctx, pac_mn)
                    fp.alt_sig = (alt_signed >> 48) & 0xFFFF

            variants = self._pac_instrumentation.instrument_stage2(tc, fix_points)
            tc_variants_per_input.append(variants)

            # Log all variant TC binaries for this input immediately after
            # stage2 — these are the exact binaries that will run on HW.
            log.header(f"VARIANT TC BINARIES  inp={inp_idx}")
            for variant, vtc in variants.items():
                vbytes = self._tc_to_bytes(vtc)
                log_tc_binary(log, variant.name, vbytes)

            # Flush after every input's data so a crash between inputs
            # leaves a complete record for the inputs already processed.
            log.ensure_flushed()

        self._last_tc_variants = tc_variants_per_input
        self._last_stage1_traces = stage1_traces
        taints = [compute_taint(cer) for cer in stage1_traces]
        ctraces = [compute_ctrace(cer) for cer in stage1_traces]
        return ctraces, taints, stage1_traces, tc_variants_per_input

    def _run_ce_on_tc(self, tc: TestCase, inp: Input, sandbox_base: int) -> Optional[List]:
        """Run CE on *tc* for *inp*; return list of trace entries or None on failure."""
        tc_bytes = self._tc_to_bytes(tc)
        data = inp.tobytes()
        tc_regs = bytearray(data[0x2000:])
        _reconstruct_pstate(memoryview(tc_regs).cast("Q"))
        try:
            execution = ContractExecution(
                tc_bytes, data[:0x2000], bytes(tc_regs),
                SimArch.RVZR_ARCH_AARCH64, 5, 10,
                req_mem_base_virt=sandbox_base,
            )
            return list(self._contract_executor.run(execution))
        except RuntimeError as exc:
            _FuzzLogger.get().w(f"  [CE ERROR] {exc}")
            return None

    @staticmethod
    def _compute_bb_map(cer_list: List) -> Tuple[List[int], Dict[int, Dict], Dict[int, int], int]:
        """Derive basic-block entries from a CE trace.

        Returns (sorted_pcs, bb_info, pc_to_id, code_base).
        bb_info[pc] = {'arch': bool, 'spec': bool, 'offset': int, 'disas': str}
        A BB entry is: the first PC, or any PC that follows a non-sequential predecessor.
        """
        if not cer_list:
            return [], {}, {}, 0
        code_base = cer_list[0].cpu.pc
        bb_entry_pcs: Set[int] = {code_base}
        for i in range(len(cer_list) - 1):
            if cer_list[i + 1].cpu.pc != cer_list[i].cpu.pc + 4:
                bb_entry_pcs.add(cer_list[i + 1].cpu.pc)

        bb_info: Dict[int, Dict] = {}
        for ite in cer_list:
            pc = ite.cpu.pc
            if pc not in bb_entry_pcs:
                continue
            if pc not in bb_info:
                bb_info[pc] = {
                    "arch": False, "spec": False,
                    "offset": pc - code_base,
                    "disas": disassemble_instruction(ite.cpu.encoding, pc) or "<unk>",
                }
            if ite.metadata.speculation_nesting == 0:
                bb_info[pc]["arch"] = True
            else:
                bb_info[pc]["spec"] = True

        for pc in bb_entry_pcs:
            if pc not in bb_info:
                bb_info[pc] = {"arch": False, "spec": False,
                               "offset": pc - code_base, "disas": "<not-reached>"}

        sorted_pcs = sorted(bb_info.keys(), key=lambda p: p - code_base)
        pc_to_id: Dict[int, int] = {pc: i + 1 for i, pc in enumerate(sorted_pcs)}
        return sorted_pcs, bb_info, pc_to_id, code_base

    def _print_ce_trace_comparison(
        self,
        inp_idx: int,
        inp: Input,
        stage1_cer: ContractExecutionResult,
        tc_variants: Dict,
        sandbox_base: int,
    ) -> None:
        """Print a side-by-side CE trace table: stage1 vs TC1/TC2/TC3.

        Columns: STAGE1 | TC1 STRIP | TC2 CORRECT | TC3 WRONG
        Each row: [nesting]+offset  disas  src-reg=value...  EA=addr (if mem)
        Rows where TC3 diverges from TC1 in control-flow are marked with  <<<
        Output goes to both stdout and /home/gal_k_1_1998/revizor_crash.log.
        """

        def _run_ce_on_tc(tc: TestCase):
            tc_bytes = self._tc_to_bytes(tc)
            data = inp.tobytes()
            tc_regs = bytearray(data[0x2000:])
            _reconstruct_pstate(memoryview(tc_regs).cast('Q'))
            try:
                execution = ContractExecution(
                    tc_bytes, data[:0x2000], bytes(tc_regs),
                    SimArch.RVZR_ARCH_AARCH64, 5, 10,
                    req_mem_base_virt=sandbox_base,
                )
                return list(self._contract_executor.run(execution))
            except RuntimeError as exc:
                return None

        def _fmt_regs(ite) -> str:
            srcs, _ = get_srcs_dests_operands(ite.cpu.encoding, ite.cpu.pc)
            parts = []
            for r in srcs[:3]:
                rl = r.lower()
                if rl == 'sp':
                    parts.append(f"sp={ite.cpu.sp:016x}")
                elif rl.startswith('x') and rl[1:].isdigit():
                    n = int(rl[1:])
                    if 0 <= n <= 30:
                        parts.append(f"x{n}={ite.cpu.gpr[n]:016x}")
            return " ".join(parts)

        def _fmt_entry(ite, base: int) -> str:
            if ite is None:
                return "<end>"
            disas = disassemble_instruction(ite.cpu.encoding, ite.cpu.pc) or "<unk>"
            offset = ite.cpu.pc - base
            nest = ite.metadata.speculation_nesting
            regs = _fmt_regs(ite)
            ea = (f"  EA={ite.metadata.memory_access.effective_address:016x}"
                  if ite.metadata.has_memory_access else "")
            # keep disas at fixed width so regs align across rows
            return f"[{nest}]+{offset:04x}  {disas:<26}  {regs}{ea}"

        # Collect all four traces.
        # stage1_cer comes from trace_test_case_with_taints (already run).
        named_traces: List[Tuple[str, List, int]] = []  # (label, entries, code_base)

        s1_list = list(stage1_cer) if stage1_cer else []
        s1_base = s1_list[0].cpu.pc if s1_list else 0
        named_traces.append(("STAGE1", s1_list, s1_base))

        for variant, label in [
            (PACVariant.STRIP_ONLY,   "TC1 STRIP"),
            (PACVariant.AUTH_CORRECT, "TC2 CORRECT"),
            (PACVariant.AUTH_WRONG,   "TC3 WRONG"),
        ]:
            tc = tc_variants.get(variant)
            if tc is None:
                named_traces.append((label, [], 0))
                continue
            cer = _run_ce_on_tc(tc)
            if cer:
                base = cer[0].cpu.pc
            else:
                base = 0
                cer = []
            named_traces.append((label, cer, base))

        labels   = [t[0] for t in named_traces]
        traces   = [t[1] for t in named_traces]
        bases    = [t[2] for t in named_traces]
        n_rows   = max((len(t) for t in traces), default=0)

        COL_W = 75
        header = f"  {'ROW':>4}  | " + " | ".join(f"{lbl:<{COL_W}}" for lbl in labels)
        sep    = "-" * len(header)

        tc1_entries = traces[1]  # TC1 STRIP
        tc3_entries = traces[3]  # TC3 WRONG
        tc1_base    = bases[1]
        tc3_base    = bases[3]

        out_lines = [
            "",
            "=" * 80,
            f"  CE Trace Comparison — inp={inp_idx}",
            "=" * 80,
            header,
            sep,
        ]

        for row in range(n_rows):
            cells = [
                _fmt_entry(tr[row] if row < len(tr) else None, base)
                for tr, base in zip(traces, bases)
            ]

            # Mark rows where TC3's PC-offset diverges from TC1.
            tc1_e = tc1_entries[row] if row < len(tc1_entries) else None
            tc3_e = tc3_entries[row] if row < len(tc3_entries) else None
            if (tc1_e is None) != (tc3_e is None):
                flag = "  <<<"
            elif tc1_e is not None and tc3_e is not None:
                flag = "  <<<" if (tc1_e.cpu.pc - tc1_base) != (tc3_e.cpu.pc - tc3_base) else ""
            else:
                flag = ""

            out_lines.append(
                f"  {row:>4}  | " + " | ".join(f"{c:<{COL_W}}" for c in cells) + flag
            )

        if _FuzzLogger._VERBOSITY < 2:
            return
        log = _FuzzLogger.get()
        for line in out_lines:
            log.w(line, ch="pac_comparison")
        log.ensure_flushed()

    def trace_test_case_variants_hw(
        self,
        inputs: List[Input],
        n_reps: int,
    ) -> List[Dict[PACVariant, HTrace]]:
        """Run all TC variants for each input on hardware with stage-1-compatible mistraining.

        For each input[i] and each variant in _last_tc_variants[i], executes the variant TC
        on hardware using i._arch_trace for BPU mistraining.  All variants share the same
        branch layout as stage-1, so the same arch_trace applies to all of them.

        Returns per-input: {PACVariant → HTrace}.
        """
        assert self._last_tc_variants is not None and len(self._last_tc_variants) == len(inputs), \
            "trace_test_case_with_taints() must be called before trace_test_case_variants_hw()"

        STAT.executor_reruns += n_reps * len(inputs)

        sandbox_base, _ = self.read_base_addresses()
        _AUTH_MNEMONICS = {'autia', 'autib', 'autda', 'autdb', 'autiza', 'autizb', 'autdza', 'autdzb'}

        log = _FuzzLogger.get()

        # ----------------------------------------------------------------
        # Phase 1: CE analysis — run CE on EVERY variant for EVERY input,
        # log all traces and BB maps, and build BB-instrumented binaries.
        # This entire phase completes BEFORE any hardware execution starts,
        # so the log is complete even if a subsequent HW run crashes the machine.
        # ----------------------------------------------------------------
        stage1_traces = self._last_stage1_traces or ([None] * len(inputs))

        # pre_hw[inp_idx] = {PACVariant: (cer_list, instrumented_binary_bytes)}
        pre_hw: List[Dict[PACVariant, Tuple[Optional[List], bytes]]] = []

        for inp_idx, (inp, variants) in enumerate(zip(inputs, self._last_tc_variants)):
            inp_data = inp.tobytes()
            entry_x7 = int.from_bytes(inp_data[7 * 8: 8 * 8], "little")
            per_inp: Dict[PACVariant, Tuple[Optional[List], bytes]] = {}

            log.header(f"PRE-HW ANALYSIS  inp={inp_idx}")

            # Log stage1 trace (already computed in trace_test_case_with_taints).
            s1_cer = stage1_traces[inp_idx]
            if s1_cer:
                log_ce_trace(log, "STAGE1", inp_idx, list(s1_cer))

            for variant, tc in variants.items():
                vname = variant.name
                cer = self._run_ce_on_tc(tc, inp, sandbox_base)
                if cer is not None:
                    log_ce_trace(log, vname, inp_idx, cer)
                else:
                    log.w(f"  CE failed for {vname} inp={inp_idx}")

                # Compute and log/print BB map for this variant.
                sorted_pcs, bb_info, pc_to_id, code_base = self._compute_bb_map(cer or [])
                log_bb_map(log, vname, inp_idx, sorted_pcs, bb_info, pc_to_id,
                               code_base, entry_x7)

                # Log AUTH instructions in this variant's CE trace.
                if cer:
                    log.header(f"AUTH INSTRUCTIONS IN CE: {vname}  inp={inp_idx}")
                    for ite in cer:
                        disas = disassemble_instruction(ite.cpu.encoding, ite.cpu.pc) or ""
                        mn = disas.split()[0].lower() if disas else ""
                        if mn in _AUTH_MNEMONICS:
                            nest = ite.metadata.speculation_nesting
                            tag = "ARCH" if nest == 0 else f"SPEC({nest})"
                            log.wp(f"  {tag}  {disas}  pc=0x{ite.cpu.pc:016x}")

                per_inp[variant] = (cer, self._tc_to_bytes(tc))

            # Also log comparison table when TC3 differs from TC1 in control flow (verbosity 2).
            if _FuzzLogger._VERBOSITY >= 2:
                tc1_pair = per_inp.get(PACVariant.STRIP_ONLY)
                tc3_pair = per_inp.get(PACVariant.AUTH_WRONG)
                if tc1_pair and tc3_pair:
                    tc1_cer, _ = tc1_pair
                    tc3_cer, _ = tc3_pair
                    if tc1_cer and tc3_cer and len(tc1_cer) != len(tc3_cer):
                        log.wp(f"  *** TC1 vs TC3 trace length differs: "
                               f"TC1={len(tc1_cer)} TC3={len(tc3_cer)} — control flow divergence!")
                    self._print_ce_trace_comparison(inp_idx, inp, s1_cer, variants, sandbox_base)

            pre_hw.append(per_inp)

        # All CE analysis is done.  Force an OS-level flush so the entire
        # log is physically on disk before the first HW instruction executes.
        # A machine crash after this point leaves a complete pre-crash record.
        log.ensure_flushed()
        log.wp("[HW PHASE] All pre-HW logging complete — starting hardware execution")

        # ----------------------------------------------------------------
        # Phase 2: Hardware execution — iterate inputs/variants and run HW.
        # ----------------------------------------------------------------

        counter_log = defaultdict(lambda: defaultdict(list))
        expected_log = {}

        results: List[Dict[PACVariant, HTrace]] = []
        for inp_idx, (inp, variants) in enumerate(zip(inputs, self._last_tc_variants)):
            per_variant: Dict[PACVariant, HTrace] = {}
            self.local_executor.discard_all_inputs()
            iid = self.local_executor.allocate_iid()
            self.local_executor.checkout_region(InputRegion(iid))
            self.local_executor.write(ExecutorMemory(_input_bytes_with_pstate(inp)))

            per_inp = pre_hw[inp_idx]

            for variant, tc in variants.items():
                vname = variant.name
                _, instr_bin = per_inp.get(variant, (None, None))

                log.wp(f"[HW] inp={inp_idx} variant={vname}")

                # Write the BB-instrumented binary (works for all variants).
                if instr_bin is not None:
                    self.local_executor.checkout_region(TestCaseRegion())
                    self.local_executor.write(instr_bin)
                else:
                    self._write_prebuilt_tc_to_executor(tc)

                trace_list = []
                for rep in range(n_reps):
                    if hasattr(inp, "_arch_trace"):
                        entries = self.set_branch_mistraining(inp._arch_trace)
                        if rep == 0:
                            log_mistraining(log, self._tc_counter, inp_idx, entries,
                                            inp._arch_trace)
                    if inp_idx not in expected_log:
                        expected_log[inp_idx] = len(entries)
                    assert expected_log[inp_idx] == len(entries)
                    self.local_executor.trace()
                    self.local_executor.checkout_region(InputRegion(iid))
                    hwm = self.local_executor.hardware_measurement()
                    trace_list.append(hwm.htrace)
                    counter_log[variant][inp_idx].append(hwm.pfcs[2])
                ht = HTrace(trace_list=trace_list)
                per_variant[variant] = ht
                log.w(f"  htrace=0x{ht.raw[0]:016x}  n_reps={len(trace_list)}", ch="pac_hw")
            results.append(per_variant)

        for variant, d in counter_log.items():
            for idx, vals in d.items():
                avg = 100 * (((1.0 * sum(vals)) / len(vals)) / expected_log[idx]) if expected_log[idx] > 0 else "NA"
                log.w(f"  variant={variant.name} input={idx} miss-rate: {sum(vals)/len(vals):.1f}/{expected_log[idx]}"
                      f" = {avg}%  reps={vals}", ch="pac_hw")


        return results


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

    def load_test_case(self, test_case: TestCase):
        result = super().load_test_case(test_case)
        self._run_stage1()
        return result

    def _run_stage1(self) -> None:
        """Instrument test_case with NOP placeholders (stage-1) and cache layout."""
        log = _FuzzLogger.get()
        log.register("mte_flow",   "mte/flow.log",   min_verbosity=1)
        log.register("mte_stage1", "mte/stage1.log", min_verbosity=2)
        log.register("mte_hw",     "mte/hw.log",     min_verbosity=1)
        log.wp(f"[MTE] test case #{self._tc_counter}")

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

        # Build sandboxed TC once — CE on this produces branch offsets matching
        # what trace_test_case() loads into the kernel module for mistraining.
        sandboxed_patched = copy.deepcopy(self.test_case)
        pass_on_test_case(sandboxed_patched, [Aarch64SandboxPass()])
        sandboxed_layout = Aarch64ASMLayout(sandboxed_patched)
        sandboxed_assembly = Aarch64Printer(self.target_desc).print_layout(sandboxed_layout)
        sandboxed_tc_bytes = ConfigurableGenerator.in_memory_assemble(sandboxed_assembly)
        sandboxed_traces: List[ContractExecutionResult] = []

        log = _FuzzLogger.get()
        for inp_idx, inp in enumerate(inputs):
            mte_inp_ch = log.register(f"mte_inp_{inp_idx:03d}",
                                      f"mte/inputs/inp_{inp_idx:03d}.log",
                                      min_verbosity=2)
            if mte_inp_ch:
                log.use(mte_inp_ch)
                log_input(log, inp_idx, inp, ch=mte_inp_ch)

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

            # Run CE on the sandboxed TC for mistraining-compatible branch offsets.
            sandboxed_execution = ContractExecution(
                sandboxed_tc_bytes, tc_memory, tc_regs,
                SimArch.RVZR_ARCH_AARCH64, nesting, 10,
                req_mem_base_virt=sandbox_base,
            )
            try:
                sandboxed_cer = self._contract_executor.run(sandboxed_execution)
            except RuntimeError:
                sandboxed_cer = cer  # fall back to stage-1 trace if sandboxed CE crashes
            sandboxed_traces.append(sandboxed_cer)

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

            tc_variants_per_input.append(self._mte.instrument_stage2(tc, fix_points, sandbox_base))

        # Use sandboxed-TC traces for ctrace/taint/mistraining — their branch offsets
        # match the sandboxed TC that trace_test_case() loads into the kernel module.
        taints = [compute_taint(cer) for cer in sandboxed_traces]
        ctraces = [compute_ctrace(cer) for cer in sandboxed_traces]

        return ctraces, taints, sandboxed_traces, tc_variants_per_input


def _reconstruct_pstate(view: memoryview) -> None:
    """Convert per-flag NZCV encoding in slot 6 to ARM PSTATE format via NZCVScheme."""
    view[6] = NZCVScheme.to_pstate(int(view[6]))


def _input_bytes_with_pstate(inp) -> bytes:
    """Return inp.tobytes() with slot-6 converted from per-flag to PSTATE format."""
    data = bytearray(inp.tobytes())
    _reconstruct_pstate(memoryview(data)[0x2000:].cast('Q'))
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


