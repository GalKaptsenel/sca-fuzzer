"""
File: Implementation of executor for AArch64 architecture
  - Interfacing with the kernel module
  - Aggregation of the results
"""
from __future__ import annotations
import copy
import random
import os.path

import numpy as np
from typing import List, Tuple, Set, Optional, Dict
from collections import defaultdict
from enum import Enum, auto
from pathlib import Path


from .aarch64_generator import Aarch64Generator
from .aarch64_printer import Aarch64Printer, Aarch64ASMLayout
from ..interfaces import (HTrace, Input, TestCase, Executor, HardwareTracingError, CTrace,
                          InputTaint, GeneratorException)
from ..config import CONF
from ..util import Logger, STAT, FuzzLogger
from .aarch64_target_desc import Aarch64TargetDesc, SANDBOX_BASE_REGISTER
from .aarch64_kernel import LocalHWExecutor, TestCaseRegion, InputRegion, ExecutorMemory
from .aarch64_generator import Pass, Aarch64SandboxPass
from .aarch64_seal import SealedNIInstrumentation, is_speculative
from .aarch64_pac import PACInstrumentation, PACFixPoint, AUTH_SLOT_POS, _AUTH_TO_PAC
from .aarch64_contract_executor import (ContractExecution, ContractExecutorService,
                                        ExecutionClause, SUPPORTED_EXECUTION_CLAUSES,
                                        BranchPredictor, EXECUTION_CLAUSE_MAP, SimArch)
from .aarch64_disasm import disassemble_instruction, decode_reg_accesses, is_conditional_branch
from .aarch64_mte import MteTagState, mte_tag_store_effect, MTEFixPoint
from .aarch64_seal_factory import make_seal_pass
from .aarch64_trace import compute_ctrace, compute_taint, ContractExecutionResult
from .aarch64_input_layout import _input_bytes_with_pstate, REGISTER_REGION_OFFSET
from .aarch64_log import (log_start_test_case, log_input, log_ce_trace, log_bb_map,
                          log_tc_binary, log_pac_op, log_slot, log_mistraining, log_ni_table)

# ==================================================================================================
# Helper functions
# ==================================================================================================
def pass_on_test_case(test_case: TestCase, passes: List[Pass]):
    for p in passes:
        p.run_on_test_case(test_case)


def _ce_memory_regs(inp) -> Tuple[bytes, bytes]:
    """Split an input into (memory, PSTATE-reconstructed registers) for a CE run."""
    full = _input_bytes_with_pstate(inp)
    return full[:REGISTER_REGION_OFFSET], full[REGISTER_REGION_OFFSET:]


def _read_reg(cpu, reg: Optional[str]) -> Optional[int]:
    """Read a register value (x0-x30 or sp) from a CE CPUState; None for a None reg."""
    if reg is None:
        return None
    return cpu.sp if reg == 'sp' else cpu.gpr[int(reg[1:])]


# ==================================================================================================
# NON-INTERFERENCE: per-input TC variant result containers
# Used by the non-interference executors; regular-Revizor mode never produces these.
# Maps variant enum → TestCase.
# ==================================================================================================
TCVariants = Dict[Enum, TestCase]


class NIVariant(Enum):
    """The test cases an NI engine compares per input: the all-genuine reference and a decoy
    instance. Any hardware-trace difference between them is a speculative leak."""
    BASELINE = auto()  # engine.baseline() — genuine everywhere
    DECOY    = auto()  # engine.decoys()   — decoys on speculative slots


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

    def _is_smt_enabled(self) -> bool:
        """Whether SMT is enabled on the current CPU (sibling threads share uarch state -> false
        positives). True if enabled, False otherwise."""
        smt_file = Path('/sys/devices/system/cpu/smt/control')
        if smt_file.is_file():
            result = smt_file.read_text().strip()
            return 'on' in result or '1' in result
        return False

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
        pass  # intentional no-op on aarch64: no quick-and-dirty stabilization-skip is implemented

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

    def __init__(self, *args, **kwargs):
        Aarch64Executor.__init__(self, *args, **kwargs)

        self.test_case: Optional[TestCase] = None
        self._tc_counter: int = 0
        self._sandboxed_cache = None
        self.local_executor = LocalHWExecutor('/dev/executor', '/sys/executor')

        if self.target_desc.cpu_desc.vendor.lower() != "aarch64":
            self.LOG.error(
                "Attempting to run the AArch64 executor on a non-AArch64 CPU!\n"
                "Change the `executor` configuration option to the appropriate vendor value.")

        contract_executor_bin = os.path.join(
            os.path.dirname(__file__), "contract_executor", "contract_executor")
        self._contract_executor = ContractExecutorService(contract_executor_bin)

    def read_base_addresses(self):
        """
        Read the base addresses of the code and the sandbox from the kernel module.
        This function is used to synchronize the memory layout between the executor and the model
        :return: a tuple (sandbox_base, code_base)
        """
        return self.local_executor.sandbox_base, self.local_executor.code_base

    def branch_mistraining_entries(self, cer) -> List[Tuple[int, bool]]:
        """Compute the per-branch mistraining config from a CE trace, without touching the device.

        For each architectural conditional branch in cer, the base predictor must be saturated
        in the opposite direction so the first HW execution always mispredicts it.

        :param cer: ContractExecutionResult for the input, or None
        :return: list of (byte_offset, train_taken) entries; empty if no branches to train
        """
        # Disabled by default: the applied training currently saturates the branch toward its
        # architectural direction, suppressing the natural misprediction Spectre-v1 needs (see
        # CONF.enable_branch_mistraining). Empty -> apply_branch_mistraining() clears training.
        if not CONF.enable_branch_mistraining:
            return []
        if cer is None or len(cer) == 0:
            return []

        ce_code_base = cer[0].cpu.pc
        entries = []
        for i, ite in enumerate(cer):
            if ite.metadata.speculation_nesting != 0:
                continue
            if not is_conditional_branch(ite.cpu.encoding):
                continue
            # arch direction from the next nest-0 entry: cer[i+1] may be a speculative
            # (opposite-direction) entry at nesting>1, which would invert the training.
            succ_pc = None
            for j in range(i + 1, len(cer)):
                if cer[j].metadata.speculation_nesting == 0:
                    succ_pc = cer[j].cpu.pc
                    break
            if succ_pc is None:
                continue
            taken = (succ_pc != ite.cpu.pc + 4)
            byte_offset = ite.cpu.pc - ce_code_base
            entries.append((byte_offset, not taken))  # opposite → guaranteed mispredict
        return entries

    def apply_branch_mistraining(self, entries: List[Tuple[int, bool]]) -> None:
        """Apply a precomputed mistraining config to the device for the next trace.

        An empty list clears any training left over from a previous input, so an input
        with no branches to train is always measured with a clean predictor state.
        """
        if entries:
            self.local_executor.write_branch_training_config(entries)
        else:
            self.local_executor.clear_branch_training()

    def load_test_case(self, test_case: TestCase):
        self._tc_counter += 1
        log = FuzzLogger.get()
        log.register("basic_flow", "basic/flow.log", min_verbosity=1)
        log.register("basic_hw",   "basic/hw.log",  min_verbosity=1)
        log.wp(f"[TC #{self._tc_counter}] loading test case")
        written_tc = self._write_test_case(test_case)
        self.curr_test_case = test_case
        self.ignore_list = set()
        self._sandboxed_cache = None  # invalidate cached sandboxed binary for the new TC
        return written_tc

    def _write_test_case(self, test_case: TestCase) -> None:
        self.test_case = test_case

    def _write_modified_test_case(self, label: str, passes: List[Pass]):
        if self._sandboxed_cache is None:
            patched = copy.deepcopy(self.test_case)
            pass_on_test_case(patched, passes)
            tc_bytes = self._assemble_tc(patched)[0]
            patched.asm_path, patched.obj_path, patched.bin_path = \
                f"{label}.asm", f"{label}.o", f"{label}.bin"
            self._sandboxed_cache = (tc_bytes, patched)

        tc_bytes, patched = self._sandboxed_cache
        self.local_executor.checkout_region(TestCaseRegion())
        self.local_executor.write(tc_bytes)

        return patched

    def _assemble_tc(self, tc: TestCase) -> Tuple[bytes, Aarch64ASMLayout]:
        """Assemble a TestCase to binary; return (bytes, layout)."""
        layout = Aarch64ASMLayout(tc)
        assembly = Aarch64Printer(self.target_desc).print_layout(layout)
        return Aarch64Generator.in_memory_assemble(assembly), layout

    def _make_ce_execution(self, tc_bytes: bytes, inp: Input, sandbox_base: int, nesting: int,
                           max_mispred_instructions: int, ct: ExecutionClause,
                           bp: BranchPredictor = BranchPredictor.NONE) -> ContractExecution:
        tc_memory, tc_regs = _ce_memory_regs(inp)
        return ContractExecution(
            tc_bytes, tc_memory, tc_regs, SimArch.RVZR_ARCH_AARCH64,
            nesting,
            max_mispred_instructions,  # unused by CE
            req_mem_base_virt=sandbox_base,
            execution_clauses=ct,
            branch_predictor=bp,
        )

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

        sandboxed_test_case = self._write_modified_test_case("sandboxed_test_case", [Aarch64SandboxPass()])

        input_to_trace_list = defaultdict(list)
        counter_log = defaultdict(list)
        pfc_log = defaultdict(list)
        expected_log = {}
        log = FuzzLogger.get()

        payloads = [ExecutorMemory(_input_bytes_with_pstate(i)) for i in inputs]
        train_entries = [self.branch_mistraining_entries(getattr(i, "_arch_trace", None))
                         for i in inputs]
        for idx, entries in enumerate(train_entries):
            expected_log[idx] = len(entries)
            if entries:
                log_mistraining(log, self._tc_counter, idx, entries,
                                inputs[idx]._arch_trace, ch="basic_hw")

        # Batch inputs sharing a mistraining config (the kernel applies one global config per
        # trace and measures every loaded input in that trace).
        groups: Dict[Tuple, List[int]] = defaultdict(list)
        for idx, entries in enumerate(train_entries):
            groups[tuple(entries)].append(idx)

        for entries, idxs in groups.items():
            self.local_executor.discard_all_inputs()
            iids = []
            for idx in idxs:
                iid = self.local_executor.allocate_iid()
                self.local_executor.checkout_region(InputRegion(iid))
                self.local_executor.write(payloads[idx])
                iids.append(iid)
            self.apply_branch_mistraining(list(entries))
            for _ in range(n_reps):
                self.local_executor.trace()
                for iid, idx in zip(iids, idxs):
                    self.local_executor.checkout_region(InputRegion(iid))
                    hwm = self.local_executor.hardware_measurement()
                    input_to_trace_list[idx].append(hwm.htrace)
                    pfc_log[idx].append(hwm.pfcs)
                    counter_log[idx].append(hwm.pfcs[2])

        for idx, vals in counter_log.items():
            avg_m = sum(vals) / len(vals)
            exp = expected_log[idx]
            rate = f" = {100 * avg_m / exp:.0f}% of expected" if exp > 0 else ""
            log.w(f"  input={idx} mispredicts: avg={avg_m:.2f}{rate}  reps={vals}",
                  ch="basic_hw")
            # Speculative-execution view: pfc[0]=INST_RETIRED, pfc[1]=INST_SPEC; their
            # difference = instructions executed speculatively but not retired (wrong-path).
            pf = pfc_log[idx]
            ret = sum(p[0] for p in pf) / len(pf)
            spec = sum(p[1] for p in pf) / len(pf)
            log.w(f"  input={idx} spec: retired={ret:.0f} inst_spec={spec:.0f} "
                  f"wrongpath={spec - ret:.1f}", ch="basic_hw")

        results = self._aggregate_htraces(len(inputs), n_reps, input_to_trace_list, pfc_log)
        return results, len(inputs) * [sandboxed_test_case]

    def _aggregate_htraces(self, n_inputs: int, n_reps: int,
                           input_to_trace_list: Dict[int, List[int]],
                           pfc_log: Dict[int, List]) -> List[HTrace]:
        results = []
        for idx in range(n_inputs):
            assert len(input_to_trace_list[idx]) == n_reps, \
                f"input {idx}: collected {len(input_to_trace_list[idx])} traces, expected {n_reps}"
            # ignored (priming) inputs run but are excluded from analysis: report a zero htrace
            trace_list = [0] * n_reps if idx in self.ignore_list else input_to_trace_list[idx]
            results.append(HTrace(trace_list=trace_list, perf_counters=np.array(pfc_log[idx])))
        return results

    def trace_test_case_with_taints(self, inputs: List[Input], nesting: int) -> Tuple[List[CTrace], List[InputTaint], List[ContractExecutionResult], List]:
        """
        Sandbox the test case, run CE per input, and return cache-set CTraces and input
        taints. No TC variants are produced here.

        :return: (ctraces, taints, ce_traces, [])  — tc_variants is always empty here
        """
        if not inputs or self.test_case is None:
            return [], [], [], []

        patched = copy.deepcopy(self.test_case)
        pass_on_test_case(patched, [Aarch64SandboxPass()])
        tc_bytes = self._assemble_tc(patched)[0]

        sandbox_base, _ = self.read_base_addresses()

        ct = ExecutionClause.SEQ
        bp = BranchPredictor.NONE
        for c in CONF.contract_execution_clause:
            if c not in EXECUTION_CLAUSE_MAP:
                raise ValueError(f"contract_execution_clause not supported on aarch64: {c!r}")
            bit, bp = EXECUTION_CLAUSE_MAP[c]
            ct |= bit
        if ct not in SUPPORTED_EXECUTION_CLAUSES:
            raise ValueError(f"unsupported execution-clause combination {ct!r} "
                             f"from {list(CONF.contract_execution_clause)}; "
                             f"supported: {sorted(SUPPORTED_EXECUTION_CLAUSES)}")
        nesting_depth = 0 if ct == ExecutionClause.SEQ else nesting

        traces: List[ContractExecutionResult] = []

        for inp in inputs:
            execution = self._make_ce_execution(tc_bytes, inp, sandbox_base, nesting_depth,
                                                 CONF.model_max_spec_window, ct, bp=bp)
            cer = self._contract_executor.run(execution)
            traces.append(cer)

        taints = [compute_taint(cer) for cer in traces]
        ctraces = [compute_ctrace(cer) for cer in traces]

        return ctraces, taints, traces, []


# ==================================================================================================
# Common non-interference base: generic CE-analysis and CE-trace table-printing helpers,
# shared by the PAC and MTE non-interference executors.
# ==================================================================================================
class Aarch64NonInterferenceExecutor(Aarch64LocalExecutor):
    """Shared base for the PAC/MTE non-interference executors: generic CE run, BB-map,
    and CE-trace comparison helpers — no variant-specific logic."""

    # (baseline, candidate) variant enums whose control-flow divergence is highlighted in
    # the CE-trace comparison. Each NI executor must set this.
    _compare_variants: Tuple[Enum, Enum]

    def _assemble_and_write_test_case(self, tc: TestCase) -> None:
        self.local_executor.checkout_region(TestCaseRegion())
        self.local_executor.write(self._assemble_tc(tc)[0])

    def _run_ce_on_tc(self, tc: TestCase, inp: Input, sandbox_base: int) -> Optional[List]:
        """Run CE on *tc* for *inp*; return list of trace entries or None on failure."""
        tc_bytes = self._assemble_tc(tc)[0]
        try:
            execution = self._make_ce_execution(tc_bytes, inp, sandbox_base, CONF.model_max_nesting,
                                                 CONF.model_max_spec_window,
                                                 ExecutionClause.COND)
            return list(self._contract_executor.run(execution))
        except RuntimeError as exc:
            FuzzLogger.get().w(f"  [CE ERROR] {exc}")
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
        Output goes to the FuzzLogger "pac_comparison" channel.
        """
        if FuzzLogger._VERBOSITY < 2:  # nothing is logged below this verbosity; skip the CE runs
            return

        def _run_ce_on_tc(tc: TestCase):
            tc_bytes = self._assemble_tc(tc)[0]
            try:
                execution = self._make_ce_execution(tc_bytes, inp, sandbox_base,
                                                     CONF.model_max_nesting,
                                                     CONF.model_max_spec_window,
                                                     ExecutionClause.COND)
                return list(self._contract_executor.run(execution))
            except RuntimeError as exc:
                FuzzLogger.get().w(f"  [CE ERROR] {exc}")
                return None

        def _fmt_regs(ite) -> str:
            srcs, _ = decode_reg_accesses(ite.cpu.encoding, ite.cpu.pc)
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

        # STAGE1 trace (from trace_test_case_with_taints) plus one column per variant.
        named_traces: List[Tuple[str, List, int]] = []  # (label, entries, code_base)
        by_variant: Dict[Enum, Tuple[List, int]] = {}

        s1_list = list(stage1_cer) if stage1_cer else []
        s1_base = s1_list[0].cpu.pc if s1_list else 0
        named_traces.append(("STAGE1", s1_list, s1_base))

        for variant, tc in tc_variants.items():
            cer = _run_ce_on_tc(tc) if tc is not None else None
            entries = cer if cer else []
            base = cer[0].cpu.pc if cer else 0
            named_traces.append((variant.name, entries, base))
            by_variant[variant] = (entries, base)

        labels   = [t[0] for t in named_traces]
        traces   = [t[1] for t in named_traces]
        bases    = [t[2] for t in named_traces]
        n_rows   = max((len(t) for t in traces), default=0)

        COL_W = 75
        header = f"  {'ROW':>4}  | " + " | ".join(f"{lbl:<{COL_W}}" for lbl in labels)
        sep    = "-" * len(header)

        # Divergence is marked between the baseline and candidate variants.
        baseline, candidate = self._compare_variants
        base_entries, base_code = by_variant[baseline]
        cand_entries, cand_code = by_variant[candidate]

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

            # Mark rows where the last variant's PC-offset diverges from the first.
            be = base_entries[row] if row < len(base_entries) else None
            ce = cand_entries[row] if row < len(cand_entries) else None
            if (be is None) != (ce is None):
                flag = "  <<<"
            elif be is not None and ce is not None:
                flag = "  <<<" if (be.cpu.pc - base_code) != (ce.cpu.pc - cand_code) else ""
            else:
                flag = ""

            out_lines.append(
                f"  {row:>4}  | " + " | ".join(f"{c:<{COL_W}}" for c in cells) + flag
            )

        log = FuzzLogger.get()
        for line in out_lines:
            log.w(line, ch="pac_comparison")
        log.ensure_flushed()

    def _log_pre_hw_ce_analysis(self, inputs: List[Input], sandbox_base: int, log) -> None:
        """Diagnostic pre-HW pass: run CE on every variant for every input and log the CE
        traces, BB maps, and AUTH instructions, flushing to disk before any HW runs."""
        auth_mnemonics = {'autia', 'autib', 'autda', 'autdb',
                          'autiza', 'autizb', 'autdza', 'autdzb'}
        stage1_traces = self._last_stage1_traces or ([None] * len(inputs))

        for inp_idx, (inp, variants) in enumerate(zip(inputs, self._last_tc_variants)):
            entry_x7 = int.from_bytes(inp.tobytes()[7 * 8: 8 * 8], "little")
            log.header(f"PRE-HW ANALYSIS  inp={inp_idx}")

            s1_cer = stage1_traces[inp_idx]
            if s1_cer:
                log_ce_trace(log, "STAGE1", inp_idx, list(s1_cer))

            variant_cers: Dict[Enum, Optional[List]] = {}
            for variant, tc in variants.items():
                vname = variant.name
                cer = self._run_ce_on_tc(tc, inp, sandbox_base)
                variant_cers[variant] = cer
                if cer is not None:
                    log_ce_trace(log, vname, inp_idx, cer)
                else:
                    log.w(f"  CE failed for {vname} inp={inp_idx}")

                sorted_pcs, bb_info, pc_to_id, code_base = self._compute_bb_map(cer or [])
                log_bb_map(log, vname, inp_idx, sorted_pcs, bb_info, pc_to_id, code_base, entry_x7)

                if cer:
                    log.header(f"AUTH INSTRUCTIONS IN CE: {vname}  inp={inp_idx}")
                    for ite in cer:
                        disas = disassemble_instruction(ite.cpu.encoding, ite.cpu.pc) or ""
                        mn = disas.split()[0].lower() if disas else ""
                        if mn in auth_mnemonics:
                            nest = ite.metadata.speculation_nesting
                            tag = "ARCH" if nest == 0 else f"SPEC({nest})"
                            log.wp(f"  {tag}  {disas}  pc=0x{ite.cpu.pc:016x}")

            if FuzzLogger._VERBOSITY >= 2:
                baseline, candidate = self._compare_variants
                base_cer, cand_cer = variant_cers[baseline], variant_cers[candidate]
                if base_cer and cand_cer and len(base_cer) != len(cand_cer):
                    log.wp(f"  *** {baseline.name} vs {candidate.name} trace length differs: "
                           f"{len(base_cer)} vs {len(cand_cer)} — control flow divergence!")
                self._print_ce_trace_comparison(inp_idx, inp, s1_cer, variants, sandbox_base)

        log.ensure_flushed()
        log.wp("[HW PHASE] All pre-HW logging complete — starting hardware execution")


# ==================================================================================================
# Non-interference executor (PAC / MTE)
# ==================================================================================================
class Aarch64PacNonInterferenceExecutor(Aarch64NonInterferenceExecutor):
    """Executor for PAC non-interference testing.

    Lifecycle:
    1. load_test_case(tc)  — seals the test case (PAC instrumentation), caches the result.
    2. trace_test_case_with_taints(inputs, nesting)  — for each input:
         a. Run the CE on the sealed TC to capture signed pointer values per fix point.
         b. Mint the engine's baseline() + a decoy instance to compare on hardware.
       Returns ctraces/taints derived from the sealed-TC CE traces.
    """

    _compare_variants = (NIVariant.BASELINE, NIVariant.DECOY)

    def __init__(self, generator: Aarch64Generator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._generator: Aarch64Generator = generator
        self._pac_instrumentation = PACInstrumentation(
            self._generator, CONF.pac_xpac_weight, CONF.pac_auth_weight)
        self._engine: SealedNIInstrumentation = self._pac_instrumentation.make_engine()

        self._stage1_tc: Optional[TestCase] = None
        self._stage1_fix_points: Optional[List[PACFixPoint]] = None
        self._stage1_tc_bytes: Optional[bytes] = None
        self._stage1_slot_offset_to_fp: Optional[Dict[int, PACFixPoint]] = None
        self._last_tc_variants: Optional[List[TCVariants]] = None
        self._last_stage1_traces: Optional[List[ContractExecutionResult]] = None

        # Decoy-signature machinery (kernel SIGN only — never AUTH; a failed AUTH at EL1 resets the
        # box on FEAT_FPAC hardware). The PAC-field mask is the set of top-16 bits kernel SIGN sets,
        # i.e. exactly the bits AUTH checks; cached per signing mnemonic.
        self._gpr_pool: List[str] = self._generator.target_desc.registers[64]
        self._pac_mask16_cache: Dict[str, int] = {}

    def load_test_case(self, test_case: TestCase):
        log = FuzzLogger.get()
        # Register channels on first use; subsequent calls are no-ops.
        log.register("pac_flow",  "pac/flow.log",   min_verbosity=1)
        log.register("pac_stage1","pac/stage1.log", min_verbosity=2)
        log.register("pac_hw",    "pac/hw.log",     min_verbosity=1)
        result = super().load_test_case(test_case)
        log_start_test_case(log, self._tc_counter)
        # Pin PAC keys so this process's pac_sign and the CE subprocess's auth share
        # the same key material (otherwise AUTH fails against correct_sig).
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
        stage1_tc, fix_points = self._pac_instrumentation.seal_test_case(patched)
        tc_bytes, layout = self._assemble_tc(stage1_tc)

        # Map byte offset of each XPAC placeholder → PACFixPoint (pre-slot state captured here)
        xpac_offset_to_fp: Dict[int, PACFixPoint] = {}
        for fp in fix_points:
            xpac_inst = fp.slot_insts[AUTH_SLOT_POS]
            xpac_offset_to_fp[layout.instruction_address[xpac_inst]] = fp

        self._stage1_tc = stage1_tc
        self._stage1_fix_points = fix_points
        self._stage1_tc_bytes = tc_bytes
        self._stage1_slot_offset_to_fp = xpac_offset_to_fp
        FuzzLogger.get().wp(f"  [STAGE1] fix_points={len(fix_points)}")

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
        # The engine fills the same sealed TC + fix points; per input the fix points are reset and
        # refilled from the CE trace below, so baseline()/decoys() reflect that input.
        self._engine.set_sealed(tc, fix_points)

        stage1_traces: List[ContractExecutionResult] = []
        tc_variants_per_input: List[TCVariants] = []

        xpac_offset_to_fp = self._stage1_slot_offset_to_fp

        log = FuzzLogger.get()
        log.register("pac_signing",    "pac/signing.log",    min_verbosity=1)
        log.register("pac_comparison", "pac/comparison.log", min_verbosity=2)
        log.register("pac_ni_table",   "pac/ni_table.log",   min_verbosity=1)

        for inp_idx, inp in enumerate(inputs):
            # Log input before CE so it's on disk even if CE crashes.
            log_input(log, inp_idx, inp, ch="pac_flow")
            log.ensure_flushed()

            for fp in fix_points:
                fp.reset()

            execution = self._make_ce_execution(tc_bytes, inp, sandbox_base, nesting,
                                                 CONF.model_max_spec_window,
                                                 ExecutionClause.COND)
            cer = self._contract_executor.run(execution)
            stage1_traces.append(cer)

            log_ce_trace(log, "STAGE1", inp_idx, list(cer))
            log.header(f"STAGE1 PAC-SIGN OPS  inp={inp_idx}")
            self._sign_reached_fixpoints(cer, xpac_offset_to_fp, log)
            self._fill_missing_alt_sigs(fix_points, size=6)

            log.header(f"STAGE1 SLOT CLASSIFICATION  inp={inp_idx}")
            for fp in fix_points:
                log_slot(log, inp_idx, fp)

            # Fix points now hold this input's signatures/nesting; mint the variants to compare.
            variants: TCVariants = {
                NIVariant.BASELINE: self._engine.baseline(random),
                NIVariant.DECOY:    next(self._engine.decoys(random)),
            }
            tc_variants_per_input.append(variants)

            # Log all variant TC binaries for this input immediately after minting them —
            # these are the exact binaries that will run on HW.
            log.header(f"VARIANT TC BINARIES  inp={inp_idx}")
            for variant, vtc in variants.items():
                log_tc_binary(log, variant.name, self._assemble_tc(vtc)[0])

            # DEBUG (env REVIZOR_PAC_NI_TABLE): per-instruction sealed/baseline/decoy table with the
            # CE arch/spec annotation, for eyeballing the seal result without any post-hoc parsing.
            if os.environ.get("REVIZOR_PAC_NI_TABLE"):
                # correct_sig per slot, read from the fix point (not the trace); annotate both the
                # AUT*/XPAC (PAC-op) row and its MOVK/NOP sig row, 4 bytes earlier.
                info_by_off: Dict[int, str] = {}
                for auth_off, fp in xpac_offset_to_fp.items():
                    sig = f"0x{fp.correct_sig:04x}" if fp.correct_sig is not None else "None"
                    info_by_off[auth_off] = info_by_off[auth_off - 4] = sig
                log_ni_table(log, inp_idx, [
                    ("sealed",   tc_bytes),
                    ("baseline", self._assemble_tc(variants[NIVariant.BASELINE])[0]),
                    ("decoy#1",  self._assemble_tc(variants[NIVariant.DECOY])[0]),
                    ("decoy#2",  self._assemble_tc(next(self._engine.decoys(random)))[0]),
                ], list(cer), "CorrectSig", info_by_off, ch="pac_ni_table")

            # Flush after every input's data so a crash between inputs
            # leaves a complete record for the inputs already processed.
            log.ensure_flushed()

        self._last_tc_variants = tc_variants_per_input
        self._last_stage1_traces = stage1_traces
        taints = [compute_taint(cer) for cer in stage1_traces]
        ctraces = [compute_ctrace(cer) for cer in stage1_traces]
        return ctraces, taints, stage1_traces, tc_variants_per_input

    def _sign_reached_fixpoints(self, cer, xpac_offset_to_fp, log) -> None:
        """Sign every fix point the CE trace reaches: record correct_sig (the committed signature)
        and a verified pool of wrong signatures, plus the minimum speculation depth. The
        architectural occurrence (depth 0) is authoritative and overrides an earlier speculative
        one. Uses only kernel SIGN — never AUTH (a failed AUTH at EL1 resets the box on FEAT_FPAC).
        """
        if len(cer) == 0:
            return
        code_base = cer[0].cpu.pc
        for ite in cer:
            fp = xpac_offset_to_fp.get(ite.cpu.pc - code_base)
            if fp is None:
                continue
            depth = ite.metadata.speculation_nesting
            if fp.spec_nesting is None or depth < fp.spec_nesting:
                fp.spec_nesting = depth
            if fp.correct_sig is not None and depth != 0:
                continue                       # already signed; keep it unless this is the arch hit

            pac_mn  = _AUTH_TO_PAC[fp.committed_inst.name.lower()]
            value_reg = fp.committed_inst.operands[0].value
            has_ctx = len(fp.committed_inst.operands) > 1
            ctx_reg = fp.committed_inst.operands[1].value if has_ctx else None

            ptr = _read_reg(ite.cpu, value_reg)
            ctx = _read_reg(ite.cpu, ctx_reg) if has_ctx else 0

            signed = self.local_executor.pac_sign(ptr, ctx, pac_mn)
            fp.correct_sig = (signed >> 48) & 0xFFFF
            log_pac_op(log, "SIGN", pac_mn, ptr, ctx, signed)
            log.w(f"    => slot={fp.slot_id}  correct_sig=0x{fp.correct_sig:04x}")

            # The from-set: every generatable register except the committed value register
            # (signing value_reg would just reproduce correct_sig).
            from_regs = [r for r in self._gpr_pool if r != value_reg]
            fp.alt_sigs = self._build_alt_sigs(ite.cpu, pac_mn, fp.correct_sig, ctx, from_regs,
                                               size=6, tries=64)
            log.w(f"    => slot={fp.slot_id}  alt_sigs={[f'0x{s:04x}' for s in fp.alt_sigs]}")

    def _build_alt_sigs(self, cpu, pac_mn, correct_sig, ctx, from_regs, size, tries) -> List[int]:
        """Distinct wrong top-16 signatures, each differing from correct_sig within the PAC-field
        bits so AUTH provably fails (a sig differing only in TBI-ignored bits would still
        authenticate). Candidates are signatures of live values from from_regs, or — less often — of
        a random value. Pure SIGN, never AUTH."""
        mask16 = self._pac_field_mask16(pac_mn, samples=64)
        pool: List[int] = []
        for _ in range(tries):
            if from_regs and random.random() < 0.85:
                value = _read_reg(cpu, random.choice(from_regs))
            else:
                value = random.randrange(1 << 64)
            cand = (self.local_executor.pac_sign(value, ctx, pac_mn) >> 48) & 0xFFFF
            # Flip only the PAC-field bits, keeping the genuine pointer's non-PAC bits (top byte,
            # sign): XPAC then canonicalizes the forgery to exactly the genuine pointer on the
            # speculative path, while AUTH still rejects it on hardware.
            sig = (correct_sig & ~mask16) | (cand & mask16)
            if sig != correct_sig and sig not in pool:
                pool.append(sig)
                if len(pool) >= size:
                    break
        if not pool:
            raise GeneratorException(
                f"no effective PAC forgery in {tries} tries (pac_field_mask=0x{mask16:04x}); "
                "PAC appears misconfigured (signing changes no checked bits)")
        return pool

    def _pac_field_mask16(self, pac_mn, samples) -> int:
        """Return the top-16-bit mask of the PAC field — the pointer bits an AUTH actually checks —
        used to decide whether a candidate decoy signature really fails authentication.

        WHAT: sign `samples` clean low-half addresses (top 16 bits zero, so the PAC field starts
        empty) and OR together `signed ^ preimage`. The set bits are exactly the bits SIGN wrote —
        the PAC field — viewed in the top-16 window (`>> 48`). Cached per signing mnemonic.

        CORRECTNESS: SIGN inserts the code into the PAC field and AUTH recomputes and compares
        against that *same* field — architecturally the same bit positions, so "bits SIGN sets" ==
        "bits AUTH checks". SIGN never touches the address, the sign bit (55) or (under TBI) the top
        byte, and the clean pre-image leaves those zero, so they never enter the mask. We use only
        SIGN+XOR — never AUTH (a failed AUTH at EL1 resets the box) and never XPAC (its top-byte
        handling is TBI-dependent).

        SOUNDNESS: the result is always a *subset* of the true PAC field (over-counting is
        impossible by the above). A decoy is accepted only if it differs from the correct signature
        within this mask, and every masked bit is a genuine PAC bit, so an accepted decoy provably
        fails AUTH on this hardware. Under-counting can only reject valid decoys (conservative),
        never admit an ineffective one.

        UNDER-COUNT PROBABILITY: a PAC bit is ~50/50 across random signatures, so a given field bit
        is missed only if it is 0 in all `samples` of them ≈ 2**-samples; across the ~7 field bits
        the chance of missing any is <= 7 * 2**-samples (samples=64 -> ~4e-19). The assertion below
        also catches a grossly wrong mask (empty, or including the sign bit)."""
        cached = self._pac_mask16_cache.get(pac_mn)
        if cached is not None:
            return cached
        mask = 0
        for _ in range(samples):
            v = random.randrange(1 << 48)            # clean low-half address: top 16 bits are zero
            mask |= self.local_executor.pac_sign(v, random.randrange(1 << 64), pac_mn) ^ v
        mask16 = (mask >> 48) & 0xFFFF
        # The field is a non-empty run ending at bit 54; it never includes bit 55 (the sign bit, 0x80).
        assert mask16 and not (mask16 & 0x80), f"implausible PAC-field mask 0x{mask16:04x}"
        self._pac_mask16_cache[pac_mn] = mask16
        return mask16

    def _fill_missing_alt_sigs(self, fix_points, size) -> None:
        """Give each slot the CE never reached (no correct_sig) a pool of random wrong signatures.
        They are not PAC-mask-verified — there is no committed signature to differ from — but an
        unreached slot never executes on the contract path, and on hardware a forged AUTH there
        faults at EL0 (sandboxed). Pure SIGN; never AUTH."""
        for fp in fix_points:
            if fp.alt_sigs:
                continue
            pac_mn = _AUTH_TO_PAC[fp.committed_inst.name.lower()]
            sigs: Set[int] = set()
            while len(sigs) < size:
                sigs.add((self.local_executor.pac_sign(
                    random.randrange(1 << 64), random.randrange(1 << 64), pac_mn) >> 48) & 0xFFFF)
            fp.alt_sigs = list(sigs)

    def trace_test_case_variants_hw(
        self,
        inputs: List[Input],
        n_reps: int,
    ) -> List[Dict[NIVariant, HTrace]]:
        """Run all TC variants for each input on hardware with stage-1-compatible mistraining.

        For each input[i] and each variant in _last_tc_variants[i], executes the variant TC
        on hardware using i._arch_trace for BPU mistraining.  All variants share the same
        branch layout as stage-1, so the same arch_trace applies to all of them.

        Returns per-input: {NIVariant → HTrace}.
        """
        assert self._last_tc_variants is not None and len(self._last_tc_variants) == len(inputs), \
            "trace_test_case_with_taints() must be called before trace_test_case_variants_hw()"

        STAT.executor_reruns += n_reps * len(inputs)

        sandbox_base, _ = self.read_base_addresses()
        log = FuzzLogger.get()

        # Phase 1 is diagnostic only: when logging is enabled, run CE on every variant and
        # record traces / BB maps / AUTH ops BEFORE any HW runs, so a Phase-2 machine crash
        # still leaves a complete pre-crash record on disk.
        if FuzzLogger._VERBOSITY >= 1:
            self._log_pre_hw_ce_analysis(inputs, sandbox_base, log)

        # Phase 2: hardware execution.
        counter_log = defaultdict(lambda: defaultdict(list))
        expected_log = {}

        results: List[Dict[NIVariant, HTrace]] = []
        for inp_idx, (inp, variants) in enumerate(zip(inputs, self._last_tc_variants)):
            per_variant: Dict[NIVariant, HTrace] = {}
            self.local_executor.discard_all_inputs()
            iid = self.local_executor.allocate_iid()
            self.local_executor.checkout_region(InputRegion(iid))
            self.local_executor.write(ExecutorMemory(_input_bytes_with_pstate(inp)))

            # Mistraining depends only on the input's arch trace (shared by all variants and
            # reps), and the kernel re-applies the saved config on every trace, so set it once.
            entries = self.branch_mistraining_entries(getattr(inp, "_arch_trace", None))
            self.apply_branch_mistraining(entries)
            expected_log[inp_idx] = len(entries)
            if entries:
                log_mistraining(log, self._tc_counter, inp_idx, entries, inp._arch_trace)

            for variant, tc in variants.items():
                log.wp(f"[HW] inp={inp_idx} variant={variant.name}")
                self._assemble_and_write_test_case(tc)

                trace_list = []
                for rep in range(n_reps):
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
class Aarch64MteNonInterferenceExecutor(Aarch64NonInterferenceExecutor):
    """
    Executor for MTE non-interference testing.

    Lifecycle:
    1. load_test_case(tc)  — seals the test case (MTE instrumentation), caches the result.
    2. trace_test_case_with_taints(inputs, nesting)  — for each input:
         a. Run the sealed TC through CE (CONTRACT_ALWAYS_MISPREDICT) to discover which
            memory-access slots have spec_nesting > 0 (speculative) vs 0 (architectural).
            Uses arch_seen semantics: once a slot fires at nesting==0 it is permanently arch.
         b. Mint the engine's baseline() (correct tag everywhere) + a decoy instance (speculative
            slots retagged via IRG / MOVK) to compare on hardware.
    """

    _compare_variants = (NIVariant.BASELINE, NIVariant.DECOY)

    def __init__(self, generator: Aarch64Generator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._generator: Aarch64Generator = generator
        self._mte = make_seal_pass(generator, {"mte"})
        # The sandbox clamp is part of the seal now but must never be decoyed (always in-bounds);
        # the tag is decoyed on speculative slots.
        self._engine: SealedNIInstrumentation = self._mte.make_engine(
            should_decoy=lambda fp, seal: seal.name != "sandbox" and is_speculative(fp))

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
        log = FuzzLogger.get()
        log.register("mte_flow",   "mte/flow.log",   min_verbosity=1)
        log.register("mte_stage1", "mte/stage1.log", min_verbosity=2)
        log.register("mte_hw",     "mte/hw.log",     min_verbosity=1)
        log.register("mte_ni_table", "mte/ni_table.log", min_verbosity=1)
        log.wp(f"[MTE] test case #{self._tc_counter}")

        patched = copy.deepcopy(self.test_case)
        stage1_tc, fix_points = self._mte.seal_test_case(patched)
        tc_bytes, layout = self._assemble_tc(stage1_tc)

        # The tag placeholder (last slot inst) sits immediately before its memory access, whatever
        # the seal — so: mem_access is at tag_offset + 4.
        mem_access_offset_to_fp: Dict[int, MTEFixPoint] = {}
        for fp in fix_points:
            tag_offset = layout.instruction_address[fp.slot_insts[-1]]
            mem_access_offset_to_fp[tag_offset + 4] = fp

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

        self._engine.set_sealed(tc, fix_points)

        # Build sandboxed TC once — CE on this produces branch offsets matching
        # what trace_test_case() loads into the kernel module for mistraining.
        sandboxed_patched = copy.deepcopy(self.test_case)
        pass_on_test_case(sandboxed_patched, [Aarch64SandboxPass()])
        sandboxed_tc_bytes = self._assemble_tc(sandboxed_patched)[0]
        sandboxed_traces: List[ContractExecutionResult] = []

        log = FuzzLogger.get()
        for inp_idx, inp in enumerate(inputs):
            mte_inp_ch = log.register(f"mte_inp_{inp_idx:03d}",
                                      f"mte/inputs/inp_{inp_idx:03d}.log",
                                      min_verbosity=2)
            if mte_inp_ch:
                log.use(mte_inp_ch)
                log_input(log, inp_idx, inp, ch=mte_inp_ch)

            for fp in fix_points:
                fp.reset()

            # ALWAYS_MISPREDICT so the CE reveals speculative paths (NI relies on this).
            execution = self._make_ce_execution(tc_bytes, inp, sandbox_base, nesting,
                                                CONF.model_max_spec_window,
                                                ExecutionClause.COND)
            cer = self._contract_executor.run(execution)
            traces.append(cer)

            # Run CE on the sandboxed TC for mistraining-compatible branch offsets.
            sandboxed_execution = self._make_ce_execution(sandboxed_tc_bytes, inp, sandbox_base,
                                                          nesting, CONF.model_max_spec_window,
                                                          ExecutionClause.COND)
            sandboxed_cer = self._contract_executor.run(sandboxed_execution)
            sandboxed_traces.append(sandboxed_cer)

            self._classify_mte_slots(cer, offset_to_fp, (sandbox_base >> 56) & 0xF)
            # Fix points now hold this input's per-slot spec_nesting; mint the variants to compare.
            variants = {
                NIVariant.BASELINE: self._engine.baseline(random),
                NIVariant.DECOY:    next(self._engine.decoys(random)),
            }
            tc_variants_per_input.append(variants)

            # DEBUG (env REVIZOR_MTE_NI_TABLE): per-instruction sealed/baseline/decoy table with the
            # CE arch/spec annotation and each slot's committed tag (from fp.correct_tag, not the
            # trace), on the tag-op row (4 bytes before its memory access).
            if os.environ.get("REVIZOR_MTE_NI_TABLE"):
                info_by_off: Dict[int, str] = {}
                for mem_off, fp in offset_to_fp.items():
                    info_by_off[mem_off - 4] = (f"0x{fp.correct_tag:x}"
                                                if fp.correct_tag is not None else "None")
                log_ni_table(log, inp_idx, [
                    ("sealed",   tc_bytes),
                    ("baseline", self._assemble_tc(variants[NIVariant.BASELINE])[0]),
                    ("decoy#1",  self._assemble_tc(variants[NIVariant.DECOY])[0]),
                    ("decoy#2",  self._assemble_tc(next(self._engine.decoys(random)))[0]),
                ], list(cer), "CorrectTag", info_by_off, ch="mte_ni_table")

        # Use sandboxed-TC traces for ctrace/taint/mistraining — their branch offsets
        # match the sandboxed TC that trace_test_case() loads into the kernel module.
        taints = [compute_taint(cer) for cer in sandboxed_traces]
        ctraces = [compute_ctrace(cer) for cer in sandboxed_traces]

        return ctraces, taints, sandboxed_traces, tc_variants_per_input

    @staticmethod
    def _classify_mte_slots(cer, offset_to_fp, default_tag: int = 0) -> None:
        """From the sealed-TC trace, record per-slot spec_nesting and correct_tag (in place).
        spec_nesting: arch_seen — a slot seen at nesting 0 is permanently architectural.
        correct_tag: the allocation tag of the accessed cell, tracked by MteTagState (architectural
        tag stores commit; speculative ones are ignored)."""
        if len(cer) == 0:
            return
        code_base = cer[0].cpu.pc
        arch_seen: Set[int] = set()
        tags = MteTagState(default_tag)
        for ite in cer:
            nest = ite.metadata.speculation_nesting
            tags.to_depth(nest)             # track depth: deeper copies, unwind pops (reverts)
            store = mte_tag_store_effect(ite)
            if store is not None:
                tags.set(*store)            # at the current depth (arch at 0, else speculative)
            if not ite.metadata.has_memory_access:
                continue
            fp = offset_to_fp.get(ite.cpu.pc - code_base)
            if fp is None:
                continue
            ea = ite.metadata.memory_access.effective_address
            cell_tag, ptr_tag = tags.tag_at(ea), (ea >> 56) & 0xF
            if nest == 0:
                fp.spec_nesting = 0
                fp.correct_tag, fp.ptr_tag = cell_tag, ptr_tag
                arch_seen.add(fp.slot_id)
            elif fp.slot_id not in arch_seen and fp.spec_nesting is None:
                fp.spec_nesting = int(nest)
                if fp.correct_tag is None:
                    fp.correct_tag, fp.ptr_tag = cell_tag, ptr_tag
