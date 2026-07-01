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
from .aarch64_pac import PacSigner
from .aarch64_sealer import make_sealer, SealedTestCase, ResolvedSealingTestCase
from .aarch64_contract_executor import (ContractExecution, ContractExecutorService,
                                        ExecutionClause, SUPPORTED_EXECUTION_CLAUSES,
                                        BranchPredictor, EXECUTION_CLAUSE_MAP, SimArch)
from .aarch64_disasm import disassemble_instruction, decode_reg_accesses, is_conditional_branch
from .aarch64_trace import compute_ctrace, compute_taint, ContractExecutionResult
from .aarch64_input_layout import _input_bytes_with_pstate, REGISTER_REGION_OFFSET
from .aarch64_input_wire import serialize_input, MTE_TAG_COUNT
from .aarch64_mte import MTE_INITIAL_DEFAULT_TAG
from .aarch64_log import (log_start_test_case, log_input, log_ce_trace, log_bb_map,
                          log_tc_binary, log_mistraining, log_ni_table)

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
                           bp: BranchPredictor = BranchPredictor.NONE,
                           mte_tags: Optional[List[int]] = None) -> ContractExecution:
        tc_memory, tc_regs = _ce_memory_regs(inp)
        return ContractExecution(
            tc_bytes, tc_memory, tc_regs, SimArch.RVZR_ARCH_AARCH64,
            nesting,
            max_mispred_instructions,  # CE per-window instruction cap
            req_mem_base_virt=sandbox_base,
            execution_clauses=ct,
            branch_predictor=bp,
            mte_tags=mte_tags,
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

        payloads = [ExecutorMemory(serialize_input(i)) for i in inputs]
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
    """The non-interference executor for whatever primitives the instruction categories enable
    (PAC and/or MTE). It seals the test case with the active value-seals — the memory pass composes
    [Sandbox(, PacSign)(, MteTag)], and PAC also seals AUT* sites — then per input runs the contract
    executor once, fills each fix point's data (PAC: sign; MTE: classify tags), and compares the
    baseline vs decoy TC variants on hardware. Adding a primitive means adding its seal + its fill
    step here; the orchestration is primitive-agnostic. (Below: generic CE/BB-map/comparison helpers,
    then the lifecycle.)"""

    # (baseline, candidate) variant enums whose control-flow divergence is highlighted in the
    # CE-trace comparison.
    _compare_variants: Tuple[Enum, Enum] = (NIVariant.BASELINE, NIVariant.DECOY)

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
        sealed_cer: ContractExecutionResult,
        tc_variants: Dict,
        sandbox_base: int,
    ) -> None:
        """Print a side-by-side CE trace table: sealed vs TC1/TC2/TC3.

        Columns: SEALED | TC1 STRIP | TC2 CORRECT | TC3 WRONG
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

        # SEALED trace (from trace_test_case_with_taints) plus one column per variant.
        named_traces: List[Tuple[str, List, int]] = []  # (label, entries, code_base)
        by_variant: Dict[Enum, Tuple[List, int]] = {}

        s1_list = list(sealed_cer) if sealed_cer else []
        s1_base = s1_list[0].cpu.pc if s1_list else 0
        named_traces.append(("SEALED", s1_list, s1_base))

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
        ce_traces = self._last_ce_traces or ([None] * len(inputs))

        for inp_idx, (inp, variants) in enumerate(zip(inputs, self._last_tc_variants)):
            entry_x7 = int.from_bytes(inp.tobytes()[7 * 8: 8 * 8], "little")
            log.header(f"PRE-HW ANALYSIS  inp={inp_idx}")

            s1_cer = ce_traces[inp_idx]
            if s1_cer:
                log_ce_trace(log, "SEALED", inp_idx, list(s1_cer))

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

    # ==============================================================================================
    # Lifecycle: detect active primitives, seal, fill per input, compare variants on hardware
    def __init__(self, generator: Aarch64Generator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._generator: Aarch64Generator = generator
        cats = CONF.instruction_categories or []
        self._primitives: Set[str] = set()
        if any(c.startswith("PAC") for c in cats):
            self._primitives.add("pac")
        if any(c.startswith("MTE") for c in cats):
            self._primitives.add("mte")
        if not self._primitives:
            raise GeneratorException(
                "non-interference fuzzing needs a PAC or MTE instruction category enabled")

        # PAC signing capability (kernel SIGN only — never AUTH; a failed AUTH at EL1 resets the box).
        signer = PacSigner(self.local_executor.pac_sign) if "pac" in self._primitives else None

        # The sealer owns all sealing + resolution; it traces via our _seal_trace (assemble + CE run).
        self._sealer = make_sealer(generator, self._seal_trace, self._primitives, signer)

        # Per-test-case sealed placeholder, populated by load_test_case(); the CE-trace context the
        # sealer's resolve uses (captured per with_taints call).
        self._sealed: Optional[SealedTestCase] = None
        self._sandbox_base: Optional[int] = None
        self._nesting: int = CONF.model_max_nesting
        self._last_tc_variants: Optional[List[TCVariants]] = None
        self._last_ce_traces: Optional[List[ContractExecutionResult]] = None

    def _mte_tags_for(self, inp: Input) -> Optional[List[int]]:
        """Uniform initial tags the sandbox is loaded with when MTE is active (else None — leave tag
        memory untouched). The kernel HW, the CE tag memory, and the model's MteTagState all seed
        from this same uniform tag."""
        if "mte" not in self._primitives:
            return None
        return [MTE_INITIAL_DEFAULT_TAG] * MTE_TAG_COUNT

    def _seal_trace(self, tc: TestCase, inp: Input) -> ContractExecutionResult:
        """One CE trace of `tc` for `inp` — the trace_fn the sealer resolves over. Owns assembly +
        the CE setup so the sealer never sees the assembler."""
        assert self._sandbox_base is not None, \
            "sandbox base not resolved — read_base_addresses() must run before any seal trace"
        tc_bytes, _ = self._assemble_tc(tc)
        return self._contract_executor.run(self._make_ce_execution(
            tc_bytes, inp, self._sandbox_base, self._nesting, CONF.model_max_spec_window,
            ExecutionClause.COND, mte_tags=self._mte_tags_for(inp)))

    # ------------------------------------------------------------------ lifecycle
    def load_test_case(self, test_case: TestCase):
        log = FuzzLogger.get()
        log.register("ni_flow",     "ni/flow.log",     min_verbosity=1)
        log.register("pac_signing", "ni/signing.log",  min_verbosity=1)  # log_pac_op/log_slot channel
        log.register("ni_table",    "ni/ni_table.log", min_verbosity=1)
        log.register("pac_hw",      "ni/hw.log",       min_verbosity=1)
        result = super().load_test_case(test_case)
        log_start_test_case(log, self._tc_counter)
        if "pac" in self._primitives:
            # Pin PAC keys so this process's pac_sign and the CE subprocess's auth share the same
            # key material (else AUTH fails against correct_sig).
            self.local_executor.set_pac_keys(self.local_executor.get_pac_keys())
        self._sealed = self._sealer.seal(self.test_case)
        log.ensure_flushed()
        return result

    def trace_test_case_with_taints(
        self,
        inputs: List[Input],
        nesting: int,
    ) -> Tuple[List[CTrace], List[InputTaint], List[ContractExecutionResult], List[TCVariants]]:
        if not inputs or self.test_case is None:
            return [], [], [], []
        assert self._sealed is not None, "sealed TC not built — load_test_case() not called"

        self._nesting = nesting
        self._sandbox_base, _ = self.read_base_addresses()

        log = FuzzLogger.get()
        ce_traces: List[ContractExecutionResult] = []
        tc_variants_per_input: List[TCVariants] = []

        for inp_idx, inp in enumerate(inputs):
            log_input(log, inp_idx, inp, ch="ni_flow")
            log.ensure_flushed()

            # Resolve the placeholder for this input, then mint genuine ONCE (it carries the random
            # XPAC strip). taint/ctrace are the GENUINE baseline's CE trace — the TC that actually runs
            # on hardware — NOT the placeholder, whose slots behave differently.
            resolved = self._sealed.resolve(inp)
            genuine = resolved.genuine()
            cer = self._seal_trace(genuine, inp)
            ce_traces.append(cer)

            variants = self._mint_variants(resolved, genuine)
            tc_variants_per_input.append(variants)

            log_ce_trace(log, "BASELINE", inp_idx, list(cer))
            log.header(f"VARIANT TC BINARIES  inp={inp_idx}")
            for variant, vtc in variants.items():
                log_tc_binary(log, variant.name, self._assemble_tc(vtc)[0])
            self._maybe_log_ni_table(log, inp_idx, cer, variants)
            log.ensure_flushed()

        self._last_tc_variants = tc_variants_per_input
        self._last_ce_traces = ce_traces
        taints = [compute_taint(cer) for cer in ce_traces]
        ctraces = [compute_ctrace(cer) for cer in ce_traces]
        return ctraces, taints, ce_traces, tc_variants_per_input

    def _mint_variants(self, resolved: ResolvedSealingTestCase, genuine: TestCase) -> TCVariants:
        """The TC variants to run on hardware for the resolved input. NI compares the already-minted
        genuine baseline against a fresh decoy; the regular subclass overrides this."""
        return {NIVariant.BASELINE: genuine, NIVariant.DECOY: resolved.decoy()}

    def _maybe_log_ni_table(self, log, inp_idx, cer, variants) -> None:
        """DEBUG (env REVIZOR_NI_TABLE): per-variant byte columns for the sealed/baseline/decoy table.
        The per-slot Sig/Tag annotation is dropped — it read the old fix points, which no longer exist
        (the resolved values live inside the sealer's ResolvedSealingTestCase)."""
        if not os.environ.get("REVIZOR_NI_TABLE"):
            return
        log_ni_table(log, inp_idx, self._ni_table_columns(variants),
                     list(cer), "", {}, ch="ni_table")

    def _ni_table_columns(self, variants):
        """(name, bytes) columns for the debug table: this input's baseline and decoy."""
        return [(variant.name.lower(), self._assemble_tc(tc)[0])
                for variant, tc in variants.items()]


    # ------------------------------------------------------------------ hardware: compare variants
    def trace_test_case_variants_hw(
        self,
        inputs: List[Input],
        n_reps: int,
    ) -> List[Dict[NIVariant, HTrace]]:
        """Run each input's baseline + decoy variants on hardware with the input's arch-trace
        mistraining (shared by all variants — same branch layout). Returns per-input {variant: HTrace}."""
        assert self._last_tc_variants is not None and len(self._last_tc_variants) == len(inputs), \
            "trace_test_case_with_taints() must be called before trace_test_case_variants_hw()"
        STAT.executor_reruns += n_reps * len(inputs)
        sandbox_base, _ = self.read_base_addresses()
        log = FuzzLogger.get()
        if FuzzLogger._VERBOSITY >= 1:
            self._log_pre_hw_ce_analysis(inputs, sandbox_base, log)

        counter_log = defaultdict(lambda: defaultdict(list))
        expected_log: Dict[int, int] = {}
        results: List[Dict[NIVariant, HTrace]] = []
        for inp_idx, (inp, variants) in enumerate(zip(inputs, self._last_tc_variants)):
            per_variant: Dict[NIVariant, HTrace] = {}
            self.local_executor.discard_all_inputs()
            iid = self.local_executor.allocate_iid()
            self.local_executor.checkout_region(InputRegion(iid))
            self.local_executor.write(ExecutorMemory(serialize_input(inp, mte_tags=self._mte_tags_for(inp))))

            entries = self.branch_mistraining_entries(getattr(inp, "_arch_trace", None))
            self.apply_branch_mistraining(entries)
            expected_log[inp_idx] = len(entries)
            if entries:
                log_mistraining(log, self._tc_counter, inp_idx, entries, inp._arch_trace)

            for variant, tc in variants.items():
                log.wp(f"[HW] inp={inp_idx} variant={variant.name}")
                self._assemble_and_write_test_case(tc)
                trace_list = []
                for _ in range(n_reps):
                    self.local_executor.trace()
                    self.local_executor.checkout_region(InputRegion(iid))
                    hwm = self.local_executor.hardware_measurement()
                    trace_list.append(hwm.htrace)
                    counter_log[variant][inp_idx].append(hwm.pfcs[2])
                ht = HTrace(trace_list=trace_list)
                per_variant[variant] = ht
                log.w(f"  htrace=0x{ht.raw[0]:016x}  n_reps={len(trace_list)}", ch="pac_hw")
            results.append(per_variant)
        return results


class Aarch64RegularSealedExecutor(Aarch64NonInterferenceExecutor):
    """Regular fuzzing with PAC/MTE support.

    Reuses the NI seal/resolve pipeline. Inputs are grouped into *sealing classes* — identical
    resolved (correct_sig, correct_tag, spec_nesting) across all fix points — and each class runs ONE
    sealed TC, minted once (with the real rng) and shared by every member. The form is a per-class
    coin flip:
      - decoy: genuine on every architectural slot, decoyed (wrong signature / retag) on the
        speculative slots — exposes the speculative PAC/MTE leak.
      - genuine: genuine everywhere.
    Either way the architectural slots are always correct, so the TC is arch-safe (no failed AUTH at
    EL1, no tag fault), and the representative's arch AUT*/tag match every member's pointer.

    Sharing one TC per class gives, for free, a consistent decoy decision at each slot across all the
    class's inputs (they run the identical program); an input never runs another class's TC. The seal
    slots are register-only, so sealing differences between classes never enter the cache htrace,
    leaving the standard regular by-contract-trace leak comparison valid. (A contract trace fixes the
    accessed addresses, hence the pointers, hence the signatures/tags — so a contract-trace class is
    contained in a sealing class and likewise shares one TC.)"""

    _DECOY_PROB = 0.5   # per sealing class: probability of the speculative-decoy form vs all-genuine

    def __init__(self, generator: Aarch64Generator, *args, **kwargs):
        super().__init__(generator, *args, **kwargs)
        self._class_tc: Dict[tuple, TCVariants] = {}    # sealing key -> the class's one shared TC
        self._tc_by_input: Dict[int, TCVariants] = {}   # id(input) -> its TC (from the last resolve)

    def load_test_case(self, test_case: TestCase):
        self._class_tc = {}                             # fresh per-class TCs for each test case
        self._tc_by_input = {}
        super().load_test_case(test_case)

    def _mint_variants(self, resolved: ResolvedSealingTestCase, genuine: TestCase) -> TCVariants:
        """The TC for the resolved input's sealing class — minted once per class with a real-random
        per-class coin (decoy vs all-genuine) and shared by every member, so caveat-4 merges (Ti ==
        Tt) fall out automatically. Reuses the already-minted `genuine` for the BASELINE form."""
        key = resolved.collapse_key
        if key not in self._class_tc:
            if random.random() < self._DECOY_PROB:
                self._class_tc[key] = {NIVariant.DECOY: resolved.decoy()}
            else:
                self._class_tc[key] = {NIVariant.BASELINE: genuine}
        return self._class_tc[key]

    def trace_test_case_with_taints(self, inputs, nesting):
        result = super().trace_test_case_with_taints(inputs, nesting)
        # remember each input's TC so trace_test_case can find it without re-resolving
        self._tc_by_input = {id(inp): var for inp, var in zip(inputs, self._last_tc_variants)}
        return result

    def _resolve_input_tc(self, inp: Input) -> TCVariants:
        """Resolve one input over the placeholder sealed TC and return its sealing class's TC. Used
        by trace_test_case for inputs the last with_taints pass didn't cover (slow-path priming,
        reuse_ctraces). genuine()'s arch slots are always correct (never a forged/failed AUTH)."""
        resolved = self._sealed.resolve(inp)
        return self._mint_variants(resolved, resolved.genuine())

    def trace_test_case(self, inputs: List[Input], n_reps: int) -> Tuple[List[HTrace], List[TestCase]]:
        """Run each input on its sealing class's one shared TC. Every input maps to its TC here (from
        the with_taints cache, else re-resolved), so this is correct for any input list the slow path
        passes (priming subsets, reuse_ctraces, larger samples) and never runs a wrong-class TC."""
        if not inputs or self.test_case is None:
            return [], []
        self._sandbox_base, _ = self.read_base_addresses()
        STAT.executor_reruns += n_reps * len(inputs)

        # map each input -> its class TC, then group inputs sharing a TC (one TC written per class)
        per_input_tcs: List[TestCase] = []
        classes: Dict[int, List[int]] = defaultdict(list)
        tc_by_id: Dict[int, TestCase] = {}
        for i, inp in enumerate(inputs):
            var = self._tc_by_input.get(id(inp)) or self._resolve_input_tc(inp)
            tc = next(iter(var.values()))
            per_input_tcs.append(tc)
            classes[id(tc)].append(i)
            tc_by_id[id(tc)] = tc

        input_to_trace_list: Dict[int, List[int]] = defaultdict(list)
        pfc_log: Dict[int, List] = defaultdict(list)
        for tcid, idxs in classes.items():
            self._assemble_and_write_test_case(tc_by_id[tcid])
            for i in idxs:
                inp = inputs[i]
                self.local_executor.discard_all_inputs()
                iid = self.local_executor.allocate_iid()
                self.local_executor.checkout_region(InputRegion(iid))
                self.local_executor.write(ExecutorMemory(serialize_input(inp, mte_tags=self._mte_tags_for(inp))))
                self.apply_branch_mistraining(
                    self.branch_mistraining_entries(getattr(inp, "_arch_trace", None)))
                for _ in range(n_reps):
                    self.local_executor.trace()
                    self.local_executor.checkout_region(InputRegion(iid))
                    hwm = self.local_executor.hardware_measurement()
                    input_to_trace_list[i].append(hwm.htrace)
                    pfc_log[i].append(hwm.pfcs)
        htraces = self._aggregate_htraces(len(inputs), n_reps, input_to_trace_list, pfc_log)
        return htraces, per_input_tcs
