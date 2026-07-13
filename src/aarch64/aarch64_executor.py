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
from typing import List, Tuple, Set, Optional, Dict, Sequence
from collections import defaultdict
from pathlib import Path


from .aarch64_generator import Aarch64Generator
from .aarch64_printer import Aarch64Printer, Aarch64ASMLayout
from ..interfaces import (HTrace, Input, TestCase, Executor, HardwareTracingError, CTrace,
                          InputTaint, GeneratorException, Measurement)
from ..config import CONF
from ..util import Logger, STAT, FuzzLogger
from .aarch64_target_desc import Aarch64TargetDesc
from .aarch64_kernel import LocalHWExecutor, TestCaseRegion, InputRegion, ExecutorMemory, PacKeys
from .aarch64_generator import Pass, Aarch64SandboxPass
from .seal.pac import PacSigner
from .seal.sealer import make_sealer, SealedTestCase, ResolvedSealingTestCase
from .aarch64_relocations import apply_relocations, Relocation
from .aarch64_contract_executor import (ContractExecution, ContractExecutorService,
                                        ExecutionClause, SUPPORTED_EXECUTION_CLAUSES,
                                        BranchPredictor, EXECUTION_CLAUSE_MAP, SimArch)
from .aarch64_disasm import disassemble_instruction, is_conditional_branch
from .aarch64_trace import compute_ctrace, compute_taint, ContractExecutionResult
from .aarch64_input_layout import _input_bytes_with_pstate, REGISTER_REGION_OFFSET
from .aarch64_executor_input_encoder import ExecutorInput, MTE_TAG_COUNT
from .aarch64_mte import MTE_INITIAL_DEFAULT_TAG
from .aarch64_log import (log_start_test_case, log_input, log_ce_trace, log_bb_map,
                          log_tc_binary, log_ni_table)

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
# NON-INTERFERENCE: per-input variants, keyed by name (variant name -> reloc plan)
# ==================================================================================================


class NIVariant:
    """Per-input variant names compared on hardware. `DECOY` is a template; `decoy_n(i)` names the nth
    decoy."""
    BASELINE = "baseline"
    DECOY = "decoy{i}"

    @staticmethod
    def decoy_n(i: int) -> str:
        return NIVariant.DECOY.format(i=i)


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
        # WIP, off by default (CONF.enable_branch_mistraining) pending hardware efficacy confirmation.
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
                           mte_tags: Optional[List[int]] = None,
                           pac_keys: Optional[PacKeys] = None) -> ContractExecution:
        tc_memory, tc_regs = _ce_memory_regs(inp)
        return ContractExecution(
            tc_bytes, tc_memory, tc_regs, SimArch.RVZR_ARCH_AARCH64,
            nesting,
            max_mispred_instructions,  # CE per-window instruction cap
            req_mem_base_virt=sandbox_base,
            execution_clauses=ct,
            branch_predictor=bp,
            mte_tags=mte_tags,
            pac_keys=pac_keys,
        )

    def _batch_trace(self, payloads: List[ExecutorMemory], n_reps: int) -> List[HTrace]:
        """Stage all `payloads` into one trace (each carries its own BPU-training section), repeat
        n_reps times, and aggregate one HTrace per payload."""
        log = FuzzLogger.get()
        input_to_trace_list = defaultdict(list)
        counter_log = defaultdict(list)
        pfc_log = defaultdict(list)

        self.local_executor.discard_all_inputs()
        iids = []
        for payload in payloads:
            iid = self.local_executor.allocate_iid()
            self.local_executor.checkout_region(InputRegion(iid))
            self.local_executor.write(payload)
            iids.append(iid)
        for _ in range(n_reps):
            self.local_executor.trace()
            for idx, iid in enumerate(iids):
                self.local_executor.checkout_region(InputRegion(iid))
                hwm = self.local_executor.hardware_measurement()
                input_to_trace_list[idx].append(hwm.htrace)
                pfc_log[idx].append(hwm.pfcs)
                counter_log[idx].append(hwm.pfcs[2])

        for idx, vals in counter_log.items():
            log.w(f"  input={idx} mispredicts: avg={sum(vals) / len(vals):.2f}  reps={vals}",
                  ch="basic_hw")
            pf = pfc_log[idx]   # pfc[0]=INST_RETIRED, pfc[1]=INST_SPEC; diff = wrong-path spec
            ret = sum(p[0] for p in pf) / len(pf)
            spec = sum(p[1] for p in pf) / len(pf)
            log.w(f"  input={idx} spec: retired={ret:.0f} inst_spec={spec:.0f} "
                  f"wrongpath={spec - ret:.1f}", ch="basic_hw")

        return self._aggregate_htraces(len(payloads), n_reps, input_to_trace_list, pfc_log)

    def _bpu_entries(self, inp: Input) -> Tuple[Tuple[int, bool], ...]:
        """Per-input branch training from the input's CE trace (empty when unavailable)."""
        return tuple(self.branch_mistraining_entries(getattr(inp, "_arch_trace", None)))

    def as_executor_input(self, inp: Input) -> ExecutorInput:
        """Convert one arch input into its kernel input file — the fuzzer's boost seam."""
        return ExecutorInput(inp, bpu_training=self._bpu_entries(inp))

    def trace_test_case(self, exec_inputs: List[ExecutorInput],
                        n_reps: int) -> Tuple[List[HTrace], List[TestCase]]:
        """Collect hardware traces for the loaded test case over `exec_inputs` — one upload, many
        inputs."""
        if not exec_inputs or self.test_case is None:
            return [], []
        STAT.executor_reruns += n_reps * len(exec_inputs)
        reported = self._write_modified_test_case("sandboxed_test_case", [Aarch64SandboxPass()])
        payloads = [ExecutorMemory(ei.serialize()) for ei in exec_inputs]
        return self._batch_trace(payloads, n_reps), [reported] * len(exec_inputs)

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

    def trace_test_case_with_taints(self, inputs: List[Input], nesting: int) -> Tuple[List[CTrace], List[InputTaint], List[ContractExecutionResult]]:
        """Sandbox the test case, run CE per input, and return cache-set CTraces, input taints, and the
        CE traces (the last feed per-input BPU mistraining)."""
        if not inputs or self.test_case is None:
            return [], [], []

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

        return ctraces, taints, traces


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

    def _upload_skeleton(self) -> None:
        """Upload the placeholder skeleton object code once; each input then carries its own code
        relocations, which the kernel splices in before that input runs."""
        self.local_executor.checkout_region(TestCaseRegion())
        self.local_executor.write(self._sealed.object_code)

    def _run_ce_on_tc(self, tc_bytes: bytes, inp: Input, sandbox_base: int) -> Optional[List]:
        """Run CE on the machine code *tc_bytes* for *inp*; return trace entries or None on failure."""
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

    def _log_pre_hw_ce_analysis(self, exec_inputs: List[ExecutorInput], sandbox_base: int, log) -> None:
        """Diagnostic pre-HW pass: run CE on each kernel input file's enacted code and log the CE
        trace, BB map, and AUTH instructions, flushing to disk before any HW runs."""
        auth_mnemonics = {'autia', 'autib', 'autda', 'autdb',
                          'autiza', 'autizb', 'autdza', 'autdzb'}
        for idx, ei in enumerate(exec_inputs):
            inp = ei.input_
            entry_x7 = int.from_bytes(inp.tobytes()[7 * 8: 8 * 8], "little")
            log.header(f"PRE-HW ANALYSIS  variant={idx}")

            tc = apply_relocations(self._sealed.object_code, list(ei.code_reloc))
            cer = self._run_ce_on_tc(tc, inp, sandbox_base)
            if cer is not None:
                log_ce_trace(log, f"v{idx}", idx, cer)
            else:
                log.w(f"  CE failed for variant={idx}")

            sorted_pcs, bb_info, pc_to_id, code_base = self._compute_bb_map(cer or [])
            log_bb_map(log, f"v{idx}", idx, sorted_pcs, bb_info, pc_to_id, code_base, entry_x7)

            if cer:
                log.header(f"AUTH INSTRUCTIONS IN CE: variant={idx}")
                for ite in cer:
                    disas = disassemble_instruction(ite.cpu.encoding, ite.cpu.pc) or ""
                    mn = disas.split()[0].lower() if disas else ""
                    if mn in auth_mnemonics:
                        nest = ite.metadata.speculation_nesting
                        tag = "ARCH" if nest == 0 else f"SPEC({nest})"
                        log.wp(f"  {tag}  {disas}  pc=0x{ite.cpu.pc:016x}")

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

        # The campaign PAC keys, generated once by the input generator, embedded in every PAC input
        # and passed with every sign request — the kernel keeps no key state of its own.
        self._pac_keys: Optional[PacKeys] = self._campaign_pac_keys() if "pac" in self._primitives \
            else None

        # PAC signing capability (kernel SIGN only — never AUTH; a failed AUTH at EL1 resets the box).
        signer = PacSigner(self.local_executor.pac_sign, self._pac_keys) \
            if "pac" in self._primitives else None

        # The sealer owns all sealing + resolution; it traces via _seal_trace and assembles object
        # code via _assemble_tc.
        self._sealer = make_sealer(generator, self._seal_trace, lambda tc: self._assemble_tc(tc)[0],
                                   self._primitives, signer)

        # Per-test-case sealed placeholder, populated by load_test_case(); the CE-trace context the
        # sealer's resolve uses (captured per with_taints call).
        self._sealed: Optional[SealedTestCase] = None
        self._sandbox_base: Optional[int] = None
        # Memo of the pure resolve, keyed by input. Reset per test case.
        self._resolve_cache: Dict[bytes, ResolvedSealingTestCase] = {}

    def _resolve(self, inp: Input) -> ResolvedSealingTestCase:
        key = inp.tobytes()
        if key not in self._resolve_cache:
            self._resolve_cache[key] = self._sealed.resolve(inp)
        return self._resolve_cache[key]

    def _mte_tags_for(self, inp: Input) -> Optional[List[int]]:
        """Uniform initial tags the sandbox is loaded with when MTE is active (else None — leave tag
        memory untouched). The kernel HW, the CE tag memory, and the model's MteTagState all seed
        from this same uniform tag."""
        if "mte" not in self._primitives:
            return None
        return [MTE_INITIAL_DEFAULT_TAG] * MTE_TAG_COUNT

    def _campaign_pac_keys(self) -> PacKeys:
        """The deterministic per-campaign PAC keys, generated once by the input generator (seeded from
        the config), shared by every input so a sealing class's one shared signature set verifies."""
        from .. import factory
        return PacKeys(*factory.get_input_generator(CONF.input_gen_seed).generate_pac_keys())

    def _pac_keys_words(self) -> Optional[List[int]]:
        """The campaign PAC keys as the 10-word PAC_KEYS section (else None when PAC is inactive), so
        every input carries the keys its baked signatures were signed under."""
        if "pac" not in self._primitives:
            return None
        assert self._pac_keys is not None
        return self._pac_keys.words()

    def _ce_trace(self, tc_bytes: bytes, inp: Input) -> ContractExecutionResult:
        """One CE trace of the machine code `tc_bytes` for `inp`. Sealing always traces at max
        speculation depth: a shallower pass would classify deep-speculative slots as unreached and
        under-decoy them, missing leaks."""
        if self._sandbox_base is None:
            self._sandbox_base, _ = self.read_base_addresses()
        return self._contract_executor.run(self._make_ce_execution(
            tc_bytes, inp, self._sandbox_base, CONF.model_max_nesting, CONF.model_max_spec_window,
            ExecutionClause.COND, mte_tags=self._mte_tags_for(inp), pac_keys=self._pac_keys))

    def _seal_trace(self, tc: TestCase, inp: Input) -> ContractExecutionResult:
        """The sealer's trace_fn: assemble the placeholder TC and run one CE trace."""
        return self._ce_trace(self._assemble_tc(tc)[0], inp)

    # ------------------------------------------------------------------ lifecycle
    def load_test_case(self, test_case: TestCase):
        log = FuzzLogger.get()
        log.register("ni_flow",     "ni/flow.log",     min_verbosity=1)
        log.register("pac_signing", "ni/signing.log",  min_verbosity=1)  # log_pac_op/log_slot channel
        log.register("ni_table",    "ni/ni_table.log", min_verbosity=1)
        log.register("pac_hw",      "ni/hw.log",       min_verbosity=1)
        result = super().load_test_case(test_case)
        log_start_test_case(log, self._tc_counter)
        self._sealed = self._sealer.seal(self.test_case)
        self._resolve_cache = {}
        log.ensure_flushed()
        return result

    def trace_test_case_with_taints(
        self,
        inputs: List[Input],
        nesting: int,
    ) -> Tuple[List[CTrace], List[InputTaint], List[ContractExecutionResult]]:
        """CE pass: ctraces + taints from each input's genuine baseline."""
        if not inputs or self.test_case is None:
            return [], [], []
        assert self._sealed is not None, "sealed TC not built — load_test_case() not called"

        log = FuzzLogger.get()
        ce_traces: List[ContractExecutionResult] = []
        for inp_idx, inp in enumerate(inputs):
            log_input(log, inp_idx, inp, ch="ni_flow")
            log.ensure_flushed()

            resolved = self._resolve(inp)
            cer = self._ce_trace(apply_relocations(resolved.object_code, resolved.genuine()), inp)
            ce_traces.append(cer)

            log_ce_trace(log, "BASELINE", inp_idx, list(cer))
            if FuzzLogger.on():
                self._log_variant_binaries(log, inp_idx, cer, resolved)
            log.ensure_flushed()

        taints = [compute_taint(cer) for cer in ce_traces]
        ctraces = [compute_ctrace(cer) for cer in ce_traces]
        return ctraces, taints, ce_traces

    def variants_for_input(self, inp: Input) -> Dict[str, ExecutorInput]:
        """One kernel input file per variant of `inp`. Deterministic, so the CE pass, the HW pass, and
        priming build the identical set."""
        resolved = self._resolve(inp)
        tags = self._mte_tags_for(inp)
        bpu = self._bpu_entries(inp)
        keys = self._pac_keys_words()
        return {name: ExecutorInput(inp, code_reloc=plan, mte_tags=tags, pac_keys=keys,
                                    bpu_training=bpu)
                for name, plan in self._variants_for(resolved).items()}

    def sealing_class_of(self, inp: Input) -> tuple:
        """The input's sealing class (its resolved per-slot signatures/tags). Variants collide only
        when this agrees, not just the ctrace."""
        return self._resolve(inp).collapse_key

    def _variants_for(self, resolved: ResolvedSealingTestCase) -> Dict[str, Tuple[Relocation, ...]]:
        """One boosting class: the genuine baseline plus `CONF.inputs_per_class - 1` decoys."""
        plans = {NIVariant.BASELINE: resolved.genuine()}
        for i in range(CONF.inputs_per_class - 1):
            rng = random.Random(hash((resolved.collapse_key, self._sealed.salt, i)))
            plans[NIVariant.decoy_n(i)] = resolved.decoy(rng)
        return plans

    def _log_variant_binaries(self, log, inp_idx, cer,
                              resolved: ResolvedSealingTestCase) -> None:
        columns = [(name, apply_relocations(resolved.object_code, list(plan)))
                   for name, plan in self._variants_for(resolved).items()]
        log.header(f"VARIANT TC BINARIES  inp={inp_idx}")
        for name, code in columns:
            log_tc_binary(log, name, code)
        log_ni_table(log, inp_idx, columns, list(cer), "", {}, ch="ni_table")


    # ------------------------------------------------------------------ hardware: sealed execution
    def trace_test_case(self, exec_inputs: List[ExecutorInput],
                        n_reps: int) -> Tuple[List[HTrace], List[TestCase]]:
        """Trace each kernel input file on the shared skeleton (its code-relocation table spliced in by
        the kernel); returns one htrace per element plus the loaded TestCase per element (as the base
        executor does, so the artifact store can print each measurement's asm). The exact machine code a
        boosted variant ran differs per seal, but that is recorded separately via
        reconstruct_enacted_code(); Measurement.test_case must stay a TestCase, not bytes."""
        if not exec_inputs or self.test_case is None:
            return [], []
        return self._trace_exec_inputs(exec_inputs, n_reps), [self.test_case] * len(exec_inputs)

    def as_executor_input(self, inp: Input) -> ExecutorInput:
        """Convert one arch input into its sealed kernel input file — the regular fuzzer's boost seam."""
        return self._seal_input(inp)

    def _seal_input(self, inp: Input) -> ExecutorInput:
        """Extend an arch input to its baseline kernel input file — genuine relocations here; the regular
        subclass overrides to the input's sealing-class plan."""
        return ExecutorInput(inp, code_reloc=self._resolve(inp).genuine(),
                             mte_tags=self._mte_tags_for(inp), pac_keys=self._pac_keys_words(),
                             bpu_training=self._bpu_entries(inp))

    def _trace_exec_inputs(self, exec_inputs: List[ExecutorInput], n_reps: int) -> List[HTrace]:
        """Upload the skeleton, then trace each kernel input file — the kernel splices that input's
        code-relocation table into the shared skeleton before it runs and reverts afterwards."""
        if not exec_inputs:
            return []
        assert self._sealed is not None, "sealed TC not built — load_test_case() not called"
        STAT.executor_reruns += n_reps * len(exec_inputs)
        if FuzzLogger.on():
            sandbox_base, _ = self.read_base_addresses()
            self._log_pre_hw_ce_analysis(exec_inputs, sandbox_base, FuzzLogger.get())
        self._upload_skeleton()
        payloads = [ExecutorMemory(ei.serialize()) for ei in exec_inputs]
        return self._batch_trace(payloads, n_reps)

    def reconstruct_enacted_code(self, ei: ExecutorInput) -> bytes:  # TEMP(enacted-reloc)
        """Rebuild the exact machine code that ran for a boosted variant: re-resolve its arch input
        over the current sealed TC (object_code is deterministic) and re-apply its relocations."""
        resolved = self._resolve(ei.input_)
        return apply_relocations(resolved.object_code, list(ei.code_reloc))


class Aarch64RegularSealedExecutor(Aarch64NonInterferenceExecutor):
    """Regular fuzzing with PAC/MTE support.

    Reuses the NI seal/resolve pipeline. Inputs are grouped into *sealing classes* — identical
    resolved (correct_sig, correct_tag, spec_nesting) across all fix points — and each class runs ONE
    sealed form, chosen deterministically from the class + per-test-case salt (so every member renders
    it identically). The form is a per-class coin flip:
      - decoy: genuine on every architectural slot, decoyed (wrong signature / retag) on the
        speculative slots — exposes the speculative PAC/MTE leak.
      - genuine: genuine everywhere.
    Either way the architectural slots are always correct, so the TC is arch-safe (no failed AUTH at
    EL1, no tag fault), and the representative's arch AUT*/tag match every member's pointer.

    One form per class gives, for free, a consistent decoy decision at each slot across all the class's
    inputs (they run the identical program); an input never runs another class's form. The seal slots
    are register-only, so sealing differences between classes never enter the cache htrace, leaving the
    standard regular by-contract-trace leak comparison valid. (A contract trace fixes the accessed
    addresses, hence the pointers, hence the signatures/tags — so a contract-trace class is contained
    in a sealing class and likewise shares one form.)"""

    _DECOY_PROB = 0.5   # per sealing class: probability of the speculative-decoy form vs all-genuine

    def _seal_input(self, inp: Input) -> ExecutorInput:
        """Extend an arch input to its sealing class's kernel input file. The decoy-vs-genuine coin and
        the decoy itself are a pure function of the sealing class + per-test-case salt, so every member
        of a class runs the identical program (caveat-4 merges fall out); genuine()'s arch slots are
        always correct (never a forged AUTH)."""
        resolved = self._resolve(inp)
        rng = random.Random(hash((resolved.collapse_key, self._sealed.salt)))
        plan = resolved.decoy(rng) if rng.random() < self._DECOY_PROB else resolved.genuine()
        return ExecutorInput(inp, code_reloc=plan, mte_tags=self._mte_tags_for(inp),
                             pac_keys=self._pac_keys_words(), bpu_training=self._bpu_entries(inp))
