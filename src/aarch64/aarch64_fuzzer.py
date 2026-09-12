"""
File: AArch64 fuzzer
"""
from typing import List, Generator
from contextlib import contextmanager
import os
import shutil
import copy

from ..fuzzer import FuzzerGeneric, NoninterferenceFuzzer
from ..interfaces import TestCase, Input, HardwareTracingError
from ..analyser import MergedBitmapAnalyser
from ..util import STAT
from ..config import CONF, ConfigException
from .aarch64_executor import Aarch64Executor, pass_on_test_case
from .aarch64_generator import Aarch64DsbSyPass
from .leftover import GeneralizedPrimingDetector, LeftoverFinding


# ==================================================================================================
# Helper functions
# ==================================================================================================

@contextmanager
def quick_and_dirty_mode(executor: Aarch64Executor) -> Generator[None, None, None]:
    """
    Context manager that enables us to use quick and dirty mode in the form of `with` statement
    """
    try:
        executor.set_quick_and_dirty(True)
        yield
    finally:
        executor.set_quick_and_dirty(False)


def create_fenced_test_case(test_case: TestCase) -> TestCase:
    """Return a copy of the test case with a DSB SY speculation barrier inserted after every
    instruction (at the IR level), so it executes with speculation suppressed."""
    fenced = copy.deepcopy(test_case)
    pass_on_test_case(fenced, [Aarch64DsbSyPass()])
    return fenced


# ==================================================================================================
# Fuzzer classes
# ==================================================================================================
class Aarch64Fuzzer(FuzzerGeneric):
    executor: Aarch64Executor

    # AArch64 saves inputs as the executor-ready REIF container (flags already in PSTATE form).
    input_file_extension: str = "reif"

    def _boost_inputs(self, inputs, nesting):
        # The aarch64 fuzzer's input unit is the ExecutorInput; convert once boosting is done.
        boosted, ctraces = super()._boost_inputs(inputs, nesting)
        return list(map(self.executor.as_executor_input, boosted)), ctraces

    def _save_input(self, input_, path: str) -> None:
        from .aarch64_executor_input_encoder import ExecutorInput
        ExecutorInput(input_).save(path)

    def filter(self, test_case: TestCase, inputs: List[Input]) -> bool:
        """
        This function implements a multi-stage algorithm that gradually filters out
        uninteresting test cases

        :param test_case: the target test case
        :param inputs: list of inputs to be tested
        :return: True if the test case should be filtered out; False otherwise
        """
        # Exit if no filtering is enabled
        if not CONF.enable_speculation_filter and not CONF.enable_observation_filter:
            return False

        # Number of repetitions for each input
        reps = CONF.executor_filtering_repetitions

        # Enable quick and dirty mode to speed up the process
        with quick_and_dirty_mode(self.executor):
            # Collect hardware traces for the test case
            try:
                self.executor.load_test_case(test_case)
                exec_inputs = list(map(self.executor.as_executor_input, inputs))
                org_htraces, _ = self.executor.trace_test_case(exec_inputs, reps)
            except HardwareTracingError as e:
                STAT.hw_tracing_errors += 1
                self.LOG.warning("fuzzer", f"hardware tracing failed, filtering test case: {e}")
                return True

            # 1. Speculation filter:
            # Execute on the test case on the HW and monitor PFCs
            # if there are no mispredictions, this test case is unlikely
            # to produce a violation, so just move on to the next one
            if CONF.enable_speculation_filter:
                # aarch64 PFCs: [0]=INST_RETIRED, [1]=INST_SPEC, [2]=BR_MIS_PRED. A test case can
                # only leak speculatively if some branch mispredicted; INST_SPEC >= INST_RETIRED
                # always, so the x86-style spec-vs-retired comparison does not apply here.
                for htrace in org_htraces:
                    pfc_values = htrace.perf_counters_max
                    if pfc_values[0] == 0:  # nothing retired => measurement error, can't filter
                        break
                    if pfc_values[2] > 0:   # mispredicted branch => speculation occurred
                        break
                else:
                    STAT.spec_filter += 1
                    return True

            # 2. Observation filter:
            # Check if any of the htraces contain a speculative cache eviction
            # for this create a fenced version of the test case and collect traces for it
            if CONF.enable_observation_filter:
                fenced_test_case = create_fenced_test_case(test_case)
                try:
                    self.executor.load_test_case(fenced_test_case)
                    exec_inputs = list(map(self.executor.as_executor_input, inputs))
                    fenced_htraces, _ = self.executor.trace_test_case(exec_inputs, reps)
                except HardwareTracingError:
                    return True

                traces_match = True
                for i, _ in enumerate(inputs):
                    if not self.analyser.htraces_are_equivalent(fenced_htraces[i], org_htraces[i]):
                        traces_match = False
                        break
                if traces_match:
                    STAT.observ_filter += 1
                    return True

            return False


class Aarch64NoninterferenceFuzzer(NoninterferenceFuzzer):
    """AArch64 non-interference fuzzer.

    Adds the cross-input speculative leftover detector (generalized priming + hybrid tipping-point
    search, `leftover.py`) as the sole NI leftover-detection algorithm. Before each normal NI round it
    builds the genuine/bad seal lanes for the input batch and runs the search under the leftover
    regime; the regime is captured and restored around the search so the normal NI round that follows
    sees the executor's usual regime. Regular (non-NI) fuzzing is a different fuzzer entirely and is
    unaffected. The search itself lives in `leftover.py` and knows nothing about the fuzzer; the only
    coupling is this thin seam (lane construction, the measure closure, the regime, and reporting)."""

    # The executor sysfs regime the leftover search requires: SSBS on for the store-bypass window, no
    # per-input flushing/rotation (would wipe the trained BTB entry), and unpinned execution (pinning
    # measurably raises the non-canonical residual -- experimenter bonus finding). Captured-and-restored.
    _LEFTOVER_REGIME = (("enable_ssbs", "1"), ("enable_pre_run_flush", "0"),
                        ("enable_phr_flush", "0"), ("enable_view_rotation", "0"),
                        ("pin_to_core", "-1"))

    def initialize_modules(self) -> None:
        super().initialize_modules()
        # Fail fast, loud: the search's soundness depends on the leftover regime, which needs local
        # sysfs control. Do not silently run it against an executor that cannot apply the regime.
        if CONF.enable_leftover_detection and not self._regime_controllable():
            raise ConfigException(
                "enable_leftover_detection requires a local HW executor with sysfs regime control "
                "(view_rotation / pin_to_core); set enable_leftover_detection = False for remote "
                "executors.")

    def _regime_controllable(self) -> bool:
        dev = getattr(self.executor, "device", None)
        return dev is not None and hasattr(dev, "_read_sysfs") and hasattr(dev, "_write_sysfs")

    def fuzzing_round(self, test_case: TestCase, inputs: List[Input],
                      ignore_list=None):
        if CONF.enable_leftover_detection and len(inputs) >= 2:
            self._detect_leftovers(test_case, inputs)
        return super().fuzzing_round(test_case, inputs, ignore_list)

    def _detect_leftovers(self, test_case: TestCase, inputs: List[Input]) -> None:
        """Run the generalized-priming leftover search on the batch's genuine/bad seal lanes."""
        self.executor.load_test_case(test_case)
        genuine = [self.executor.genuine_variant(inp) for inp in inputs]
        bad = [self.executor.noncanon_variant(inp) for inp in inputs]
        outlier_threshold = CONF.analyser_outliers_threshold

        detector = GeneralizedPrimingDetector(
            measure=lambda batch, reps: self.executor.trace_test_case(batch, reps)[0],
            key=lambda trace: MergedBitmapAnalyser.merged_bitmap(trace, outlier_threshold),
            reps=CONF.leftover_reps)

        findings: List[LeftoverFinding] = []
        with self._leftover_regime():
            try:
                findings = detector.detect(genuine, bad)
            except HardwareTracingError as e:  # transient device failure -> skip this round's search
                self.LOG.warning("fuzzer", f"leftover detection: hardware tracing failed: {e}")

        for f in findings:
            self.LOG.warning("fuzzer", f"LEFTOVER: prober input {f.prober} <- tipping point "
                             f"{f.tipping_point} (violating block {list(f.block)})")
            self._save_leftover_artifact(test_case, f, genuine, bad)

    @contextmanager
    def _leftover_regime(self) -> Generator[None, None, None]:
        """Apply the leftover-search sysfs regime, capturing and restoring the executor's values."""
        dev = self.executor.device
        saved = {name: dev._read_sysfs(name) for name, _ in self._LEFTOVER_REGIME}
        try:
            for name, value in self._LEFTOVER_REGIME:
                dev._write_sysfs(name, value.encode())
            yield
        finally:
            for name, value in saved.items():
                dev._write_sysfs(name, value.encode())

    def _save_leftover_artifact(self, test_case: TestCase, finding: LeftoverFinding,
                                genuine: List, bad: List) -> None:
        """Save the generating test case and the block's genuine/bad variants, for reproduction."""
        try:
            path = os.path.join(self.work_dir,
                                f"leftover_p{finding.prober}_t{finding.tipping_point}_{STAT.test_cases}")
            os.makedirs(path, exist_ok=True)
            for attr in ("bin_path", "asm_path"):
                src = getattr(test_case, attr, None)
                if src and os.path.exists(src):
                    shutil.copy(src, os.path.join(path, os.path.basename(src)))
            for slot in finding.block:
                for tag, lane in (("genuine", genuine), ("bad", bad)):
                    with open(os.path.join(path, f"input{slot}_{tag}.reif"), "wb") as f:
                        f.write(lane[slot].serialize())
            self.LOG.warning("fuzzer", f"LEFTOVER artifacts saved: {path}")
        except Exception as e:            # artifact saving must never abort a fuzzing round
            self.LOG.warning("fuzzer", f"LEFTOVER artifact save failed: {e}")
