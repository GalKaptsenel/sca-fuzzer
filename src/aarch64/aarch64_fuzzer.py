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
from ..analyser import MergedBitmapAnalyser, ChiSquaredAnalyser
from ..util import STAT
from ..config import CONF, ConfigException
from .aarch64_executor import Aarch64Executor, pass_on_test_case
from .aarch64_generator import Aarch64DsbSyPass
from .leftover import GeneralizedPrimingDetector, LeftoverFinding
from .boosted_lanes import lanes_of, detecting_position_and_lanes


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


def regime_controllable(executor: Aarch64Executor) -> bool:
    """The leftover searches need local sysfs regime control (view_rotation / phr_flush / pinning). Only a
    local HW executor exposes it; a remote executor cannot apply the regime."""
    dev = getattr(executor, "device", None)
    return dev is not None and hasattr(dev, "_read_sysfs") and hasattr(dev, "_write_sysfs")


@contextmanager
def executor_regime(executor: Aarch64Executor, regime) -> Generator[None, None, None]:
    """Apply a sysfs regime (a sequence of (name, value) pairs) to the executor for the duration of the
    `with` block, capturing the prior values and restoring them on exit."""
    dev = executor.device
    saved = {name: dev._read_sysfs(name) for name, _ in regime}
    try:
        for name, value in regime:
            dev._write_sysfs(name, value.encode())
        yield
    finally:
        for name, value in saved.items():
            dev._write_sysfs(name, value.encode())


# ==================================================================================================
# Fuzzer classes
# ==================================================================================================
class Aarch64Fuzzer(FuzzerGeneric):
    executor: Aarch64Executor

    # AArch64 saves inputs as the executor-ready REIF container (flags already in PSTATE form).
    input_file_extension: str = "reif"

    # Regime for the boosted-lane leftover search (enable_boosted_leftover). Reset the BPU/PHR ONCE per
    # trace (the post-fix phr_flush; see measurement.c) and keep cross-input training within a trace
    # (view_rotation off, so all inputs share one code view). Do not use the legacy combined knob.
    # Captured and restored around the search so the surrounding fuzzing keeps its usual regime.
    _BOOSTED_LEFTOVER_REGIME = (("enable_pre_run_flush", "0"), ("enable_view_rotation", "0"),
                                ("enable_phr_flush", "1"))

    def _boost_inputs(self, inputs, nesting):
        # The aarch64 fuzzer's input unit is the ExecutorInput; convert once boosting is done.
        boosted, ctraces = super()._boost_inputs(inputs, nesting)
        return list(map(self.executor.as_executor_input, boosted)), ctraces

    def _save_input(self, input_, path: str) -> None:
        from .aarch64_executor_input_encoder import ExecutorInput
        ExecutorInput(input_).save(path)

    def _priming(self, violations: List, inputs: List[Input]) -> List:
        """Regular fuzzing: when enable_boosted_leftover is set, REPLACE standard priming with the
        generalized-priming leftover search over boosted lanes. Priming asks "does the divergence follow
        the detecting pair's own input?"; this asks the strictly more general "which (earlier or own)
        ct-equal input causes it?" -- localizing the flagged violation's own detecting pair across its
        two diverging lanes. A found leaking pair (self- or cross-input) confirms the violation and
        reports the [leaker..detector] chain; none means a false positive, exactly as priming. Off by
        default -> standard priming (unchanged)."""
        if not CONF.enable_boosted_leftover:
            return super()._priming(violations, inputs)
        if not regime_controllable(self.executor):
            raise ConfigException("enable_boosted_leftover requires a local HW executor with sysfs "
                                  "regime control; set enable_boosted_leftover = False for remote.")
        n_orig = len(inputs) // CONF.inputs_per_class
        lanes = lanes_of(inputs, n_orig)
        detector = self._make_leftover_detector()
        with executor_regime(self.executor, self._BOOSTED_LEFTOVER_REGIME):
            try:
                for violation in reversed(violations):        # priming pops the stack from the end
                    located = self._localize_boosted_violation(violation, lanes, n_orig, detector)
                    if located is not None:
                        finding, prefix_lane, suffix_lane = located
                        # Attach for the artifact report; do not print (priming runs once per sample
                        # size, so a print here would repeat). The report.txt carries the explanation.
                        violation.leftover = (finding, prefix_lane, suffix_lane, n_orig)
                        self.LOG.dbg("fuzzer", f"boosted leftover: leaking pair {finding.leaking_pair}"
                                     f" -> detecting pair {finding.detecting_pair}")
                        return [violation]
            except HardwareTracingError as e:                 # transient device failure -> skip the round
                self.LOG.warning("fuzzer", f"boosted leftover: hardware tracing failed: {e}")
        return []

    def _make_leftover_detector(self) -> GeneralizedPrimingDetector:
        """The generalized-priming detector wired to this executor, with the production key/confirm seams
        (denoised-consensus key for the bisection; robust chi-squared for the re-verify)."""
        outlier = CONF.analyser_outliers_threshold
        robust = ChiSquaredAnalyser()
        return GeneralizedPrimingDetector(
            measure=lambda batch, reps: self.executor.trace_test_case(batch, reps)[0],
            key=lambda trace: MergedBitmapAnalyser.merged_bitmap(trace, outlier),
            confirm=lambda a, b: not robust.htraces_are_equivalent(a, b),
            reps=CONF.leftover_reps, verify_reps=CONF.leftover_verify_reps)

    def _localize_boosted_violation(self, violation, lanes: List[list], n_orig: int,
                                    detector: GeneralizedPrimingDetector):
        """Focused localization of one flagged violation: find its detecting position and two diverging
        lanes, then run the toggle search on that lane pair from both bases (genuine- and decoy-prefix).
        Returns (finding, prefix_lane, suffix_lane) -- the lanes that hold the sequence before the leaking
        pair and the toggled chain [leaking..detecting] from it onward -- or None if not localizable."""
        located = detecting_position_and_lanes(violation.htrace_groups, n_orig)
        if located is None:
            return None
        detecting, lane_a, lane_b = located
        for prefix_lane, suffix_lane, prefix_genuine in ((lane_a, lane_b, True), (lane_b, lane_a, False)):
            finding = detector.find_leaking_pair(lanes[prefix_lane], lanes[suffix_lane],
                                                 detecting, prefix_genuine)
            if finding is not None:
                return finding, prefix_lane, suffix_lane
        return None

    def _store_violation_artifact(self, test_case: TestCase, violation, path: str) -> str:
        """Extend the standard artifact: when the violation was localized by the boosted-lane leftover
        search, append the two 'complex inputs' to report.txt so it reads on its own."""
        violation_dir = super()._store_violation_artifact(test_case, violation, path)
        leftover = getattr(violation, "leftover", None)
        if leftover is not None:
            with open(f"{violation_dir}/report.txt", "a") as f:
                f.write(self._leftover_report_section(*leftover))
        return violation_dir

    def _leftover_report_section(self, finding: LeftoverFinding, prefix_lane: int, suffix_lane: int,
                                 n_orig: int) -> str:
        """The counterexample as two sequences of input ids (the reduced 'complex inputs'): both are
        ct-equal (each position holds a member of the same input class), they differ ONLY across positions
        [leaking pair .. detecting pair], yet the detecting pair's hardware trace differs -- so an earlier
        input leaves a microarchitectural leftover that changes a later input's readout. The input classes
        (ct-equal members) are listed explicitly so the two sequences are readable on their own."""
        lo, hi = finding.leaking_pair, finding.detecting_pair
        num_lanes = CONF.inputs_per_class
        members = lambda p: [lane * n_orig + p for lane in range(num_lanes)]            # class p's ct-equal ids
        seq_a = [prefix_lane * n_orig + p for p in range(hi + 1)]                       # all prefix lane
        seq_b = [(prefix_lane if p < lo else suffix_lane) * n_orig + p for p in range(hi + 1)]
        kind = "self-dependent (the detecting pair's own input)" if lo == hi \
            else f"cross-input (input {lo} leaves a leftover that changes input {hi})"
        classes = "".join(
            f"    position {p:2d}: class {p} = {members(p)}"
            + (f"   (A picks {seq_a[p]}, B picks {seq_b[p]})\n" if p >= lo else "   (both pick %d)\n" % seq_a[p])
            for p in range(hi + 1))
        return (
            "\n## Cross-input leftover (boosted-lane localization)\n"
            f"* Leaking pair:   position {lo} (input class {lo})\n"
            f"* Detecting pair: position {hi} (input class {hi})\n"
            f"* Kind: {kind}\n"
            f"* The two sequences below are each ct-equal (every position holds a member of the same input\n"
            f"  class), differ only in positions {lo}..{hi}, yet the detecting pair's hardware trace differs.\n"
            f"* Sequence A: {seq_a}\n"
            f"* Sequence B: {seq_b}\n"
            f"* Input classes (ct-equal members), position by position:\n{classes}")

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
        if CONF.enable_leftover_detection and not regime_controllable(self.executor):
            raise ConfigException(
                "enable_leftover_detection requires a local HW executor with sysfs regime control "
                "(view_rotation / pin_to_core); set enable_leftover_detection = False for remote "
                "executors.")

    def fuzzing_round(self, test_case: TestCase, inputs: List[Input],
                      ignore_list=None):
        if CONF.enable_leftover_detection and len(inputs) >= 2:
            self._detect_leftovers(test_case, inputs)
        return super().fuzzing_round(test_case, inputs, ignore_list)

    def _detect_leftovers(self, test_case: TestCase, inputs: List[Input]) -> None:
        """Run the generalized-priming leftover search on the batch's genuine/decoy seal lanes."""
        self.executor.load_test_case(test_case)
        genuine = [self.executor.genuine_variant(inp) for inp in inputs]
        bad = [self.executor.noncanon_variant(inp) for inp in inputs]
        outlier_threshold = CONF.analyser_outliers_threshold
        robust = ChiSquaredAnalyser()  # jitter-tolerant re-verify, independent of the configured analyser

        detector = GeneralizedPrimingDetector(
            measure=lambda batch, reps: self.executor.trace_test_case(batch, reps)[0],
            key=lambda trace: MergedBitmapAnalyser.merged_bitmap(trace, outlier_threshold),
            confirm=lambda a, b: not robust.htraces_are_equivalent(a, b),
            reps=CONF.leftover_reps, verify_reps=CONF.leftover_verify_reps)

        findings: List[LeftoverFinding] = []
        with executor_regime(self.executor, self._LEFTOVER_REGIME):
            try:
                findings = detector.detect(genuine, bad)
            except HardwareTracingError as e:  # transient device failure -> skip this round's search
                self.LOG.warning("fuzzer", f"leftover detection: hardware tracing failed: {e}")

        for f in findings:
            base = "genuine" if f.prefix_genuine else "decoy"
            self.LOG.warning("fuzzer", f"LEFTOVER: leaking pair {f.leaking_pair} -> detecting pair "
                             f"{f.detecting_pair} (chain {list(f.chain)}, {base}-prefix base)")
            self._save_leftover_artifact(test_case, f, genuine, bad)

    def _save_leftover_artifact(self, test_case: TestCase, finding: LeftoverFinding,
                                genuine: List, bad: List) -> None:
        """Save the test case and the counterexample chain: each input's variant as it runs (the base
        lane before the leaking pair, the toggle lane from it onward), plus the leaking pair's base
        variant -- the flip that removes the leak. The base lane is genuine for a genuine-prefix finding
        and decoy for the mirror decoy-prefix finding (see LeftoverFinding.prefix_genuine)."""
        try:
            base, toggle = (genuine, bad) if finding.prefix_genuine else (bad, genuine)
            path = os.path.join(
                self.work_dir,
                f"leftover_l{finding.leaking_pair}_d{finding.detecting_pair}_{STAT.test_cases}")
            os.makedirs(path, exist_ok=True)
            for attr in ("bin_path", "asm_path"):
                src = getattr(test_case, attr, None)
                if src and os.path.exists(src):
                    shutil.copy(src, os.path.join(path, os.path.basename(src)))
            for slot in range(finding.detecting_pair + 1):
                lane = toggle if slot in finding.chain else base
                with open(os.path.join(path, f"input{slot}.reif"), "wb") as f:
                    f.write(lane[slot].serialize())
            with open(os.path.join(path, f"input{finding.leaking_pair}.flip.reif"), "wb") as f:
                f.write(base[finding.leaking_pair].serialize())
            self.LOG.warning("fuzzer", f"LEFTOVER artifacts saved: {path}")
        except Exception as e:            # artifact saving must never abort a fuzzing round
            self.LOG.warning("fuzzer", f"LEFTOVER artifact save failed: {e}")
