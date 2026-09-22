"""
File: AArch64 fuzzer
"""
from typing import List, Generator, Dict, Tuple, Optional, Sequence, TYPE_CHECKING
from contextlib import contextmanager
import copy

from ..fuzzer import FuzzerGeneric, NoninterferenceFuzzer
from ..interfaces import TestCase, Input, Violation, HardwareTracingError
from ..analyser import MergedBitmapAnalyser, ChiSquaredAnalyser
from ..util import STAT
from ..config import CONF, ConfigException
from .aarch64_executor import Aarch64Executor, Aarch64LocalExecutor, pass_on_test_case
from .aarch64_generator import Aarch64DsbSyPass
from .cross_input import (GeneralizedPrimingDetector, CrossInputFinding, Localizer,
                          linear_scan, exponential_search)
from .boosted_lanes import lanes_of, detecting_position_and_lanes
from .aarch64_kernel import LocalHWExecutor

# Map cross_input_priming_localizer to the cross_input.py localizer strategy: "any" = galloping + bisection
# (some leaking pair), "optimal" = linear scan (t_max, the largest leaking class).
_CROSS_INPUT_PRIMING_LOCALIZERS: Dict[str, Localizer] = {
    "any": exponential_search, "optimal": linear_scan}


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


def regime_controllable(executor: Aarch64LocalExecutor) -> bool:
    """Cross-input priming needs local sysfs regime control (view_rotation / phr_flush / pinning). Only a
    local HW executor exposes it; a remote executor cannot apply the regime."""
    return isinstance(executor.device, LocalHWExecutor)


@contextmanager
def executor_regime(executor: Aarch64LocalExecutor,
                    regime: Sequence[Tuple[str, str]]) -> Generator[None, None, None]:
    """Apply a sysfs regime (a sequence of (name, value) pairs) to the executor for the duration of the
    `with` block, capturing the prior values and restoring them on exit."""
    dev = executor.device
    if not isinstance(dev, LocalHWExecutor):
        raise ConfigException("cross-input priming requires a local HW executor with sysfs regime "
                              "control; set enable_cross_input_priming = False for remote.")
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
if TYPE_CHECKING:                  # the mixin is only ever combined with a FuzzerGeneric subclass, so
    _MixinBase = FuzzerGeneric     # tell the type checker (self.executor / LOG / super() resolve there);
else:                              # at runtime it stays a pure mixin whose bases come from the subclass.
    _MixinBase = object


class CrossInputPrimingMixin(_MixinBase):
    """Cross-input priming (generalized priming): replace standard priming with a localization over the
    boosted lanes that finds the *leaking pair* -- an earlier ct-equal input whose microarchitectural
    residue surfaces as the *detecting pair*'s cache divergence. Shared by the regular fuzzer (boosted
    random-data lanes) and the non-interference fuzzer (boosted seal-variant lanes -> canonicality/BTB).
    The search itself is `src/aarch64/cross_input.py`; this mixin is the thin seam (reps and localizer
    selection, lane construction, the executor regime, and reporting). Subclasses set
    `_CROSS_INPUT_PRIMING_REGIME`. Off by default -> the subclass's standard priming (via super())."""

    executor: Aarch64LocalExecutor

    # Default regime: reset the BPU/PHR ONCE per trace (post-fix phr_flush; see measurement.c) and keep
    # cross-input training within a trace (view_rotation off, so all inputs share one code view).
    # Captured and restored around the search so the surrounding fuzzing keeps its usual regime. The NI
    # fuzzer overrides this with the seal-lane regime.
    _CROSS_INPUT_PRIMING_REGIME: Tuple[Tuple[str, str], ...] = (
        ("enable_pre_run_flush", "0"), ("enable_view_rotation", "0"), ("enable_phr_flush", "1"))

    def _priming(self, violations: List[Violation], inputs: List[Input]) -> List[Violation]:
        """When enable_cross_input_priming is set, REPLACE standard priming with the generalized-priming
        localization over boosted lanes. Priming asks "does the divergence follow the detecting pair's own
        input?"; this asks the strictly more general "which (earlier or own) ct-equal input causes it?" --
        localizing the flagged violation's own detecting pair across its two diverging lanes. A found
        leaking pair (self- or cross-input) confirms the violation and reports the [leaker..detector]
        chain; none means a false positive, exactly as priming. Off by default -> standard priming."""
        if not CONF.enable_cross_input_priming:
            return super()._priming(violations, inputs)
        if not violations:
            return []
        if not regime_controllable(self.executor):
            raise ConfigException("enable_cross_input_priming requires a local HW executor with sysfs "
                                  "regime control; set enable_cross_input_priming = False for remote.")
        # Reuse standard priming's sample sizes: the current stage size (read off the violation, as
        # _prime_one does) for localization, and the largest configured size to re-verify a boundary.
        reps = len(violations[0].measurements[0].htrace.raw)
        verify_reps = CONF.executor_sample_sizes[-1]
        localizer = _CROSS_INPUT_PRIMING_LOCALIZERS[CONF.cross_input_priming_localizer]
        n_orig = len(inputs) // CONF.inputs_per_class
        lanes = lanes_of(inputs, n_orig)
        detector = self._make_cross_input_detector(reps, verify_reps, localizer)
        with executor_regime(self.executor, self._CROSS_INPUT_PRIMING_REGIME):
            try:
                for violation in reversed(violations):        # priming pops the stack from the end
                    located = self._localize_boosted_violation(violation, lanes, n_orig, detector)
                    if located is not None:
                        finding, prefix_lane, suffix_lane = located
                        # Attach for the artifact report; do not print (priming runs once per sample
                        # size, so a print here would repeat). The report.txt carries the explanation.
                        violation.cross_input_finding = (finding, prefix_lane, suffix_lane, n_orig)
                        self.LOG.dbg("fuzzer", f"cross-input priming: leaking pair {finding.leaking_pair}"
                                     f" -> detecting pair {finding.detecting_pair}")
                        return [violation]
            except HardwareTracingError as e:                 # transient device failure -> skip the round
                self.LOG.warning("fuzzer", f"cross-input priming: hardware tracing failed: {e}")
        return []

    def _make_cross_input_detector(
            self, reps: int, verify_reps: int,
            localizer: Localizer) -> GeneralizedPrimingDetector:
        """The generalized-priming detector wired to this executor, with the production key/confirm seams
        (denoised-consensus key for the bisection; robust chi-squared for the re-verify) and the selected
        localizer strategy."""
        outlier = CONF.analyser_outliers_threshold
        robust = ChiSquaredAnalyser()
        return GeneralizedPrimingDetector(
            measure=lambda batch, n: self.executor.trace_test_case(batch, n)[0],
            key=lambda trace: MergedBitmapAnalyser.merged_bitmap(trace, outlier),
            confirm=lambda a, b: not robust.htraces_are_equivalent(a, b),
            reps=reps, verify_reps=verify_reps, localizer=localizer)

    def _localize_boosted_violation(self, violation: Violation, lanes: List[list], n_orig: int,
                                    detector: GeneralizedPrimingDetector
                                    ) -> Optional[Tuple[CrossInputFinding, int, int]]:
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

    def _store_violation_artifact(self, test_case: TestCase, violation: Violation, path: str) -> str:
        """Extend the standard artifact: when the violation was localized by cross-input priming, append
        the two 'complex inputs' to report.txt so it reads on its own."""
        violation_dir = super()._store_violation_artifact(test_case, violation, path)
        residue = violation.cross_input_finding
        if residue is not None:
            with open(f"{violation_dir}/report.txt", "a") as f:
                f.write(self._cross_input_report_section(*residue))
        return violation_dir

    def _cross_input_report_section(
            self, finding: CrossInputFinding, prefix_lane: int, suffix_lane: int,
            n_orig: int) -> str:
        """The counterexample as two sequences of input ids (the reduced 'complex inputs'): both are
        ct-equal (each position holds a member of the same input class), they differ ONLY across positions
        [leaking pair .. detecting pair], yet the detecting pair's hardware trace differs -- so an earlier
        input leaves a microarchitectural residue that changes a later input's readout. The input classes
        (ct-equal members) are listed explicitly so the two sequences are readable on their own."""
        lo, hi = finding.leaking_pair, finding.detecting_pair
        num_lanes = CONF.inputs_per_class

        def members(p):                                # class p's ct-equal input ids
            return [lane * n_orig + p for lane in range(num_lanes)]

        seq_a = [prefix_lane * n_orig + p for p in range(hi + 1)]                       # all prefix lane
        seq_b = [(prefix_lane if p < lo else suffix_lane) * n_orig + p for p in range(hi + 1)]
        kind = "self-dependent (the detecting pair's own input)" if lo == hi \
            else f"cross-input (input {lo} leaves a residue that changes input {hi})"
        classes = "".join(
            f"    position {p:2d}: class {p} = {members(p)}"
            + (f"   (A picks {seq_a[p]}, B picks {seq_b[p]})\n" if p >= lo else "   (both pick %d)\n" % seq_a[p])
            for p in range(hi + 1))
        return (
            "\n## Cross-input priming (leaker/detector localization)\n"
            f"* Leaking pair:   position {lo} (input class {lo})\n"
            f"* Detecting pair: position {hi} (input class {hi})\n"
            f"* Kind: {kind}\n"
            f"* The two sequences below are each ct-equal (every position holds a member of the same input\n"
            f"  class), differ only in positions {lo}..{hi}, yet the detecting pair's hardware trace differs.\n"
            f"* Sequence A: {seq_a}\n"
            f"* Sequence B: {seq_b}\n"
            f"* Input classes (ct-equal members), position by position:\n{classes}")


class Aarch64Fuzzer(CrossInputPrimingMixin, FuzzerGeneric):
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


class Aarch64NoninterferenceFuzzer(CrossInputPrimingMixin, NoninterferenceFuzzer):
    """AArch64 non-interference fuzzer. NI boosting fills each input class with its seal variants (the
    genuine baseline plus decoys), so when enable_cross_input_priming is set, cross-input priming
    localizes canonicality/BTB residues over the seal lanes: the input whose genuine/decoy seal trains a
    predictor entry that surfaces as a later input's cache divergence. Uses the seal-lane regime."""

    # Seal-lane regime: SSBS on for the store-bypass window, no per-input flushing/rotation (would wipe
    # the trained BTB entry), and unpinned execution (pinning measurably raises the non-canonical
    # residual). Captured and restored around the search.
    _CROSS_INPUT_PRIMING_REGIME = (("enable_ssbs", "1"), ("enable_pre_run_flush", "0"),
                                   ("enable_phr_flush", "0"), ("enable_view_rotation", "0"),
                                   ("pin_to_core", "-1"))
