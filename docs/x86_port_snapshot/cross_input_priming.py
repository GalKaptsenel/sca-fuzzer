"""
File: Cross-input priming -- the fuzzing-round stage built on top of the cross-input search.

This module is the seam between `rvzr/cross_input.py`, which implements the architecture- and
fuzzer-independent search, and the rest of Revizor. It holds the three things the search itself
must not know about:

* the *lane layout*: how the boosted input sequence maps onto input classes and lanes, and how a
  flagged violation maps onto the lane pair that diverged;
* the *measurement context*: the initialization sequence (IS) that makes every measured input
  sequence start from the same microarchitectural state;
* the *seams*: which executor call takes the measurements, and how two hardware traces are
  compared while searching and while re-verifying a result.

Boosting (`DataGenerator.generate_boosted`) lays the inputs out as `inputs_per_class` lanes of
`n_classes` inputs each, order-preserving:

```
[ I0, I1, ..., I(n-1),   I0', I1', ..., I(n-1)',   I0'', ..., I(n-1)'' ]
  \\------ lane 0 -----/  \\------ lane 1 -------/   \\----- lane 2 ----/
```

Lane `r` is `boosted[r * n : (r + 1) * n]`. The members at position `j` across lanes
(`boosted[r * n + j]` for every `r`) belong to the same input class, so they are contract-equal by
construction. The toggle that cross-input priming performs at class `j` is therefore "`Ij` versus
one of its own boostings" -- across lanes at the *same* position, never across positions.

Copyright (C) Microsoft Corporation
SPDX-License-Identifier: MIT
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, TYPE_CHECKING

from .config import CONF
from .analyser import merged_bitmap
from .cross_input import (CrossInputFinding, GeneralizedPrimingDetector, Key, LOCALIZERS,
                          ToggleValidator, Trace)
from .logs import dbg, warning

if TYPE_CHECKING:
    from .analyser import Analyser
    from .executor import Executor
    from .logs import FuzzLogger
    from .tc_components.test_case_data import InputData
    from .traces import CTrace, HardwareEqClass, HTrace, Violation


# ==================================================================================================
# Public: Boosted-lane bookkeeping
# ==================================================================================================
def lanes_of(boosted: Sequence[Any], n_classes: int) -> List[List[Any]]:
    """
    Split the flat boosted input list into lanes.

    :param boosted: the boosted input sequence, as produced by `DataGenerator.generate_boosted`
    :param n_classes: the number of input classes, i.e. the number of original inputs
    :return: `len(boosted) // n_classes` lanes of `n_classes` inputs each
    """
    assert n_classes >= 1, "need at least one input class"
    assert boosted and len(boosted) % n_classes == 0, \
        "the boosted list must hold a whole number of lanes"
    n_lanes = len(boosted) // n_classes
    return [list(boosted[r * n_classes:(r + 1) * n_classes]) for r in range(n_lanes)]


def detecting_candidates(hw_classes: Sequence[HardwareEqClass],
                         n_classes: int) -> List[Tuple[int, int, int]]:
    """
    Map a flagged violation onto every candidate set of search arguments.

    `hw_classes` are the violation's measurements -- one contract-equivalence class, i.e. one
    contract trace, across lanes -- partitioned by hardware trace. Two measurements in DIFFERENT
    hardware classes that sit at the SAME class position are two lanes whose readout at that
    position diverged, and each such pair is a usable starting point for the search. Every
    measurement carries an `input_id` into the flat boosted list, so `position = input_id %
    n_classes` and `lane = input_id // n_classes`.

    Every candidate is returned, not just one, because a single arbitrary choice is often unusable:
    position 0 has no history to search at all, and a position whose readout happens to be unstable
    yields nothing even when another position would succeed. Candidates are ordered by DESCENDING
    position, so the ones with the most history -- and hence the best chance of exposing a
    cross-input leak -- are tried first.

    :param hw_classes: the violation's hardware equivalence classes
    :param n_classes: the number of input classes
    :return: a list of (detecting class, base lane, toggle lane) triples, best candidate first;
             empty if the violation cannot be localized
    """
    groups = [g for g in hw_classes if len(g) > 0]
    if len(groups) < 2:
        return []

    # class position -> (hardware class index, lane) of every measurement seen at that position
    seen: Dict[int, List[Tuple[int, int]]] = {}
    for group_id, group in enumerate(groups):
        for measurement in group:
            position = measurement.input_id % n_classes
            seen.setdefault(position, []).append((group_id, measurement.input_id // n_classes))

    candidates: List[Tuple[int, int, int]] = []
    for position in sorted(seen, reverse=True):
        for i, (group_a, lane_a) in enumerate(seen[position]):
            pair = next(((gb, lb) for gb, lb in seen[position][i + 1:] if gb != group_a), None)
            if pair is not None:
                candidates.append((position, lane_a, pair[1]))
                break
    return candidates


@dataclass(frozen=True)
class LaneCounterexample:
    """
    A `CrossInputFinding` expressed in terms of the boosted input sequence: the two input sequences
    that form the counterexample, and the input classes they are drawn from.
    """

    finding: CrossInputFinding
    """ the located boundary """

    prefix_lane: int
    """ the lane that supplies the classes before the leaking class """

    suffix_lane: int
    """ the lane that supplies the classes from the leaking class onwards """

    n_classes: int
    """ the number of input classes, i.e. the lane length """

    n_lanes: int
    """ the number of lanes, i.e. CONF.inputs_per_class """

    # ==============================================================================================
    # Public Methods
    def class_members(self, position: int) -> List[int]:
        """ The boosted input IDs of all contract-equal members of the class at `position` """
        return [lane * self.n_classes + position for lane in range(self.n_lanes)]

    def sequence_a(self) -> List[int]:
        """ The boosted input IDs of the first counterexample sequence: the prefix lane
        throughout """
        detecting = self.finding.detecting_class
        return [self.prefix_lane * self.n_classes + p for p in range(detecting + 1)]

    def sequence_b(self) -> List[int]:
        """ The boosted input IDs of the second counterexample sequence: the prefix lane up to the
        leaking class, then the suffix lane """
        leaking = self.finding.leaking_class
        detecting = self.finding.detecting_class
        return [(self.prefix_lane if p < leaking else self.suffix_lane) * self.n_classes + p
                for p in range(detecting + 1)]

    def full_str(self) -> str:
        """ Render the counterexample as a violation report section """
        leaking = self.finding.leaking_class
        detecting = self.finding.detecting_class
        seq_a = self.sequence_a()
        seq_b = self.sequence_b()

        if self.finding.is_self_dependent:
            kind = "self-dependent (the detecting input leaks about itself)"
        else:
            kind = (f"cross-input (input class {leaking} leaves a microarchitectural residue that "
                    f"changes the hardware trace of input class {detecting})")

        classes = ""
        for p in range(detecting + 1):
            picks = f"both pick {seq_a[p]}" if p < leaking \
                else f"A picks {seq_a[p]}, B picks {seq_b[p]}"
            classes += f"    class {p:3d}: members {self.class_members(p)}   ({picks})\n"

        return ("\n## Cross-Input Priming\n"
                f"* Leaking input class: {leaking}\n"
                f"* Detecting input class: {detecting}\n"
                f"* Kind: {kind}\n"
                "* The two sequences below hold one member of every input class, so they are\n"
                f"  contract-equal position by position. They differ only in classes "
                f"{leaking}..{detecting},\n"
                "  yet the hardware trace of the detecting class differs between them.\n"
                f"* Sequence A: {seq_a}\n"
                f"* Sequence B: {seq_b}\n"
                f"* Input classes (contract-equal members), class by class:\n{classes}")


# ==================================================================================================
# Public: Measurement context
# ==================================================================================================
@contextmanager
def initialization_sequence(executor: Executor) -> Iterator[None]:
    """
    Configure the executor so that every batch it measures is preceded by the initialization
    sequence (IS) and by nothing else.

    The IS is the routine that drives the microarchitectural state to one reproducible value before
    each input sequence, so that two sequences differing in a single input are compared from the
    same starting point. The executor kernel module already implements it: `run_experiment` flushes
    the microarchitectural state once per batch and then executes `executor_warmups` warm-up
    rounds, and -- crucially -- it does *not* flush between the inputs inside the batch. That
    intra-batch cascade is precisely the channel that cross-input priming reads, so the IS must
    separate the measured sequences and must not be inserted between the inputs within one.

    Hence this context manager only has to (a) make sure that the flush is enabled and (b) clear
    the ignore list, whose input IDs index the boosted input sequence and would otherwise zero out
    the traces of unrelated slots in the (shorter) spliced sequences. Both are restored on exit.

    :param executor: the executor to configure
    """
    saved_ignore_list = executor.get_ignore_list()
    executor.set_ignore_list([])
    executor.set_pre_run_flush(True)
    try:
        yield
    finally:
        executor.set_pre_run_flush(CONF.enable_pre_run_flush)
        executor.set_ignore_list(saved_ignore_list)


# ==================================================================================================
# Public: The fuzzing-round stage
# ==================================================================================================
_KEY_OUTLIER_THRESHOLD: float = 0.5
""" Fraction of repetitions a sample must reach to enter the localizer's key.

Deliberately NOT CONF.analyser_outliers_threshold, because the key and the analyser have different
jobs. The analyser decides equivalence and is conservative: unioning in a rare sample can only make
it declare two traces equivalent, never flag a violation. The key, by contrast, must be a STABLE
fingerprint, and the merged bitmap is monotone in the sample size -- the more repetitions, the more
bits get set. With the analyser's default of 0.1 a sample seen in a tenth of the runs still enters
the key, so at large sample sizes every splice saturates to the same bitmap and no boundary can be
localized at all. Requiring a majority keeps the key pinned to the dominant behaviour, which is what
makes it comparable across splices and across sample sizes. """


def _validated(htrace: HTrace) -> HTrace:
    """
    Reject a corrupted or ignored hardware trace. The search cannot draw any conclusion from one,
    so the round is abandoned, exactly as the standard priming check does.

    :param htrace: the hardware trace to validate
    :return: the same trace, if it is usable
    :raises IOError: if the trace is empty, corrupted or ignored
    """
    if htrace.is_empty() or htrace.is_corrupted_or_ignored():
        raise IOError("Corrupted hardware trace during cross-input priming")
    return htrace


def _contract_equal_toggle(base_ctraces: List[CTrace],
                           toggle_ctraces: List[CTrace]) -> ToggleValidator:
    """
    Build the validator that accepts a boundary only if its two inputs are contract-equal.

    :param base_ctraces: contract traces of the lane supplying the prefix
    :param toggle_ctraces: contract traces of the lane supplying the toggled suffix
    :return: a predicate over class positions
    """
    return lambda t: bool(base_ctraces[t] == toggle_ctraces[t])


class CrossInputPrimingCheck:
    """
    The cross-input priming stage of a fuzzing round; a generalization of the standard priming
    check (`_RoundManager._priming_check`).

    Goal: the same as the standard check -- tell a genuine violation apart from cross-talk between
    inputs -- but without rejecting the violations whose leakage is carried between inputs by
    microarchitectural state that the executor cannot read directly (a branch-target-buffer entry,
    branch history, the TLB, the PHT, the RSB, the prefetcher). The standard check keeps a
    violation only if the divergence follows the detecting input's own data, so it discards every
    such leak as cross-talk.

    Approach: the divergence is caused by *some* input in the sequence; find out which one. Let the
    two lanes that diverged be `a` and `b` (a lane holds one member of every input class), and let
    the divergence be observed at class `d`. Measure the spliced sequences
    `sigma(t) = a[0..t) ++ b[t..d]` for varying `t`, reading the trace of class `d` each time.
    Since `sigma(d + 1)` is lane `a` and `sigma(0)` is lane `b`, the two ends disagree, so some
    neighbouring pair must disagree too. That pair differs in exactly one input -- the two
    contract-equal members of class `t` -- and runs the same sequence afterwards, so it is a
    genuine contract counterexample regardless of which microarchitectural structure carried the
    leakage. `t == d` recovers the standard priming check; `t < d` is the generalization. Setting
    CONF.cross_input_leaks_only keeps only the latter, by narrowing the searched interval to the
    classes before `d` -- see `GeneralizedPrimingDetector.find_leaking_class`.

    If no such `t` exists, the divergence does not follow any input in the sequence, and the
    violation is a false positive -- exactly as in the standard check.
    """

    def __init__(self, executor: Executor, analyser: Analyser, log: FuzzLogger) -> None:
        """
        :param executor: the executor that takes the measurements
        :param analyser: the analyser used to re-verify a located result
        :param log: the logger of the fuzzing round
        """
        self._executor = executor
        self._analyser = analyser
        self._log = log

    # ==============================================================================================
    # Public Interface
    def run(self, violations: List[Violation], org_inputs: List[InputData],
            boosted_inputs: List[InputData], ctraces: List[CTrace]) -> List[Violation]:
        """
        Localize the leaking input class behind the flagged violations.

        :param violations: the violations flagged by the analyser
        :param org_inputs: the original (non-boosted) inputs; one per input class
        :param boosted_inputs: the boosted input sequence that was measured
        :param ctraces: the contract traces of `boosted_inputs`, used to check that a located
               boundary really toggles between two contract-equal inputs
        :return: a single-element list with the first localized violation, which then carries the
                 counterexample in its `cross_input_finding` field; or an empty list if none of the
                 violations could be localized
        """
        if not violations:
            return []

        n_classes = len(org_inputs)
        lanes = lanes_of(boosted_inputs, n_classes)
        ctrace_lanes = lanes_of(ctraces, n_classes)
        detector = self._make_detector(violations[0].measurements[0].htrace.sample_size())

        with initialization_sequence(self._executor):
            try:
                for checked, violation in enumerate(reversed(violations)):
                    self._log.priming(len(violations) - checked)
                    counterexample = self._localize(violation, lanes, ctrace_lanes, detector)
                    if counterexample is None:
                        continue

                    # the violation was localized; it's a genuine violation
                    violation.cross_input_finding = counterexample
                    dbg(
                        "fuzzer", "Cross-input priming: input class "
                        f"{counterexample.finding.leaking_class} leaks into input class "
                        f"{counterexample.finding.detecting_class}")
                    return [violation]
            except IOError:
                warning("fuzzer",
                        "Tracing error during cross-input priming. Skipping this test case")

        # no input in the sequence caused the divergence, so it's a false positive
        return []

    # ==============================================================================================
    # Private Interface
    def _make_detector(self, reps: int) -> GeneralizedPrimingDetector:
        """
        Create the search and wire it to this round's executor and analyser.

        The search uses two different comparisons of hardware traces, for two different purposes:

        * while localizing, traces are compared by their merged bitmap, denoised at
          `_KEY_OUTLIER_THRESHOLD` rather than at the analyser's threshold (see that constant). The
          bitmap is a canonical form, so the comparison is transitive, which the bisection relies
          on; the statistical comparisons implemented by the analysers are not.
        * a located result is re-measured at the largest configured sample size and re-checked with
          the configured analyser, which is noise-tolerant. This is the same test that declared the
          violation in the first place, so an accepted result is a divergence of the very kind the
          fuzzer reports.

        :param reps: the sample size to localize with; normally the size at which the violation was
               detected
        :return: the configured detector
        """
        analyser = self._analyser

        def key(trace: Trace) -> Key:
            return merged_bitmap(_validated(trace), _KEY_OUTLIER_THRESHOLD)

        def confirm(trace1: Trace, trace2: Trace) -> bool:
            return not analyser.htraces_are_equivalent(_validated(trace1), _validated(trace2))

        return GeneralizedPrimingDetector(
            measure=self._executor.trace_test_case,
            key=key,
            confirm=confirm,
            reps=reps,
            verify_reps=CONF.executor_sample_sizes[-1],
            localizer=LOCALIZERS[CONF.cross_input_priming_localizer])

    def _localize(self, violation: Violation, lanes: List[List[InputData]],
                  ctrace_lanes: List[List[CTrace]],
                  detector: GeneralizedPrimingDetector) -> Optional[LaneCounterexample]:
        """
        Localize the leaking input class behind a single flagged violation.

        The violation is first mapped onto the class at which the divergence was observed and the
        two lanes that diverged there. EVERY such candidate is tried, best first, because one
        arbitrary choice is often unusable: position 0 has no history at all, and an unstable
        readout yields nothing even when another position would succeed. Each candidate is searched
        from both lanes as the base, mirroring the standard priming check's symmetric swap: the
        divergence may be caused by either lane's prefix.

        A located boundary is additionally required to toggle between two CONTRACT-EQUAL inputs.
        Boosting makes the lanes contract-equal position by position only as far as the taint
        tracker pins every contract-relevant input bit; where it does not (under `l1d`, for example,
        no branch condition is tainted at all), a lane pair at some position may have different
        contract traces, and toggling it proves nothing -- a contract is free to let architecturally
        different inputs leave different microarchitectural residue.

        :param violation: the violation to localize
        :param lanes: the boosted input sequence, split into lanes
        :param ctrace_lanes: the contract traces of `lanes`, with the same layout
        :param detector: the detector to run the search with
        :return: the located counterexample, or None if the violation could not be localized
        """
        n_classes = len(lanes[0])
        candidates = detecting_candidates(violation.get_hw_classes(), n_classes)
        for detecting, lane_a, lane_b in candidates:
            directions = ((lane_a, lane_b, True), (lane_b, lane_a, False))
            for prefix_lane, suffix_lane, from_first in directions:
                finding = detector.find_leaking_class(
                    lanes[prefix_lane],
                    lanes[suffix_lane],
                    detecting,
                    from_first,
                    exclude_self_dependence=CONF.cross_input_leaks_only,
                    is_valid_toggle=_contract_equal_toggle(
                        ctrace_lanes[prefix_lane], ctrace_lanes[suffix_lane]))
                if finding is not None:
                    return LaneCounterexample(finding, prefix_lane, suffix_lane, n_classes,
                                              len(lanes))
        return None
