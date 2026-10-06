"""
File: Cross-input priming, a generalization of the priming check.

Standard priming (see `_RoundManager._priming_check`) discards a suspected violation unless the
divergence follows the *detecting pair*'s own input data. That rejects a whole class of genuine
leaks: the ones whose only footprint is outside the cache that the executor reads (a
branch-target-buffer entry, branch history, the TLB, the PHT, the RSB, the prefetcher). Such a
leak is left behind by an *earlier* input and only becomes observable as a *later* input's cache
divergence. Cross-input priming certifies exactly those, without ever reading the leftover
state directly: it lets one input's leftover propagate into the cache of a later run in the same
batch, and reads it there.

Terminology (mirrors the write-up "Generalized priming"):

* class        an input class, i.e. a set of inputs that share a contract trace. Boosting
               (`DataGenerator.generate_boosted`) produces `inputs_per_class` members per class.
* lane         one input per class, indexed by class `0..k`; `base` and `toggle` below. Two lanes
               meet every class with different -- but contract-equal -- members.
* IS           the initialization sequence: a fixed routine that drives the microarchitectural
               state to one reproducible value before each lane. It separates lanes but is *not*
               executed between the classes inside a lane -- that intra-lane forward cascade is
               the leftover channel being exploited. See `_initialization_sequence` in fuzzer.py
               for how the IS is realized on top of the executor.
* splice       `sigma_{base->toggle}(t)`: classes `[0, t)` from `base`, `[t, detecting]` from
               `toggle`. Neighbouring splices `sigma(t)` and `sigma(t + 1)` differ in exactly one
               input (class `t`) and share everything else, including the whole suffix.
* r(t)         the class-`detecting` hardware trace of `sigma(..., t)`. A localizer sees the chain
               only through its key, `r_key(t) = key(r(t))`.
* boundary     a `t` with `r_key(t) != r_key(t + 1)`: toggling class `t` alone flips the detecting
               class's readout, so the two members of class `t` are a genuine counterexample.
* t_max        the largest boundary. `linear_scan` returns it; `exponential_search` returns some
               boundary (the gallop biases it towards the detecting class, the most recent
               trainer).

Soundness: `sigma(t)` and `sigma(t + 1)` run identical histories except for the one input at class
`t`, and then run the same suffix, yet the detecting class reads differently. Hence every boundary
is a genuine violation, whatever microarchitectural structure carries the leftover -- the search
never models the channel, it only compares opaque traces. A boundary always exists whenever the
two ends disagree, and there may be several, each a distinct leaking class.

The search is self-contained -- it knows nothing about the fuzzer or the executor -- and is driven
by three injected seams:

* `measure(batch, reps) -> [trace per slot]` -- the only hardware seam.
* `key(trace) -> hashable` -- a canonical, denoised form used *during* the search. Two traces are
  equal iff their keys are equal, so equality is transitive, which the bisection relies on.
* `confirm(a, b) -> bool` -- a robust, noise-tolerant test of "do these really differ", used only
  to re-verify a located boundary at a larger sample size, where transitivity is not needed but
  tolerance to microarchitectural jitter is.

Copyright (C) Microsoft Corporation
SPDX-License-Identifier: MIT
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

Variant = Any
""" An opaque per-slot input variant that the `measure` callable understands """

Trace = Any
""" An opaque hardware trace that the `key` and `confirm` callables understand """

Key = Any
""" A hashable canonical form of a trace """

Lane = Sequence[Variant]
""" One input variant per class, indexed by class """

RKey = Callable[[int], Key]
""" The oracle `r_key(t) = key(r(t))` that a localizer queries """

Localizer = Callable[[RKey, int, int], int]
""" Given `r_key` and an interval `[lo, hi)` with `r_key(lo) != r_key(hi)`, return a boundary
`t` (`lo <= t < hi`) such that `r_key(t) != r_key(t + 1)` """

ToggleValidator = Callable[[int], bool]
""" Whether class `t`'s two members are a legitimate toggle, i.e. whether they are contract-equal.
Only the boundary class has to satisfy this: the neighbouring splices `sigma(t)` and `sigma(t + 1)`
agree on every other class, so those classes are literally the same input in both batches. """


# ==================================================================================================
# Public: Splicing
# ==================================================================================================
def splice(base: Lane, toggle: Lane, detecting: int, t: int) -> List[Variant]:
    """
    Build the spliced lane `sigma_{base->toggle}(t)`, truncated to the classes `[0, detecting]`.

    Classes `[0, t)` are taken from `base` and classes `[t, detecting]` from `toggle`. Classes
    after `detecting` keep their `base` variant, as they cannot affect the class-`detecting`
    readout. At `t = detecting + 1` the toggled range is empty, which is the all-`base` end.

    :param base: the lane that supplies the prefix
    :param toggle: the lane that supplies the toggled suffix
    :param detecting: the class whose hardware trace is read out
    :param t: the first class taken from `toggle`
    :return: the spliced input sequence, one variant per class
    """
    assert 0 <= t <= detecting + 1, "t must be within [0, detecting + 1]"
    batch = list(base)
    for c in range(t, detecting + 1):
        batch[c] = toggle[c]
    return batch


# ==================================================================================================
# Public: Localizers
# ==================================================================================================
def linear_scan(r_key: RKey, lo: int, hi: int) -> int:
    """
    Return `t_max`, the largest boundary in `[lo, hi)`.

    Read `r` downwards from the all-`base` end `hi`; the first `t` whose value differs from the
    class above it is `t_max`, because `r` is constant above `t_max` (no boundary lies there).
    Finding the largest boundary is inherently linear, so this costs up to `hi - lo + 1` probes.

    :param r_key: the oracle `r_key(t)`
    :param lo: the lower end of the search interval
    :param hi: the all-`base` end of the search interval; requires `r_key(lo) != r_key(hi)`
    :return: the largest boundary in `[lo, hi)`
    """
    upper = r_key(hi)
    for t in range(hi - 1, lo - 1, -1):
        cur = r_key(t)
        if cur != upper:
            return t
        upper = cur
    return lo  # unreachable, given that r_key(lo) != r_key(hi)


def exponential_search(r_key: RKey, lo: int, hi: int) -> int:
    """
    Return some boundary in `[lo, hi)`, found by galloping from the all-`base` end and then
    bisecting.

    Probe `t = hi - 1, hi - 2, hi - 4, ...` until `r` leaves the all-`base` value `ref`; the last
    doubling window then brackets a boundary (`r` differs at `t` and equals `ref` at `prev`).
    Bisecting that window is the plain binary search, so the returned boundary is genuine. The
    gallop only picks the bracket, and it biases the result towards the detecting class -- the
    most recent trainer, and the most likely leaker. Costs `O(log d)` probes in the distance `d`
    to that boundary, never worse than a bisection of the full interval.

    :param r_key: the oracle `r_key(t)`
    :param lo: the lower end of the search interval
    :param hi: the all-`base` end of the search interval; requires `r_key(lo) != r_key(hi)`
    :return: a boundary in `[lo, hi)`
    """
    ref = r_key(hi)
    gap, prev = 1, hi  # invariant: r_key(prev) == ref
    while True:
        t = hi - gap
        if t <= lo:
            low_end = lo  # r_key(lo) != ref by the caller's promise
            break
        if r_key(t) != ref:
            low_end = t
            break
        prev = t
        gap *= 2

    # bracket [low_end, high_end]: r_key(low_end) != ref == r_key(high_end)
    high_end = prev
    while high_end - low_end > 1:
        mid = (low_end + high_end) // 2
        if r_key(mid) != ref:
            low_end = mid
        else:
            high_end = mid
    return low_end


LOCALIZERS: Dict[str, Localizer] = {
    "any": exponential_search,
    "optimal": linear_scan,
}
""" The localizer strategies selectable via `CONF.cross_input_priming_localizer` """


# ==================================================================================================
# Public: Detector
# ==================================================================================================
@dataclass(frozen=True)
class CrossInputFinding:
    """
    A located boundary: toggling the two contract-equal members of class `leaking_class` flips the
    hardware trace of class `detecting_class`, with the rest of the sequence held fixed.
    """

    leaking_class: int
    """ the class whose two members flip the readout """

    detecting_class: int
    """ the class whose hardware trace was read out """

    prefix_from_first: bool
    """ which lane supplied the prefix: True if the first lane supplied the classes before
    `leaking_class` and the second lane supplied the classes from it onwards; False for the
    mirrored direction """

    @property
    def is_self_dependent(self) -> bool:
        """ Whether the leaking class is the detecting class itself, i.e. the input leaks about
        itself. This is the case that standard priming already detects. """
        return self.leaking_class == self.detecting_class

    @property
    def chain(self) -> range:
        """ The classes `[leaking_class .. detecting_class]` that form the counterexample """
        return range(self.leaking_class, self.detecting_class + 1)


class GeneralizedPrimingDetector:
    """
    Localizes the leaking class behind a detected divergence by splicing two lanes and probing the
    detecting class's hardware trace. See the module docstring for the algorithm and its seams.
    """

    def __init__(self,
                 measure: Callable[[List[Variant], int], List[Trace]],
                 key: Callable[[Trace], Key],
                 confirm: Callable[[Trace, Trace], bool],
                 *,
                 reps: int,
                 verify_reps: int,
                 localizer: Localizer = exponential_search) -> None:
        """
        :param measure: runs a batch of variants `reps` times and returns one trace per slot
        :param key: maps a trace to a hashable canonical form with transitive equality
        :param confirm: robustly tests whether two traces genuinely differ
        :param reps: the sample size used while localizing
        :param verify_reps: the (larger) sample size used to re-verify a located boundary
        :param localizer: the search strategy over the splice chain
        """
        assert reps >= 1 and verify_reps >= 1, "sample sizes must be positive"
        self._measure = measure
        self._key = key
        self._confirm = confirm
        self._reps = reps
        self._verify_reps = verify_reps
        self._localizer = localizer

    # ==============================================================================================
    # Public Interface
    def find_leaking_class(self,
                           base: Lane,
                           toggle: Lane,
                           detecting: int,
                           prefix_from_first: bool = True,
                           exclude_self_dependence: bool = False,
                           is_valid_toggle: Optional[ToggleValidator] = None
                           ) -> Optional[CrossInputFinding]:
        """
        Locate the leaking class behind the divergence at class `detecting`, sweeping the prefix
        from `base` towards `toggle` with the injected localizer.

        Each splice is measured at most once (memoized), so `r_key` is a stable value that the
        localizer may query freely. A located boundary is re-measured at `verify_reps` and
        confirmed with the robust test, which rejects boundaries that only appeared under
        measurement jitter.

        `exclude_self_dependence` does not merely filter the result: it narrows the searched
        interval to the classes *strictly before* the detecting one, and compares `r(detecting)`
        (the splice whose only toggled class is the detecting one) against `r(0)` as its ends. That
        matters, because a chain may hold several boundaries. Were the interval left at
        `[0, detecting]`, a self-dependent boundary would be found -- `exponential_search`
        gallops from the all-base end, so it is biased towards exactly that one -- and the whole
        violation would then be discarded even though a cross-input boundary also existed further
        down the chain.

        If the narrowed ends agree, no strictly-cross-input boundary can be *localized*. Note that
        this does not prove that none exists: a history-folded predictor can hide an interior
        witness between coinciding ends, and guaranteeing detection there is inherently linear.

        :param base: the lane that supplies the prefix
        :param toggle: the lane that supplies the toggled suffix
        :param detecting: the class whose hardware trace is read out
        :param prefix_from_first: recorded in the finding; see `CrossInputFinding`
        :param exclude_self_dependence: if True, search only the classes strictly before the
               detecting one, so that the finding is strictly cross-input
        :param is_valid_toggle: if given, a located boundary is accepted only if this returns True
               for it. The caller uses this to assert that the boundary class's two members really
               are contract-equal: a boundary whose two inputs have DIFFERENT contract traces is
               not a counterexample at all, because a contract is free to let architecturally
               different inputs leave different microarchitectural residue. Lanes built by boosting
               are contract-equal position by position only as far as the taint tracker pins every
               contract-relevant input bit, which does not hold under every observation clause --
               under `l1d`, for instance, no branch condition is tainted at all.
        :return: the located finding, or None if the divergence could not be localized
        """
        assert len(base) == len(toggle), "the two lanes must have equal length"
        assert 0 <= detecting < len(base), "the detecting class must be within the lanes"
        lo = 0
        hi = detecting if exclude_self_dependence else detecting + 1
        if hi <= lo:  # the detecting class is the first one; it has no history to search
            return None
        cache: Dict[int, Key] = {}

        def r_key(t: int) -> Key:
            if t not in cache:
                cache[t] = self._key(self._probe(base, toggle, detecting, t, self._reps))
            return cache[t]

        # the readout does not depend on the searched part of the history -> nothing to localize
        if r_key(lo) == r_key(hi):
            return None

        boundary = self._localizer(r_key, lo, hi)
        if is_valid_toggle is not None and not is_valid_toggle(boundary):
            return None
        if not self._reverify(base, toggle, detecting, boundary):
            return None
        return CrossInputFinding(boundary, detecting, prefix_from_first)

    def detect(self,
               lane_a: Lane,
               lane_b: Lane,
               exclude_self_dependence: bool = False) -> List[CrossInputFinding]:
        """
        Scan every class as a detecting class and collect all the findings.

        For each detecting class, the prefix is swept from both lanes as the base (`a -> b` and
        `b -> a`), mirroring priming's symmetric swap: the divergence may be caused by either
        lane's prefix. Neither lane is assumed to be a baseline. Findings are de-duplicated by
        (leaking class, detecting class).

        Note that the fuzzer does not use this entry point -- it localizes the one violation that
        the analyser flagged, see `_RoundManager._cross_input_priming_check`. This method is the
        exhaustive variant, useful for offline analysis of a known-interesting input sequence.

        :param lane_a: the first lane
        :param lane_b: the second lane
        :param exclude_self_dependence: keep only strictly cross-input findings
        :return: the list of located findings
        """
        assert len(lane_a) == len(lane_b), "the two lanes must have equal length"
        findings: Dict[Any, CrossInputFinding] = {}
        for detecting in range(1, len(lane_a)):
            for base, toggle, prefix_from_first in ((lane_a, lane_b, True), (lane_b, lane_a,
                                                                             False)):
                finding = self.find_leaking_class(base, toggle, detecting, prefix_from_first,
                                                  exclude_self_dependence)
                if finding is not None:
                    findings.setdefault((finding.leaking_class, finding.detecting_class), finding)
        return list(findings.values())

    # ==============================================================================================
    # Private Interface
    def _probe(self, base: Lane, toggle: Lane, detecting: int, t: int, reps: int) -> Trace:
        """ Measure `r(t)`: the detecting class's trace under the splice `sigma(t)` """
        return self._measure(splice(base, toggle, detecting, t), reps)[detecting]

    def _reverify(self, base: Lane, toggle: Lane, detecting: int, t: int) -> bool:
        """ Re-measure the two sides of the boundary at `verify_reps` and robustly confirm that
        they differ """
        trace_low = self._probe(base, toggle, detecting, t, self._verify_reps)
        trace_high = self._probe(base, toggle, detecting, t + 1, self._verify_reps)
        return self._confirm(trace_low, trace_high)
