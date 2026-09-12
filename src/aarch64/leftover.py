"""
File: AArch64 non-interference cross-input leftover detector.

Generalized priming for cross-input speculative leftovers. Revizor's priming asks "does a detected
cache divergence follow the *detecting pair*'s own input?" and discards it otherwise. This generalizes
that: when the divergence is caused by a *different*, earlier input -- the "leaking pair", whose leak
lives in the microarchitecture (e.g. a branch-target-buffer entry) and only surfaces as the detecting
pair's cache divergence -- we use the detecting pair's signal to find the leaking pair, with priming's
own definition (same starting microarchitectural state, toggle one pair's genuine/decoy seal, see if
the signal follows).

The search is self-contained -- it knows nothing about the fuzzer or executor -- and is driven by:
  measure(batch, reps) -> per-slot hardware traces of a batch whose slot s holds the genuine or the
                          decoy variant of input s;
  key(trace)           -> a hashable canonical form; two traces are equal iff their keys are equal, so
                          equality is transitive, which the bisection invariant relies on;
  confirm(a, b)        -> whether two traces genuinely differ, by a robust (noise-tolerant) test; used
                          only to re-verify a found boundary, where transitivity is not needed but
                          tolerance to microarchitectural jitter is.

For a detecting pair `d`, over its history [0, d), with `d` held at its decoy variant:
  * Localize (hybrid) -- let H_k = [genuine 0..k, decoy k..d]. With the invariant key(H_lo) != key(H_hi)
    (lo=0, hi=d), bisect to a tipping point k where toggling slot k's genuine/decoy flips d's signal:
    the leaking pair. A chain of priming swaps; needs no predictor model and no monotonicity.
  * Re-verify -- re-measure H_k and H_{k+1} fresh at a larger sample and confirm they robustly differ,
    rejecting a boundary that only appeared under measurement jitter.

A found leaking pair is a valid ct-seq counterexample for ANY predictor (TAGE or trivial) -- the
search never models how the prediction forms, so its *validity* is predictor-agnostic. Two SECONDARY
properties are narrower and hold for a single-entry, most-recent-wins predictor (the audited N3 BTB):
scanning by the two endpoints (all-decoy vs all-genuine history) is guaranteed to *find* an existing
leak -- a history-folded predictor could hide an interior witness between coinciding endpoints, and
guaranteeing detection there is inherently linear, not O(log n); and running the chain standalone
reproduces the leak only where the genuine prefix is replaceable. The reported counterexample is the
whole chain [leaking pair .. detecting pair] (genuine before the leaking pair, decoy from it onward),
one per detecting pair.
"""
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence

Variant = Any        # an opaque per-slot input variant the `measure` callable understands
Trace = Any          # an opaque hardware trace the `key` and `confirm` callables understand
Key = Any            # a hashable canonical form of a trace


@dataclass(frozen=True)
class LeftoverFinding:
    """Toggling input `leaking_pair`'s genuine/decoy seal flips the speculative readout of input
    `detecting_pair`, with the rest of the sequence fixed. `chain` = [leaking_pair .. detecting_pair]
    is the self-contained counterexample: genuine before the leaking pair, decoy from it onward."""
    leaking_pair: int
    detecting_pair: int
    chain: range


class GeneralizedPrimingDetector:
    def __init__(self, measure: Callable[[List[Variant], int], List[Trace]],
                 key: Callable[[Trace], Key], confirm: Callable[[Trace, Trace], bool],
                 *, reps: int, verify_reps: int) -> None:
        assert reps >= 1 and verify_reps >= 1, "reps must be positive"
        self._measure = measure
        self._key = key
        self._confirm = confirm
        self._reps = reps
        self._verify_reps = verify_reps

    def detect(self, genuine: Sequence[Variant], bad: Sequence[Variant]) -> List[LeftoverFinding]:
        """Scan every input as a detecting pair and return each one's leaking pair, if any."""
        assert len(genuine) == len(bad), "genuine and decoy lanes must have equal length"
        found = (self.find_leaking_pair(genuine, bad, d) for d in range(1, len(genuine)))
        return [f for f in found if f is not None]

    def find_leaking_pair(self, genuine: Sequence[Variant], bad: Sequence[Variant],
                          detecting: int) -> Optional[LeftoverFinding]:
        """Locate the leaking pair for detecting pair `detecting`, or None if it has none."""
        lo, hi = 0, detecting
        lo_key = self._probe_key(genuine, bad, detecting, lo)          # all-decoy history
        if lo_key == self._probe_key(genuine, bad, detecting, hi):     # all-genuine history
            return None                                                # signal independent of history
        while hi - lo > 1:                                             # invariant: key(H_lo) == lo_key != key(H_hi)
            mid = (lo + hi) // 2
            if self._probe_key(genuine, bad, detecting, mid) == lo_key:
                lo = mid
            else:
                hi = mid
        if not self._reverify(genuine, bad, detecting, lo):
            return None
        return LeftoverFinding(lo, detecting, range(lo, detecting + 1))

    def _probe(self, genuine: Sequence[Variant], bad: Sequence[Variant],
               detecting: int, k: int, reps: int) -> Trace:
        """detecting pair's trace under H_k: slots [0, k) genuine, [k, detecting] decoy (detecting
        pair held decoy). Slots after it are irrelevant under in-order execution; they stay genuine."""
        batch = list(genuine)
        for s in range(k, detecting + 1):
            batch[s] = bad[s]
        return self._measure(batch, reps)[detecting]

    def _probe_key(self, genuine: Sequence[Variant], bad: Sequence[Variant],
                   detecting: int, k: int) -> Key:
        return self._key(self._probe(genuine, bad, detecting, k, self._reps))

    def _reverify(self, genuine: Sequence[Variant], bad: Sequence[Variant],
                  detecting: int, k: int) -> bool:
        """Fresh, larger-sample, robust re-measurement of the boundary that toggles the leaking pair."""
        h_k = self._probe(genuine, bad, detecting, k, self._verify_reps)
        h_k1 = self._probe(genuine, bad, detecting, k + 1, self._verify_reps)
        return self._confirm(h_k, h_k1)
