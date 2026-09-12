"""
File: AArch64 non-interference cross-input leftover detector.

Generalized-priming search for cross-input speculative leftovers: a predecessor input speculatively
trains a predictor entry (e.g. a branch-target-buffer target) that a later "prober" input then
speculatively reads. The search is self-contained -- it knows nothing about the fuzzer or executor --
and is driven by two injected callables:

  measure(batch, reps) -> the per-slot hardware traces of a batch whose slot `s` holds one of the two
                          supplied variants of input `s` (its genuine or its bad variant);
  key(trace)           -> a hashable canonical form of a trace. Two traces are equal iff their keys
                          are equal, so trace equality is transitive by construction -- which the
                          bisection below relies on (a statistical test would not be transitive).

For each prober `q`, over its history slots [0, q):
  * Detect (generalized priming): compare the prober, held to its genuine variant, under an all-good
    history vs an all-bad history. Equal keys => the prober does not depend on its history => no
    leftover. Different keys => that surviving dependence is the cross-input leak (ordinary priming
    discards it as a context artifact; we report it).
  * Localize (hybrid tipping point): let H_k = [genuine 0..k, bad k..q, genuine prober]. With the
    invariant key(H_lo) != key(H_hi) (lo=0, hi=q), bisect: keep lo wherever key(H_mid) == key(H_lo),
    otherwise move hi. It terminates at hi = lo+1, a tipping point k where toggling slot k's
    genuine/bad state flips the prober. This needs no model of the predictor and no monotonicity, so
    it holds for any predictor.
  * Re-verify: re-measure H_k and H_{k+1} fresh and confirm they still differ, rejecting a boundary
    that only appeared under measurement noise -- which the invariant alone cannot catch.

The search returns one tipping point per leaking prober: a valid violating counterexample, not an
enumeration of every contributor (recursing into the sub-intervals to enumerate all of them is
unsound -- a sub-interval whose two endpoints share a key hides every tip inside it).
"""
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence

Variant = Any        # an opaque per-slot input variant the `measure` callable understands
Trace = Any          # an opaque hardware trace the `key` callable understands
Key = Any            # a hashable canonical form of a trace


@dataclass(frozen=True)
class LeftoverFinding:
    """Toggling the genuine/bad state of input `tipping_point` flips the speculative readout of input
    `prober`, holding the rest of the sequence fixed. `block` = [tipping_point .. prober] is the
    self-contained violating sub-sequence."""
    prober: int
    tipping_point: int
    block: range


class GeneralizedPrimingDetector:
    def __init__(self, measure: Callable[[List[Variant], int], List[Trace]],
                 key: Callable[[Trace], Key], *, reps: int) -> None:
        assert reps >= 1, "reps must be positive"
        self._measure = measure
        self._key = key
        self._reps = reps

    def detect(self, genuine: Sequence[Variant], bad: Sequence[Variant]) -> List[LeftoverFinding]:
        """Scan every prober q in [1, n) for a cross-input leftover from its history [0, q)."""
        assert len(genuine) == len(bad), "genuine and bad lanes must have equal length"
        found = (self._detect_one(genuine, bad, q) for q in range(1, len(genuine)))
        return [f for f in found if f is not None]

    def _probe_key(self, genuine: Sequence[Variant], bad: Sequence[Variant], q: int, k: int) -> Key:
        """key of prober q's trace under H_k: slots [0, k) genuine, [k, q) bad, slot q genuine."""
        batch = list(genuine)
        for s in range(k, q):
            batch[s] = bad[s]
        return self._key(self._measure(batch, self._reps)[q])

    def _detect_one(self, genuine: Sequence[Variant], bad: Sequence[Variant],
                    q: int) -> Optional[LeftoverFinding]:
        lo, hi = 0, q
        lo_key = self._probe_key(genuine, bad, q, lo)              # all-bad history
        if lo_key == self._probe_key(genuine, bad, q, hi):        # all-good history
            return None                                           # prober independent of its history
        while hi - lo > 1:                                        # invariant: key(H_lo) == lo_key != key(H_hi)
            mid = (lo + hi) // 2
            if self._probe_key(genuine, bad, q, mid) == lo_key:
                lo = mid
            else:
                hi = mid
        if self._probe_key(genuine, bad, q, lo) == self._probe_key(genuine, bad, q, lo + 1):
            return None                                           # fresh re-measurement: boundary was noise
        return LeftoverFinding(q, lo, range(lo, q + 1))
