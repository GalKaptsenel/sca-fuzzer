"""
File: AArch64 non-interference cross-input leftover detector (generalized priming).

Revizor's priming asks "does a detected cache divergence follow the *detecting pair*'s own input?" and
discards it otherwise. This generalizes that: when the divergence is caused by a *different*, earlier
input -- the "leaking pair", whose leak lives in the microarchitecture (e.g. a branch-target-buffer
entry) and only surfaces as the detecting pair's cache divergence -- we use the detecting pair's signal
to find the leaking pair, under priming's own definition (same starting microarchitectural state, toggle
one pair's two ct-equal variants, see if the signal follows).

The search is self-contained -- it knows nothing about the fuzzer or executor -- and is driven by:
  measure(batch, reps) -> per-slot hardware traces of a batch whose slot s holds the base or the toggle
                          variant of input s;
  key(trace)           -> a hashable canonical form; two traces are equal iff their keys are equal, so
                          equality is transitive, which the bisection invariant relies on;
  confirm(a, b)        -> whether two traces genuinely differ, by a robust (noise-tolerant) test; used
                          only to re-verify a found boundary, where transitivity is not needed but
                          tolerance to microarchitectural jitter is.

Objects mirror the write-up "Generalized priming":
  * lane          L^(l): one input variant per class, indexed by class 0..k (`base` and `toggle` below).
  * splice        sigma_{base->toggle}(t): classes [0, t) from `base`, [t, detecting] from `toggle`.
  * r(t)          the class-`detecting` hardware trace of splice(..., t); a localizer sees the chain only
                  through its key, r_key(t) = key(r(t)).
  * boundary      a t with r_key(t) != r_key(t+1): toggling class t flips the detecting pair's readout,
                  so (base[t], toggle[t]) is a leaking pair.
  * t_max         the largest boundary. `linear_scan` returns it; `exponential_search` returns some
                  boundary (galloping biases it toward the detecting pair, the most-recent trainer).

Two localizers, injected as `localizer` (both share the splice/r/re-verify core):
  * linear_scan       returns t_max. Reading r downward from the all-base end, the first change is t_max
                      (no boundary lies above it). Finding the largest boundary is inherently linear.
  * exponential_search (default) galloping + bisection. Probe t = hi-1, hi-2, hi-4, ... until r leaves
                      the all-base value, bracketing a boundary in the last doubling window, then bisect
                      that window. Galloping only *finds the bracket*; the bisection inside is the proven
                      binary search, so the returned boundary is genuine. Cost O(log d) in the distance
                      d to that boundary, never worse than a plain bisection.

For a detecting pair `d`, over its history [0, d), searched from BOTH bases (mirroring priming's symmetric
swap: the divergence may be caused by the prefix being base OR toggle). A boundary re-verify then
re-measures r(t) and r(t+1) fresh at a larger sample and confirms they robustly differ, rejecting a
boundary that only appeared under measurement jitter.

A boundary at the detecting pair itself (leaking pair == detecting pair) is the pair's OWN variant
flipping its OWN readout -- the own-target confound. It is a valid finding by default (an input that
leaks about itself); pass exclude_self_dependence to keep only strictly cross-input leftovers.

A found leaking pair is a valid ct-seq counterexample for ANY predictor -- the search never models how the
prediction forms, so its *validity* is predictor-agnostic. Guaranteeing that the endpoint promise
(all-base vs all-toggle) *finds* an existing leak is a narrower property of a single-entry, most-recent-
wins predictor (the audited N3 BTB); a history-folded predictor could hide an interior witness between
coinciding endpoints, and guaranteeing detection there is inherently linear. The reported counterexample
is the whole chain [leaking pair .. detecting pair] (the base lane before the leaking pair, the toggle
lane from it onward), de-duplicated by (leaking pair, detecting pair) across the two bases.
"""
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence

Variant = Any        # an opaque per-slot input variant the `measure` callable understands
Trace = Any          # an opaque hardware trace the `key` and `confirm` callables understand
Key = Any            # a hashable canonical form of a trace
Lane = Sequence[Variant]

# A localizer sees the chain only through r_key(t) = key(r(t)). Given r_key(lo) != r_key(hi), it returns
# a boundary t (lo <= t < hi) with r_key(t) != r_key(t+1).
RKey = Callable[[int], Key]
Localizer = Callable[[RKey, int, int], int]


def splice(base: Lane, toggle: Lane, detecting: int, t: int) -> List[Variant]:
    """sigma_{base->toggle}(t) truncated to [0, detecting]: classes [0, t) from `base`, [t, detecting] from
    `toggle`; classes after `detecting` stay `base` (they do not affect the class-`detecting` readout). At
    t = detecting + 1 the toggled range is empty -- the all-base end."""
    batch = list(base)
    for c in range(t, detecting + 1):
        batch[c] = toggle[c]
    return batch


def linear_scan(r_key: RKey, lo: int, hi: int) -> int:
    """Return t_max, the LARGEST boundary in [lo, hi). Read r downward from the all-base end `hi`; the
    first t whose value differs from the class above it is t_max, since r is constant above t_max (no
    boundary lies there). Assumes r_key(lo) != r_key(hi)."""
    upper = r_key(hi)
    for t in range(hi - 1, lo - 1, -1):
        cur = r_key(t)
        if cur != upper:
            return t
        upper = cur
    return lo                                        # unreachable given r_key(lo) != r_key(hi)


def exponential_search(r_key: RKey, lo: int, hi: int) -> int:
    """Gallop from the all-base end `hi`, then bisect the bracket. Probe t = hi-1, hi-2, hi-4, ... until r
    leaves the all-base value `ref`; the last window [t, prev] then brackets a boundary (r differs at t,
    equals ref at prev). Bisecting that window is the proven binary search. Assumes r_key(lo) != r_key(hi),
    so the gallop always finds a differing t (at worst t = lo)."""
    ref = r_key(hi)                                  # the all-base end value
    gap, prev = 1, hi                                # invariant: r_key(prev) == ref
    while True:
        t = hi - gap
        if t <= lo:
            lo2 = lo                                 # r_key(lo) != ref by the caller's promise
            break
        if r_key(t) != ref:
            lo2 = t
            break
        prev = t
        gap *= 2
    hi2 = prev                                       # bracket [lo2, hi2]: r_key(lo2) != ref == r_key(hi2)
    while hi2 - lo2 > 1:
        mid = (lo2 + hi2) // 2
        if r_key(mid) != ref:
            lo2 = mid
        else:
            hi2 = mid
    return lo2                                        # boundary: r_key(lo2) != r_key(lo2 + 1)


@dataclass(frozen=True)
class LeftoverFinding:
    """Toggling input `leaking_pair`'s two ct-equal variants flips the readout of input `detecting_pair`,
    the rest of the sequence fixed. `chain` = [leaking_pair .. detecting_pair] is the self-contained
    counterexample. `prefix_genuine` is the base the search swept from: True = genuine before the leaking
    pair and decoy from it on; False = the mirror (decoy before, genuine from it on)."""
    leaking_pair: int
    detecting_pair: int
    chain: range
    prefix_genuine: bool


class GeneralizedPrimingDetector:
    def __init__(self, measure: Callable[[List[Variant], int], List[Trace]],
                 key: Callable[[Trace], Key], confirm: Callable[[Trace, Trace], bool],
                 *, reps: int, verify_reps: int, localizer: Localizer = exponential_search) -> None:
        assert reps >= 1 and verify_reps >= 1, "reps must be positive"
        self._measure = measure
        self._key = key
        self._confirm = confirm
        self._reps = reps
        self._verify_reps = verify_reps
        self._localizer = localizer

    def detect(self, genuine: Sequence[Variant], bad: Sequence[Variant],
               exclude_self_dependence: bool = False) -> List[LeftoverFinding]:
        """Scan every input as a detecting pair. For each, search from BOTH bases -- genuine-prefix and
        decoy-prefix -- mirroring priming's symmetric swap: the divergence may be caused by the prefix
        being good OR bad. De-duplicate by (leaking pair, detecting pair). A self-dependent finding
        (leaking pair == detecting pair -- the pair leaks about itself) is a valid result by default;
        pass exclude_self_dependence to drop it and keep only strictly cross-input leftovers."""
        assert len(genuine) == len(bad), "the two lanes must have equal length"
        findings: dict = {}
        for detecting in range(1, len(genuine)):
            for base, toggle, prefix_genuine in ((genuine, bad, True), (bad, genuine, False)):
                f = self.find_leaking_pair(base, toggle, detecting, prefix_genuine,
                                           exclude_self_dependence)
                if f is not None:
                    findings.setdefault((f.leaking_pair, f.detecting_pair), f)
        return list(findings.values())

    def find_leaking_pair(self, base: Sequence[Variant], toggle: Sequence[Variant], detecting: int,
                          prefix_genuine: bool = True,
                          exclude_self_dependence: bool = False) -> Optional[LeftoverFinding]:
        """Locate the leaking pair for `detecting`, sweeping the prefix from `base` toward `toggle` with the
        injected localizer. Each split is measured at most once (memoized), so r_key is a stable value the
        localizer may query freely. A tip at the detecting pair itself (leaking pair == detecting pair) is
        self-dependence -- the pair's own variant flips its own readout; returned by default, dropped when
        exclude_self_dependence."""
        lo, hi = 0, detecting + 1
        cache: dict = {}

        def r_key(t: int) -> Key:                    # r(t)'s key; each split measured once, so it is stable
            if t not in cache:
                cache[t] = self._probe_key(base, toggle, detecting, t)
            return cache[t]

        if r_key(lo) == r_key(hi):                   # signal independent of history -> nothing to localize
            return None
        boundary = self._localizer(r_key, lo, hi)
        if exclude_self_dependence and boundary == detecting:   # own-target confound, not cross-input
            return None
        if not self._reverify(base, toggle, detecting, boundary):
            return None
        return LeftoverFinding(boundary, detecting, range(boundary, detecting + 1), prefix_genuine)

    def _probe(self, base: Sequence[Variant], toggle: Sequence[Variant],
               detecting: int, k: int, reps: int) -> Trace:
        """detecting pair's trace r(k) under the splice sigma_{base->toggle}(k)."""
        return self._measure(splice(base, toggle, detecting, k), reps)[detecting]

    def _probe_key(self, base: Sequence[Variant], toggle: Sequence[Variant],
                   detecting: int, k: int) -> Key:
        return self._key(self._probe(base, toggle, detecting, k, self._reps))

    def _reverify(self, base: Sequence[Variant], toggle: Sequence[Variant],
                  detecting: int, k: int) -> bool:
        """Fresh, larger-sample, robust re-measurement of the boundary that toggles the leaking pair."""
        h_k = self._probe(base, toggle, detecting, k, self._verify_reps)
        h_k1 = self._probe(base, toggle, detecting, k + 1, self._verify_reps)
        return self._confirm(h_k, h_k1)
