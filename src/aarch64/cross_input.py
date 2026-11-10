"""AArch64 speculative-exception cross-input leak detector (generalized priming).

Standard priming keeps a cache divergence only when it follows the detecting pair's own input. This
also finds divergences caused by an earlier leaking pair whose leak lives in the microarchitecture (for
example a branch-target-buffer entry) and only surfaces as the detecting pair's cache divergence. It is
self-contained (it knows nothing about the fuzzer or executor) and driven by three callables:
  measure(batch, reps): per-slot hardware traces of a batch whose slot s holds input s's base or toggle variant
  key(trace):           a hashable canonical form, with transitive equality (the search relies on it)
  confirm(a, b):        a noise-tolerant difference test, used only to re-verify a found boundary

For a detecting pair d it searches the history [0, d) from both bases for a boundary t where toggling
class t flips the detecting readout, so (base[t], toggle[t]) is a leaking pair. Two localizers are
injected as `localizer`: linear_scan returns the largest boundary t_max, exponential_search (default)
gallops to bracket a boundary then bisects it in O(log distance). A boundary at the detecting pair
itself is the own-target confound (an input leaking about itself), kept by default or dropped via
exclude_self_dependence. The reported counterexample is the chain [leaking pair .. detecting pair]."""
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
    `toggle`. classes after `detecting` stay `base` (they do not affect the class-`detecting` readout). At
    t = detecting + 1 the toggled range is empty -- the all-base end."""
    batch = list(base)
    for c in range(t, detecting + 1):
        batch[c] = toggle[c]
    return batch


def linear_scan(r_key: RKey, lo: int, hi: int) -> int:
    """Return t_max, the LARGEST boundary in [lo, hi). Read r downward from the all-base end `hi`. the
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
    leaves the all-base value `ref`. the last window [t, prev] then brackets a boundary (r differs at t,
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


def binary_search(r_key: RKey, lo: int, hi: int) -> int:
    """Pure bisection over the full range [lo, hi) -- no galloping. Given r_key(lo) != r_key(hi), probe the
    midpoint and keep the half that still straddles a change, returning a boundary t with r_key(t) !=
    r_key(t+1). Cost is ceil(log2(hi - lo)) probes regardless of where the boundary lies -- the
    position-independent counterpart of exponential_search, which is cheaper only for a near boundary."""
    ref = r_key(hi)                                  # the all-base end value
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if r_key(mid) != ref:
            lo = mid
        else:
            hi = mid
    return lo                                        # boundary: r_key(lo) != ref == r_key(lo + 1)


@dataclass(frozen=True)
class CrossInputFinding:
    """Toggling input `leaking_pair`'s two ct-equal variants flips the readout of input `detecting_pair`,
    the rest of the sequence fixed. `chain` = [leaking_pair .. detecting_pair] is the self-contained
    counterexample. `prefix_from_first` records which lane supplied the prefix: True = the first
    lane before the leaking pair and the second from it on. False = the mirror."""
    leaking_pair: int
    detecting_pair: int
    chain: range
    prefix_from_first: bool


class GeneralizedPrimingDetector:
    def __init__(self, measure: Callable[[List[Variant], int], List[Trace]],
                 key: Callable[[Trace], Key], confirm: Callable[[Trace, Trace], bool],
                 *, reps: int, verify_reps: int, localizer: Localizer = exponential_search,
                 is_valid_toggle: Optional[Callable[[Variant, Variant], bool]] = None) -> None:
        assert reps >= 1 and verify_reps >= 1, "reps must be positive"
        self._measure = measure
        self._key = key
        self._confirm = confirm
        self._reps = reps
        self._verify_reps = verify_reps
        self._localizer = localizer
        self._is_valid_toggle = is_valid_toggle

    def detect(self, lane_a: Sequence[Variant], lane_b: Sequence[Variant],
               exclude_self_dependence: bool = False) -> List[CrossInputFinding]:
        """Scan every input as a detecting pair. For each, sweep the prefix from BOTH lanes as the base
        (a->b and b->a) -- priming's symmetric swap: the divergence may be caused by either lane's
        prefix. Neither lane is assumed to be a genuine baseline. they are simply the two lanes the
        detected inputs belong to. De-duplicate by (leaking pair, detecting pair). A self-dependent
        finding (leaking pair == detecting pair -- the pair leaks about itself) is a valid result by
        default. pass exclude_self_dependence to drop it and keep only strictly cross-input leaks."""
        assert len(lane_a) == len(lane_b), "the two lanes must have equal length"
        findings: dict = {}
        for detecting in range(1, len(lane_a)):
            directions = ((lane_a, lane_b, True), (lane_b, lane_a, False))
            for base, toggle, prefix_from_first in directions:
                f = self.find_leaking_pair(base, toggle, detecting, prefix_from_first,
                                           exclude_self_dependence)
                if f is not None:
                    findings.setdefault((f.leaking_pair, f.detecting_pair), f)
        return list(findings.values())

    def find_leaking_pair(self, base: Sequence[Variant], toggle: Sequence[Variant], detecting: int,
                          prefix_from_first: bool = True,
                          exclude_self_dependence: bool = False) -> Optional[CrossInputFinding]:
        """Locate the leaking pair for `detecting`, sweeping the prefix from `base` toward `toggle` with the
        injected localizer. Each split is measured at most once (memoized), so r_key is a stable value the
        localizer may query freely. A tip at the detecting pair itself (leaking pair == detecting pair) is
        self-dependence -- the pair's own variant flips its own readout. returned by default, excluded by
        searching only [0, detecting) when exclude_self_dependence."""
        lo, hi = 0, (detecting if exclude_self_dependence else detecting + 1)
        cache: dict = {}

        def r_key(t: int) -> Key:                    # r(t)'s key. each split measured once, so it is stable
            if t not in cache:
                cache[t] = self._probe_key(base, toggle, detecting, t)
            return cache[t]

        if r_key(lo) == r_key(hi):                   # signal independent of history -> nothing to localize
            return None
        boundary = self._localizer(r_key, lo, hi)
        if self._is_valid_toggle is not None and not self._is_valid_toggle(base[boundary], toggle[boundary]):
            return None                              # members not contract-equal: not a counterexample
        if not self._reverify(base, toggle, detecting, boundary):
            return None
        chain = range(boundary, detecting + 1)
        return CrossInputFinding(boundary, detecting, chain, prefix_from_first)

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
