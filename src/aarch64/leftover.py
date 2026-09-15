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

For a detecting pair `d`, over its history [0, d), searched from BOTH bases (mirroring priming's
symmetric swap: the divergence may be caused by the prefix being genuine OR decoy):
  * Localize (hybrid) -- let H_k = [base 0..k, toggle k..d], with the detecting pair toggled with the
    suffix. With the invariant key(H_lo) != key(H_hi) (lo=0 all-toggle, hi=d+1 all-base), bisect to a
    tipping point k where toggling slot k flips d's signal: the leaking pair. A chain of priming swaps;
    needs no predictor model and no monotonicity. The genuine-prefix base sweeps decoy->genuine; the
    decoy-prefix base is the mirror -- a prefix of decoy inputs can itself be the cause.
  * Re-verify -- re-measure H_k and H_{k+1} fresh at a larger sample and confirm they robustly differ,
    rejecting a boundary that only appeared under measurement jitter.
A boundary at the detecting pair itself (leaking pair == detecting pair) is the pair's OWN seal flipping
its OWN readout -- the own-target confound. It is a valid finding by default (an input that leaks about
itself); pass exclude_self_dependence to keep only strictly cross-input leftovers.

A found leaking pair is a valid ct-seq counterexample for ANY predictor (TAGE or trivial) -- the
search never models how the prediction forms, so its *validity* is predictor-agnostic. Two SECONDARY
properties are narrower and hold for a single-entry, most-recent-wins predictor (the audited N3 BTB):
scanning by the two endpoints (all-decoy vs all-genuine history) is guaranteed to *find* an existing
leak -- a history-folded predictor could hide an interior witness between coinciding endpoints, and
guaranteeing detection there is inherently linear, not O(log n); and running the chain standalone
reproduces the leak only where the swept prefix is replaceable. The reported counterexample is the
whole chain [leaking pair .. detecting pair] (the base lane before the leaking pair, the toggle lane
from it onward), de-duplicated by (leaking pair, detecting pair) across the two bases -- so up to two
findings per detecting pair, one from each base.
"""
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence

Variant = Any        # an opaque per-slot input variant the `measure` callable understands
Trace = Any          # an opaque hardware trace the `key` and `confirm` callables understand
Key = Any            # a hashable canonical form of a trace


@dataclass(frozen=True)
class LeftoverFinding:
    """Toggling input `leaking_pair`'s two ct-equal variants flips the readout of input
    `detecting_pair`, the rest of the sequence fixed. `chain` = [leaking_pair .. detecting_pair] is the
    self-contained counterexample. `prefix_genuine` is the base the search swept from: True = genuine
    before the leaking pair and decoy from it on; False = the mirror (decoy before, genuine from it on)."""
    leaking_pair: int
    detecting_pair: int
    chain: range
    prefix_genuine: bool


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
        """Locate the leaking pair for `detecting`, sweeping the prefix from `base` toward `toggle`.
        The detecting pair is toggled with the suffix, so the all-`base` end is hi = detecting + 1. A tip
        at the detecting pair itself (leaking pair == detecting pair) is self-dependence -- the pair's
        own seal flips its own readout; returned by default, dropped when exclude_self_dependence."""
        lo, hi = 0, detecting + 1
        lo_key = self._probe_key(base, toggle, detecting, lo)          # all-toggle history
        if lo_key == self._probe_key(base, toggle, detecting, hi):     # all-base history
            return None                                                # signal independent of history
        while hi - lo > 1:                                             # invariant: key(H_lo) == lo_key != key(H_hi)
            mid = (lo + hi) // 2
            if self._probe_key(base, toggle, detecting, mid) == lo_key:
                lo = mid
            else:
                hi = mid
        if exclude_self_dependence and lo == detecting:               # own-target confound, not cross-input
            return None
        if not self._reverify(base, toggle, detecting, lo):
            return None
        return LeftoverFinding(lo, detecting, range(lo, detecting + 1), prefix_genuine)

    def _probe(self, base: Sequence[Variant], toggle: Sequence[Variant],
               detecting: int, k: int, reps: int) -> Trace:
        """detecting pair's trace under H_k: slots [0, k) from `base`, [k, detecting] from `toggle`. At
        the top endpoint k = detecting + 1 the toggled range is empty (the all-`base` end). Slots after
        `detecting` stay `base` (in-order irrelevant)."""
        batch = list(base)
        for s in range(k, detecting + 1):
            batch[s] = toggle[s]
        return self._measure(batch, reps)[detecting]

    def _probe_key(self, base: Sequence[Variant], toggle: Sequence[Variant],
                   detecting: int, k: int) -> Key:
        return self._key(self._probe(base, toggle, detecting, k, self._reps))

    def _reverify(self, base: Sequence[Variant], toggle: Sequence[Variant],
                  detecting: int, k: int) -> bool:
        """Fresh, larger-sample, robust re-measurement of the boundary that toggles the leaking pair."""
        h_k = self._probe(base, toggle, detecting, k, self._verify_reps)
        h_k1 = self._probe(base, toggle, detecting, k + 1, self._verify_reps)
        return self._confirm(h_k, h_k1)
