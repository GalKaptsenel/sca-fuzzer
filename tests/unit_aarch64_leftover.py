"""
Logical tests for the AArch64 non-interference cross-input leftover detector (src/aarch64/leftover.py).

They drive the algorithm with MOCK measurement oracles that simulate predictors, so they test the
algorithm's LOGIC -- localization, the full-chain finding, the robust re-verify, and the transitivity
of the trace key -- with no hardware. Run from the repo root:
    python -m unittest tests.unit_aarch64_leftover
"""
import os
import sys
import random
import unittest
from collections import namedtuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.aarch64.leftover import (GeneralizedPrimingDetector,        # noqa: E402
                                  linear_scan, exponential_search)
from src.analyser import MergedBitmapAnalyser, ChiSquaredAnalyser    # noqa: E402

H = namedtuple("H", "raw")          # minimal trace-like object: just `.raw`
OUTLIER = 0.1
BG = 1 << 20                        # a baseline bit every detecting-pair readout carries


def key(trace):
    """The production transitive key: the denoised consensus bitmap."""
    return MergedBitmapAnalyser.merged_bitmap(trace, OUTLIER)


def robust_differ(a, b):
    """The production re-verify test: a robust (chi-squared) 'do these genuinely differ'."""
    return not ChiSquaredAnalyser().htraces_are_equivalent(a, b)


def detector(measure, reps=200, verify_reps=500, localizer=exponential_search):
    return GeneralizedPrimingDetector(measure, key, robust_differ, reps=reps, verify_reps=verify_reps,
                                      localizer=localizer)


def lanes(n):
    """The genuine and decoy lanes for an n-input sequence, as opaque per-slot markers."""
    return [("g", s) for s in range(n)], [("b", s) for s in range(n)]


class MostRecentWinsBTB:
    """Single-entry, most-recent-wins BTB oracle. A GENUINE trainer slot allocates its target; a decoy
    slot faults before allocating. The detecting pair is held decoy (its own target faults, so its
    readout is purely the leftover), and reads the target of the most-recent GENUINE trainer among its
    predecessors, or none. Deterministic -- a noiseless oracle."""

    def __init__(self, targets):
        self.targets = targets          # {slot: target_bit}

    def measure(self, batch, reps):
        out = []
        for d in range(len(batch)):
            leftover = 0
            for s in range(d):           # predecessors, most-recent genuine trainer wins
                kind, _ = batch[s]
                if kind == "g" and s in self.targets:
                    leftover = self.targets[s]
            out.append(H([BG | leftover] * reps))
        return out


class HistoryFoldOracle:
    """A history-folded predictor: the detecting pair's readout depends on the *parity* of genuine
    slots in its prefix, not the most-recent one. This is the documented case where the O(log n)
    endpoint gate MISSES -- the two endpoints (all-decoy, all-genuine) can coincide while an interior
    configuration differs. Any pair actually found is still a valid counterexample; guaranteeing
    detection here is linear, not O(log n)."""

    def measure(self, batch, reps):
        out = []
        for d in range(len(batch)):
            genuine_prefix = sum(1 for s in range(d) if batch[s][0] == "g")
            out.append(H([BG | (1 << (genuine_prefix % 2))] * reps))
        return out


class SelfSealOracle:
    """Each slot's readout depends only on its OWN seal (genuine vs decoy), never on a predecessor.
    This is the own-target confound: toggling the detecting pair's own seal flips its own readout, but
    there is no cross-input leftover. The detector must report nothing -- a leftover is by definition
    caused by a DIFFERENT, earlier input."""

    def measure(self, batch, reps):
        return [H([BG | (1 if batch[d][0] == "g" else 0)] * reps) for d in range(len(batch))]


class ChainOracle:
    """A fully-scripted chain: the detecting pair's readout under the splice is chain[t], where t is the
    split (the first toggled class). This gives exact control over r(t) = chain[t], so a non-monotone
    chain can be built to separate the two localizers (linear_scan finds t_max; exponential_search may
    gallop past it to another, still-genuine, boundary). Used only genuine-prefix (base 'g', toggle 'b')."""

    def __init__(self, chain):
        self.chain = chain                          # chain[t] = r(t), for t in 0 .. len(chain) - 1 = hi

    def measure(self, batch, reps):
        t = next((s for s, v in enumerate(batch) if v[0] == "b"), len(self.chain) - 1)   # the split
        val = self.chain[min(t, len(self.chain) - 1)]
        return [H([BG | val] * reps) for _ in batch]


def valid_tip(oracle, detecting, k, prefix_genuine=True, reps=200):
    """A leaking pair is valid iff toggling slot k flips the detecting pair's readout, in the direction
    the finding was made (genuine-prefix base, or the mirror decoy-prefix base)."""
    det = detector(oracle.measure)
    g, b = lanes(detecting + 1)
    base, toggle = (g, b) if prefix_genuine else (b, g)
    return key(det._probe(base, toggle, detecting, k, reps)) != key(det._probe(base, toggle, detecting, k + 1, reps))


class DetectorTest(unittest.TestCase):

    def test_distinct_targets_every_detecting_pair_leaks(self):
        oracle = MostRecentWinsBTB({0: 1 << 0, 1: 1 << 1, 2: 1 << 2})
        findings = detector(oracle.measure).detect(*lanes(4))
        # the genuine-prefix base leaks every detecting pair; with distinct targets the readout changes at
        # every class, so the leaking pair the default localizer returns is the most-recent trainer d-1.
        gp = [f for f in findings if f.prefix_genuine]
        self.assertEqual([f.detecting_pair for f in gp], [1, 2, 3])
        self.assertTrue(all(f.leaking_pair == f.detecting_pair - 1 for f in gp))
        for f in gp:
            self.assertEqual(f.chain, range(f.leaking_pair, f.detecting_pair + 1))   # chain reported
        # every reported finding is a genuine boundary (both bases are searched; identical (leaking,
        # detecting) results from the two bases are de-duplicated to a single report).
        for f in findings:
            self.assertTrue(valid_tip(oracle, f.detecting_pair, f.leaking_pair, f.prefix_genuine))

    def test_inert_prefix_is_skipped(self):
        oracle = MostRecentWinsBTB({2: 1 << 2})               # slots 0,1 inert; 2 trains
        findings = detector(oracle.measure).detect(*lanes(4))
        self.assertEqual(len(findings), 1)
        self.assertEqual((findings[0].leaking_pair, findings[0].detecting_pair), (2, 3))
        self.assertEqual(findings[0].chain, range(2, 4))

    def test_no_trainers_no_violation(self):
        self.assertEqual(detector(MostRecentWinsBTB({}).measure).detect(*lanes(5)), [])

    def test_self_dependence_reported_by_default_and_optionally_excluded(self):
        # Own-target confound: each slot reacts only to its own seal, so every detecting pair bisects to
        # lo == detecting (the pair itself). By default these self-pairs are valid findings; passing
        # exclude_self_dependence drops them, leaving only strictly cross-input leftovers (none here).
        det = detector(SelfSealOracle().measure)
        default = det.detect(*lanes(6))
        self.assertEqual([(f.leaking_pair, f.detecting_pair) for f in default],
                         [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)])
        self.assertEqual(det.detect(*lanes(6), exclude_self_dependence=True), [])

    def test_same_target_masking_still_valid(self):
        oracle = MostRecentWinsBTB({0: 1 << 2, 1: 1 << 1, 2: 1 << 2})   # slots 0,2 share a target
        f = next(x for x in detector(oracle.measure).detect(*lanes(4)) if x.detecting_pair == 3)
        self.assertTrue(valid_tip(oracle, 3, f.leaking_pair, f.prefix_genuine))

    def test_multi_contributor_returns_one_valid_pair(self):
        # Two independent contributors into detecting pair 15. A single search returns ONE valid
        # leaking pair (incomplete-by-design; enumeration is deliberately NOT implemented).
        oracle = MostRecentWinsBTB({6: 1 << 3, 14: 1 << 4})
        f = next(x for x in detector(oracle.measure).detect(*lanes(16)) if x.detecting_pair == 15)
        self.assertIn(f.leaking_pair, (6, 14))
        self.assertTrue(valid_tip(oracle, 15, f.leaking_pair, f.prefix_genuine))

    def test_returned_pairs_are_always_valid(self):
        # Randomized: over many most-recent-wins configs, EVERY reported leaking pair must be a genuine
        # boundary. The search may be incomplete, but must never return a garbage pair.
        rng = random.Random(20260912)
        for _ in range(800):
            n = rng.randint(3, 30)
            symbols = [1 << i for i in range(rng.randint(1, 4))]
            slots = rng.sample(range(n - 1), rng.randint(1, min(8, n - 1)))
            oracle = MostRecentWinsBTB({s: rng.choice(symbols) for s in slots})
            for f in detector(oracle.measure).detect(*lanes(n)):
                self.assertTrue(valid_tip(oracle, f.detecting_pair, f.leaking_pair, f.prefix_genuine))

    def test_history_fold_endpoint_coincidence_is_missed_but_a_valid_witness_exists(self):
        # Documented most-recent-wins scope: for a history-folded predictor the two endpoints can
        # coincide (all-decoy == all-genuine) so the O(log n) endpoint gate misses -- even though an
        # interior leaking pair exists and would be valid. This is a completeness gap, not unsoundness.
        oracle = HistoryFoldOracle()
        det = detector(oracle.measure)
        g, b = lanes(5)
        self.assertIsNone(det.find_leaking_pair(g, b, 4))     # endpoints coincide -> missed
        self.assertTrue(valid_tip(oracle, 4, 0))              # yet slot 0 is a genuine interior witness


class ScriptedMeasure:
    """Returns a scripted raw trace for the detecting slot on each call, so re-verification can be
    driven adversarially. Non-detecting slots are constant, so only the detecting slot decides."""
    CONST = 1 << 30

    def __init__(self, detecting, raws):
        self.detecting = detecting
        self.raws = list(raws)          # one raw trace per call
        self.calls = 0

    def measure(self, batch, reps):
        raw = self.raws[self.calls]
        self.calls += 1
        return [H(list(raw)) if s == self.detecting else H([self.CONST]) for s in range(len(batch))]


class ReverifyTest(unittest.TestCase):
    # These drive ONE search direction (find_leaking_pair) to isolate the re-verify path with a
    # deterministic call count. detecting=1, so hi=2 (the detector is toggled). Call order: localize
    # probe(k=0), probe(k=2), probe(k=1); then re-verify probe(k=0), probe(k=1) -- 5 calls.
    C, V = 0b1, 0b10                    # background bit + a bit straddling the 10% cutoff at 500 reps

    def test_reverify_drops_straddle_jitter(self):
        # Localization sees a difference (distinct search values -> candidate). But the fresh re-verify
        # sees two SAME-distribution traces whose straddle bit lands 52/500 vs 48/500: the exact-bitmap
        # KEY differs (the old exact-bitmap re-verify would KEEP it), yet the robust test finds them
        # equivalent, so it is DROPPED. This actually pins the exact-bitmap -> robust re-verify fix
        # (the earlier all-identical [7,7] script did not -- the old check drops identical traces too).
        C, V = self.C, self.V
        straddle_hi, straddle_lo = [C | V] * 52 + [C] * 448, [C | V] * 48 + [C] * 452
        self.assertNotEqual(key(H(straddle_hi)), key(H(straddle_lo)))   # exact-bitmap alone would keep it
        m = ScriptedMeasure(1, [[0b100] * 200, [0b1000] * 200, [0b1000] * 200, straddle_hi, straddle_lo])
        g, b = lanes(2)
        self.assertIsNone(detector(m.measure).find_leaking_pair(g, b, 1))
        self.assertEqual(m.calls, 5)

    def test_reverify_keeps_confirmed_boundary(self):
        # A genuine boundary: the detecting pair's readout robustly differs between the two histories.
        m = ScriptedMeasure(1, [[0b100] * 200, [0b1000] * 200, [0b1000] * 200, [0b1] * 500, [0b1000000] * 500])
        g, b = lanes(2)
        f = detector(m.measure).find_leaking_pair(g, b, 1)
        self.assertEqual((f.leaking_pair, f.detecting_pair), (0, 1))
        self.assertTrue(f.prefix_genuine)


class TraceKeyTest(unittest.TestCase):

    def test_key_filters_low_frequency_jitter(self):
        # a 1.5%-frequency residual (below the 10% outlier threshold) is filtered out.
        self.assertEqual(key(H([0b100] * 200)), key(H([0b010] * 3 + [0b100] * 197)))

    def test_key_equality_is_an_equivalence_relation(self):
        # The bisection invariant relies on trace equality being reflexive/symmetric/TRANSITIVE; exact
        # equality of the denoised bitmap key is, by construction. A statistical test would not be.
        sample = [H([0b100] * 200), H([0b010] * 3 + [0b100] * 197),
                  H([0b100] * 200), H([0b001] * 200)]
        eq = lambda a, c: key(a) == key(c)
        for a in sample:
            self.assertTrue(eq(a, a))
            for c in sample:
                self.assertEqual(eq(a, c), eq(c, a))
                for d in sample:
                    if eq(a, c) and eq(c, d):
                        self.assertTrue(eq(a, d))


class LocalizerTest(unittest.TestCase):
    """The two injected localizers: linear_scan returns t_max (the largest boundary), exponential_search
    (galloping + bisection) returns some genuine boundary, possibly not t_max."""

    def _find(self, chain, localizer):
        det = len(chain) - 2                          # hi = det + 1 = last chain index
        g = [("g", s) for s in range(det + 1)]
        b = [("b", s) for s in range(det + 1)]
        f = detector(ChainOracle(chain).measure, localizer=localizer).find_leaking_pair(g, b, det)
        return None if f is None else f.leaking_pair

    @staticmethod
    def _is_boundary(chain, t):
        return chain[t] != chain[t + 1]

    def test_linear_returns_tmax_exponential_may_skip_to_another_boundary(self):
        # Non-monotone chain over classes 0..9 (det = 8, hi = 9); boundaries at 0, 5, 6, so t_max = 6.
        A, B = 1 << 1, 1 << 2
        chain = [B, A, A, A, A, A, B, A, A, A]
        self.assertEqual(self._find(chain, linear_scan), 6)          # optimal: the largest boundary
        exp = self._find(chain, exponential_search)
        self.assertNotEqual(exp, 6)                                  # galloping's probes stepped over t_max
        self.assertIn(exp, (0, 5, 6))
        self.assertTrue(self._is_boundary(chain, exp))               # but it is always a genuine boundary

    def test_localizers_agree_on_a_single_boundary_chain(self):
        A, B = 1 << 1, 1 << 2
        chain = [A, A, A, A, A, B, B, B]                             # det = 6, hi = 7; sole boundary at 4
        self.assertEqual(self._find(chain, linear_scan), 4)
        self.assertEqual(self._find(chain, exponential_search), 4)

    def test_endpoints_equal_returns_none_for_both(self):
        chain = [1, 1, 1, 1, 1]                                      # no boundary; r(lo) == r(hi)
        self.assertIsNone(self._find(chain, linear_scan))
        self.assertIsNone(self._find(chain, exponential_search))


if __name__ == "__main__":
    unittest.main(verbosity=2)
