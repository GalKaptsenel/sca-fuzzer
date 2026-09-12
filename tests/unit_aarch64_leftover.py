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
from src.aarch64.leftover import GeneralizedPrimingDetector          # noqa: E402
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


def detector(measure, reps=200, verify_reps=500):
    return GeneralizedPrimingDetector(measure, key, robust_differ, reps=reps, verify_reps=verify_reps)


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


def valid_tip(oracle, detecting, k, reps=200):
    """A leaking pair is valid iff toggling slot k's genuine/decoy flips the detecting pair's readout."""
    det = detector(oracle.measure)
    g, b = lanes(detecting + 1)
    return key(det._probe(g, b, detecting, k, reps)) != key(det._probe(g, b, detecting, k + 1, reps))


class DetectorTest(unittest.TestCase):

    def test_distinct_targets_every_detecting_pair_leaks(self):
        oracle = MostRecentWinsBTB({0: 1 << 0, 1: 1 << 1, 2: 1 << 2})
        findings = detector(oracle.measure).detect(*lanes(4))
        self.assertEqual([f.detecting_pair for f in findings], [1, 2, 3])
        # first-tip anchor: distinct targets, no inert prefix -> leaking pair is slot 0.
        self.assertTrue(all(f.leaking_pair == 0 for f in findings))
        for f in findings:
            self.assertEqual(f.chain, range(0, f.detecting_pair + 1))   # whole chain reported

    def test_inert_prefix_is_skipped(self):
        oracle = MostRecentWinsBTB({2: 1 << 2})               # slots 0,1 inert; 2 trains
        findings = detector(oracle.measure).detect(*lanes(4))
        self.assertEqual(len(findings), 1)
        self.assertEqual((findings[0].leaking_pair, findings[0].detecting_pair), (2, 3))
        self.assertEqual(findings[0].chain, range(2, 4))

    def test_no_trainers_no_violation(self):
        self.assertEqual(detector(MostRecentWinsBTB({}).measure).detect(*lanes(5)), [])

    def test_same_target_masking_still_valid(self):
        oracle = MostRecentWinsBTB({0: 1 << 2, 1: 1 << 1, 2: 1 << 2})   # slots 0,2 share a target
        f = next(x for x in detector(oracle.measure).detect(*lanes(4)) if x.detecting_pair == 3)
        self.assertTrue(valid_tip(oracle, 3, f.leaking_pair))

    def test_multi_contributor_returns_one_valid_pair(self):
        # Two independent contributors into detecting pair 15. A single search returns ONE valid
        # leaking pair (incomplete-by-design; enumeration is deliberately NOT implemented).
        oracle = MostRecentWinsBTB({6: 1 << 3, 14: 1 << 4})
        f = next(x for x in detector(oracle.measure).detect(*lanes(16)) if x.detecting_pair == 15)
        self.assertIn(f.leaking_pair, (6, 14))
        self.assertTrue(valid_tip(oracle, 15, f.leaking_pair))

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
                self.assertTrue(valid_tip(oracle, f.detecting_pair, f.leaking_pair))

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
    """Returns a scripted value for the detecting slot on each call, so re-verification can be driven
    adversarially. Non-detecting slots are constant, so only the detecting slot decides the result."""
    CONST = 1 << 30

    def __init__(self, detecting, script):
        self.detecting = detecting
        self.script = list(script)
        self.calls = 0

    def measure(self, batch, reps):
        value = self.script[self.calls]
        self.calls += 1
        return [H([value] * reps) if s == self.detecting else H([self.CONST]) for s in range(len(batch))]


class ReverifyTest(unittest.TestCase):
    # detecting=1. Call order: localize probe(k=0), probe(k=1); re-verify probe(k=0), probe(k=1).

    def test_reverify_drops_straddle_jitter(self):
        # Localization sees a difference (1 vs 2 -- a jitter-flipped key), but the fresh robust
        # re-verify sees the same distribution (7 vs 7) -> DROP. This is fable's straddle-jitter case:
        # a boundary that only appeared under measurement jitter must not be reported.
        m = ScriptedMeasure(detecting=1, script=[1, 2, 7, 7])
        self.assertEqual(detector(m.measure).detect(*lanes(2)), [])
        self.assertEqual(m.calls, 4)

    def test_reverify_keeps_confirmed_boundary(self):
        m = ScriptedMeasure(detecting=1, script=[1, 2, 2, 1])    # re-verify 2 vs 1 -> robustly differ -> KEEP
        findings = detector(m.measure).detect(*lanes(2))
        self.assertEqual(len(findings), 1)
        self.assertEqual((findings[0].leaking_pair, findings[0].detecting_pair), (0, 1))


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
