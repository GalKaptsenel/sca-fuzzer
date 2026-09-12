"""
Logical tests for the AArch64 non-interference cross-input leftover detector (src/aarch64/leftover.py).

They drive the algorithm with MOCK measurement oracles that simulate predictors, so they test the
algorithm's LOGIC -- detection, tipping-point localization, re-verification, and the transitivity of
the trace key -- with no hardware. Run from the repo root:
    python -m unittest tests.unit_aarch64_leftover
"""
import os
import sys
import random
import unittest
from collections import namedtuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.aarch64.leftover import GeneralizedPrimingDetector       # noqa: E402
from src.analyser import MergedBitmapAnalyser                     # noqa: E402

H = namedtuple("H", "raw")          # minimal trace-like object: just `.raw`
OUTLIER = 0.1


def key(trace):
    """The production trace key used by the seam: the denoised consensus bitmap."""
    return MergedBitmapAnalyser.merged_bitmap(trace, OUTLIER)


def detector(measure, reps=200):
    return GeneralizedPrimingDetector(measure, key, reps=reps)


def lanes(n):
    """The genuine and bad lanes for an n-input sequence, as opaque per-slot markers."""
    return [("g", s) for s in range(n)], [("b", s) for s in range(n)]


class MostRecentWinsBTB:
    """Single-entry, most-recent-wins, PC-indexed BTB oracle. A slot that is GENUINE and a trainer
    allocates its target; a BAD slot faults before allocating (trains nothing). A prober's speculative
    readout bitmap = background | own_bit | (target of the most-recent GENUINE trainer among its
    predecessors, or 0). Deterministic -- a noiseless oracle."""
    OWN = 1 << 40
    BG = 1 << 41

    def __init__(self, targets):
        self.targets = targets          # {slot: target_bit}

    def measure(self, batch, reps):
        out = []
        for q in range(len(batch)):
            leftover = 0
            for s in range(q):           # predecessors, most-recent genuine trainer wins
                kind, _ = batch[s]
                if kind == "g" and s in self.targets:
                    leftover = self.targets[s]
            out.append(H([self.BG | self.OWN | leftover] * reps))
        return out


def assert_valid_tip(test, oracle, q, k):
    """A tip is valid iff toggling slot k's genuine/bad state flips prober q, holding the rest fixed."""
    det = detector(oracle.measure)
    g, b = lanes(q + 1)
    test.assertNotEqual(det._probe_key(g, b, q, k), det._probe_key(g, b, q, k + 1),
                        f"tip {k} for prober {q} is not a real boundary")


class DetectorTest(unittest.TestCase):

    def test_single_entry_distinct_targets_every_prober_leaks(self):
        oracle = MostRecentWinsBTB({0: 1 << 0, 1: 1 << 1, 2: 1 << 2})
        findings = detector(oracle.measure).detect(*lanes(4))
        self.assertEqual([f.prober for f in findings], [1, 2, 3])
        # first-tip anchor: distinct targets, no inert prefix -> the boundary is slot 0.
        self.assertTrue(all(f.tipping_point == 0 for f in findings))
        for f in findings:
            self.assertEqual(f.block, range(0, f.prober + 1))

    def test_inert_prefix_is_skipped(self):
        oracle = MostRecentWinsBTB({2: 1 << 2})               # slots 0,1 inert; 2 trains
        findings = detector(oracle.measure).detect(*lanes(4))
        self.assertEqual(len(findings), 1)
        self.assertEqual((findings[0].prober, findings[0].tipping_point), (3, 2))
        self.assertEqual(findings[0].block, range(2, 4))

    def test_no_trainers_no_violation(self):
        self.assertEqual(detector(MostRecentWinsBTB({}).measure).detect(*lanes(5)), [])

    def test_same_target_masking_still_valid_tip(self):
        oracle = MostRecentWinsBTB({0: 1 << 2, 1: 1 << 1, 2: 1 << 2})   # slots 0,2 share a target
        findings = detector(oracle.measure).detect(*lanes(4))
        assert_valid_tip(self, oracle, 3, next(f.tipping_point for f in findings if f.prober == 3))

    def test_multi_contributor_returns_one_valid_tip(self):
        # Two independent contributors into prober 15. A single search returns ONE valid tip
        # (incomplete-by-design; enumeration is deliberately NOT implemented -- see leftover.py).
        oracle = MostRecentWinsBTB({6: 1 << 3, 14: 1 << 4})
        f = next(x for x in detector(oracle.measure).detect(*lanes(16)) if x.prober == 15)
        self.assertIn(f.tipping_point, (6, 14))
        assert_valid_tip(self, oracle, 15, f.tipping_point)

    def test_returned_tips_are_always_valid_boundaries(self):
        # Randomized: over many most-recent-wins configs, EVERY reported tip must be a genuine
        # adjacent boundary. The search may be incomplete, but must never return a garbage tip.
        rng = random.Random(20260912)
        for _ in range(800):
            n = rng.randint(3, 30)
            symbols = [1 << i for i in range(rng.randint(1, 4))]
            slots = rng.sample(range(n - 1), rng.randint(1, min(8, n - 1)))
            oracle = MostRecentWinsBTB({s: rng.choice(symbols) for s in slots})
            for f in detector(oracle.measure).detect(*lanes(n)):
                assert_valid_tip(self, oracle, f.prober, f.tipping_point)


class ScriptedMeasure:
    """Returns a scripted value for the probe slot on each call, so re-verification can be driven
    adversarially. Non-probe slots are constant, so only the probe decides equality."""
    CONST = 1 << 50

    def __init__(self, probe, script):
        self.probe = probe
        self.script = list(script)
        self.calls = 0

    def measure(self, batch, reps):
        value = self.script[self.calls]
        self.calls += 1
        return [H([value] * reps) if s == self.probe else H([self.CONST]) for s in range(len(batch))]


class ReverifyTest(unittest.TestCase):
    # n=2, prober 1. Call order: detect H_lo(k=0), H_hi(k=1); re-verify H_k(k=0), H_{k+1}(k=1).

    def test_reverify_drops_spurious_boundary(self):
        # Detection differs (1 != 2), but the fresh re-verification sees the same value (7 == 7) -> DROP.
        m = ScriptedMeasure(probe=1, script=[1, 2, 7, 7])
        self.assertEqual(detector(m.measure).detect(*lanes(2)), [])
        self.assertEqual(m.calls, 4)

    def test_reverify_keeps_confirmed_boundary(self):
        m = ScriptedMeasure(probe=1, script=[1, 2, 2, 1])    # re-verify 2 != 1 -> KEEP
        findings = detector(m.measure).detect(*lanes(2))
        self.assertEqual(len(findings), 1)
        self.assertEqual((findings[0].prober, findings[0].tipping_point), (1, 0))


class TraceKeyTest(unittest.TestCase):

    def test_key_filters_low_frequency_jitter(self):
        # a 1.5%-frequency residual (below the 10% outlier threshold) is filtered out, so a trace with
        # the BTB's run-to-run overwrite jitter keys the same as the clean one.
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
