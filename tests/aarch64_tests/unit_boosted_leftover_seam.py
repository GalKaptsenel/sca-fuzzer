"""
Unit tests for the boosted-lane focused-localization seam (Aarch64Fuzzer._localize_boosted_violation).
Pure logic, no hardware: a MOCK detector records how the seam drives find_leaking_pair, so the test
pins the mapping violation -> (detecting position, lane pair) and the both-bases fall-through. Run:
    python -m unittest tests.aarch64_tests.unit_boosted_leftover_seam
"""
import os
import sys
import unittest
from collections import namedtuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.aarch64.aarch64_fuzzer import Aarch64Fuzzer         # noqa: E402
from src.aarch64.leftover import LeftoverFinding             # noqa: E402

M = namedtuple("M", "input_id")


def violation(groups):
    v = type("V", (), {})()
    v.htrace_groups = groups
    return v


class MockDetector:
    """Records each find_leaking_pair call; returns a scripted finding for the base it is told to hit."""

    def __init__(self, hit_on_prefix_genuine=None):
        self.hit_on = hit_on_prefix_genuine       # None -> never finds; True/False -> finds on that base
        self.calls = []

    def find_leaking_pair(self, base, toggle, detecting, prefix_genuine=True,
                          exclude_self_dependence=False):
        self.calls.append((base, toggle, detecting, prefix_genuine))
        if self.hit_on is not None and prefix_genuine == self.hit_on:
            return LeftoverFinding(0, detecting, range(0, detecting + 1), prefix_genuine)
        return None


class LocalizeSeamTest(unittest.TestCase):
    # 3 classes (n_orig=3), 2 lanes: lane 0 = [a0,a1,a2], lane 1 = [b0,b1,b2].
    def setUp(self):
        self.fz = object.__new__(Aarch64Fuzzer)   # bypass __init__ (no HW); the method uses no other state
        self.lanes = [["a0", "a1", "a2"], ["b0", "b1", "b2"]]
        self.n_orig = 3

    def _localize(self, det, groups):
        return self.fz._localize_boosted_violation(violation(groups), self.lanes, self.n_orig, det)

    def test_genuine_prefix_hit_uses_right_position_and_lane_pair(self):
        det = MockDetector(hit_on_prefix_genuine=True)
        # class j=2 diverged between lane 0 (id 2) and lane 1 (id 5).
        f, prefix_lane, suffix_lane = self._localize(det, [[M(2)], [M(5)]])
        self.assertEqual((f.leaking_pair, f.detecting_pair, f.prefix_genuine), (0, 2, True))
        self.assertEqual((prefix_lane, suffix_lane), (0, 1))      # prefix=lane0, suffix=lane1
        self.assertEqual(len(det.calls), 1)                       # found on the first base, no fall-through
        base, toggle, detecting, prefix_genuine = det.calls[0]
        self.assertEqual((base, toggle, detecting, prefix_genuine),
                         (self.lanes[0], self.lanes[1], 2, True))  # base=lane0, toggle=lane1, detecting=2

    def test_falls_through_to_decoy_prefix_base(self):
        det = MockDetector(hit_on_prefix_genuine=False)
        f, prefix_lane, suffix_lane = self._localize(det, [[M(1)], [M(4)]])   # class j=1, lanes 0 and 1
        self.assertEqual((f.detecting_pair, f.prefix_genuine), (1, False))
        self.assertEqual((prefix_lane, suffix_lane), (1, 0))      # decoy-prefix: lanes swapped
        self.assertEqual(len(det.calls), 2)                       # genuine base missed, decoy base hit
        self.assertEqual(det.calls[1][:3], (self.lanes[1], self.lanes[0], 1))  # swapped base/toggle

    def test_no_finding_returns_none(self):
        det = MockDetector(hit_on_prefix_genuine=None)
        self.assertIsNone(self._localize(det, [[M(2)], [M(5)]]))
        self.assertEqual(len(det.calls), 2)                       # both bases tried

    def test_unlocalizable_violation_never_probes(self):
        det = MockDetector(hit_on_prefix_genuine=True)
        self.assertIsNone(self._localize(det, [[M(2), M(5)]]))    # single group -> not localizable
        self.assertEqual(det.calls, [])                           # detector never called


if __name__ == "__main__":
    unittest.main(verbosity=2)
