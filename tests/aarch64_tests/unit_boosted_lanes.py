"""
Unit tests for the boosted-lane data seam (src/aarch64/boosted_lanes.py). Pure logic, no hardware:
they check that the flat boosted-input layout is sliced into the right lanes and that lane-pair
enumeration matches the "lane 0 vs each boosting" scheme. Run from the repo root:
    python -m unittest tests.aarch64_tests.unit_boosted_lanes
"""
import os
import sys
import unittest
from collections import namedtuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.aarch64.boosted_lanes import (lanes_of, reference_lane_pairs,        # noqa: E402
                                       detecting_position_and_lanes)

M = namedtuple("M", "input_id")     # minimal Measurement-like: just an input_id into the boosted list


class LanesOfTest(unittest.TestCase):

    def test_layout_matches_boosting_order(self):
        # 3 classes, 3 lanes (1 original + 2 boostings), order-preserving.
        boosted = ["I0", "I1", "I2", "I0'", "I1'", "I2'", "I0''", "I1''", "I2''"]
        self.assertEqual(lanes_of(boosted, 3),
                         [["I0", "I1", "I2"], ["I0'", "I1'", "I2'"], ["I0''", "I1''", "I2''"]])

    def test_position_j_across_lanes_is_one_class(self):
        # Column j of the lane matrix = boosted[r*n + j] for each r = one input class.
        n = 4
        boosted = list(range(n * 3))                      # 3 lanes
        lanes = lanes_of(boosted, n)
        for j in range(n):
            column = [lanes[r][j] for r in range(3)]
            self.assertEqual(column, [j, j + n, j + 2 * n])   # same class, one per lane

    def test_single_lane_when_not_boosted(self):
        self.assertEqual(lanes_of(["a", "b"], 2), [["a", "b"]])

    def test_non_multiple_is_rejected(self):
        with self.assertRaises(AssertionError):
            lanes_of([1, 2, 3, 4, 5], 2)                   # 5 is not a whole number of lanes


class LanePairsTest(unittest.TestCase):

    def test_reference_lane_is_zero_against_each_boosting(self):
        self.assertEqual(reference_lane_pairs(4), [(0, 1), (0, 2), (0, 3)])

    def test_single_lane_has_no_pairs(self):
        self.assertEqual(reference_lane_pairs(1), [])       # nothing to toggle against


class DetectingPositionTest(unittest.TestCase):
    # 4 classes (n_orig=4), 3 lanes. Class position j lives at input_ids j, 4+j, 8+j.

    def test_maps_position_and_diverging_lanes(self):
        # Class j=2 diverged: lane 0's member (id 2) vs lane 2's member (id 10) are in different groups.
        groups = [[M(2)], [M(10)]]
        self.assertEqual(detecting_position_and_lanes(groups, 4), (2, 0, 2))   # j=2, base lane 0, toggle 2

    def test_picks_a_differing_group_member_at_the_same_position(self):
        # First group is lane 1 (id 5, j=1); a differing group has lane 2 at the same j (id 9).
        groups = [[M(5)], [M(9)]]
        self.assertEqual(detecting_position_and_lanes(groups, 4), (1, 1, 2))

    def test_single_group_is_not_localizable(self):
        self.assertIsNone(detecting_position_and_lanes([[M(2), M(6), M(10)]], 4))   # all agree -> no toggle

    def test_ignores_empty_groups(self):
        self.assertIsNone(detecting_position_and_lanes([[M(2)], []], 4))


if __name__ == "__main__":
    unittest.main(verbosity=2)
