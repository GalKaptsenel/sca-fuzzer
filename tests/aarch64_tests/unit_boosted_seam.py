"""
Unit tests for the boosted-lane focused-localization seam (Aarch64Fuzzer._localize_boosted_violation).
Pure logic, no hardware: a MOCK detector records how the seam drives find_leaking_pair, so the test
pins the mapping violation -> (detecting position, lane pair) and the both-bases fall-through. Run:
    python -m unittest tests.aarch64_tests.unit_boosted_seam
"""
import contextlib
import copy
import os
import sys
import unittest
from unittest import mock
from collections import namedtuple

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
from src.config import CONF                                   # noqa: E402
from src.aarch64.aarch64_fuzzer import Aarch64Fuzzer         # noqa: E402
from src.aarch64.cross_input import CrossInputFinding             # noqa: E402

M = namedtuple("M", "input_id")


def violation(groups):
    v = type("V", (), {})()
    v.htrace_groups = groups
    return v


class MockDetector:
    """Records each find_leaking_pair call; returns a scripted finding for the base it is told to hit."""

    def __init__(self, hit_on_prefix_from_first=None):
        self.hit_on = hit_on_prefix_from_first       # None -> never finds; True/False -> finds on that base
        self.calls = []

    def find_leaking_pair(self, base, toggle, detecting, prefix_from_first=True,
                          exclude_self_dependence=False):
        self.calls.append((base, toggle, detecting, prefix_from_first))
        if self.hit_on is not None and prefix_from_first == self.hit_on:
            return CrossInputFinding(0, detecting, range(0, detecting + 1), prefix_from_first)
        return None


class LocalizeSeamTest(unittest.TestCase):
    # 3 classes (n_orig=3), 2 lanes: lane 0 = [a0,a1,a2], lane 1 = [b0,b1,b2].
    @classmethod
    def setUpClass(cls):
        cls._saved_conf = copy.deepcopy(CONF._borg_shared_state)
        CONF.load(os.path.join(_ROOT, "config_pac.yml"))      # the seam reads aarch64 options

    @classmethod
    def tearDownClass(cls):
        CONF._borg_shared_state.clear()
        CONF._borg_shared_state.update(cls._saved_conf)

    def setUp(self):
        self.fz = object.__new__(Aarch64Fuzzer)   # bypass __init__ (no HW); the method uses no other state
        self.lanes = [["a0", "a1", "a2"], ["b0", "b1", "b2"]]
        self.n_orig = 3

    def _localize(self, det, groups):
        return self.fz._localize_boosted_violation(violation(groups), self.lanes, self.n_orig, det)

    def test_genuine_prefix_hit_uses_right_position_and_lane_pair(self):
        det = MockDetector(hit_on_prefix_from_first=True)
        # class j=2 diverged between lane 0 (id 2) and lane 1 (id 5).
        f, prefix_lane, suffix_lane = self._localize(det, [[M(2)], [M(5)]])
        self.assertEqual((f.leaking_pair, f.detecting_pair, f.prefix_from_first), (0, 2, True))
        self.assertEqual((prefix_lane, suffix_lane), (0, 1))      # prefix=lane0, suffix=lane1
        self.assertEqual(len(det.calls), 1)                       # found on the first base, no fall-through
        base, toggle, detecting, prefix_from_first = det.calls[0]
        self.assertEqual((base, toggle, detecting, prefix_from_first),
                         (self.lanes[0], self.lanes[1], 2, True))  # base=lane0, toggle=lane1, detecting=2

    def test_falls_through_to_decoy_prefix_base(self):
        det = MockDetector(hit_on_prefix_from_first=False)
        f, prefix_lane, suffix_lane = self._localize(det, [[M(1)], [M(4)]])   # class j=1, lanes 0 and 1
        self.assertEqual((f.detecting_pair, f.prefix_from_first), (1, False))
        self.assertEqual((prefix_lane, suffix_lane), (1, 0))      # decoy-prefix: lanes swapped
        self.assertEqual(len(det.calls), 2)                       # genuine base missed, decoy base hit
        self.assertEqual(det.calls[1][:3], (self.lanes[1], self.lanes[0], 1))  # swapped base/toggle

    def test_no_finding_returns_none(self):
        det = MockDetector(hit_on_prefix_from_first=None)
        self.assertIsNone(self._localize(det, [[M(2)], [M(5)]]))
        self.assertEqual(len(det.calls), 2)                       # both bases tried

    def test_unlocalizable_violation_never_probes(self):
        det = MockDetector(hit_on_prefix_from_first=True)
        self.assertIsNone(self._localize(det, [[M(2), M(5)]]))    # single group -> not localizable
        self.assertEqual(det.calls, [])                           # detector never called


class PrimingOutcomeLogTest(unittest.TestCase):
    """_priming reports every candidate's outcome at INFO: self-dependent or cross-input, or none."""

    @classmethod
    def setUpClass(cls):
        cls._saved_conf = copy.deepcopy(CONF._borg_shared_state)
        CONF.load(os.path.join(_ROOT, "config_pac.yml"))
        CONF.enable_cross_input_priming = True
        CONF.cross_input_leaks_only = False

    @classmethod
    def tearDownClass(cls):
        CONF._borg_shared_state.clear()
        CONF._borg_shared_state.update(cls._saved_conf)

    def _prime(self, finding):
        fz = object.__new__(Aarch64Fuzzer)
        fz.executor, fz.LOG = mock.Mock(), mock.Mock()
        det = mock.Mock()
        det.find_leaking_pair.return_value = finding
        fz._make_cross_input_detector = lambda reps, verify, loc: det
        v = violation([[M(2)], [M(5)]])
        v.measurements = [mock.Mock(htrace=mock.Mock(raw=[0] * 4))]
        with mock.patch("src.aarch64.aarch64_fuzzer.regime_controllable", return_value=True), \
                mock.patch("src.aarch64.aarch64_fuzzer.executor_regime", return_value=contextlib.nullcontext()), \
                mock.patch("src.aarch64.aarch64_fuzzer.lanes_of",
                           return_value=[["a0", "a1", "a2"], ["b0", "b1", "b2"]]):
            out = fz._priming([v], ["i"] * 6)
        return out, [c.args[1] for c in fz.LOG.inform.call_args_list]

    def test_self_dependent_and_cross_input_are_reported(self):
        for leaking, kind in ((2, "self-dependent"), (0, "cross-input")):
            out, msgs = self._prime(CrossInputFinding(leaking, 2, range(leaking, 3), True))
            self.assertEqual(len(out), 1)
            self.assertIn("any localizer, self-dependent + cross-input", msgs[0])
            self.assertIn(f"CONFIRMED {kind} leak", msgs[-1])

    def test_no_leaking_pair_is_a_false_positive(self):
        out, msgs = self._prime(None)
        self.assertEqual(out, [])
        self.assertIn("no leaking pair -> false positive", msgs[-1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
