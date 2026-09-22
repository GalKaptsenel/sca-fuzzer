"""
Tests for the cross-input priming CONFIG KNOBS and fuzzer WIRING (the localizer algorithms themselves
are covered by tests.unit_aarch64_leftover.LocalizerTest). Run from the repo root:
    python -m unittest tests.aarch64_tests.unit_cross_input_priming

The config half touches only src.config (no hardware, no capstone). The wiring half imports the aarch64
fuzzer (needs capstone) lazily inside its tests, so a capstone-less environment still runs the config
tests and only errors the wiring ones.
"""
import os
import sys
import copy
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import CONF, ConfigException                       # noqa: E402


class CrossInputPrimingConfigTest(unittest.TestCase):
    """The two knobs: enable_cross_input_priming and cross_input_priming_localizer."""

    def setUp(self):
        self._saved = copy.deepcopy(CONF._borg_shared_state)
        CONF._borg_shared_state.clear()
        CONF.__init__()
        CONF.instruction_set = "aarch64"
        CONF.set_to_arch_defaults()

    def tearDown(self):
        CONF._borg_shared_state.clear()
        CONF._borg_shared_state.update(self._saved)

    def test_defaults(self):
        # off by default (standard priming), and the default localizer is the "any" (galloping) one.
        self.assertEqual(CONF.enable_cross_input_priming, False)
        self.assertEqual(CONF.cross_input_priming_localizer, "any")

    def test_removed_leftover_knobs_are_gone(self):
        for gone in ("enable_leftover_detection", "leftover_reps", "leftover_verify_reps",
                     "enable_boosted_leftover"):
            self.assertFalse(hasattr(CONF, gone), f"{gone} should have been removed")

    def test_localizer_accepts_any_and_optimal(self):
        for v in ("any", "optimal"):
            CONF.safe_set("cross_input_priming_localizer", v)         # config-file load path
            self.assertEqual(CONF.cross_input_priming_localizer, v)

    def test_localizer_rejects_unknown_value(self):
        with self.assertRaises(ConfigException):
            CONF.safe_set("cross_input_priming_localizer", "bogus")

    def test_enable_is_boolean(self):
        CONF.safe_set("enable_cross_input_priming", True)
        self.assertTrue(CONF.enable_cross_input_priming)
        with self.assertRaises(ConfigException):
            CONF.safe_set("enable_cross_input_priming", "yes")        # wrong type -> loud


class CrossInputPrimingWiringTest(unittest.TestCase):
    """The fuzzer seam: the localizer name->strategy map, the shared mixin, and the NI seal regime."""

    def test_localizer_name_to_strategy_map(self):
        from src.aarch64 import aarch64_fuzzer as F
        from src.aarch64.leftover import exponential_search, linear_scan
        self.assertIs(F._CROSS_INPUT_PRIMING_LOCALIZERS["any"], exponential_search)
        self.assertIs(F._CROSS_INPUT_PRIMING_LOCALIZERS["optimal"], linear_scan)
        # every configured value must resolve to a strategy (no silent gap)
        for v in ("any", "optimal"):
            self.assertTrue(callable(F._CROSS_INPUT_PRIMING_LOCALIZERS[v]))

    def test_both_fuzzers_share_the_mixin(self):
        from src.aarch64.aarch64_fuzzer import (Aarch64Fuzzer, Aarch64NoninterferenceFuzzer,
                                                CrossInputPrimingMixin)
        for cls in (Aarch64Fuzzer, Aarch64NoninterferenceFuzzer):
            self.assertIn(CrossInputPrimingMixin, cls.__mro__)
            # the mixin's priming override must win over the generic FuzzerGeneric._priming
            self.assertIs(cls._priming, CrossInputPrimingMixin._priming)

    def test_ni_uses_the_seal_regime_basic_does_not(self):
        from src.aarch64.aarch64_fuzzer import Aarch64Fuzzer, Aarch64NoninterferenceFuzzer
        ni = dict(Aarch64NoninterferenceFuzzer._CROSS_INPUT_PRIMING_REGIME)
        basic = dict(Aarch64Fuzzer._CROSS_INPUT_PRIMING_REGIME)
        # NI (canonicality) needs SSBS on and unpinned execution; the basic (v2/BTB) regime does not.
        self.assertEqual(ni.get("enable_ssbs"), "1")
        self.assertEqual(ni.get("pin_to_core"), "-1")
        self.assertNotIn("enable_ssbs", basic)
        # both keep cross-input training within a trace (no per-input view rotation)
        self.assertEqual(ni.get("enable_view_rotation"), "0")
        self.assertEqual(basic.get("enable_view_rotation"), "0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
