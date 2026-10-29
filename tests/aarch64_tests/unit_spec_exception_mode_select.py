"""
Config + factory wiring for the speculative-exception fuzzer (stage 3): the mode is REQUIRED (no
default), enum-checked, the carrier count is bounded, the old 'non-interference' value is gone, and
the x86 'noninterference' clause is untouched by the rename. No hardware/CE. Run from root:
    python -m unittest tests.aarch64_tests.unit_spec_exception_mode_select
"""
import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import CONF, ConfigException                                  # noqa: E402
from src import factory                                                       # noqa: E402
from src.aarch64 import aarch64_fuzzer                                         # noqa: E402


class SpecExceptionConfigTest(unittest.TestCase):
    def setUp(self):
        self._saved = copy.deepcopy(CONF._borg_shared_state)
        CONF.instruction_set = "aarch64"
        CONF.set_to_arch_defaults()
        CONF.fuzzer = "speculative-exception"

    def tearDown(self):
        CONF._borg_shared_state.clear()
        CONF._borg_shared_state.update(self._saved)

    def test_mode_required_when_selected(self):
        CONF.speculative_exception_mode = None
        with self.assertRaises(ConfigException):
            CONF._value_sanity_check()

    def test_mode_set_passes_sanity_check(self):
        CONF.speculative_exception_mode = "identical"
        CONF._value_sanity_check()                       # no raise
        CONF.speculative_exception_mode = "full"
        CONF._value_sanity_check()

    def test_invalid_mode_value_rejected(self):
        with self.assertRaises(ConfigException):
            CONF.safe_set("speculative_exception_mode", "bogus")

    def test_carrier_count_out_of_range_rejected(self):
        CONF.speculative_exception_mode = "identical"
        for bad in (0, 17):
            CONF.speculative_exception_carrier_count = bad
            with self.assertRaises(ConfigException):
                CONF._value_sanity_check()

    def test_carrier_count_in_range_ok(self):
        CONF.speculative_exception_mode = "full"
        CONF.speculative_exception_carrier_count = 16
        CONF._value_sanity_check()                       # no raise


class RenameSafetyTest(unittest.TestCase):
    def setUp(self):
        self._saved = copy.deepcopy(CONF._borg_shared_state)
        CONF.instruction_set = "aarch64"
        CONF.set_to_arch_defaults()

    def tearDown(self):
        CONF._borg_shared_state.clear()
        CONF._borg_shared_state.update(self._saved)

    def test_old_fuzzer_value_rejected(self):
        with self.assertRaises(ConfigException):
            CONF.safe_set("fuzzer", "non-interference")

    def test_factory_selects_spec_exception_fuzzer(self):
        real = aarch64_fuzzer.Aarch64SpeculativeExceptionFuzzer
        aarch64_fuzzer.Aarch64SpeculativeExceptionFuzzer = lambda *a, **k: ("spec-exc", a)
        try:
            CONF.fuzzer = "speculative-exception"
            result = factory.get_fuzzer("base.json", "/tmp", None, None)
            self.assertEqual(result[0], "spec-exc")
        finally:
            aarch64_fuzzer.Aarch64SpeculativeExceptionFuzzer = real

    def test_x86_noninterference_concept_untouched(self):
        # The rename is AArch64-scoped: x86's 'noninterference' contract clause must still exist
        # (checked via config, which imports without the unicorn/x86_model dependency).
        from src.config import Conf
        self.assertIn("noninterference", Conf._option_values["contract_execution_clause"])


if __name__ == "__main__":
    unittest.main()
