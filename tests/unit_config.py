"""
Unit tests for Conf: unsupported-option handling and arch-default loading.
"""
import copy
import unittest

from src.config import CONF, ConfigException

# A name that does not exist in any production config; used to exercise the
# unsupported-option mechanism without coupling the test to real knob names.
TEST_UNSUPPORTED_OPTION = "test_only_unsupported_option"


class UnsupportedOptionsTest(unittest.TestCase):
    def setUp(self):
        # CONF is a Borg singleton; snapshot its shared state so this test
        # cannot leak instruction_set / _unsupported_options into other tests.
        self._saved_state = copy.deepcopy(CONF._borg_shared_state)
        CONF.instruction_set = "aarch64"
        CONF.set_to_arch_defaults()
        CONF._unsupported_options = [TEST_UNSUPPORTED_OPTION]

    def tearDown(self):
        CONF._borg_shared_state.clear()
        CONF._borg_shared_state.update(self._saved_state)

    def test_supported_option_still_readable(self):
        self.assertEqual(CONF.executor, "aarch64")

    def test_reading_unsupported_option_raises(self):
        with self.assertRaises(ConfigException):
            getattr(CONF, TEST_UNSUPPORTED_OPTION)

    def test_setting_unsupported_option_raises(self):
        with self.assertRaises(ConfigException):
            CONF.safe_set(TEST_UNSUPPORTED_OPTION, True)

    def test_unknown_option_still_raises(self):
        with self.assertRaises(ConfigException):
            CONF.safe_set("totally_made_up_var", 1)

    def test_switching_arch_resets_unsupported_list(self):
        self.assertIn(TEST_UNSUPPORTED_OPTION, CONF._unsupported_options)
        CONF.instruction_set = "x86-64"
        CONF.set_to_arch_defaults()
        self.assertEqual(CONF._unsupported_options, [])


class CheckOptionsFalsyTest(unittest.TestCase):
    """ Regression: a falsy invalid option value (e.g. an empty string) must be rejected, not
    silently accepted because of an `if invalid_value:` truthiness check. """

    def setUp(self):
        self._saved_state = copy.deepcopy(CONF._borg_shared_state)
        CONF.instruction_set = "aarch64"
        CONF.set_to_arch_defaults()

    def tearDown(self):
        CONF._borg_shared_state.clear()
        CONF._borg_shared_state.update(self._saved_state)

    def test_empty_string_value_rejected(self):
        with self.assertRaises(ConfigException):
            CONF._check_options("executor", "")

    def test_falsy_list_element_rejected(self):
        with self.assertRaises(ConfigException):
            CONF._check_options("logging_modes", [""])

    def test_valid_values_accepted(self):
        CONF._check_options("executor", "aarch64")
        CONF._check_options("logging_modes", ["info"])


if __name__ == "__main__":
    unittest.main()
