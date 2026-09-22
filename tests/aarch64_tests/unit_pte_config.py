"""Config-knob tests for PTE fuzzing (touches only src.config; no hardware, no capstone):
    python -m unittest tests.aarch64_tests.unit_pte_config
"""
import os
import sys
import copy
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import CONF, ConfigException                                       # noqa: E402
from src.aarch64.seal.pagetable_model import LEAF_LAYOUT                            # noqa: E402
from src.aarch64.seal.environment import PteFuzzPolicy                              # noqa: E402


class PteConfigTest(unittest.TestCase):
    def setUp(self):
        self._saved = copy.deepcopy(CONF._borg_shared_state)
        CONF._borg_shared_state.clear()
        CONF.__init__()
        CONF.instruction_set = "aarch64"
        CONF.set_to_arch_defaults()

    def tearDown(self):
        CONF._borg_shared_state.clear()
        CONF._borg_shared_state.update(self._saved)

    def test_defaults_off_and_leaf_only(self):
        self.assertEqual(CONF.enable_pte_fuzzing, False)
        self.assertEqual(CONF.pte_fuzz_table_fields, [])           # leaf-only by default

    def test_default_leaf_fields_match_the_layout(self):
        # config and the descriptor model must agree on what is fuzzable by default (no drift)
        self.assertEqual(CONF.pte_fuzz_leaf_fields, LEAF_LAYOUT.default_fuzzable_field_names)

    def test_enable_is_boolean(self):
        CONF.safe_set("enable_pte_fuzzing", True)
        self.assertTrue(CONF.enable_pte_fuzzing)
        with self.assertRaises(ConfigException):
            CONF.safe_set("enable_pte_fuzzing", "yes")             # wrong type -> loud

    def test_configured_fields_build_a_valid_policy(self):
        # the fields the config ships must be accepted by the policy (validated against the layout)
        PteFuzzPolicy(leaf_fields=CONF.pte_fuzz_leaf_fields,
                      table_fields=CONF.pte_fuzz_table_fields)


if __name__ == "__main__":
    unittest.main(verbosity=2)
