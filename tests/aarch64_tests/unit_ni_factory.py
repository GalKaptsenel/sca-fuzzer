"""
Factory dispatch for the AArch64 speculative-exception executor.

`fuzzer: speculative-exception` selects the speculative-exception fuzzer; get_speculative_exception_executor builds the single
unified speculative-exception executor, which auto-detects the active primitives (PAC and/or MTE) from the enabled
instruction categories. These tests stub the executor class so no hardware/device is touched.
"""
import copy
import unittest

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # any cwd
from src.config import CONF, ConfigException
from src import factory
from src.aarch64 import aarch64_executor


class NonInterferenceFactoryDispatchTest(unittest.TestCase):
    def setUp(self):
        self._saved = copy.deepcopy(CONF._borg_shared_state)
        CONF.instruction_set = "aarch64"
        CONF.set_to_arch_defaults()
        # stub the executor (its real __init__ opens /dev/executor)
        self._real = aarch64_executor.Aarch64SpeculativeExceptionExecutor
        aarch64_executor.Aarch64SpeculativeExceptionExecutor = lambda g, m=False: ("ni", g, m)

    def tearDown(self):
        aarch64_executor.Aarch64SpeculativeExceptionExecutor = self._real
        CONF._borg_shared_state.clear()
        CONF._borg_shared_state.update(self._saved)

    def test_builds_unified_ni_executor(self):
        result = factory.get_speculative_exception_executor("gen")
        self.assertEqual(result[0], "ni")
        self.assertEqual(result[1], "gen")

    def test_non_aarch64_raises(self):
        CONF.instruction_set = "x86-64"
        with self.assertRaises(ConfigException):
            factory.get_speculative_exception_executor("gen")


if __name__ == "__main__":
    unittest.main()
