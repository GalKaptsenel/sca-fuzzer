"""
Factory dispatch for the AArch64 non-interference executor.

`fuzzer: non-interference` selects the NI fuzzer; the separate `noninterference_mode` knob
(pac|mte) selects which NI executor the factory builds. No default — it must be set explicitly.
These tests stub the executor classes so no hardware/device is touched.
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
        # stub the executors (their real __init__ opens /dev/executor)
        self._pac = aarch64_executor.Aarch64PacNonInterferenceExecutor
        self._mte = aarch64_executor.Aarch64MteNonInterferenceExecutor
        aarch64_executor.Aarch64PacNonInterferenceExecutor = lambda g, m=False: ("pac", g, m)
        aarch64_executor.Aarch64MteNonInterferenceExecutor = lambda g, m=False: ("mte", g, m)

    def tearDown(self):
        aarch64_executor.Aarch64PacNonInterferenceExecutor = self._pac
        aarch64_executor.Aarch64MteNonInterferenceExecutor = self._mte
        CONF._borg_shared_state.clear()
        CONF._borg_shared_state.update(self._saved)

    def test_mode_defaults_to_unset(self):
        self.assertIsNone(CONF.noninterference_mode)

    def test_pac_mode_builds_pac_executor(self):
        CONF.noninterference_mode = "pac"
        self.assertEqual(factory.get_noninterference_executor("gen")[0], "pac")

    def test_mte_mode_builds_mte_executor(self):
        CONF.noninterference_mode = "mte"
        self.assertEqual(factory.get_noninterference_executor("gen")[0], "mte")

    def test_unset_mode_raises(self):
        CONF.noninterference_mode = None
        with self.assertRaises(ConfigException):
            factory.get_noninterference_executor("gen")


if __name__ == "__main__":
    unittest.main()
