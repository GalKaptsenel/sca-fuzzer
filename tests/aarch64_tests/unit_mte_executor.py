"""A contract-executor crash during NI variant tracing must propagate, not be swallowed.
Dependencies are mocked."""
import unittest
from unittest import mock

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # run from any cwd
import src.aarch64.aarch64_executor as ex_mod
from src.aarch64.aarch64_executor import Aarch64NonInterferenceExecutor


class SandboxedCeCrashTest(unittest.TestCase):
    def _executor(self):
        ex = Aarch64NonInterferenceExecutor.__new__(Aarch64NonInterferenceExecutor)
        ex.test_case = mock.Mock()
        ex._sealed_tc = mock.Mock()
        ex._fix_points = []
        ex._sealed_tc_bytes = b""
        ex._layout = None
        ex._pac_fps = []
        ex._mem_fps = []
        ex._engine = mock.Mock()
        ex.read_base_addresses = mock.Mock(return_value=(0x1000, 0x2000))
        ex._assemble_tc = mock.Mock(return_value=(b"", None))
        ex._make_ce_execution = mock.Mock()
        ex._contract_executor = mock.Mock()
        ex._contract_executor.run.side_effect = RuntimeError("CE crashed")
        return ex

    def test_ce_crash_propagates(self):
        ex = self._executor()
        with mock.patch.object(ex_mod, "pass_on_test_case"), \
             mock.patch.object(ex_mod, "Aarch64SandboxPass"), \
             mock.patch.object(ex_mod, "log_input"), \
             mock.patch.object(ex_mod.copy, "deepcopy", return_value=mock.Mock()):
            with self.assertRaises(RuntimeError):
                ex.trace_test_case_with_taints([mock.Mock()], 0)


if __name__ == "__main__":
    unittest.main()
