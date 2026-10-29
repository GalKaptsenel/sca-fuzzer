"""A contract-executor crash during speculative-exception variant tracing must propagate, not be swallowed.
Dependencies are mocked."""
import unittest
from unittest import mock

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # run from any cwd
import src.aarch64.aarch64_executor as ex_mod
from src.aarch64.aarch64_executor import Aarch64SpeculativeExceptionExecutor


class SandboxedCeCrashTest(unittest.TestCase):
    def _executor(self):
        ex = Aarch64SpeculativeExceptionExecutor.__new__(Aarch64SpeculativeExceptionExecutor)
        ex.test_case = mock.Mock()
        ex._sealed = mock.Mock()
        ex._resolve_cache = {}
        ex._nesting = 0
        # resolve() runs the CE resolution trace internally; a crash there must propagate, not be
        # swallowed.
        ex._sealed.resolve.side_effect = RuntimeError("CE crashed")
        ex.read_base_addresses = mock.Mock(return_value=(0x1000, 0x2000))
        return ex

    def test_ce_crash_propagates(self):
        ex = self._executor()
        with mock.patch.object(ex_mod, "log_input"):
            with self.assertRaises(RuntimeError):
                ex.trace_test_case_with_taints([mock.Mock()], 0)


if __name__ == "__main__":
    unittest.main()
