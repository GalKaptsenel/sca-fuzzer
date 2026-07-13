"""RemoteHWExecutor.run_batch retries transient transport errors (IOError) but fails fast on a
malformed response (a persistent ABI/format error), rather than retrying it."""
import unittest
from unittest import mock

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # run from any cwd
import src.aarch64.aarch64_kernel as kmod
from src.aarch64.aarch64_kernel import RemoteHWExecutor, RemoteExecutorConfig


class RetryPolicyTest(unittest.TestCase):
    def _executor(self, **run_mock):
        ex = RemoteHWExecutor.__new__(RemoteHWExecutor)   # skip __init__ (no device to reach)
        ex._cfg = RemoteExecutorConfig(device="d", sysfs="s", module="m", userland="u")
        ex._conn = mock.Mock()
        ex._conn.run = mock.Mock(**run_mock)
        return ex

    def test_transport_error_is_retried(self):
        ex = self._executor(side_effect=IOError("connection down"))
        with mock.patch.object(kmod.time, "sleep"):
            with self.assertRaises(IOError):
                ex.run_batch([], 1)
        self.assertEqual(ex._conn.run.call_count, RemoteHWExecutor._RETRIES)

    def test_malformed_response_is_not_retried(self):
        ex = self._executor(return_value=b"\x00" * 40)   # valid length, bad magic -> ValueError
        with mock.patch.object(kmod.time, "sleep"):
            with self.assertRaises(ValueError):
                ex.run_batch([], 1)
        self.assertEqual(ex._conn.run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
