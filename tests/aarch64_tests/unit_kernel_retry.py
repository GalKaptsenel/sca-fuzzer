"""RemoteHWExecutor.run_batch retries transient transport errors (IOError) but fails fast on a
malformed response (a persistent ABI/format error), rather than retrying it."""
import unittest
from unittest import mock

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # run from any cwd
import src.aarch64.aarch64_kernel as kmod
from src.aarch64.aarch64_kernel import RemoteHWExecutor, RemoteExecutorConfig
from src.interfaces import HardwareTracingError


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
        # A well-formed-length but bad-magic response is a persistent format error: decode_response
        # raises ValueError, which run_batch surfaces as HardwareTracingError and does NOT retry
        # (retry is IOError-only).
        ex = self._executor(return_value=b"\x00" * 40)
        with mock.patch.object(kmod.time, "sleep"):
            with self.assertRaises(HardwareTracingError):
                ex.run_batch([], 1)
        self.assertEqual(ex._conn.run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
