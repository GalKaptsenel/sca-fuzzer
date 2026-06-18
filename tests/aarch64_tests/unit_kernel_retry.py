"""hardware_measurement retries transient transport errors (IOError) but fails fast on an
unparseable executor response (a persistent ABI/format error), rather than retrying it."""
import unittest
from unittest import mock

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # run from any cwd
import src.aarch64.aarch64_kernel as kmod
from src.aarch64.aarch64_kernel import RemoteHWExecutor


class RetryPolicyTest(unittest.TestCase):
    def _executor(self):
        ex = RemoteHWExecutor.__new__(RemoteHWExecutor)
        ex.connection = mock.Mock()
        return ex

    def test_parse_failure_is_not_retried(self):
        ex = self._executor()
        ex._query_executor = mock.Mock(return_value="garbage without htrace/pfc")
        with mock.patch.object(kmod.time, "sleep"):
            with self.assertRaises(RuntimeError):
                ex.hardware_measurement()
        self.assertEqual(ex._query_executor.call_count, 1)

    def test_transport_error_is_retried(self):
        ex = self._executor()
        ex._query_executor = mock.Mock(side_effect=IOError("connection down"))
        with mock.patch.object(kmod.time, "sleep"):
            with self.assertRaises(IOError):
                ex.hardware_measurement()
        self.assertEqual(ex._query_executor.call_count, RemoteHWExecutor._RETRIES)


if __name__ == "__main__":
    unittest.main()
