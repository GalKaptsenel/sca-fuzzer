"""
Copyright (C) Microsoft Corporation
SPDX-License-Identifier: MIT
"""
import unittest
from unittest import mock

from src.interfaces import CTrace, HardwareTracingError
from src.util import STAT
from src.fuzzer import FuzzerGeneric, TracingArguments


class HardwareTracingErrorTest(unittest.TestCase):
    """ Regression: a HardwareTracingError must be surfaced (counted + logged) and the test case
    skipped, not silently swallowed as 'no violation'. """

    def test_tracing_error_is_counted_logged_and_skipped(self):
        fuzzer = FuzzerGeneric.__new__(FuzzerGeneric)
        fuzzer.LOG = mock.MagicMock()
        fuzzer.executor = mock.MagicMock()
        fuzzer.executor.trace_test_case.side_effect = HardwareTracingError("boom")

        args = TracingArguments(
            inputs=[object()], n_reps=1, model_nesting=1, ctraces=[CTrace([1])],
            record_stats=False, fast_boosting=False, update_ignore_list=False,
            reuse_ctraces=True, added_htraces=[])

        before = STAT.hw_tracing_errors
        result = fuzzer._collect_traces(args)

        self.assertEqual(result, ([], [], []))
        self.assertEqual(STAT.hw_tracing_errors, before + 1)
        fuzzer.LOG.warning.assert_called_once()


if __name__ == "__main__":
    unittest.main()
