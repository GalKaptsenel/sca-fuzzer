"""
Copyright (C) Microsoft Corporation
SPDX-License-Identifier: MIT
"""
import unittest
from unittest import mock

from src.interfaces import CTrace, HardwareTracingError
from src.util import STAT
from src.fuzzer import FuzzerGeneric, TracingArguments
from src.config import CONF
from tests.conf_isolation import setUpModule, tearDownModule  # noqa: F401  (restores CONF + cwd)


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


class SeedPinningTest(unittest.TestCase):
    """ Regression: unset seeds are pinned in CONF before any module reads them (the AArch64 PAC keys
    derive from input_gen_seed at executor construction), so the rerun config replays the same keys. """

    def setUp(self):
        self._saved = (CONF.program_generator_seed, CONF.input_gen_seed)

    def tearDown(self):
        CONF.program_generator_seed, CONF.input_gen_seed = self._saved

    def test_unset_seeds_are_drawn_and_pinned(self):
        CONF.program_generator_seed, CONF.input_gen_seed = 0, 0
        FuzzerGeneric._pin_seeds()
        self.assertNotEqual(CONF.program_generator_seed, 0)
        self.assertNotEqual(CONF.input_gen_seed, 0)

    def test_explicit_seeds_are_kept(self):
        CONF.program_generator_seed, CONF.input_gen_seed = 285283, 1613523954
        FuzzerGeneric._pin_seeds()
        self.assertEqual((CONF.program_generator_seed, CONF.input_gen_seed), (285283, 1613523954))


if __name__ == "__main__":
    unittest.main()
