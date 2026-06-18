"""
Copyright (C) Microsoft Corporation
SPDX-License-Identifier: MIT
"""
import unittest

from src.aarch64.aarch64_executor import Aarch64LocalExecutor


class IgnoreListZeroingTest(unittest.TestCase):
    """ Regression: ignored (priming) inputs must get a zeroed htrace so they are excluded
    from the equivalence analysis (matching the set_ignore_list contract and the x86 executor). """

    def _executor(self, ignore_list):
        executor = Aarch64LocalExecutor.__new__(Aarch64LocalExecutor)
        executor.ignore_list = set(ignore_list)
        return executor

    def test_ignored_input_htrace_is_zeroed(self):
        executor = self._executor({1})
        n_reps = 3
        traces = {0: [11, 11, 11], 1: [22, 22, 22], 2: [33, 33, 33]}
        pfc = {i: [[1, 2, 3]] * n_reps for i in range(3)}

        htraces = executor._aggregate_htraces(3, n_reps, traces, pfc)

        self.assertEqual(htraces[0].raw, [11, 11, 11])
        self.assertEqual(htraces[1].raw, [0, 0, 0], "ignored input htrace was not zeroed")
        self.assertEqual(htraces[2].raw, [33, 33, 33])


if __name__ == "__main__":
    unittest.main()
