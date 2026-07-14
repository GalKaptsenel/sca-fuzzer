"""
Copyright (C) Microsoft Corporation
SPDX-License-Identifier: MIT
"""
import unittest

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # run from any cwd
from src.aarch64.aarch64_executor import Aarch64LocalExecutor
from src.aarch64.aarch64_kernel import HWMeasurement


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
        per_input = [[HWMeasurement(htrace=h, pfcs=(1, 2, 3)) for _ in range(n_reps)]
                     for h in (11, 22, 33)]

        htraces = executor._aggregate_measurements(per_input)

        self.assertEqual(htraces[0].raw, [11, 11, 11])
        self.assertEqual(htraces[1].raw, [0, 0, 0], "ignored input htrace was not zeroed")
        self.assertEqual(htraces[2].raw, [33, 33, 33])


if __name__ == "__main__":
    unittest.main()
