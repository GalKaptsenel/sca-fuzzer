"""The bulk tier (_bulk_filter_candidates) measures a whole window's fast path in ONE executor
super-batch and returns only the test cases the analyser flags — non-candidates finish there. Fully
device-free: the executor and analyser are mocked."""
import unittest
from unittest import mock

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # run from any cwd
from src.fuzzer import FuzzerGeneric
from src.config import CONF
from src.util import STAT


class BulkTierTest(unittest.TestCase):
    _CONF_FIELDS = ("contract_execution_clause", "model_min_nesting",
                    "executor_sample_sizes", "inputs_per_class")

    def setUp(self):
        self._saved = {n: getattr(CONF, n) for n in self._CONF_FIELDS}
        CONF.contract_execution_clause = ["seq"]     # start_nesting = 1, no model_min_nesting dep
        CONF.executor_sample_sizes = [10, 50]
        CONF.inputs_per_class = 2

    def tearDown(self):
        for name, value in self._saved.items():
            setattr(CONF, name, value)

    def _driver(self, filter_results):
        drv = FuzzerGeneric.__new__(FuzzerGeneric)
        drv.executor = mock.Mock()
        drv.executor.make_trace_unit.side_effect = lambda boosted: ("unit", tuple(boosted))
        drv.executor.trace_batch.side_effect = lambda units, n_reps: [["htrace"] for _ in units]
        drv.analyser = mock.Mock()
        drv.analyser.filter_violations.side_effect = list(filter_results)
        # boosted = inputs_per_class round-major copies; ctraces = one per original input
        drv._boost_inputs = lambda inputs, nesting: (
            list(inputs) * CONF.inputs_per_class, [("ct", k) for k in range(len(inputs))])
        return drv

    def test_only_flagged_test_cases_become_candidates(self):
        drv = self._driver(filter_results=[[], ["a-violation"]])   # tc0 clean, tc1 flagged
        window = [("tc0", ["i0"]), ("tc1", ["i1"])]
        fast_path_before = STAT.fast_path

        candidates = drv._bulk_filter_candidates(window)

        self.assertEqual(candidates, [("tc1", ["i1"])])
        self.assertEqual([c.args[0] for c in drv.executor.load_test_case.call_args_list],
                         ["tc0", "tc1"])
        # exactly one super-batch, carrying both units, at the first sample size
        drv.executor.trace_batch.assert_called_once()
        units, n_reps = drv.executor.trace_batch.call_args.args
        self.assertEqual(len(units), 2)
        self.assertEqual(n_reps, CONF.executor_sample_sizes[0])
        self.assertEqual(STAT.fast_path - fast_path_before, 1)   # only the clean tc0

    def test_ctraces_are_replicated_per_class(self):
        drv = self._driver(filter_results=[[]])
        drv._bulk_filter_candidates([("tc0", ["i0", "i1"])])
        boosted, ctraces = drv.analyser.filter_violations.call_args.args[0:2]
        # 2 inputs x inputs_per_class boosted; ctraces replicated to match, round-major
        self.assertEqual(len(boosted), 2 * CONF.inputs_per_class)
        self.assertEqual(len(ctraces), len(boosted))
        self.assertEqual(ctraces, [("ct", 0), ("ct", 1)] * CONF.inputs_per_class)

    def test_empty_window_makes_no_device_call(self):
        drv = self._driver(filter_results=[])
        self.assertEqual(drv._bulk_filter_candidates([]), [])
        drv.executor.trace_batch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
