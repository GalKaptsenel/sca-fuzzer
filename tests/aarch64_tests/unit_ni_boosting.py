"""NI boosting collapses onto the base engine: real-CE seal variants, mocked HW htraces.
  * a class is filled with the input's seal variants, one composite ctrace per input.
  * baseline-vs-decoy htrace divergence is detected as a violation carrying the ExecutorInput variants.
  * identical variant htraces do not violate.
Needs /dev/executor + the CE.
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sealed_fixture import SealedExecutorFixture
from src.config import CONF
from src import factory
from src.interfaces import HTrace


class NiBoostingTest(SealedExecutorFixture, unittest.TestCase):
    def _fuzzer(self):
        from src.fuzzer import NoninterferenceFuzzer
        CONF.analyser = "sets"                       # deterministic set comparison for this test
        fz = NoninterferenceFuzzer.__new__(NoninterferenceFuzzer)
        fz.executor = self.ex
        fz.analyser = factory.get_analyser()
        fz.LOG = mock.Mock()
        return fz

    def _boosted_args(self, fz, inputs):
        from src.fuzzer import TracingArguments
        boosted, ctraces = fz._boost_inputs(inputs, CONF.model_max_nesting)
        args = TracingArguments(inputs=boosted, n_reps=4, model_nesting=CONF.model_max_nesting,
                                ctraces=ctraces, record_stats=False, fast_boosting=True,
                                update_ignore_list=False, reuse_ctraces=False, added_htraces=[])
        return boosted, args

    def test_boost_fills_class_with_seal_variants(self):
        fz = self._fuzzer()
        inputs = self.igen.generate(1)
        boosted, ctraces = fz._boost_inputs(inputs, CONF.model_max_nesting)
        self.assertEqual(len(boosted), CONF.inputs_per_class * len(inputs))   # class filled by seal
        self.assertTrue(all(isinstance(b, self.ExecutorInput) for b in boosted))
        self.assertEqual(len(ctraces), len(inputs))                          # one composite per input

    def test_boost_and_detect_divergence(self):
        fz = self._fuzzer()
        boosted, args = self._boosted_args(fz, self.igen.generate(1))

        # baseline (flat 0) diverges from the rest -> a violation, carrying the ExecutorInput variants
        def fake(flat, n):
            return ([HTrace([1] * 4) if i == 0 else HTrace([0] * 4) for i in range(len(flat))],
                    [None] * len(flat))
        with mock.patch.object(self.ex, "trace_test_case", side_effect=fake):
            violations, _c, _h = fz._collect_traces(args)
        self.assertTrue(violations, "baseline vs decoy divergence must be a violation")
        self.assertTrue(all(isinstance(m.input_, self.ExecutorInput)
                            for v in violations for m in v.measurements))

    def test_boost_no_violation_when_variants_agree(self):
        fz = self._fuzzer()
        _boosted, args = self._boosted_args(fz, self.igen.generate(1))
        with mock.patch.object(self.ex, "trace_test_case",
                               side_effect=lambda flat, n: ([HTrace([0] * 4)] * len(flat),
                                                            [None] * len(flat))):
            violations, _c, _h = fz._collect_traces(args)
        self.assertEqual(violations, [], "identical variant htraces must not violate")


if __name__ == "__main__":
    unittest.main()
