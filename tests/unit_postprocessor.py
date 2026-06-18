"""
Copyright (C) Microsoft Corporation
SPDX-License-Identifier: MIT
"""
import copy
import shutil
import unittest
from unittest import mock

from src.config import CONF
from src.postprocessor import MainMinimizer


class _Measurement:
    def __init__(self, input_id):
        self.input_id = input_id


class _Violation:
    def __init__(self):
        self.measurements = [_Measurement(0)]
        self.input_sequence = [object(), object()]


class _TestCase:
    def __init__(self):
        self.actors = {"main": object()}
        self.asm_path = "dummy.asm"


class InputPassAdoptionTest(unittest.TestCase):
    """ Regression for MainMinimizer.run: the input-minimization reproduction check must
    re-validate the MINIMIZED sequence (not the original), so a minimized sequence that no
    longer reproduces the violation is reverted instead of adopted. """

    def setUp(self):
        self._saved = copy.deepcopy(CONF._borg_shared_state)
        CONF.instruction_set = "aarch64"
        CONF.executor_sample_sizes = [10]
        CONF.inputs_per_class = 2
        CONF.minimizer_retries = 1

    def tearDown(self):
        CONF._borg_shared_state.clear()
        CONF._borg_shared_state.update(self._saved)

    def _make_minimizer(self, reproduce):
        """ Build a MainMinimizer with mocked collaborators. `reproduce(inputs)` decides
        whether fuzzing_round returns the violation for a given input sequence. The inputs
        passed to the instruction passes are recorded in self.seen['inputs']. """
        self.original_inputs = [object(), object()]
        self.minimized_inputs = [object()]
        violation = _Violation()
        test_case = _TestCase()

        fuzzer = mock.MagicMock()
        fuzzer.asm_parser.parse_file.return_value = test_case
        fuzzer.input_gen.generate.return_value = self.original_inputs
        fuzzer.fuzzing_round.side_effect = \
            lambda tc, inputs: violation if reproduce(inputs) else None

        minimizer = MainMinimizer(fuzzer, mock.MagicMock())
        minimizer._run_input_passes = mock.MagicMock(return_value=self.minimized_inputs)
        self.seen = {}

        def record_instruction_pass(passes, tc, inputs, viol, outfile):
            self.seen["inputs"] = inputs
            return tc
        minimizer._run_instruction_passes = record_instruction_pass
        return minimizer

    def test_non_reproducing_minimization_is_reverted(self):
        # Only the original sequence reproduces; the minimized one does not.
        minimizer = self._make_minimizer(lambda inputs: inputs is self.original_inputs)
        with mock.patch.object(shutil, "copy"):
            minimizer.run("dummy.asm", 2, "out.asm", "", 1, enable_input_seq_pass=True)
        # A non-reproducing minimization must be reverted: boosting stays enabled and the
        # original inputs continue downstream.
        self.assertEqual(CONF.inputs_per_class, 2,
                         "non-reproducing minimization was not reverted")
        self.assertIs(self.seen["inputs"], self.original_inputs)


if __name__ == "__main__":
    unittest.main()
