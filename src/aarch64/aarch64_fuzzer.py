"""
File: x86 implementation of the test case generator

Copyright (C) Microsoft Corporation
SPDX-License-Identifier: MIT
"""
from subprocess import run
from typing import List, Optional, Generator
from contextlib import contextmanager
import tempfile
import os

from ..fuzzer import FuzzerGeneric, ArchitecturalFuzzer
from ..interfaces import TestCase, Input, InstructionSetAbstract, Violation, Measurement, \
    HTrace, HardwareTracingError, CTrace
from ..util import STAT, Logger
from ..config import CONF
from .aarch64_executor import Aarch64Executor


# ==================================================================================================
# Helper functions
# ==================================================================================================

@contextmanager
def quick_and_dirty_mode(executor: Aarch64Executor) -> Generator[None, None, None]:
    """
    Context manager that enables us to use quick and dirty mode in the form of `with` statement
    """
    try:
        executor.set_quick_and_dirty(True)
        yield
    finally:
        executor.set_quick_and_dirty(False)


def create_fenced_test_case(test_case: TestCase, fenced_name: str, asm_parser) -> TestCase:
    with open(test_case.asm_path, 'r') as f:
        with open(fenced_name, 'w') as fenced_asm:
            started = False
            for line in f:
                fenced_asm.write(line + '\n')
                line = line.strip().lower()
                if line == '.test_case_enter:':
                    started = True
                    continue
                if not started:
                    continue
                if line and line[0] not in ["#", ".", "j"] \
                        and "loop" not in line \
                        and "macro" not in line:
                    fenced_asm.write('lfence\n')
    fenced_test_case = asm_parser.parse_file(fenced_name)
    return fenced_test_case


# ==================================================================================================
# Fuzzer classes
# ==================================================================================================
class Aarch64Fuzzer(FuzzerGeneric):
    executor: Aarch64Executor

    def filter(self, test_case: TestCase, inputs: List[Input]) -> bool:
        """
        This function implements a multi-stage algorithm that gradually filters out
        uninteresting test cases

        :param test_case: the target test case
        :param inputs: list of inputs to be tested
        :return: True if the test case should be filtered out; False otherwise
        """
        # Exit if not filtering is enabled
        if not CONF.enable_speculation_filter and not CONF.enable_observation_filter:
            return False

        # Number of repetitions for each input
        reps = CONF.executor_filtering_repetitions

        # Enable quick and dirty mode to speed up the process
        with quick_and_dirty_mode(self.executor):
            # Collect hardware traces for the test case
            try:
                self.executor.load_test_case(test_case)
                org_htraces = self.executor.trace_test_case(inputs, reps)
            except HardwareTracingError:
                return True

            # 1. Speculation filter:
            # Execute on the test case on the HW and monitor PFCs
            # if there are no mispredictions, this test case is unlikely
            # to produce a violation, so just move on to the next one
            if CONF.enable_speculation_filter:
                for i, htrace in enumerate(org_htraces):
                    pfc_values = htrace.perf_counters_max
                    if pfc_values[0] == 0:  # zero indicates an error; filtering is not possible
                        break
                    if pfc_values[0] > pfc_values[1] or pfc_values[2] > 0:
                        break
                else:
                    STAT.spec_filter += 1
                    return True

            # 2. Observation filter:
            # Check if any of the htraces contain a speculative cache eviction
            # for this create a fenced version of the test case and collect traces for it
            if CONF.enable_observation_filter:
                fenced = tempfile.NamedTemporaryFile(delete=False)
                fenced_test_case = create_fenced_test_case(test_case, fenced.name, self.asm_parser)
                try:
                    self.executor.load_test_case(fenced_test_case)
                    fenced_htraces = self.executor.trace_test_case(inputs, reps)
                except HardwareTracingError:
                    return True
                os.remove(fenced.name)

                traces_match = True
                for i, _ in enumerate(inputs):
                    if not self.analyser.htraces_are_equivalent(fenced_htraces[i], org_htraces[i]):
                        traces_match = False
                        break
                if traces_match:
                    STAT.observ_filter += 1
                    return True

            return False


class Aarch64ArchitecturalFuzzer(ArchitecturalFuzzer):
    pass
class Aarch64ArchDiffFuzzer(FuzzerGeneric):
    executor: Aarch64Executor

    def _build_dummy_ecls(self) -> Violation:
        inputs = [Input()]
        ctrace = CTrace.get_null()
        measurements = [Measurement(0, inputs[0], ctrace, HTrace([0]))]
        violation = Violation.from_measurements(ctrace, measurements, [], inputs)
        return violation

    def fuzzing_round(self,
                      test_case: TestCase,
                      inputs: List[Input],
                      _: List[int] = None) -> Optional[Violation]:
        if _ is None:
            _ = []

        with quick_and_dirty_mode(self.executor):
            # collect non-fenced traces
            self.arch_executor.load_test_case(test_case)
            reg_values: List[List[int]] = []
            try:
                htraces: List[HTrace] = self.arch_executor.trace_test_case(inputs, 1)
            except HardwareTracingError:
                return None
            for htrace in htraces:
                reg_values.append([htrace.raw[0]] + [int(v) for v in htrace.perf_counters[0]])

            # collect fenced traces
            fenced = tempfile.NamedTemporaryFile(delete=False)
            fenced_test_case = create_fenced_test_case(test_case, fenced.name, self.asm_parser)
            self.arch_executor.load_test_case(fenced_test_case)
            fenced_reg_values: List[List[int]] = []
            try:
                htraces = self.arch_executor.trace_test_case(inputs, 1)
            except HardwareTracingError:
                return None
            for htrace in htraces:
                fenced_reg_values.append([htrace.raw[0]]
                                         + [int(v) for v in htrace.perf_counters[0]])
            os.remove(fenced.name)

            for i, input_ in enumerate(inputs):
                if fenced_reg_values[i] == reg_values[i]:
                    if "dbg_dump_htraces" in CONF.logging_modes:
                        print(f"Input #{i}")
                        print(f"Fenced:       {[v for v in fenced_reg_values[i]]}")
                        print(f"Non-fenced:   {[v for v in reg_values[i]]}")
                    continue

                if "dbg_violation" in CONF.logging_modes:
                    print(f"Input #{i}")
                    print(f"Fenced:       {[v for v in fenced_reg_values[i]]}")
                    print(f"Non-fenced:   {[v for v in reg_values[i]]}")

                return self._build_dummy_ecls()
            return None
