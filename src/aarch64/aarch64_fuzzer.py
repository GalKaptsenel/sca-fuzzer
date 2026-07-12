"""
File: AArch64 fuzzer
"""
from typing import List, Generator
from contextlib import contextmanager
import copy

from ..fuzzer import FuzzerGeneric
from ..interfaces import TestCase, Input, HardwareTracingError
from ..util import STAT
from ..config import CONF
from .aarch64_executor import Aarch64Executor, pass_on_test_case
from .aarch64_generator import Aarch64DsbSyPass


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


def create_fenced_test_case(test_case: TestCase) -> TestCase:
    """Return a copy of the test case with a DSB SY speculation barrier inserted after every
    instruction (at the IR level), so it executes with speculation suppressed."""
    fenced = copy.deepcopy(test_case)
    pass_on_test_case(fenced, [Aarch64DsbSyPass()])
    return fenced


# ==================================================================================================
# Fuzzer classes
# ==================================================================================================
class Aarch64Fuzzer(FuzzerGeneric):
    executor: Aarch64Executor

    # AArch64 saves inputs with NZCV in the per-flag NZCVScheme encoding, not the
    # PSTATE form the kernel expects (see debugging/to_executor_input.py).
    input_artifact_tag: str = "_nzcv_scheme"

    def _boost_inputs(self, inputs, nesting):
        # The aarch64 fuzzer's input unit is the ExecutorInput; convert once boosting is done.
        boosted, ctraces = super()._boost_inputs(inputs, nesting)
        return list(map(self.executor.as_executor_input, boosted)), ctraces

    def _save_input(self, input_, path: str) -> None:
        # Persist the wire input so it loads verbatim into /dev/executor (no seal — offline generate).
        from .aarch64_executor_input_encoder import ExecutorInput
        ExecutorInput(input_).save(path)

    def filter(self, test_case: TestCase, inputs: List[Input]) -> bool:
        """
        This function implements a multi-stage algorithm that gradually filters out
        uninteresting test cases

        :param test_case: the target test case
        :param inputs: list of inputs to be tested
        :return: True if the test case should be filtered out; False otherwise
        """
        # Exit if no filtering is enabled
        if not CONF.enable_speculation_filter and not CONF.enable_observation_filter:
            return False

        # Number of repetitions for each input
        reps = CONF.executor_filtering_repetitions

        # Enable quick and dirty mode to speed up the process
        with quick_and_dirty_mode(self.executor):
            # Collect hardware traces for the test case
            try:
                self.executor.load_test_case(test_case)
                exec_inputs = list(map(self.executor.as_executor_input, inputs))
                org_htraces, _ = self.executor.trace_test_case(exec_inputs, reps)
            except HardwareTracingError as e:
                STAT.hw_tracing_errors += 1
                self.LOG.warning("fuzzer", f"hardware tracing failed, filtering test case: {e}")
                return True

            # 1. Speculation filter:
            # Execute on the test case on the HW and monitor PFCs
            # if there are no mispredictions, this test case is unlikely
            # to produce a violation, so just move on to the next one
            if CONF.enable_speculation_filter:
                # aarch64 PFCs: [0]=INST_RETIRED, [1]=INST_SPEC, [2]=BR_MIS_PRED. A test case can
                # only leak speculatively if some branch mispredicted; INST_SPEC >= INST_RETIRED
                # always, so the x86-style spec-vs-retired comparison does not apply here.
                for htrace in org_htraces:
                    pfc_values = htrace.perf_counters_max
                    if pfc_values[0] == 0:  # nothing retired => measurement error, can't filter
                        break
                    if pfc_values[2] > 0:   # mispredicted branch => speculation occurred
                        break
                else:
                    STAT.spec_filter += 1
                    return True

            # 2. Observation filter:
            # Check if any of the htraces contain a speculative cache eviction
            # for this create a fenced version of the test case and collect traces for it
            if CONF.enable_observation_filter:
                fenced_test_case = create_fenced_test_case(test_case)
                try:
                    self.executor.load_test_case(fenced_test_case)
                    exec_inputs = list(map(self.executor.as_executor_input, inputs))
                    fenced_htraces, _ = self.executor.trace_test_case(exec_inputs, reps)
                except HardwareTracingError:
                    return True

                traces_match = True
                for i, _ in enumerate(inputs):
                    if not self.analyser.htraces_are_equivalent(fenced_htraces[i], org_htraces[i]):
                        traces_match = False
                        break
                if traces_match:
                    STAT.observ_filter += 1
                    return True

            return False
