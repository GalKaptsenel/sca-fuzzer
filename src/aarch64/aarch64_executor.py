"""
File: Implementation of executor for x86 architecture
  - Interfacing with the kernel module
  - Aggregation of the results

Copyright (C) Microsoft Corporation
SPDX-License-Identifier: MIT
"""
import copy
import subprocess
import os.path
import os

import numpy as np
from typing import List, Tuple, Set, Generator, Optional, Any

from .aarch64_generator import Aarch64TagMemoryAccesses, Aarch64Printer, Aarch64MarkMemoryAccesses
from .. import ConfigurableGenerator
from ..interfaces import HTrace, Input, TestCase, Executor, HardwareTracingError
from ..config import CONF
from ..util import Logger, STAT
from .aarch64_target_desc import Aarch64TargetDesc
from .aarch64_connection import Connection, UserlandExecutorImp, TestCaseRegion, InputRegion, \
    HWMeasurement
from .aarch64_generator import Pass


# ==================================================================================================
# Helper functions
# ==================================================================================================
def km_write(value, path: str) -> None:
    subprocess.run(f"echo -n {value} > {path}", shell=True, check=True)


def km_write_bytes(value: bytes, path: str) -> None:
    with open(path, "wb") as f:
        f.write(value)


def can_set_reserved() -> bool:
    """
    Check if setting reserved bits is possible on the current CPU.
    :return: True if it's possible, False otherwise
    """
    reserved_requested = \
        any(CONF._actors[a]['data_properties']['reserved_bit'] for a in CONF._actors) or \
        any(CONF._actors[a]['data_ept_properties']['reserved_bit'] for a in CONF._actors)
    if reserved_requested:
        physical_bits = int(
            subprocess.run(
                "lscpu | grep 'Address sizes' | awk '{print $3}'",
                shell=True,
                check=True,
                capture_output=True).stdout.decode().strip())
        if physical_bits > 51:
            return False
    return True


def is_kernel_module_installed() -> bool:
    return os.path.isfile("/dev/executor")


def configure_kernel_module() -> None:
    km_write(CONF.executor_warmups, '/sys/x86_executor/warmups')
    km_write("1" if getattr(CONF, 'x86_executor_enable_ssbp_patch') else "0",
             "/sys/x86_executor/enable_ssbp_patch")
    km_write("1" if getattr(CONF, 'x86_executor_enable_prefetcher') else "0",
             "/sys/x86_executor/enable_prefetcher")
    km_write("1" if CONF.enable_pre_run_flush else "0", "/sys/x86_executor/enable_pre_run_flush")
    km_write(CONF.executor_mode, "/sys/x86_executor/measurement_mode")
    km_write("1" if getattr(CONF, 'x86_enable_hpa_gpa_collisions') else "0",
             "/sys/x86_executor/enable_hpa_gpa_collisions")


def read_trace(
    n_reps: int,
    n_inputs: int,
    enable_warnings: bool = True) -> Generator[Tuple[int, int, int, List[int]], None, None]:
    """
    Generator function that reads the traces from the kernel module.
    The generator handles the batched output of the kernel module and yields the traces one by one.
    The traces are read in reverse order.

    Example:
    Assume the kernel module output for n_reps=2 and n_inputs=2 is:
    ```
    htrace1, pfc1..5
    htrace0, pfc1..5
    done
    htrace1, pfc1..5
    htrace0, pfc1..5
    done
    ```
    then the generator will yield the following tuples:
    ```
    (0, 1, htrace1, [pfc1..5])
    (0, 0, htrace0, [pfc1..5])
    (1, 1, htrace1, [pfc1..5])
    (1, 0, htrace0, [pfc1..5])
    ```

    :param n_reps: number of repetitions of the measurements
    :param n_inputs: number of inputs
    :param enable_warnings: if True, the function will print warnings if the kernel module output is
           malformed or if it returns an error
    :return: a generator that yields a tuple (repetition, input_id, htrace, [pfc1, ..., pfc5])
    :raises HardwareTracingError: if the kernel module output is malformed
    """
    if n_inputs <= 0:
        return
    LOG = Logger()

    rep_id = 0
    last_input_id = n_inputs - 1
    while rep_id < n_reps:
        input_id: int = last_input_id
        reading_finished: bool = False
        while not reading_finished:
            # read the next batch of traces from the kernel module
            output = subprocess.check_output(
                f"taskset -c {CONF.executor_taskset} cat /sys/x86_executor/trace", shell=True)
            lines = output.decode().split("\n")

            # parse the output
            for line in lines:
                # print(rep_id, input_id, line)
                # skip empty lines
                if not line:
                    continue

                # we reached the end of the batch? read the next batch
                if 'done' in line:
                    reading_finished = True
                    break

                # transform the line into a sequence of ints
                line_words = line.split(",")
                line_ints = [int(x) for x in line_words]

                # if the line width is unexpected, it's an error
                if len(line_words) != 6:
                    if enable_warnings:
                        LOG.warning("executor", f"Unexpected line width: {len(line_words)}")
                    rewind_km_output_to_end()
                    raise HardwareTracingError()

                # if the hardware trace is zero, it's an error
                if line_ints[0] == 0:
                    if enable_warnings:
                        LOG.warning("executor", "Kernel module error; see dmesg for details")
                    rewind_km_output_to_end()
                    raise HardwareTracingError()

                # yield the trace
                yield rep_id, input_id, line_ints[0], line_ints[1:]

                # move to next input
                input_id -= 1
                if input_id < 0:
                    # if we reached the end of a repetition, restart the input counter
                    input_id = last_input_id
                    rep_id += 1
        assert input_id == last_input_id, f"input_id: {input_id}, rep_id: {rep_id}"
    return


def rewind_km_output_to_end():
    """
    Read to the end of the kernel module output, until the 'done' line.
    """
    while True:
        output = subprocess.check_output(
            f"taskset -c {CONF.executor_taskset} cat /sys/x86_executor/trace", shell=True)
        if 'done' in output.decode():
            break


# ==================================================================================================
# Main executor class
# ==================================================================================================
class Aarch64Executor(Executor):
    """
    The executor for aarch64 architecture. The executor interfaces with the kernel module to collect
    measurements.

    The high-level workflow is as follows:
    1. Load a test case into the kernel module (see _write_test_case).
    2. Load a set of inputs into the kernel module (see __write_inputs).
    3. Run the measurements by calling the kernel module (see _get_raw_measurements). Each
       measurement is repeated `n_reps` times.
    4. Aggregate the measurements into sets of traces (see _aggregate_measurements).
    """

    previous_num_inputs: int = 0
    curr_test_case: TestCase
    ignore_list: Set[int]

    def __init__(self, enable_mismatch_check_mode: bool = False):
        super().__init__(enable_mismatch_check_mode)
        self.LOG = Logger()
        self.target_desc = Aarch64TargetDesc()
        self.ignore_list = set()

        # Check the execution environment:
        if self._is_smt_enabled() and not enable_mismatch_check_mode:
            self.LOG.warning("executor", "SMT is on! You may experience false positives.")
        if not can_set_reserved():
            self.LOG.error("executor: Cannot set reserved bits on this CPU")

    def _is_smt_enabled(self) -> bool:
        """
        Check if SMT is enabled on the current CPU.

        :return: True if SMT is enabled, False otherwise
        """
        pass

    def set_vendor_specific_features(self):
        pass  # override in vendor-specific executors

    # ==============================================================================================
    # Interface: Quick and Dirty Mode
    def set_quick_and_dirty(self, state: bool):
        """
        Enable or disable the quick and dirty mode in the executor. In this mode, the executor
        will skip some of the stabilization phases, which will make the measurements faster but
        less reliable.

        :param state: True to enable the quick and dirty mode, False to disable it
        """
        pass

    # ==============================================================================================
    # Interface: Ignore List
    def set_ignore_list(self, ignore_list: List[int]):
        """
        Sets a list of inputs IDs that should be ignored by the executor.
        The executor will executed the inputs with these IDs as normal (in case they are
        necessary for priming the uarch state), but their htraces will be set to zero

        :param ignore_list: a list of input IDs to ignore
        """
        self.ignore_list = set(ignore_list)

    def extend_ignore_list(self, ignore_list: List[int]):
        """
        Add a list of new inputs IDs to the current ignore list.

        :param ignore_list: a list of input IDs to add to the ignore list
        """
        self.ignore_list.update(ignore_list)

    # ==============================================================================================
    # Interface: Base Addresses
    def read_base_addresses(self):
        """
        Read the base addresses of the code and the sandbox from the kernel module.
        This function is used to synchronize the memory layout between the executor and the model
        :return: a tuple (sandbox_base, code_base)
        """
        raise NotImplemented()

    # ==============================================================================================
    # Interface: Test Case Loading
    def load_test_case(self, test_case: TestCase):
        """
        Load a test case into the executor.
        This function must be called before calling `trace_test_case`.

        This function also sets the mismatch check mode in the kernel module if requested.
        The flag has to be set before loading the test case because the kernel module links
        the test case code with different measurement functions based on this flag.

        :param test_case: the test case object to load
        """
        # write the test case to the kernel module
        written_tc = self._write_test_case(test_case)
        self.curr_test_case = test_case

        # reset the ignore list; as we are testing a new program now, the old ignore list is not
        # relevant anymore
        self.ignore_list = set()
        return written_tc

    def _write_test_case(self, test_case: TestCase):
        raise NotImplemented()

    # ==============================================================================================
    # Interface: Test Case Tracing
    def trace_test_case(self, inputs: List[Input], n_reps: int) -> List[HTrace]:
        """
        Call the executor kernel module to collect the hardware traces for
        the test case (previously loaded with `load_test_case`) and the given inputs.

        :param inputs: list of inputs to be used for the test case
        :param n_reps: number of times to repeat each measurement
        :return: a list of HTrace objects, one for each input
        :raises HardwareTracingError: if the kernel module output is malformed
        """
        raise NotImplemented()


# ==================================================================================================
# Vendor-specific executors
# ==================================================================================================
class Aarch64RemoteExecutor(Aarch64Executor):

    def __init__(self, connection: Connection, *args):
        self.connection = connection
        super().__init__(*args)
        self.test_case: Optional[TestCase] = None
        self.tmpdir = '/data/local/tmp'
        self.userland_executor = UserlandExecutorImp(connection, f'{self.tmpdir}/executor_userland',
                                                     '/dev/executor', '/sys/executor',
                                                     f'{self.tmpdir}/revizor-executor.ko'
                                                     )
        if self.target_desc.cpu_desc.vendor.lower() != "arm":  # Technically ARM currently does not produce ARM processors, and other vendors do produce ARM processors
            self.LOG.error(
                "Attempting to run ARM executor on a non-ARM CPUs!\n"
                "Change the `executor` configuration option to the appropriate vendor value.")

    def _is_smt_enabled(self):
        result = self.connection.shell('cat /sys/devices/system/cpu/smt/control')
        return 'on' in result.lower().split()

    def set_vendor_specific_features(self):
        pass

    def read_base_addresses(self):
        """
        Read the base addresses of the code and the sandbox from the kernel module.
        This function is used to synchronize the memory layout between the executor and the model
        :return: a tuple (sandbox_base, code_base)
        """
        return self.userland_executor.sandbox_base, self.userland_executor.code_base

    def _write_test_case_with_bitmap_trace(self, test_case: TestCase) -> None:
        patched_test_case = copy.deepcopy(test_case)
        tagging_pass = Aarch64TagMemoryAccesses(memory_accesses_to_guess_tag=None)
        marking_pass = Aarch64MarkMemoryAccesses()
        Aarch64RemoteExecutor._pass_on_test_case(patched_test_case, [tagging_pass, marking_pass])

        Aarch64RemoteExecutor._assemble_local_test_case(patched_test_case, 'generated_retrieve_bitmap')

        self._load_test_case_remotely(patched_test_case)

        os.remove(patched_test_case.bin_path)
        os.remove(patched_test_case.asm_path)
        os.remove(patched_test_case.obj_path)

    @staticmethod
    def _pass_on_test_case(test_case: TestCase, passes: List[Pass]):
        for p in passes:
            p.run_on_test_case(test_case)

    def _load_test_case_remotely(self, test_case: TestCase):
        remote_filename = f'{self.tmpdir}/remote_{test_case.bin_path}'
        self.connection.push(test_case.bin_path, remote_filename)
        self.userland_executor.checkout_region(TestCaseRegion())
        self.userland_executor.write_file(remote_filename)
        self.connection.shell(f'rm {remote_filename}')

    @staticmethod
    def _assemble_local_test_case(test_case: TestCase, base_filename: str):
        printer = Aarch64Printer(Aarch64TargetDesc())
        test_case.bin_path, test_case.asm_path, test_case.obj_path = \
            (f'{base_filename}.{suffix}' for suffix in ('bin', 'asm', 'o'))

        printer.print(test_case, test_case.asm_path)

        ConfigurableGenerator.assemble(test_case.asm_path, test_case.obj_path, test_case.bin_path)

    def _write_test_with_correct_tags(self) -> TestCase:
        patched_test_case = copy.deepcopy(self.test_case)
        tagging_pass = Aarch64TagMemoryAccesses(memory_accesses_to_guess_tag=None)
        Aarch64RemoteExecutor._pass_on_test_case(patched_test_case, [tagging_pass])

        Aarch64RemoteExecutor._assemble_local_test_case(patched_test_case, 'generated_correct_tags')

        self._load_test_case_remotely(patched_test_case)

        return patched_test_case

    def _write_test_with_incorrect_tags(self, iid: int, memory_accesses_to_guess_tag: List[int]) -> TestCase:
        patched_test_case = copy.deepcopy(self.test_case)
        tagging_pass = Aarch64TagMemoryAccesses(
            memory_accesses_to_guess_tag=memory_accesses_to_guess_tag)
        Aarch64RemoteExecutor._pass_on_test_case(patched_test_case, [tagging_pass])

        Aarch64RemoteExecutor._assemble_local_test_case(patched_test_case, f'generated_patched_iid_{iid}')

        self._load_test_case_remotely(patched_test_case)

        return patched_test_case

    def _write_test_case(self, test_case: TestCase) -> None:
        self.test_case = test_case

    def _create_local_files_for_inputs(self, inputs: List[Input]):
        remote_filenames = []
        for idx, inp in enumerate(inputs):
            inpname = f"input{idx}.bin"
            remote_fname = f'{self.tmpdir}/{inpname}'
            inp.save(inpname)
            self.connection.push(inpname, remote_fname)
            os.remove(inpname)
            remote_filenames.append(remote_fname)
        return remote_filenames

    def _write_inputs_to_connection(self, inputs: List[Input], n_reps: int) -> Tuple[
        np.ndarray[int, int], List[str]]:
        array: np.ndarray[int, int] = np.zeros((len(inputs), n_reps), dtype=np.uint64)
        remote_filenames = self._create_local_files_for_inputs(inputs)

        for col in range(n_reps):
            for row, fname in enumerate(remote_filenames):
                array[row, col] = self.userland_executor.allocate_iid()
                self.userland_executor.checkout_region(InputRegion(array[row, col]))
                self.userland_executor.write_file(fname)

        return array, remote_filenames

    def trace_test_case(self, inputs: List[Input], n_reps: int) -> List[Tuple[
        Tuple[TestCase,HTrace], Tuple[TestCase,HTrace]]]:
        """
        Call the executor kernel module to collect the hardware traces for
        the test case (previously loaded with `load_test_case`) and the given inputs.

        :param inputs: list of inputs to be used for the test case
        :param n_reps: number of times to repeat each measurement
        :return: a list of HTrace objects, one for each input
        :raises HardwareTracingError: if the kernel module output is malformed
        """
        def parse_bitmap(n: str, bit: int) -> List[int]:
            result = []
            for index, b in enumerate(n):
                if bit == int(b):
                    result.append(len(n) - (index + 1))
            return result
            #return [i for i in range(n.bit_count()) if ((n >> i) & 1) == bit]

        def _measure_input(iid: int, iids: np.ndarray[int, int]) -> np.ndarray[Any, np.dtype[HWMeasurement]]:

            def checkout_and_measure(cid: int) -> HWMeasurement:
                self.userland_executor.checkout_region(InputRegion(iids[iid][cid]))
                return self.userland_executor.hardware_measurement()

            hwmeasurements = list(map(checkout_and_measure, range(iids.shape[1])))
            return np.array(hwmeasurements, dtype=object)

        # Skip if it's a dummy call
        if not inputs or self.test_case is None:
            return []

        # Store statistics
        n_inputs = len(inputs)
        STAT.executor_reruns += n_reps * n_inputs
        iids, filenames = self._write_inputs_to_connection(inputs, n_reps)


        all_correct_tags_traces: np.ndarray = np.ndarray(shape=(n_inputs, n_reps), dtype=np.uint64)
        all_incorrect_tags_traces: np.ndarray = np.ndarray(shape=(n_inputs, n_reps), dtype=np.uint64)
        all_correct_tags_pfc: np.ndarray = np.ndarray(shape=(n_inputs, n_reps, 3), dtype=np.uint64)
        all_incorrect_tags_pfc: np.ndarray = np.ndarray(shape=(n_inputs, n_reps, 3), dtype=np.uint64)
        all_not_architectural_memory_accesses: np.ndarray = np.ndarray(shape=(n_inputs,), dtype=object)
        all_tc_incorrect_tags: np.ndarray = np.ndarray(shape=(n_inputs,), dtype=TestCase)

        self._write_test_case_with_bitmap_trace(self.test_case)
        self.userland_executor.trace()

        for iid in range(iids.shape[0]):  # Iterate over rows
            self.userland_executor.checkout_region(InputRegion(iids[iid, 0]))
            architectural_memory_accesses = self.userland_executor.hardware_measurement().memory_ids
            all_not_architectural_memory_accesses[iid]: List[int] = \
                parse_bitmap(architectural_memory_accesses, bit=0)

        tc_correct_tags: TestCase = self._write_test_with_correct_tags()
        self.userland_executor.trace()
        for iid in range(iids.shape[0]):
            hwmeasurements = _measure_input(iid, iids)
            all_correct_tags_traces[iid, :] = np.array([m.htrace for m in hwmeasurements])
            all_correct_tags_pfc[iid, :, :] = np.array([m.pfcs for m in hwmeasurements])

        for iid in range(iids.shape[0]):
            all_tc_incorrect_tags[iid] = self._write_test_with_incorrect_tags(
                iid, all_not_architectural_memory_accesses[iid])
            self.userland_executor.trace()

            hwmeasurements = _measure_input(iid, iids)
            all_incorrect_tags_traces[iid, :] = np.array([m.htrace for m in hwmeasurements])
            all_incorrect_tags_pfc[iid, :, :] = np.array([m.pfcs for m in hwmeasurements])

        for filename in filenames:
            self.connection.shell(f'rm {filename}')

        self.LOG.dbg_executor_raw_traces(all_correct_tags_traces, all_correct_tags_pfc)
        self.LOG.dbg_executor_raw_traces(all_incorrect_tags_traces, all_incorrect_tags_pfc)

        # Post-process the results and check for errors
        if not self.mismatch_check_mode:  # no need to post-process in mismatch check mode
            mask = np.uint64(0x0FFFFFFFFFFFFFF0)
            for input_id in range(n_inputs):
                for rep_id in range(n_reps):
                    # Zero-out traces for ignored inputs
                    if input_id in self.ignore_list:
                        all_correct_tags_traces[input_id][rep_id] = 0
                        all_incorrect_tags_traces[input_id][rep_id] = 0
                        continue

                    # When using TSC mode, we need to mask the lower 4 bits of the trace
                    if CONF.executor_mode == 'TSC':
                        all_correct_tags_traces[input_id][rep_id] &= mask
                        all_incorrect_tags_traces[input_id][rep_id] &= mask

        # Aggregate measurements into HTrace objects
        traces = []
        for input_row in range(n_inputs):
            trace_list_correct_tags = list(all_correct_tags_traces[input_row])
            perf_counters_correct_tags = all_correct_tags_pfc[input_row]
            trace_list_incorrect_tags = list(all_incorrect_tags_traces[input_row])
            perf_counters_incorrect_tags = all_incorrect_tags_pfc[input_row]
            traces.append((
                (tc_correct_tags,
                 HTrace(trace_list=trace_list_correct_tags,
                       perf_counters=perf_counters_correct_tags)),
                (all_tc_incorrect_tags[input_row],
                 HTrace(trace_list=trace_list_incorrect_tags,
                       perf_counters=perf_counters_incorrect_tags))
            ))


        return traces
