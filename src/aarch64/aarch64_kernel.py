"""
File: AArch64 kernel-module executor control.
  - ioctl ABI (request structs, ioctl numbers, decode helper)
  - executor memory regions and measurement types
  - LocalHWExecutor: the local userland driver for the revizor-executor kernel module
  - op-timing profiling of the kernel operations
"""
from __future__ import annotations
import os
import re
import uuid
import functools
import ctypes
import fcntl
import time
from typing import List, Literal, Union, Optional, Type, Callable, Tuple
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
from ..config import CONF

op_timings = defaultdict(list)
def profile_by_opcode(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        cmd = args[1] if len(args) > 1 else None
        start = time.time()
        try:
            return func(*args, **kwargs)
        finally:
            duration = time.time() - start
            op_timings[cmd].append(duration)

    return wrapper

@contextmanager
def profile_op(name: str):
    start = time.time()
    try:
        yield
    finally:
        op_timings[name].append(time.time() - start)


def print_opcode_summary():
    if not op_timings:
        print("No opcode data collected.")
        return

    # Compute total runtime for all opcodes
    total_runtime = sum(sum(times) for times in op_timings.values())

    # Header
    print("\n=== Opcode Profiling Summary ===")
    print(f"{'Opcode':<30} | {'Count':>6} | {'Avg (ms)':>10} | {'Total (ms)':>11} | {'% of total':>10}")
    print("-" * 80)

    # Sort opcodes by total runtime descending
    sorted_ops = sorted(op_timings.items(), key=lambda x: sum(x[1]), reverse=True)

    for op, times in sorted_ops:
        count = len(times)
        total = sum(times)
        avg = total / count
        percent = (total / total_runtime) * 100 if total_runtime > 0 else 0
        print(f"{op:<30} | {count:>6} | {avg*1000:>10.3f} | {total*1000:>11.3f} | {percent:>9.2f}%")

    print("-" * 80)
    print(f"{'Total':<30} | {'':>6} | {'':>10} | {total_runtime*1000:>11.3f} | {100:>9.2f}%\n")


def read_device(fd, chunk_size=4*1024) -> bytes:
    buffer = bytearray()
    while True:
        chunk = os.read(fd, chunk_size)
        if not chunk:
            break
        buffer.extend(chunk)
    return bytes(buffer)


class ExecutorRegion(ABC):
    pass

class ExecutorMemory(bytes):
    def __new__(cls, data):
        if isinstance(data, str):
            data = data.encode()
        return super().__new__(cls, data)

    def to_hex(self, sep=" ") -> str:
        return sep.join(f'{b:02x}' for b in self)

    def read_int(self, offset, size=4, byteorder: Literal['little', 'big']="little") -> int:
        return int.from_bytes(self[offset:offset+size], byteorder=byteorder, signed=False)


class HWMeasurement:
    def __init__(self, htrace: int, pfcs: List[int]):
        self.htrace = htrace
        self.pfcs = pfcs


class HWExecutor(ABC):
    @abstractmethod
    def trace(self):
        raise NotImplementedError()

    @abstractmethod
    def checkout_region(self, region: ExecutorRegion):
        raise NotImplementedError()

    @abstractmethod
    def hardware_measurement(self) -> HWMeasurement:
        raise NotImplementedError()

    @property
    def contents(self) -> ExecutorMemory:
        raise NotImplementedError()

    @contents.deleter
    def contents(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def number_of_inputs(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def discard_all_inputs(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def allocate_iid(self) -> int:
        raise NotImplementedError()


class TestCaseRegion(ExecutorRegion):
    def __init__(self):
        super().__init__()

class InputRegion(ExecutorRegion):
    def __init__(self, iid: int):
        super().__init__()
        self.iid = iid

class UserMeasurement(ctypes.Structure):
    _fields_ = [
        ("htrace", ctypes.c_uint64 * 1),
        ("pfc", ctypes.c_uint64 * 3),
    ]

REVISOR_IOC_MAGIC = ord('r')

REVISOR_CHECKOUT_TEST_CONSTANT      = 1
REVISOR_UNLOAD_TEST_CONSTANT        = 2
REVISOR_GET_NUMBER_OF_INPUTS_CONSTANT = 3
REVISOR_CHECKOUT_INPUT_CONSTANT     = 4
REVISOR_ALLOCATE_INPUT_CONSTANT     = 5
REVISOR_FREE_INPUT_CONSTANT         = 6
REVISOR_MEASUREMENT_CONSTANT        = 7
REVISOR_TRACE_CONSTANT              = 8
REVISOR_CLEAR_ALL_INPUTS_CONSTANT   = 9
REVISOR_GET_TEST_LENGTH_CONSTANT    = 10
REVISOR_SWAP_PAC_KEYS_CONSTANT      = 12
REVISOR_GET_EXEC_PAC_KEYS_CONSTANT  = 13
REVISOR_SET_PAC_KEYS_CONSTANT       = 14
REVISOR_GET_PAC_KEYS_CONSTANT       = 15
REVISOR_MTE_TAG_REGION_CONSTANT     = 16
REVISOR_PAC_SIGN_CONSTANT           = 17
REVISOR_PAC_AUTH_CONSTANT           = 18

class MteTagRegionReq(ctypes.Structure):
    _fields_ = [
        ("sandbox_offset", ctypes.c_uint64),
        ("length",         ctypes.c_uint64),
        ("tag",            ctypes.c_uint8),
    ]

class PacSignReq(ctypes.Structure):
    _fields_ = [
        ("ptr",      ctypes.c_uint64),
        ("ctx",      ctypes.c_uint64),
        ("mnemonic", ctypes.c_char * 16),
        ("result",   ctypes.c_uint64),
    ]

class PacKeys(ctypes.Structure):
    _fields_ = [
        ("apia_lo", ctypes.c_uint64),
        ("apia_hi", ctypes.c_uint64),
        ("apib_lo", ctypes.c_uint64),
        ("apib_hi", ctypes.c_uint64),
        ("apda_lo", ctypes.c_uint64),
        ("apda_hi", ctypes.c_uint64),
        ("apdb_lo", ctypes.c_uint64),
        ("apdb_hi", ctypes.c_uint64),
        ("apga_lo", ctypes.c_uint64),
        ("apga_hi", ctypes.c_uint64),
    ]

IOCTL_NR_TO_NAME = {
    1: "REVISOR_CHECKOUT_TEST",
    2: "REVISOR_UNLOAD_TEST",
    3: "REVISOR_GET_NUMBER_OF_INPUTS",
    4: "REVISOR_CHECKOUT_INPUT",
    5: "REVISOR_ALLOCATE_INPUT",
    6: "REVISOR_FREE_INPUT",
    7: "REVISOR_MEASUREMENT",
    8: "REVISOR_TRACE",
    9: "REVISOR_CLEAR_ALL_INPUTS",
    10: "REVISOR_GET_TEST_LENGTH",
    12: "REVISOR_SWAP_PAC_KEYS",
    13: "REVISOR_GET_EXEC_PAC_KEYS",
    14: "REVISOR_SET_PAC_KEYS",
    15: "REVISOR_GET_PAC_KEYS",
    16: "REVISOR_MTE_TAG_REGION",
    17: "REVISOR_PAC_SIGN",
    18: "REVISOR_PAC_AUTH",
}


# From asm-generic/ioctl.h
_IOC_NRBITS = 8
_IOC_TYPEBITS = 8
_IOC_SIZEBITS = 14
_IOC_DIRBITS = 2

_IOC_NRSHIFT = 0
_IOC_TYPESHIFT = _IOC_NRSHIFT + _IOC_NRBITS
_IOC_SIZESHIFT = _IOC_TYPESHIFT + _IOC_TYPEBITS
_IOC_DIRSHIFT = _IOC_SIZESHIFT + _IOC_SIZEBITS

_IOC_NONE = 0
_IOC_WRITE = 1
_IOC_READ = 2

def _IOC(dir_, type_, nr, size):
    return ((dir_ << _IOC_DIRSHIFT) |
      (type_ << _IOC_TYPESHIFT) |
      (nr << _IOC_NRSHIFT) |
      (size << _IOC_SIZESHIFT))

def _IO(type_, nr):
    return _IOC(_IOC_NONE, type_, nr, 0)

def _IOR(type_, nr, datatype):
    return _IOC(_IOC_READ, type_, nr, ctypes.sizeof(datatype))

def _IOW(type_, nr, datatype):
    return _IOC(_IOC_WRITE, type_, nr, ctypes.sizeof(datatype))

def _IOWR(type_, nr, datatype):
    return _IOC(_IOC_READ | _IOC_WRITE, type_, nr, ctypes.sizeof(datatype))

def decode_ioctl(cmd):
    nr = (cmd >> _IOC_NRSHIFT) & ((1 << _IOC_NRBITS) - 1)
    type_ = (cmd >> _IOC_TYPESHIFT) & ((1 << _IOC_TYPEBITS) - 1)
    size = (cmd >> _IOC_SIZESHIFT) & ((1 << _IOC_SIZEBITS) - 1)
    dir_ = (cmd >> _IOC_DIRSHIFT) & ((1 << _IOC_DIRBITS) - 1)
    name = IOCTL_NR_TO_NAME.get(nr, f"UNKNOWN_IOCTL_{nr}")
    return {'dir': dir_, 'type': type_, 'nr': nr, 'size': size, 'name': name}


# Generate actual ioctl numbers
REVISOR_CHECKOUT_TEST = _IO(REVISOR_IOC_MAGIC, REVISOR_CHECKOUT_TEST_CONSTANT)
REVISOR_UNLOAD_TEST = _IO(REVISOR_IOC_MAGIC, REVISOR_UNLOAD_TEST_CONSTANT)
REVISOR_GET_NUMBER_OF_INPUTS = _IOR(REVISOR_IOC_MAGIC, REVISOR_GET_NUMBER_OF_INPUTS_CONSTANT, ctypes.c_uint64)
REVISOR_CHECKOUT_INPUT = _IOW(REVISOR_IOC_MAGIC, REVISOR_CHECKOUT_INPUT_CONSTANT, ctypes.c_uint64)
REVISOR_ALLOCATE_INPUT = _IOR(REVISOR_IOC_MAGIC, REVISOR_ALLOCATE_INPUT_CONSTANT, ctypes.c_uint64)
REVISOR_FREE_INPUT = _IOW(REVISOR_IOC_MAGIC, REVISOR_FREE_INPUT_CONSTANT, ctypes.c_uint64)
REVISOR_MEASUREMENT = _IOR(REVISOR_IOC_MAGIC, REVISOR_MEASUREMENT_CONSTANT, UserMeasurement)
REVISOR_TRACE = _IO(REVISOR_IOC_MAGIC, REVISOR_TRACE_CONSTANT)
REVISOR_CLEAR_ALL_INPUTS = _IO(REVISOR_IOC_MAGIC, REVISOR_CLEAR_ALL_INPUTS_CONSTANT)
REVISOR_GET_TEST_LENGTH = _IOR(REVISOR_IOC_MAGIC, REVISOR_GET_TEST_LENGTH_CONSTANT, ctypes.c_uint64)
REVISOR_PAC_SIGN = _IOWR(REVISOR_IOC_MAGIC, REVISOR_PAC_SIGN_CONSTANT,  PacSignReq)
REVISOR_PAC_AUTH = _IOWR(REVISOR_IOC_MAGIC, REVISOR_PAC_AUTH_CONSTANT,  PacSignReq)
REVISOR_SET_PAC_KEYS = _IOW(REVISOR_IOC_MAGIC, REVISOR_SET_PAC_KEYS_CONSTANT, PacKeys)
REVISOR_GET_PAC_KEYS = _IOR(REVISOR_IOC_MAGIC, REVISOR_GET_PAC_KEYS_CONSTANT, PacKeys)
REVISOR_MTE_TAG_REGION = _IOW(REVISOR_IOC_MAGIC, REVISOR_MTE_TAG_REGION_CONSTANT, MteTagRegionReq)


class LocalHWExecutor(HWExecutor):

    def __init__(self, device_path: str, sys_executor_path: str):
        self.executor_device_path: str = device_path
        self.executor_sysfs: str = sys_executor_path
        self.current_region: ExecutorRegion = TestCaseRegion()
        self.fd = os.open(self.executor_device_path, os.O_RDWR)
        self._write_sysfs("measurement_mode", CONF.executor_mode.encode())
        self._write_sysfs("pin_to_core", b"0")
        self._write_sysfs("enable_pre_run_flush", str(CONF.enable_pre_run_flush).encode())
        self._write_sysfs("warmups", str(CONF.executor_warmups).encode())
        self.discard_all_inputs()
        self.checkout_region(TestCaseRegion())
        del self.contents

    def _write_sysfs(self, filename: str, value: bytes):
        path = os.path.join(self.executor_sysfs, filename)
        with open(path, "wb") as f:
            f.write(value)

    def _read_sysfs(self, filename: str) -> str:
        path = os.path.join(self.executor_sysfs, filename)
        with open(path, "r") as f:
            return f.read().strip()

    def _ioctl(self, cmd: int, arg=None):
        with profile_op(f'ioctl {decode_ioctl(cmd)["name"]}'):
            if arg is None:
                arg = 0
            if isinstance(arg, int):
                buf = ctypes.c_uint64(arg)
                ret = fcntl.ioctl(self.fd, cmd, buf)
                return buf.value
            else:
                return fcntl.ioctl(self.fd, cmd, arg)

    def trace(self):
        self._ioctl(REVISOR_TRACE)

    def discard_all_inputs(self):
        self._ioctl(REVISOR_CLEAR_ALL_INPUTS)

    def checkout_region(self, region: ExecutorRegion):
        self.current_region = region
        if isinstance(region, TestCaseRegion):
            self._ioctl(REVISOR_CHECKOUT_TEST)
        elif isinstance(region, InputRegion):
            self._ioctl(REVISOR_CHECKOUT_INPUT, region.iid)
        else:
            raise ValueError(f'Unsupported region type: {type(region)}')

    def hardware_measurement(self) -> HWMeasurement:
        measurement = UserMeasurement()
        self._ioctl(REVISOR_MEASUREMENT, measurement)

        htrace = measurement.htrace[0]
        pfcs = list(measurement.pfc)

        return HWMeasurement(htrace=htrace, pfcs=pfcs)

    @property
    def contents(self) -> ExecutorMemory:
        with profile_op("read"):
            raw = read_device(self.fd)
            return ExecutorMemory(raw)

    def write(self, data: bytes | ExecutorMemory) -> None:
        if isinstance(data, ExecutorMemory):
            data = bytes(data)

        with profile_op('write'):
            os.write(self.fd, data)

    @contents.deleter
    def contents(self) -> None:
        if isinstance(self.current_region, TestCaseRegion):
            self._ioctl(REVISOR_UNLOAD_TEST)
        elif isinstance(self.current_region, InputRegion):
            self._ioctl(REVISOR_FREE_INPUT)
        else:
            raise ValueError(f'Unsupported region type: {type(self.current_region)}')

    def number_of_inputs(self) -> int:
        return self._ioctl(REVISOR_GET_NUMBER_OF_INPUTS)

    @property
    def test_length(self) -> int:
        return self._ioctl(REVISOR_GET_TEST_LENGTH)

    def allocate_iid(self) -> int:
        return self._ioctl(REVISOR_ALLOCATE_INPUT)

    def _pac_op(self, cmd: int, ptr: int, ctx: int, mnemonic: str) -> int:
        req = PacSignReq()
        req.ptr = ptr & 0xFFFFFFFFFFFFFFFF
        req.ctx = ctx & 0xFFFFFFFFFFFFFFFF
        req.mnemonic = mnemonic.encode()[:15]
        req.result = 0
        self._ioctl(cmd, req)
        return req.result

    def pac_sign(self, ptr: int, ctx: int, mnemonic: str) -> int:
        return self._pac_op(REVISOR_PAC_SIGN, ptr, ctx, mnemonic)

    def pac_auth(self, ptr: int, ctx: int, mnemonic: str) -> int:
        return self._pac_op(REVISOR_PAC_AUTH, ptr, ctx, mnemonic)

    def get_pac_keys(self) -> PacKeys:
        keys = PacKeys()
        self._ioctl(REVISOR_GET_PAC_KEYS, keys)
        return keys

    def set_pac_keys(self, keys: PacKeys) -> None:
        self._ioctl(REVISOR_SET_PAC_KEYS, keys)

    def mte_tag_sandbox_region(self, sandbox_offset: int, length: int, tag: int) -> None:
        req = MteTagRegionReq()
        req.sandbox_offset = sandbox_offset
        req.length = length
        req.tag = tag & 0xF
        self._ioctl(REVISOR_MTE_TAG_REGION, req)

    @property
    def sandbox_base(self) -> int:
        return int(self._read_sysfs('print_sandbox_base'), 16)

    @property
    def code_base(self) -> int:
        return int(self._read_sysfs('print_code_base'), 16)

    def write_branch_training_config(self, entries: list) -> None:
        """entries: list of (byte_offset: int, train_taken: bool)"""
        payload = ",".join(f"{off}:{1 if taken else 0}" for off, taken in entries)
        self._write_sysfs("branch_training_config", payload.encode())

    def clear_branch_training(self) -> None:
        self._write_sysfs("enable_branch_training", b"0")


def retry(max_times: int = 1,
          retry_on: Union[Type[BaseException], Tuple[Type[BaseException], ...]] = Exception,
          backoff: float = 0.0) -> Callable:

    if not isinstance(retry_on, tuple):
        retry_on = (retry_on,)

    if not all(isinstance(exc, type) and issubclass(exc, BaseException) for exc in retry_on):
        raise TypeError("All entries in 'retry_on' must be of exception types!")

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception: Optional[BaseException] = None

            for attempt in range(1, max_times + 1):
                try:
                    return func(*args, **kwargs)

                except retry_on as e:
                    last_exception = e
                    if attempt < max_times and backoff > 0.0:
                        sleep_time = backoff * (2 ** (attempt - 1))
                        time.sleep(sleep_time)

            raise last_exception
        return wrapper
    return decorator


class RemoteHWExecutor(HWExecutor):
    _RETRIES = 100

    def __init__(self, connection: Connection, userland_application_path: str,
              device_path: str, sys_executor_path: str, module_path: str
            ):
        self.connection: Connection = connection
        self.userland_application_path: str = userland_application_path
        self.executor_device_path: str = device_path

        self.executor_sysfs: str = sys_executor_path
        self.current_region: ExecutorRegion = TestCaseRegion()

        if not self.connection.is_file_present(self.executor_device_path):
            if not self.connection.is_file_present(module_path):
                self.connection.push('revizor-executor.ko', module_path)
            self.connection.shell(f'insmod {module_path}', privileged=True)

        self.connection.shell(f'echo "{CONF.executor_mode}" > /sys/executor/measurement_mode', privileged=True)
        self.connection.shell(f'echo "0" > /sys/executor/pin_to_core', privileged=True)

        if not self.connection.is_file_present(self.userland_application_path):
            self.connection.push('executor_userland', userland_application_path)

        # Discard any previous contents in the executor memory
        self.discard_all_inputs()
        self.checkout_region(TestCaseRegion())
        del self.contents

    def trace(self) -> None:
        self._query_executor(8)

    def checkout_region(self, region: ExecutorRegion):
        self.current_region = region
        if isinstance(region, TestCaseRegion):
            self._query_executor(1)
        elif isinstance(region, InputRegion):
            self._query_executor(4, region.iid)
        else:
            raise ValueError(f'Unsupported region type: {type(region)}')

    @retry(max_times=_RETRIES, retry_on=RuntimeError, backoff=0.5)
    def hardware_measurement(self) -> HWMeasurement:
        result = self._query_executor(7)
        htrace_match = re.search(r"htrace .: ([01]+)", result)
        pfc_matches = re.findall(r"pfc (\d+): (\d+)", result)

        if htrace_match is None or not pfc_matches:
            raise RuntimeError('Could not parse measurements from executor output')

        htrace = int(htrace_match.group(1), 2)
        pfc_list = [int(pfc_value) for _, pfc_value in
              sorted(pfc_matches, key=lambda x: int(x[0]))]

        return HWMeasurement(htrace=htrace, pfcs=pfc_list)


    @property
    def contents(self) -> ExecutorMemory:
        filename = f'{uuid.uuid4().hex}.bin'
        remote_filename = f'remote_{filename}'

        with profile_op('read'):
            self.connection.shell(
                    f'{self.userland_application_path} {self.executor_device_path} r {remote_filename}',
                    privileged=True)

        self.connection.pull(remote_filename, filename)
        self.connection.shell(f'rm {remote_filename}')

        with open(filename) as f:
            return ExecutorMemory(f.read())

    def write_file(self, filename: str) -> None:
        with profile_op('write'):
            self.connection.shell(f'{self.userland_application_path} {self.executor_device_path} w {filename}', privileged=True)


    @profile_by_opcode
    def _query_executor(self, qid: int, *args) -> str:
        return self.connection.shell(f'{self.userland_application_path} {self.executor_device_path} {qid} {" ".join(str(arg) for arg in args)}', privileged=True)

    @contents.deleter
    def contents(self) -> None:
        if isinstance(self.current_region, TestCaseRegion):
            self._query_executor(2)
        elif isinstance(self.current_region, InputRegion):
            self._query_executor(6)
        else:
            raise ValueError(f'Unsupported region type: {type(self.current_region)}')

    @retry(max_times=_RETRIES, retry_on=RuntimeError, backoff=0.5)
    def number_of_inputs(self) -> int:
        result = self._query_executor(3)
        number_of_inputs_match = re.search(r"Number Of Inputs: (\d+)", result)
        if number_of_inputs_match is None:
            raise RuntimeError("Could not find number of inputs")
        return int(number_of_inputs_match.group(1))

    def discard_all_inputs(self) -> None:
        self._query_executor(9)

    @property
    @retry(max_times=_RETRIES, retry_on=RuntimeError, backoff=0.5)
    def test_length(self) -> int:
        result = self._query_executor(10)
        test_length_match = re.search(r"Test Length: (\d+)", result)
        if test_length_match is None:
            raise RuntimeError("Could not find test length")
        return int(test_length_match.group(1))

    @retry(max_times=_RETRIES, retry_on=RuntimeError, backoff=0.5)
    def allocate_iid(self) -> int:
        result = self._query_executor(5)
        iid_matching = re.search(r"Allocated Input ID: (\d+)", result)
        if iid_matching is None:
            raise RuntimeError("Could not find allocated input ID")
        return int(iid_matching.group(1))

    @property
    def sandbox_base(self) -> int:
        base = self.connection.shell(f'cat {self.executor_sysfs}/print_sandbox_base', privileged=True)
        return int(base, 16)

    @property
    def code_base(self) -> int:
        base = self.connection.shell(f'cat {self.executor_sysfs}/print_code_base', privileged=True)
        return int(base, 16)
