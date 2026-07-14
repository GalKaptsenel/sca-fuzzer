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
import time
from typing import (
    List, Literal, Union, Optional, Type, Callable, Tuple, Dict, Any, NoReturn, TYPE_CHECKING)
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import defaultdict

from .aarch64_batch import TraceUnit, HWMeasurement, encode_request, decode_response
from contextlib import contextmanager
from functools import wraps
from ..config import CONF
from ..util import STAT   # TEMP(perf-metrics): remove with the traces/s instrumentation

if TYPE_CHECKING:
    from .aarch64_connection import Connection

# TEMP(op-profiling): ad-hoc per-ioctl timing profiler; remove later (grep this tag).
# Off by default — set REVIZOR_PROFILE_OPS=1 to collect. It records a running
# [count, total_seconds] aggregate per op name (not a per-call sample list), so even when enabled a
# long run (millions of ioctls) can't grow it without bound; the summary only needs count and total.
_PROFILE_OPS = os.environ.get("REVIZOR_PROFILE_OPS") == "1"
op_timings: Dict[Any, list] = defaultdict(lambda: [0, 0.0])


def _record_op(name, duration: float) -> None:
    if not _PROFILE_OPS:
        return
    agg = op_timings[name]
    agg[0] += 1
    agg[1] += duration


def profile_by_opcode(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        cmd = args[1] if len(args) > 1 else None
        start = time.time()
        try:
            return func(*args, **kwargs)
        finally:
            _record_op(cmd, time.time() - start)

    return wrapper

@contextmanager
def profile_op(name: str):
    start = time.time()
    try:
        yield
    finally:
        _record_op(name, time.time() - start)


def print_opcode_summary():   # TEMP(op-profiling): remove later
    if not op_timings:
        print("No opcode data collected." if _PROFILE_OPS
              else "Op profiling disabled (set REVIZOR_PROFILE_OPS=1 to collect).")
        return

    # Compute total runtime for all opcodes (op_timings[op] == [count, total_seconds])
    total_runtime = sum(total for _, total in op_timings.values())

    # Header
    print("\n=== Opcode Profiling Summary ===")
    print(f"{'Opcode':<30} | {'Count':>6} | {'Avg (ms)':>10} | {'Total (ms)':>11} | {'% of total':>10}")
    print("-" * 80)

    # Sort opcodes by total runtime descending
    sorted_ops = sorted(op_timings.items(), key=lambda x: x[1][1], reverse=True)

    for op, (count, total) in sorted_ops:
        avg = total / count if count else 0
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


@dataclass(frozen=True)
class TargetInfo:
    """Static target addresses the generation side needs (e.g. the sandbox base to seal against)."""
    sandbox_base: int
    code_base: int


def _parse_midr_el1(cpu_info: str) -> int:
    """Pull MIDR_EL1 out of the module's `system/cpu_info` sysfs text."""
    for line in cpu_info.splitlines():
        if line.strip().startswith("MIDR_EL1"):
            return int(line.split(":", 1)[1].strip(), 16)
    raise ValueError(f"no MIDR_EL1 in cpu_info:\n{cpu_info}")


class HWExecutor(ABC):
    """The device backend the fuzzer measures through: report the target's addresses and measure a
    batch of (test case, inputs) units. Local (/dev/executor) or remote (over a Connection)."""

    @abstractmethod
    def target_info(self) -> TargetInfo:
        raise NotImplementedError()

    @abstractmethod
    def run_batch(self, units: List[TraceUnit], n_reps: int) -> List[List[List[HWMeasurement]]]:
        """Measure each unit's inputs `n_reps` times; returns unit -> input -> repetition."""
        raise NotImplementedError()

    @abstractmethod
    def cpu_midr(self) -> int:
        """MIDR_EL1 of the measuring core — its implementation identity (hence its PAC/QARMA)."""
        raise NotImplementedError()


class TestCaseRegion(ExecutorRegion):
    def __init__(self):
        super().__init__()

class InputRegion(ExecutorRegion):
    def __init__(self, iid: int):
        super().__init__()
        self.iid = iid

# Mirror of userapi/executor_user_api.h — keep in sync with that header (the single source).
HTRACE_WIDTH = 1
NUM_PFC = 3


class UserMeasurement(ctypes.Structure):
    _fields_ = [
        ("htrace", ctypes.c_uint64 * HTRACE_WIDTH),
        ("pfc", ctypes.c_uint64 * NUM_PFC),
    ]

# Mirror of src/aarch64/executor/userapi/executor_ioctl_nr.h — keep the numbers
# in sync with that header (the canonical, consecutive source of truth).
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
REVISOR_PAC_SIGN_CONSTANT           = 11
REVISOR_PAC_AUTH_CONSTANT           = 12
REVISOR_PAC_XPAC_CONSTANT           = 13

class PacKeys(ctypes.Structure):
    """The five PAC keys (instruction A/B, data A/B, generic), each {lo,hi} — 80 bytes / 10 words."""
    _fields_ = [
        ("apia_lo", ctypes.c_uint64), ("apia_hi", ctypes.c_uint64),
        ("apib_lo", ctypes.c_uint64), ("apib_hi", ctypes.c_uint64),
        ("apda_lo", ctypes.c_uint64), ("apda_hi", ctypes.c_uint64),
        ("apdb_lo", ctypes.c_uint64), ("apdb_hi", ctypes.c_uint64),
        ("apga_lo", ctypes.c_uint64), ("apga_hi", ctypes.c_uint64),
    ]

    def words(self) -> List[int]:
        return [getattr(self, name) for name, _ in self._fields_]


assert ctypes.sizeof(PacKeys) == 80


class PacSignReq(ctypes.Structure):
    """Sign/auth carry their keys with the request; the kernel keeps no key state of its own."""
    _fields_ = [
        ("ptr",          ctypes.c_uint64),
        ("ctx",          ctypes.c_uint64),
        ("mnemonic",     ctypes.c_char * 16),
        ("result",       ctypes.c_uint64),
        ("keys_present", ctypes.c_uint64),
        ("keys",         PacKeys),
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
    11: "REVISOR_PAC_SIGN",
    12: "REVISOR_PAC_AUTH",
    13: "REVISOR_PAC_XPAC",
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
REVISOR_PAC_XPAC = _IOWR(REVISOR_IOC_MAGIC, REVISOR_PAC_XPAC_CONSTANT,  PacSignReq)


def _raise_access_error(path: str, fix_target: str, err: OSError) -> "NoReturn":
    """Re-raise an EACCES on an executor path with the exact chmod that fixes it.

    The kernel module creates /dev/executor and /sys/executor root-only; an
    unprivileged fuzzer run needs them opened up (or must run as root). The grant
    resets whenever the module is reloaded, so it is not a one-time setup step.
    """
    recursive = "-R " if os.path.isdir(fix_target) else ""
    raise PermissionError(
        f"no access to {path}: {err.strerror}. "
        f"Fix with:  sudo chmod {recursive}a+rw {fix_target}  "
        f"(resets on module reload; or run the fuzzer as root)") from err


class LocalHWExecutor(HWExecutor):

    def __init__(self, device_path: str, sys_executor_path: str):
        import fcntl  # Linux-only; imported lazily so the module stays importable off-Linux
        self._fcntl = fcntl
        self.executor_device_path: str = device_path
        self.executor_sysfs: str = sys_executor_path
        self.current_region: ExecutorRegion = TestCaseRegion()
        try:
            self.fd = os.open(self.executor_device_path, os.O_RDWR)
        except PermissionError as e:
            _raise_access_error(self.executor_device_path, self.executor_device_path, e)
        self._write_sysfs("measurement_mode", CONF.executor_mode.encode())
        self._write_sysfs("pin_to_core", b"0")
        # sysfs parses an int (sscanf %u); emit 1/0 so a bool writes correctly.
        self._write_sysfs("enable_pre_run_flush", str(int(CONF.enable_pre_run_flush)).encode())
        self._write_sysfs("enable_ssbs", str(int(CONF.enable_speculative_store_bypass)).encode())
        self._write_sysfs("warmups", str(CONF.executor_warmups).encode())
        self.discard_all_inputs()
        self.checkout_region(TestCaseRegion())
        del self.contents

    def _write_sysfs(self, filename: str, value: bytes):
        path = os.path.join(self.executor_sysfs, filename)
        try:
            with open(path, "wb") as f:
                f.write(value)
        except PermissionError as e:
            _raise_access_error(path, self.executor_sysfs, e)

    def _read_sysfs(self, filename: str) -> str:
        path = os.path.join(self.executor_sysfs, filename)
        try:
            with open(path, "r") as f:
                return f.read().strip()
        except PermissionError as e:
            _raise_access_error(path, self.executor_sysfs, e)

    def _ioctl(self, cmd: int, arg=None):
        with profile_op(f'ioctl {decode_ioctl(cmd)["name"]}'):
            if arg is None:
                arg = 0
            if isinstance(arg, int):
                buf = ctypes.c_uint64(arg)
                ret = self._fcntl.ioctl(self.fd, cmd, buf)
                return buf.value
            else:
                return self._fcntl.ioctl(self.fd, cmd, arg)

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
        STAT.num_traces += 1   # TEMP(perf-metrics): remove with the traces/s instrumentation
        measurement = UserMeasurement()
        self._ioctl(REVISOR_MEASUREMENT, measurement)

        htrace = measurement.htrace[0]
        return HWMeasurement(htrace=htrace, pfcs=tuple(measurement.pfc))

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
            self._ioctl(REVISOR_FREE_INPUT, self.current_region.iid)
        else:
            raise ValueError(f'Unsupported region type: {type(self.current_region)}')

    def number_of_inputs(self) -> int:
        return self._ioctl(REVISOR_GET_NUMBER_OF_INPUTS)

    @property
    def test_length(self) -> int:
        return self._ioctl(REVISOR_GET_TEST_LENGTH)

    def allocate_iid(self) -> int:
        return self._ioctl(REVISOR_ALLOCATE_INPUT)

    def _pac_op(self, cmd: int, ptr: int, ctx: int, mnemonic: str, keys: Optional[PacKeys]) -> int:
        req = PacSignReq()
        req.ptr = ptr & 0xFFFFFFFFFFFFFFFF
        req.ctx = ctx & 0xFFFFFFFFFFFFFFFF
        req.mnemonic = mnemonic.encode()[:15]
        req.result = 0
        if keys is not None:
            req.keys = keys
            req.keys_present = 1
        self._ioctl(cmd, req)
        return req.result

    def pac_sign(self, ptr: int, ctx: int, mnemonic: str, keys: PacKeys) -> int:
        return self._pac_op(REVISOR_PAC_SIGN, ptr, ctx, mnemonic, keys)

    def pac_auth(self, ptr: int, ctx: int, mnemonic: str, keys: PacKeys) -> int:
        return self._pac_op(REVISOR_PAC_AUTH, ptr, ctx, mnemonic, keys)

    def pac_xpac(self, ptr: int, mnemonic: str) -> int:
        # Strips the PAC field (never faults, key-independent); mnemonic is "xpaci" or "xpacd".
        return self._pac_op(REVISOR_PAC_XPAC, ptr, 0, mnemonic, None)

    @property
    def sandbox_base(self) -> int:
        return int(self._read_sysfs('print_sandbox_base'), 16)

    @property
    def code_base(self) -> int:
        return int(self._read_sysfs('print_code_base'), 16)

    def target_info(self) -> TargetInfo:
        return TargetInfo(sandbox_base=self.sandbox_base, code_base=self.code_base)

    def cpu_midr(self) -> int:
        return _parse_midr_el1(self._read_sysfs("system/cpu_info"))

    def run_batch(self, units: List[TraceUnit], n_reps: int) -> List[List[List[HWMeasurement]]]:
        results = []
        for unit in units:
            self.checkout_region(TestCaseRegion())
            self.write(unit.test_case)
            self.discard_all_inputs()
            iids = []
            for inp in unit.inputs:
                iid = self.allocate_iid()
                self.checkout_region(InputRegion(iid))
                self.write(inp)
                iids.append(iid)
            per_input: List[List[HWMeasurement]] = [[] for _ in unit.inputs]
            for _ in range(n_reps):
                self.trace()
                for idx, iid in enumerate(iids):
                    self.checkout_region(InputRegion(iid))
                    per_input[idx].append(self.hardware_measurement())
            results.append(per_input)
        return results


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


# Local artifacts shipped to the device machine so it runs exactly this codebase.
_LOCAL_USERLAND = os.path.join(os.path.dirname(__file__), "..", "executor_userland", "executor_userland")
_LOCAL_MODULE = os.path.join(os.path.dirname(__file__), "executor", "revizor-executor.ko")

def _abi_version_from_header() -> int:
    header = os.path.join(os.path.dirname(__file__), "executor", "userapi", "executor_user_api.h")
    with open(header) as f:
        match = re.search(r"#define\s+REVISOR_EXECUTOR_ABI_VERSION\s+\(?(\d+)\)?", f.read())
    return int(match.group(1))


EXECUTOR_ABI_VERSION = _abi_version_from_header()


@dataclass(frozen=True)
class RemoteExecutorConfig:
    """The executor's paths on the device machine (the transport is a separate `Connection`)."""
    device: str
    sysfs: str
    module: str
    userland: str


class RemoteHWExecutor(HWExecutor):
    """A device backend reached over a `Connection` (SSH/ADB, or local subprocess). Each measurement
    window crosses as one streamed super-batch (stdin -> `executor_userland batch` -> stdout)."""
    _RETRIES = 5

    def __init__(self, connection: Connection, config: RemoteExecutorConfig):
        self._conn = connection
        self._cfg = config
        self._info: Optional[TargetInfo] = None   # queried once, cached for the campaign
        self._midr: Optional[int] = None
        self._ensure_ready()

    def _ensure_ready(self) -> None:
        conn, cfg = self._conn, self._cfg
        # Ship our binaries so the device runs this exact codebase, then load the module if absent.
        conn.push(_LOCAL_USERLAND, cfg.userland)
        if not conn.is_file_present(cfg.device):
            if not conn.is_file_present(cfg.module):
                conn.push(_LOCAL_MODULE, cfg.module)
            conn.shell(f"insmod {cfg.module}", privileged=True)
        self._check_abi_version()
        for name, value in (
            ("measurement_mode", CONF.executor_mode),
            ("pin_to_core", 0),
            ("enable_pre_run_flush", int(CONF.enable_pre_run_flush)),
            ("enable_ssbs", int(CONF.enable_speculative_store_bypass)),
            ("warmups", CONF.executor_warmups),
        ):
            conn.shell(f'echo "{value}" > {cfg.sysfs}/{name}', privileged=True)

    def _check_abi_version(self) -> None:
        reported = self._conn.shell(f"cat {self._cfg.sysfs}/system/abi_version", privileged=True)
        if int(reported) != EXECUTOR_ABI_VERSION:
            raise RuntimeError(f"device executor ABI version {reported.strip()} != "
                               f"expected {EXECUTOR_ABI_VERSION}; reload the module on the device")

    def cpu_midr(self) -> int:
        if self._midr is None:
            info = self._conn.shell(f"cat {self._cfg.sysfs}/system/cpu_info", privileged=True)
            self._midr = _parse_midr_el1(info)
        return self._midr

    def target_info(self) -> TargetInfo:
        if self._info is None:
            sandbox = self._conn.shell(f"cat {self._cfg.sysfs}/print_sandbox_base", privileged=True)
            code = self._conn.shell(f"cat {self._cfg.sysfs}/print_code_base", privileged=True)
            self._info = TargetInfo(sandbox_base=int(sandbox, 16), code_base=int(code, 16))
        return self._info

    @retry(max_times=_RETRIES, retry_on=IOError, backoff=0.5)
    def run_batch(self, units: List[TraceUnit], n_reps: int) -> List[List[List[HWMeasurement]]]:
        request = encode_request(units, n_reps)
        cmd = f"{self._cfg.userland} {self._cfg.device} batch - -"
        with profile_op("batch"):
            response = self._conn.run(cmd, request, privileged=True)
        return decode_response(response)


def _make_remote_connection() -> "Connection":
    """Build the transport to the device machine from CONF.executor_remote_*."""
    from .aarch64_connection import LocalConnection, SSHConnection, ADBConnection
    transport = CONF.executor_remote_transport
    if transport == "local":
        return LocalConnection()
    if transport == "ssh":
        return SSHConnection(host=CONF.executor_remote_host, port=CONF.executor_remote_port,
                             username=CONF.executor_remote_user or None, password=None,
                             key_filename=CONF.executor_remote_key or None)
    if transport == "adb":
        return ADBConnection(host=CONF.executor_remote_host, port=CONF.executor_remote_port,
                             serial=CONF.executor_remote_serial or None)
    raise ValueError(f"unknown executor_remote_transport {transport!r} "
                     "(expected 'local', 'ssh', or 'adb')")


def make_hw_executor() -> HWExecutor:
    """The AArch64 hardware-measurement backend selected by CONF: the local /dev/executor, or a remote
    device driven over a Connection. Returned to the AArch64 executor as `self.device`; x86 is
    unrelated (it drives its own module directly)."""
    if not CONF.executor_remote:
        return LocalHWExecutor("/dev/executor", "/sys/executor")
    config = RemoteExecutorConfig(device=CONF.executor_remote_device,
                                  sysfs=CONF.executor_remote_sysfs,
                                  module=CONF.executor_remote_module,
                                  userland=CONF.executor_remote_userland)
    return RemoteHWExecutor(_make_remote_connection(), config)
