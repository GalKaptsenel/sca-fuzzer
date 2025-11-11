import re
import uuid
import tempfile
import random
import string
from defer import return_value
from ppadb.client import Client as AdbClient
from typing import List, Literal, Union, Optional, Type, Callable, Tuple, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, asdict, MISSING
from enum import Enum
import functools
import time
import paramiko
import os

import time
import ctypes
from functools import wraps
from collections import defaultdict
from contextlib import contextmanager
import fcntl


op_timings = defaultdict(list)
def profile_by_opcode(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        cmd = args[1] if len(args) > 0 else None
        print(f"running profiler with command {cmd}")
        start = time.time()
        try:
            return func(*args, **kwargs)
        finally:
            duration = time.time() - start
            op_timings[cmd].append(duration)
            print(f"done {cmd}")

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


class Connection:
    def __init__(self):
        pass

    def shell(self, command: str, privileged = False) -> str:
        raise NotImplementedError

    def push(self, src, dst):
        raise NotImplementedError

    def pull(self, src, dst):
        raise NotImplementedError

    def is_file_present(self, filename: str) -> bool:
        raise NotImplementedError


class SSHConnection(Connection):
    def __init__(self, host: str = '127.0.0.1', port: int = 22, username: str = None, password: str = None, key_filename: str = None):
        super(SSHConnection, self).__init__()
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            self.client.connect(
                    hostname=self.host,
                    port=self.port,
                    username=self.username,
                    password=password,
                    key_filename=key_filename,
                    look_for_keys=True
                )
        except Exception as e:
            raise IOError(f'Could not connect to SSH server: {e}')

        try:
            self.sftp = self.client.open_sftp()
        except Exception as e:
            self.client.close()
            raise IOError(f'Could not open SFTP session: {e}')

    def shell(self, cmd: str, privileged = False) -> str:
        cmd = f'sudo -S bash -c "{cmd}"' if privileged else cmd
        stdin, stdout, stderr = self.client.exec_command(cmd)

        if privileged and stdin is not None:
            password = self.password if self.password else ""
            stdin.write(password + "\n")
            stdin.flush()

        out = stdout.read().decode().strip()
        err = stderr.read().decode().strip()
        return out if out else err

    def push(self, src, dst):
        print(f'push {src} -> {dst}')
        with profile_op('push'):
            return self.sftp.put(src, dst)

    def pull(self, src, dst):
        print(f'pull {src} -> {dst}')
        with profile_op('pull'):
            return self.sftp.get(src, dst)

    def is_file_present(self, filename: str) -> bool:
        try:
            self.sftp.stat(filename)
            return True
        except FileNotFoundError:
            return False

    def close(self):
        if self.sftp: self.sftp.close()
        if self.client: self.client.close()

class ADBConnection(Connection):
    def __init__(self, host: str = '127.0.0.1', port: int = 5037, serial: str = None):
        super(USBConnection, self).__init__()
        self.host = host
        self.port = port
        self.client = AdbClient(host=self.host, port=self.port)

        if self.client is None:
            raise IOError('Could not connect to AdbClient')

        if len(self.client.devices()) == 0:
            raise IOError('Could not find devices for ADB')

        if serial is not None:
            for device in self.client.devices():
                if device.serial == serial:
                    self.device = device
        else:
            self.device = self.client.devices()[0]


    def shell(self, cmd: str, privileged = False) -> str:
        cmd = f'su -c "{cmd}"' if privileged else cmd
        return self.device.shell(cmd)

    def push(self, src, dst):
        return self.device.push(src, dst)

    def pull(self, src, dst):
        return self.device.pull(src, dst)

    def is_file_present(self, filename: str) -> bool:
        return 'No such file or directory' not in self.shell(f'ls {filename}')

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
    def __init__(self, htrace: int, pfcs: List[int], memory_ids: str):
        self.htrace = htrace
        self.pfcs = pfcs
        self.memory_ids = memory_ids


class UserlandExecutor(ABC):
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

    @contents.setter
    def contents(self, data: Union[str, ExecutorMemory]) -> None:
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

    @property
    @abstractmethod
    def aux_buffer(self) -> bytes:
        raise NotImplementedError()

class TestCaseRegion(ExecutorRegion):
    def __init__(self):
        super().__init__()

class InputRegion(ExecutorRegion):
    def __init__(self, iid: int):
        super().__init__()
        self.iid = iid

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
                        print(f"[retry] Attempt {attempt} failed with {e!r}, retrying in {sleep_time:.2f}s...")
                        time.sleep(sleep_time)

            raise last_exception
        return wrapper
    return decorator


class ExecutorBatch:

    def __init__(self):
        self._inputs: List[str] = []
        self._tests: List[str] = []
        self._repeats: int = 1
        self._output: Optional[str] = None

    def add_input(self, input_path: str):
        self._inputs.append(input_path)

    def add_test(self, test_path: str):
        self._tests.append(test_path)

    @property
    def repeats(self):
        return self._repeats

    @repeats.setter
    def repeats(self, reps: int):
        self._repeats = reps

    @property
    def output(self):
        return self._output

    @output.setter
    def output(self, filename: Optional[str]):
        self._output = filename 

    def __str__(self):
        tests_header = "<tests>"
        inputs_header = "<inputs>"
        repeats_header = "<repeats>"
        output_header = "<output>"
        tests = functools.reduce(lambda x, y: x + "\n" + y, self._tests, tests_header)
        inputs = functools.reduce(lambda x, y: x + "\n" + y, self._inputs, inputs_header)
        repeats = str(self._repeats) + "\n"

        config = tests + "\n" + inputs + "\n" + repeats_header + "\n" + repeats

        if self._output is not None:
            config += output_header + "\n" + self._output

        return config 


class UserlandExecutorImp(UserlandExecutor):
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

		self.connection.shell(f'echo "P" > /sys/executor/measurement_mode', privileged=True) # Use Prime And Probe
		self.connection.shell(f'echo "0" > /sys/executor/pin_to_core', privileged=True) # Use Prime And Probe

		if not self.connection.is_file_present(self.userland_application_path):
			self.connection.push('executor_userland', userland_application_path)

		# Discard any previous contents in the executor memory
		self.discard_all_inputs()
		self.checkout_region(TestCaseRegion())
		del self.contents 

	def trace(self, executor_batch: Optional[ExecutorBatch] = None) -> Optional[str]:
		def random_filename(length: int = 15) -> str:
			return ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=length))

		if executor_batch is not None:
			remote_batch_file_path  = '/tmp/' + random_filename()
			self._upload_batch(executor_batch, remote_batch_file_path)
			cmd = f'{self.userland_application_path} {self.executor_device_path} c {remote_batch_file_path}',
			with profile_op('batch execution'):
				output = self.connection.shell(cmd, privileged=True)
			self.connection.shell(f'rm {remote_batch_file_path}', privileged=True)

			if executor_batch.output is not None:
				with tempfile.NamedTemporaryFile(mode='w+') as tmp_file:
					self.connection.pull(executor_batch.output, tmp_file.name)
					tmp_file.seek(0)
					return tmp_file.read()

		else:
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
		architectural_memory_accesses_bitmap_match = re.search(
				r"architectural memory access bitmap: ([01]+)", result)

		if htrace_match is None or pfc_matches is None or architectural_memory_accesses_bitmap_match is None:
			raise RuntimeError('Could not measurements')

		htrace = int(htrace_match.group(1), 2)
		assert htrace is not None
		pfc_list = [int(pfc_value) for _, pfc_value in
			  sorted(pfc_matches, key=lambda x: int(x[0]))]
		architectural_memory_accesses_bitmap = architectural_memory_accesses_bitmap_match.group(1)

		return HWMeasurement(htrace=htrace, pfcs=pfc_list, memory_ids=architectural_memory_accesses_bitmap)

	@property
	def aux_buffer(self) -> bytes:
		result = self._query_executor(12)
		hex_lines = re.findall(r'^\s*([0-9A-Fa-f]+)\s*:\s*(.*?)\s*(?:\|.*)?$', result, flags=re.MULTILINE)
		hex_bytes = []
		for _, bytes_part in hex_lines:
			hex_bytes.extend(re.findall(r'\b[0-9a-fA-F]{2}\b', bytes_part.strip()))

		try:
			return bytes(int(b, 16) for b in hex_bytes)
		except ValueError as e:
			raise ValueError(f"Invalid hexdump format: {e}")

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

		ret_data = ""
		with open(filename) as f:
			ret_data = ExecutorMemory(f.read())

		return ExecutorMemory(ret_data)

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

	@retry(max_times=_RETRIES, retry_on=RuntimeError, backoff=0.5)
	@property
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

	def _upload_batch(self, executor_batch: ExecutorBatch, remote_batch_file_path: str):
		with tempfile.NamedTemporaryFile(mode='w') as tmp_file:
			tmp_file.write(str(executor_batch))
			tmp_file.flush()
			self.connection.push(tmp_file.name, remote_batch_file_path)

class AuxBufferIoctl(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("data", ctypes.c_void_p),
    ]

class UserMeasurement(ctypes.Structure):
    _fields_ = [
        ("htrace", ctypes.c_uint64 * 1),
        ("pfc", ctypes.c_uint64 * 3),
        ("memory_ids_bitmap", ctypes.c_uint64 * 2),
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
REVISOR_BATCHED_INPUTS_CONSTANT     = 11
REVISOR_GET_AUX_BUFFER_CONSTANT     = 12

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
    11: "REVISOR_BATCHED_INPUTS",
    12: "REVISOR_GET_AUX_BUFFER",
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

def _IOC(dir, type, nr, size):
	return ((dir << _IOC_DIRSHIFT) |
	  (type << _IOC_TYPESHIFT) |
	  (nr << _IOC_NRSHIFT) |
	  (size << _IOC_SIZESHIFT))

def _IO(type, nr):
	return _IOC(_IOC_NONE, type, nr, 0)

def _IOR(type, nr, datatype):
	return _IOC(_IOC_READ, type, nr, ctypes.sizeof(datatype))

def _IOW(type, nr, datatype):
	return _IOC(_IOC_WRITE, type, nr, ctypes.sizeof(datatype))

def _IOWR(type, nr, datatype):
	return _IOC(_IOC_READ | _IOC_WRITE, type, nr, ctypes.sizeof(datatype))

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
REVISOR_BATCHED_INPUTS = _IOWR(REVISOR_IOC_MAGIC, REVISOR_BATCHED_INPUTS_CONSTANT, ctypes.c_uint64)  # adjust struct
REVISOR_GET_AUX_BUFFER = _IOWR(REVISOR_IOC_MAGIC, REVISOR_GET_AUX_BUFFER_CONSTANT, AuxBufferIoctl)


class LocalExecutorImp(UserlandExecutor):

	def __init__(self, device_path: str, sys_executor_path: str, module_path: str):
		self.executor_device_path: str = device_path
		self.executor_sysfs: str = sys_executor_path
		self.current_region: ExecutorRegion = TestCaseRegion()
		self.fd = os.open(self.executor_device_path, os.O_RDWR)
		self._write_sysfs("measurement_mode", b"P")
		self._write_sysfs("pin_to_core", b"0")
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
		memory_ids = 0
		for i, val in enumerate(measurement.memory_ids_bitmap):
			memory_ids |= val << (i * 64)

		return HWMeasurement(htrace=htrace, pfcs=pfcs, memory_ids=format(memory_ids, '0{}b'.format(128)))

	@property
	def aux_buffer(self) -> bytes:
		buf = AuxBufferIoctl()
		buf.data = 0 # NULL to get size
		self._ioctl(REVISOR_GET_AUX_BUFFER, buf)
		size_needed = buf.size
		buf_data = (ctypes.c_uint8 * size_needed)()
		buf.data = ctypes.cast(buf_data, ctypes.c_void_p)
		buf.size = size_needed
		self._ioctl(REVISOR_GET_AUX_BUFFER, buf)
		return bytes(buf_data)

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

	@property
	def sandbox_base(self) -> int:
		return int(self._read_sysfs('print_sandbox_base'), 16)

	@property
	def code_base(self) -> int:
		return int(self._read_sysfs('print_code_base'), 16)


# Auxiliary Buffer Managment
class AuxBufferType(Enum):
	RAW_BYTES = "raw_bytes"
	BITMAP_TAINTS = "bitmap_taints"
	FULL_TRACE = "full_trace"

class ExecutorAuxBuffer(ABC):
	"""Base class for all executor auxiliary buffers."""

	def __init__(self, buffer_type: AuxBufferType):
		self.buffer_type = buffer_type

	@classmethod
	def from_json(cls, data: dict):
		"""
		Generic from_json constructor: maps JSON keys to dataclass fields.
		Ignores extra keys and sets missing keys to default.
		"""
		if not hasattr(cls, "__dataclass_fields__"):
			raise TypeError(f"{cls.__name__} must be a dataclass")

		init_kwargs = {}

		for name, field in cls.__dataclass_fields__.items():
			if name in data:
				init_kwargs[name] = data[name]
			elif field.default is not MISSING:
				init_kwargs[name] = field.default
			elif field.default_factory is not MISSING:
				init_kwargs[name] = field.default_factory()
			else:
				raise ValueError(f"Missing required field '{name}' for {cls.__name__}")

		obj = cls(**init_kwargs)
		if "buffer_type" not in cls.__dataclass_fields__:
			obj.buffer_type = AuxBufferType(data["type"])

		return obj

	@classmethod
	@abstractmethod
	def from_bytes(cls, data: bytes):
		"""Construct buffer from raw byte array (must be implemented in subclass)."""
		raise NotImplementedError

	@abstractmethod
	def to_bytes(self) -> bytes:
		"""Serialize this buffer into raw bytes (must be implemented in subclass)."""
		raise NotImplementedError


	def to_dict(self) -> dict:
		"""Generic serialization to dict/JSON."""
		d = asdict(self)
		d["type"] = getattr(self, "buffer_type", self.__class__.__name__).value
		return d

AUX_BUFFER_TYPES: dict[AuxBufferType, Type[ExecutorAuxBuffer]] = {}

def register_aux_buffer(buffer_type: AuxBufferType):

	def wrapper(cls: Type[ExecutorAuxBuffer]):
		AUX_BUFFER_TYPES[buffer_type] = cls
		return cls

	return wrapper

def aux_buffer_from_json(data: dict) -> ExecutorAuxBuffer:
	buffer_type_str = data.get("type")
	if not buffer_type_str:
		raise ValueError("JSON missing 'type' field")

	try:
		buffer_type = AuxBufferType(buffer_type_str)
	except ValueError:
		raise ValueError(f"Unknown buffer type: {buffer_type_str}")

	cls = AUX_BUFFER_TYPES.get(buffer_type)
	if cls is None:
		raise ValueError(f"No registered class for buffer type: {buffer_type}")
	return cls.from_json(data)

def aux_buffer_from_bytes(buffer_type: AuxBufferType, data: bytes) -> ExecutorAuxBuffer:
	cls = AUX_BUFFER_TYPES.get(buffer_type)
	if cls is None:
		raise ValueError(f"No registered class for buffer type: {buffer_type}")
	return cls.from_bytes(data)


@register_aux_buffer(AuxBufferType.RAW_BYTES)
@dataclass
class RawBytesAuxBuffer(ExecutorAuxBuffer):
	data: bytes = b""

	def __post_init__(self):
		super().__init__(AuxBufferType.RAW_BYTES)

	@classmethod
	def from_bytes(cls, data: bytes):
		return cls(data=data)

	@classmethod
	def from_json(cls, data: dict):
		# support JSON too, e.g., base64-encoded bytes
		import base64
		raw = base64.b64decode(data.get("data", ""))
		return cls(data=raw)

	def to_dict(self) -> dict:
		import base64
		return {
			"type": self.buffer_type.value,
			"data": base64.b64encode(self.data).decode()
		}

	def __repr__(self):
		return f"<RawBytesAuxBuffer len={len(self.data)} bytes>"

