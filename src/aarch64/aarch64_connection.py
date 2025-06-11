import re
import uuid
import tempfile
from defer import return_value
from ppadb.client import Client as AdbClient
from typing import List, Literal, Union, Optional, Type, Callable, Tuple
from abc import ABC, abstractmethod
import functools
import time

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

class USBConnection(Connection):
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
    def __len__(self):
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


class UserlandExecutorImp:
    _RETRIES = 100

    def __init__(self, connection: Connection, userland_application_path: str,
                 device_path: str, sys_executor_path: str, module_path: str):
        self.remote_batch_file_path: Optional[str] = None
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

        if not self.connection.is_file_present(self.userland_application_path):
            self.connection.push('executor_userland', userland_application_path)

        # Discard any previous contents in the executor memory
        self.discard_all_inputs()
        self.checkout_region(TestCaseRegion())
        del self.contents 

    def trace(self):

        if self.remote_batch_file_path is not None:
            self.connection.shell(
            f'{self.userland_application_path} {self.executor_device_path} c {remote_batch_file_path}',
            priviledged=True)
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
            print(f"{result=}")
            raise RuntimeError('Could not measurements')

        htrace = int(htrace_match.group(1), 2)
        assert htrace is not None
        pfc_list = [int(pfc_value) for _, pfc_value in
                    sorted(pfc_matches, key=lambda x: int(x[0]))]
        architectural_memory_accesses_bitmap = architectural_memory_accesses_bitmap_match.group(1)

        return HWMeasurement(htrace=htrace, pfcs=pfc_list, memory_ids=architectural_memory_accesses_bitmap)

    @property
    def contents(self) -> ExecutorMemory:
        filename = f'{uuid.uuid4().hex}.bin'
        remote_filename = f'remote_{filename}'

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
        self.connection.shell(f'{self.userland_application_path} {self.executor_device_path} w {filename}', privileged=True)


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
            print(f"{result=}")
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
            print(f"{result=}")
            raise RuntimeError("Could not find test length")
        return int(test_length_match.group(1))

    @retry(max_times=_RETRIES, retry_on=RuntimeError, backoff=0.5)
    def allocate_iid(self) -> int:
        result = self._query_executor(5)
        iid_matching = re.search(r"Allocated Input ID: (\d+)", result)
        if iid_matching is None:
            print(f"{result=}")
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

    def upload_batch(executor_batch: ExecutorBatch, dest_path: Optional[str] = '/data/local/tmp/executor_batch'):
        if self.remote_batch_file_path is not None:
            self.connection.shell(f'rm {self.remote_batch_file_path}')

        if dest_path is not None:
            with tempfile.NamedTemporaryFile() as tmp_file:
                tmp_file.write(str(conf_file))
                self.connection.push(dest_path, tmp_file.name)

        self.remote_batch_file_path = dest_path


class ExecutorBatch:
    def __init__():
        self._inputs: List[str] = []
        self._tests: List[str] = []
        self._repeats: int = 1

    def add_input(self, input_path: str):
        inputs.append(input_path)

    def add_test(self, test_path: str):
        tests.append(test_path)

    @property
    def repeats(self):
        return self._repeats

    @property.setter
    def repeats(self, reps: int):
        self._repeats = reps

    def __str__(self):
        tests_header = "<tests>"
        inputs_header = "<inputs>"
        repeats_header = "<repeats>"
        tests = reduce("", lambda x: x + "\n", tests_header)
        inputs = reduce("", lambda x: x + "\n", tests_header)
        repeats = str(self._repeats) + "\n"
        return tests_header + "\n" + tests + "\n" + inputs_header + "\n" + inputs + "\n" + repeats_heaader + "\n" + repeats

