import subprocess
import tempfile
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from functools import reduce
from enum import IntFlag
import operator
import time
from .contract_executor.stream_ipc import *



class SimFlags(IntFlag):
    RVZR_FLAG_NONE          = 0
    RVZR_FLAG_HAS_CODE      = 1 << 0
    RVZR_FLAG_HAS_REGS      = 1 << 1
    RVZR_FLAG_HAS_MEMORY    = 1 << 2


class ConfigFlags(IntFlag):
    CONFIG_FLAG_NONE                = 0
    CONFIG_FLAG_REQ_CODE_BASE_PHYS  = 1 << 0
    CONFIG_FLAG_REQ_CODE_BASE_VIRT  = 1 << 1
    CONFIG_FLAG_REQ_MEM_BASE_PHYS   = 1 << 2
    CONFIG_FLAG_REQ_MEM_BASE_VIRT   = 1 << 3


class SimArch(IntFlag):
    RVZR_ARCH_X86_64    = 1
    RVZR_ARCH_AARCH64   = 2

class SimVersion(IntFlag):
    VER_1    = 1


RVZR_MAGIC      = b"RVZR" # 0x525A5652u

@dataclass(slots=True)
class ContractExecution:
    machine_code: bytes
    memory: bytes
    registers: bytes
    arch: SimArch
    max_misspred_branch_nesting: int
    max_misspred_instructions: int
    req_code_base_phys: Optional[int]   = None
    req_code_base_virt: Optional[int]   = None
    req_mem_base_phys: Optional[int]    = None
    req_mem_base_virt: Optional[int]    = None
    version: SimVersion = SimVersion.VER_1

    def encode(self) -> bytes:
        """
        Serialize execution into the executor's expected format.
        """
        assert isinstance(self.max_misspred_branch_nesting, int)
        assert isinstance(self.max_misspred_instructions, int)

        assert isinstance(self.memory, bytes)
        assert isinstance(self.registers, bytes)
        assert isinstance(self.machine_code, bytes)
        sim_flags = SimFlags.RVZR_FLAG_HAS_CODE | SimFlags.RVZR_FLAG_HAS_REGS | SimFlags.RVZR_FLAG_HAS_MEMORY

        config_flags = ConfigFlags.CONFIG_FLAG_NONE

        req_code_base_phys = 0
#        if self.req_code_base_phys is not None:
#            req_code_base_phys = self.req_code_base_phys
#            config_flags |= ConfigFlags.CONFIG_FLAG_REQ_CODE_BASE_PHYS
#
        req_code_base_virt = 0
#        if self.req_code_base_virt is not None:
#            req_code_base_virt = self.req_code_base_virt
#            config_flags |= ConfigFlags.CONFIG_FLAG_REQ_CODE_BASE_VIRT
#
        req_mem_base_phys = 0
#        if self.req_mem_base_phys is not None:
#            req_mem_base_phys = self.req_mem_base_phys
#            config_flags |= ConfigFlags.CONFIG_FLAG_REQ_MEM_BASE_PHYS
#
        req_mem_base_virt = 0
#        if self.req_mem_base_virt is not None:
#            req_mem_base_virt = self.req_mem_base_virt
#            config_flags |= ConfigFlags.CONFIG_FLAG_REQ_MEM_BASE_VIRT

        code_size: int = len(self.machine_code)
        mem_size: int = len(self.memory)
        regs_size: int = len(self.registers)

        data = bytearray()
        data += RVZR_MAGIC
        data += (self.version).to_bytes(2, 'little')
        data += (self.arch).to_bytes(2, 'little')
        data += (sim_flags).to_bytes(8, 'little')

        data += (config_flags).to_bytes(8, 'little')
        data += (self.max_misspred_branch_nesting).to_bytes(8, 'little')
        data += (self.max_misspred_instructions).to_bytes(8, 'little')
        data += (req_code_base_phys).to_bytes(8, 'little')
        data += (req_code_base_virt).to_bytes(8, 'little')
        data += (req_mem_base_phys).to_bytes(8, 'little')
        data += (req_mem_base_virt).to_bytes(8, 'little')

        data += code_size.to_bytes(8, 'little')
        data += mem_size.to_bytes(8, 'little')
        data += regs_size.to_bytes(8, 'little')

        data += (0).to_bytes(8, 'little') # Reserved

        data += self.machine_code
        data += self.memory
        data += self.registers

        return bytes(data)

class ContractExecutionResult:
    def decode(self) -> bytes:
        pass

class ContractExecutorService:
    def __init__(self, binary: Path):
        self._binary = binary

        self._proc: subprocess.Popen | None = None
        self._shm: ShmRegion | None = None
        self._mmap: mmap.mmap | None = None
        self._req_ring: RingBuffer | None = None
        self._resp_ring: RingBuffer | None = None

    def _attach_shm_busy_loop(self, shm_name: str, timeout: float | None = None):
        start = time.time()
        while True:
            try:
                return attach_shm(shm_name)
            except FileNotFoundError:
                if timeout is not None and (time.time() - start) > timeout:
                    raise TimeoutError(f"Timed out waiting for shm '{name}'")
                time.sleep(0.001)  # 1 ms backoff

    def start(self, shm_name: str):
        assert isinstance(shm_name, str) and len(shm_name) > 0
        shm_name = shm_name if shm_name[0] == "/" else "/" + shm_name
        self._proc = subprocess.Popen(
                [self._binary, shm_name],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
        )

        self._shm, self._mmap, req_buf, resp_buf = self._attach_shm_busy_loop(shm_name, 1)
        self._req_ring = RingBuffer(self._shm.req, req_buf)
        self._resp_ring = RingBuffer(self._shm.resp, resp_buf)
    
    def stop(self):
        if self.is_running():
            self._proc.terminate()
            self._proc.wait()
            self._proc = None
            self._shm = None
            self._mmap = None
            self._req_ring = None
            self._resp_ring = None


    def is_running(self):
        return self._proc is not None

    def run(self, execution: ContractExecution) -> ContractExecutionResult:
        """
        Run a single execution and return raw result.
        """
        data = execution.encode()
        self._req_ring.send(data, 1)
        msg_type, reply = self._resp_ring.recv()
        return ContractExecutionResult(reply)


