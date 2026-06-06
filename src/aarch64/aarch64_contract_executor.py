"""
File: AArch64 contract-executor (CE) service — request encoding and CE process management.
"""
import select
import subprocess
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from enum import IntFlag, IntEnum
from .contract_executor.stream_ipc import StreamIPC
from .aarch64_trace import ContractExecutionResult

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


class SimArch(IntEnum):
    RVZR_ARCH_X86_64    = 1
    RVZR_ARCH_AARCH64   = 2

class SimVersion(IntEnum):
    VER_1    = 1

class ContractType(IntEnum):
    ALWAYS_MISPREDICT   = 0   # explore every mispredicted branch (default)
    ARCH_ONLY           = 1   # follow architectural path, no speculation
    BPU_NEOVERSE_N3     = 2   # mispredict when TAGE (Neoverse N3 model) disagrees


RVZR_MAGIC      = b"RVZR" # 0x525A5652u

# IPC message types exchanged with the CE over StreamIPC.
_MSG_REQUEST  = 1
_MSG_RESPONSE = 2

# GPRs per CE trace entry: x0..x30 (x30=lr at gpr[30]); pinned by the CE's simulation_state.h.
AARCH64_NUM_GPRS = 31

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
    contract_type: ContractType = ContractType.ALWAYS_MISPREDICT

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
        if self.req_code_base_phys is not None:
            req_code_base_phys = self.req_code_base_phys
            config_flags |= ConfigFlags.CONFIG_FLAG_REQ_CODE_BASE_PHYS

        req_code_base_virt = 0
        if self.req_code_base_virt is not None:
            req_code_base_virt = self.req_code_base_virt
            config_flags |= ConfigFlags.CONFIG_FLAG_REQ_CODE_BASE_VIRT

        req_mem_base_phys = 0
        if self.req_mem_base_phys is not None:
            req_mem_base_phys = self.req_mem_base_phys
            config_flags |= ConfigFlags.CONFIG_FLAG_REQ_MEM_BASE_PHYS

        req_mem_base_virt = 0
        if self.req_mem_base_virt is not None:
            req_mem_base_virt = self.req_mem_base_virt
            config_flags |= ConfigFlags.CONFIG_FLAG_REQ_MEM_BASE_VIRT

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
        data += (int(self.contract_type)).to_bytes(8, 'little')

        data += code_size.to_bytes(8, 'little')
        data += mem_size.to_bytes(8, 'little')
        data += regs_size.to_bytes(8, 'little')

        data += (0).to_bytes(8, 'little') # Reserved

        data += self.machine_code
        data += self.memory
        data += self.registers

        return bytes(data)

class ContractExecutorService:
    def __init__(self, binary: Path):
        self._binary = binary
        self._proc: subprocess.Popen = self._spawn()

    def _spawn(self) -> subprocess.Popen:
        MB: int = 1 << 20
        proc = subprocess.Popen(
                [self._binary],
                bufsize=MB,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,   # inherit parent stderr — avoids pipe-full blocking on CE debug prints
                pipesize=MB,
        )
        self._stream_ipc = StreamIPC(proc.stdin, proc.stdout, proc)
        return proc

    def stop(self):
        if self.is_running():
            self._proc.terminate()
            self._proc.wait()
            self._proc = None

    def is_running(self):
        return self._proc is not None

    def _drain_stderr(self) -> str:
        if self._proc.stderr is None:
            return "(stderr inherited by parent process)"
        stderr_output = []
        while True:
            ready, _, _ = select.select([self._proc.stderr], [], [], 0.1)
            if not ready:
                break
            chunk = self._proc.stderr.read1(4096)
            if not chunk:
                break
            stderr_output.append(chunk.decode(errors='replace'))
        return ''.join(stderr_output)

    def run(self, execution: ContractExecution) -> ContractExecutionResult:
        """
        Run a single execution and return raw result.
        On CE crash or hang: log stderr, restart the process, and re-raise so the
        caller can decide whether to skip the test case.
        """
        data = execution.encode()
        self._stream_ipc.send_req(_MSG_REQUEST, data)
        try:
            msg_type, reply = self._stream_ipc.recv_resp()
        except EOFError:
            self._proc.wait(timeout=2)
            stderr = self._drain_stderr()
            rc = self._proc.returncode
            self._proc = self._spawn()
            raise RuntimeError(
                f"contract_executor crashed (exit code {rc}, signal {-rc if rc and rc < 0 else 'none'}).\n"
                f"stderr:\n{stderr or '(empty)'}"
            ) from None
        except RuntimeError as e:
            # CE hung (alive but no response within timeout) — kill it, drain stderr, restart
            stderr = self._drain_stderr()
            self._proc.kill()
            try:
                self._proc.wait(timeout=2)
            except Exception:
                pass
            self._proc = self._spawn()
            raise RuntimeError(
                f"contract_executor hung.\n"
                f"  hang details: {e}\n"
                f"stderr before kill:\n{stderr or '(empty)'}"
            ) from None
        if msg_type != _MSG_RESPONSE:
            raise RuntimeError(f"unexpected CE message type {msg_type} (expected {_MSG_RESPONSE})")
        return ContractExecutionResult(reply, AARCH64_NUM_GPRS)
