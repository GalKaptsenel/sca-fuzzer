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
from .aarch64_executor_input_encoder import build_input_init
from .aarch64_kernel import PacKeys
from ..interfaces import MAIN_AREA_SIZE, GPR_SUBREGION_SIZE

class SimFlags(IntFlag):
    RVZR_FLAG_NONE          = 0
    RVZR_FLAG_HAS_CODE      = 1 << 0
    RVZR_FLAG_HAS_INPUT     = 1 << 1   # the shared input initialization (executor_input_format) follows code


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

class ExecutionClause(IntFlag):
    """Composable execution clauses (a bitmask). seq == no clauses (SEQ == 0)."""
    SEQ     = 0
    COND    = 1   # mispredict conditional branches
    BPAS    = 2   # speculative store bypass
    BPU     = 4   # mispredict per an injected branch predictor
    BARRIER = 8   # honor barriers: cut speculation a fencing barrier stops


# The only execution-clause combinations the CE supports. Arbitrary bitmask mixes are not
# meaningful contracts (e.g. COND|BPU is two conflicting branch models) and are rejected.
# BARRIER is a modifier: only valid alongside a speculation clause it can cut.
SUPPORTED_EXECUTION_CLAUSES = frozenset({
    ExecutionClause.SEQ,
    ExecutionClause.COND,
    ExecutionClause.BPAS,
    ExecutionClause.BPU,
    ExecutionClause.COND | ExecutionClause.BPAS,   # cond-bpas
    ExecutionClause.BPAS | ExecutionClause.BARRIER,
    ExecutionClause.COND | ExecutionClause.BARRIER,
    ExecutionClause.BPU  | ExecutionClause.BARRIER,
    ExecutionClause.COND | ExecutionClause.BPAS | ExecutionClause.BARRIER,
})


class BranchPredictor(IntEnum):
    """Predictor the BPU clause uses (selected via the input). Mirror of enum branch_predictor_id."""
    NONE        = 0
    NEOVERSE_N3 = 1


# Maps an accepted contract_execution_clause string to (clause bit it sets, predictor it implies).
EXECUTION_CLAUSE_MAP = {
    "seq":                          (ExecutionClause.SEQ,  BranchPredictor.NONE),
    "no_speculation":               (ExecutionClause.SEQ,  BranchPredictor.NONE),
    "cond":                         (ExecutionClause.COND, BranchPredictor.NONE),
    "conditional_br_misprediction": (ExecutionClause.COND, BranchPredictor.NONE),
    "bpas":                         (ExecutionClause.BPAS, BranchPredictor.NONE),
    "bpu_neoverse_n3":              (ExecutionClause.BPU,  BranchPredictor.NEOVERSE_N3),
    "barrier":                      (ExecutionClause.BARRIER, BranchPredictor.NONE),
}


RVZRCE_MAGIC    = 0x4543525A5652  # "RVZRCE" (matches RVZRCE_MAGIC in common_msg_constants.h)

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
    execution_clauses: ExecutionClause = ExecutionClause.COND
    branch_predictor: BranchPredictor = BranchPredictor.NONE
    mte_tags: Optional[list] = None       # per-input MTE tags (one per 16B granule), or None
    pac_keys: Optional[PacKeys] = None    # per-input PAC keys, or None

    def encode(self) -> bytes:
        """
        Serialize execution into the executor's expected format.
        """
        assert isinstance(self.max_misspred_branch_nesting, int)
        assert isinstance(self.max_misspred_instructions, int)

        assert isinstance(self.memory, bytes)
        assert isinstance(self.registers, bytes)
        assert isinstance(self.machine_code, bytes)
        sim_flags = SimFlags.RVZR_FLAG_HAS_CODE | SimFlags.RVZR_FLAG_HAS_INPUT

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

        # memory = main‖faulty; the gpr register slots are the GPR section (CE reads only those).
        input_init = build_input_init(self.memory[:MAIN_AREA_SIZE], self.memory[MAIN_AREA_SIZE:],
                                self.registers[:GPR_SUBREGION_SIZE],
                                mte_tags=self.mte_tags,
                                pac_keys=self.pac_keys.words() if self.pac_keys is not None else None)
        code_size: int = len(self.machine_code)
        input_init_size: int = len(input_init)

        data = bytearray()
        data += (RVZRCE_MAGIC).to_bytes(8, 'little')
        data += (self.version).to_bytes(8, 'little')
        data += (self.arch).to_bytes(8, 'little')
        data += (sim_flags).to_bytes(8, 'little')

        data += (config_flags).to_bytes(8, 'little')
        data += (self.max_misspred_branch_nesting).to_bytes(8, 'little')
        data += (self.max_misspred_instructions).to_bytes(8, 'little')
        data += (req_code_base_phys).to_bytes(8, 'little')
        data += (req_code_base_virt).to_bytes(8, 'little')
        data += (req_mem_base_phys).to_bytes(8, 'little')
        data += (req_mem_base_virt).to_bytes(8, 'little')
        data += (int(self.execution_clauses)).to_bytes(8, 'little')
        data += (int(self.branch_predictor)).to_bytes(8, 'little')

        data += code_size.to_bytes(8, 'little')
        data += input_init_size.to_bytes(8, 'little')

        data += (0).to_bytes(8, 'little') # Reserved

        data += self.machine_code
        data += input_init

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
        A CE crash or hang is a fatal bug, not a recoverable condition: drain stderr for diagnostics
        and raise. It must never be skipped or masked — do not add a respawn/retry fallback here.
        """
        data = execution.encode()
        self._stream_ipc.send_req(_MSG_REQUEST, data)
        try:
            msg_type, reply = self._stream_ipc.recv_resp()
        except EOFError:
            try:
                self._proc.wait(timeout=2)
            except Exception:
                self._proc.kill()
            stderr = self._drain_stderr()
            rc = self._proc.returncode
            raise RuntimeError(
                f"contract_executor crashed (exit code {rc}, signal {-rc if rc and rc < 0 else 'none'}).\n"
                f"stderr:\n{stderr or '(empty)'}"
            ) from None
        except RuntimeError as e:
            # CE hung (alive but no response within timeout) — kill it, drain stderr, and fail
            stderr = self._drain_stderr()
            self._proc.kill()
            try:
                self._proc.wait(timeout=2)
            except Exception:
                pass
            raise RuntimeError(
                f"contract_executor hung.\n"
                f"  hang details: {e}\n"
                f"stderr before kill:\n{stderr or '(empty)'}"
            ) from None
        if msg_type != _MSG_RESPONSE:
            raise RuntimeError(f"unexpected CE message type {msg_type} (expected {_MSG_RESPONSE})")
        result = ContractExecutionResult(reply, AARCH64_NUM_GPRS)
        return result
