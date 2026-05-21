import subprocess
import struct
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from enum import IntFlag
import time
from .contract_executor.stream_ipc import StreamIPC

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

class ContractType(IntFlag):
    ALWAYS_MISPREDICT   = 0   # explore every mispredicted branch (default)
    ARCH_ONLY           = 1   # follow architectural path, no speculation
    BPU_NEOVERSE_N3     = 2   # mispredict when TAGE (Neoverse N3 model) disagrees


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

U64 = struct.Struct("<Q")
SIZE_T = struct.Struct("<Q")   # size_t = 8

class MemAccess:
    _STRUCT = struct.Struct("<QQQQQ")

    def __init__(self, buf: memoryview, offset: int):
        (self.effective_address,
        self.before,
        self.after,
        self.element_size,
        self.is_write) = self._STRUCT.unpack_from(buf, offset)

        self.size = self._STRUCT.size

    def __repr__(self):
        return (f"<MemAccess ea=0x{self.effective_address:x} "
                f"before=0x{self.before:x} after=0x{self.after:x} "
                f"size={self.element_size} "
                f"{'WRITE' if self.is_write else 'READ'}>")

    def pretty_print(self, indent=0):
        prefix = " " * indent
        print(f"{prefix}MemAccess:")
        print(f"{prefix}  EA: 0x{self.effective_address:x}")
        print(f"{prefix}  Before: 0x{self.before:x}")
        print(f"{prefix}  After: 0x{self.after:x}")
        print(f"{prefix}  Size: {self.element_size}")
        print(f"{prefix}  Type: {'WRITE' if self.is_write else 'READ'}")




class CPUState:
    def __init__(self, buf: memoryview, offset: int, num_gprs: int):
        self._buf = buf
        self._base = offset

        off = offset

        # gprs
        self.gpr = list(struct.unpack_from(
            f"<{num_gprs}Q", buf, off
        ))
        off += U64.size * num_gprs

        self.sp, = U64.unpack_from(buf, off); off += U64.size
        self.pc, = U64.unpack_from(buf, off); off += U64.size
        self.nzcv, = U64.unpack_from(buf, off); off += U64.size
        self.encoding, = U64.unpack_from(buf, off); off += U64.size

        self.extra_data_size, = SIZE_T.unpack_from(buf, off); off += SIZE_T.size

        self.extra_data = buf[off:off + self.extra_data_size]
        off += self.extra_data_size

        self.size = off - offset

    def __repr__(self):
        gprs_str = ", ".join(f"x{i}=0x{val:x}" for i, val in enumerate(self.gpr))
        return (f"<CPUState {gprs_str} SP=0x{self.sp:x} PC=0x{self.pc:x} "
                f"NZCV=0x{self.nzcv:x} ENC=0x{self.encoding:x} "
                f"extra_data_len={self.extra_data_size}>")

    def pretty_print(self, indent=0):
        from .aarch64_executor import disassemble_instruction
        prefix = " " * indent
        print(f"{prefix}CPUState:")
        for i, val in enumerate(self.gpr):
            print(f"{prefix}  x{i:02d}: 0x{val:016x}")
        print(f"{prefix}  SP : 0x{self.sp:016x}")
        print(f"{prefix}  PC : 0x{self.pc:016x}")
        print(f"{prefix}  NZCV: 0x{self.nzcv:016x}")
        print(f"{prefix}  Encoding: 0x{self.encoding:08x} ({disassemble_instruction(self.encoding, self.pc)})")
        print(f"{prefix}  Extra data size: {self.extra_data_size}")


class InstrMetadata:
    _STRUCT = struct.Struct("<QQQ")

    def __init__(self, buf: memoryview, offset: int):
        off = offset

        (self.instr_index, self.has_memory_access, self.speculation_nesting) = self._STRUCT.unpack_from(buf, offset)
        off += self._STRUCT.size

        self.memory_access = MemAccess(buf, off)
        off += self.memory_access.size


        self.size = off - offset

    def __repr__(self):
        return (f"<InstrMetadata idx={self.instr_index} "
                f"speculation_nesting={self.speculation_nesting} "
                f"has_mem={self.has_memory_access} "
                f"{self.memory_access}>")

    def pretty_print(self, indent=0):
        prefix = " " * indent
        print(f"{prefix}InstrMetadata:")
        print(f"{prefix}  Index: {self.instr_index}")
        print(f"{prefix}  Speculation nesting: {self.speculation_nesting}")
        print(f"{prefix}  Has memory access: {self.has_memory_access}")
        self.memory_access.pretty_print(indent + 2)



class InstrTraceEntry:
    def __init__(self, buf: memoryview, offset: int, num_gprs: int):
        self.cpu = CPUState(buf, offset, num_gprs)
        off = offset + self.cpu.size

        self.metadata = InstrMetadata(buf, off)
        off += self.metadata.size

        self.size = off - offset

    def __repr__(self):
        return f"<InstrTraceEntry {self.cpu} {self.metadata}>"

    def pretty_print(self, indent=0):
        prefix = " " * indent
        print(f"{prefix}=== InstrTraceEntry ===")
        self.cpu.pretty_print(indent + 2)
        self.metadata.pretty_print(indent + 2)
        print()


class ContractExecutionResult:
    def __init__(self, blob: bytes, num_gprs: int):
        self._buf = memoryview(blob)

        off = 0
        self.entry_count, = SIZE_T.unpack_from(self._buf, off)
        off += SIZE_T.size

        self.entries = []
        for _ in range(self.entry_count):
            entry = InstrTraceEntry(self._buf, off, num_gprs)
            self.entries.append(entry)
            off += entry.size

        self.size = off

    def __len__(self) -> int:
        return self.entry_count

    def __iter__(self) -> InstrTraceEntry:
        for i in range(self.entry_count):
            yield self[i]

    def __getitem__(self, idx) -> InstrTraceEntry:
        if not (0 <= idx < self.entry_count):
            raise IndexError(idx)

        return self.entries[idx]

    def __repr__(self):
        return f"<ContractExecutionResult entries={self.entry_count}>"

    def pretty_print(self):
        print(f"ContractExecutionResult: {self.entry_count} entries")
        for i, entry in enumerate(self.entries):
            print(f"\n--- Entry {i} ---")
            entry.pretty_print(indent=2)


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
        import select
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
        self._stream_ipc.send_req(1, data)
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
        assert msg_type == 2
        return ContractExecutionResult(reply, 31) # NUM GPRS of aarch64 is 31 (x0 to x30)
