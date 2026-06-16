"""
File: AArch64 execution-trace analysis.
  - Derive CTrace (cache-set footprint) and InputTaint from an execution trace
  - show_context: human-readable view of a trace window
"""
import numpy as np
import struct
from typing import Iterable, Iterator, List, Set
from itertools import chain

from ..interfaces import CTrace, InputTaint
from .aarch64_disasm import disassemble_instruction, decode_reg_accesses
from .aarch64_input_layout import map_register_to_offsets
from .aarch64_target_desc import Aarch64TargetDesc, SANDBOX_BASE_REGISTER


U64 = struct.Struct("<Q")
SIZE_T = struct.Struct("<Q")   # size_t = 8

_SANDBOX_BASE_GPR = int(Aarch64TargetDesc.reg_normalized[SANDBOX_BASE_REGISTER])

class MemAccess:
    _STRUCT = struct.Struct("<QQQQQQ")

    def __init__(self, buf: memoryview, offset: int):
        (self.effective_address,
        self.before,
        self.after,
        self.element_size,
        self.is_write,
        self.is_atomic) = self._STRUCT.unpack_from(buf, offset)

        self.size = self._STRUCT.size

    def _kind(self) -> str:
        return "RMW" if self.is_atomic else ("WRITE" if self.is_write else "READ")

    def __repr__(self):
        return (f"<MemAccess ea=0x{self.effective_address:x} "
                f"before=0x{self.before:x} after=0x{self.after:x} "
                f"size={self.element_size} {self._kind()}>")

    def pretty_print(self, indent=0):
        prefix = " " * indent
        print(f"{prefix}MemAccess:")
        print(f"{prefix}  EA: 0x{self.effective_address:x}")
        print(f"{prefix}  Before: 0x{self.before:x}")
        print(f"{prefix}  After: 0x{self.after:x}")
        print(f"{prefix}  Size: {self.element_size}")
        print(f"{prefix}  Type: {self._kind()}")


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
    _STRUCT = struct.Struct("<QQQQ")

    def __init__(self, buf: memoryview, offset: int):
        off = offset

        (self.instr_index, self.has_memory_access,
         self.speculation_nesting, self.is_pair) = self._STRUCT.unpack_from(buf, offset)
        off += self._STRUCT.size

        self.memory_access = MemAccess(buf, off)
        off += self.memory_access.size

        # Second access (pair element 1); always present in the fixed-size entry, valid iff is_pair.
        self.memory_access2 = MemAccess(buf, off)
        off += self.memory_access2.size

        self.size = off - offset

    def accesses(self):
        """The memory accesses this instruction performs (0, 1, or 2 for LDP/STP)."""
        if not self.has_memory_access:
            return []
        return [self.memory_access, self.memory_access2] if self.is_pair else [self.memory_access]

    def __repr__(self):
        return (f"<InstrMetadata idx={self.instr_index} "
                f"speculation_nesting={self.speculation_nesting} "
                f"has_mem={self.has_memory_access} pair={self.is_pair} "
                f"{self.memory_access}>")

    def pretty_print(self, indent=0):
        prefix = " " * indent
        print(f"{prefix}InstrMetadata:")
        print(f"{prefix}  Index: {self.instr_index}")
        print(f"{prefix}  Speculation nesting: {self.speculation_nesting}")
        print(f"{prefix}  Has memory access: {self.has_memory_access}")
        for acc in self.accesses():
            acc.pretty_print(indent + 2)


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

    def __iter__(self) -> Iterator[InstrTraceEntry]:
        yield from self.entries

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


class _TaintTracker:
    """
    Tracks must-preserve input byte offsets across a DFS-ordered CE execution trace.

    A depth-indexed write stack mirrors speculation nesting: writes at depth D are
    visible to reads at depth <= D and are discarded when nesting decreases (squashed).
    Depth-0 writes are architectural and persist for the whole trace.

    Extensible: instantiate one _TaintTracker per input region (GPR, memory, SIMD, …).
    """

    def __init__(self) -> None:
        self._written: List[Set[int]] = [set()]
        self.must_preserve: Set[int] = set()

    def set_depth(self, depth: int) -> None:
        while len(self._written) <= depth:
            self._written.append(set())
        while len(self._written) > depth + 1:
            self._written.pop()

    def on_read(self, offsets: Iterable[int], depth: int) -> None:
        written = set().union(*self._written[:depth + 1])
        self.must_preserve.update(o for o in offsets if o not in written)

    def on_write(self, offsets: Iterable[int], depth: int) -> None:
        self._written[depth].update(offsets)


def compute_taint(cer: ContractExecutionResult) -> InputTaint:
    """
    Derive input taint from a single CE execution result.

    A byte is marked must-preserve iff any execution path (arch or speculative)
    reads it before writing to it.
    """
    input_taint = InputTaint()
    sandbox_u8 = input_taint.view(np.uint8)
    mem_size = (input_taint[0]["main"].view(np.uint8).size
                + input_taint[0]["faulty"].view(np.uint8).size)
    gpr_u8 = input_taint[0]["gpr"].view(np.uint8)

    gpr_tracker = _TaintTracker()
    mem_tracker = _TaintTracker()

    for ite in cer:
        depth = ite.metadata.speculation_nesting
        gpr_tracker.set_depth(depth)
        mem_tracker.set_depth(depth)

        for ma in ite.metadata.accesses():
            offsets = [o for o in (ma.effective_address + b - ite.cpu.gpr[_SANDBOX_BASE_GPR]
                                   for b in range(ma.element_size))
                       if 0 <= o < mem_size]
            if ma.is_atomic:
                # read-modify-write: the cell is read (a source) and then written
                mem_tracker.on_read(offsets, depth)
                mem_tracker.on_write(offsets, depth)
            elif ma.is_write:
                mem_tracker.on_write(offsets, depth)
            else:
                mem_tracker.on_read(offsets, depth)

        srcs, dests = decode_reg_accesses(ite.cpu.encoding, ite.cpu.pc)
        gpr_tracker.on_read(chain.from_iterable(map(map_register_to_offsets, srcs)), depth)
        gpr_tracker.on_write(chain.from_iterable(map(map_register_to_offsets, dests)), depth)

    for offset in gpr_tracker.must_preserve:
        gpr_u8[offset] = True
    for offset in mem_tracker.must_preserve:
        sandbox_u8[offset] = True

    return input_taint


def compute_ctrace(cer: ContractExecutionResult) -> CTrace:
    # Ordered sequence of cache sets, in execution order (architectural and speculative accesses
    # interleaved at their execution point). Order distinguishes architectural from speculative
    # observations by position — like the x86 contract tracer — so no arch/spec value offset is
    # needed. Within one access, consecutive bytes in the same line collapse to a single entry.
    line_size, num_sets = 64, 64
    trace: List[int] = []
    for ite in cer:
        base = ite.cpu.gpr[_SANDBOX_BASE_GPR]
        for ma in ite.metadata.accesses():
            prev = None
            for byte_idx in range(ma.element_size):
                cache_set = ((ma.effective_address + byte_idx - base) // line_size) % num_sets
                if cache_set != prev:
                    trace.append(cache_set)
                    prev = cache_set
    return CTrace(raw_trace=trace)


def show_context(trace, idx, window=-1):
    if window < 0:
        window = len(trace)
    start = max(0, idx - window)
    end = min(len(trace), idx + window + 1)
    for j in range(start, end):
        insn = trace[j]
        disas = disassemble_instruction(insn.cpu.encoding, insn.cpu.pc)
        marker = "→" if j == idx else " "
        print(f"{marker} [{j:03d}] 0x{insn.cpu.pc:016x}: {disas}")
