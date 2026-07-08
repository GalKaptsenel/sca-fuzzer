"""
File: AArch64 execution-trace analysis.
  - Derive CTrace (cache-set footprint) and InputTaint from an execution trace
  - show_context: human-readable view of a trace window
"""
import numpy as np
import struct
from typing import Dict, Iterable, Iterator, List, Set
from itertools import chain

from ..interfaces import CTrace, InputTaint
from ..config import CONF, ConfigException
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
        assert self.extra_data_size == 0, \
            f"CE emitted extra_data_size={self.extra_data_size}; the parser assumes a fixed entry stride"

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
    _STRUCT = struct.Struct("<QQQQQ")

    def __init__(self, buf: memoryview, offset: int):
        off = offset

        (self.instr_index, self.has_memory_access,
         self.speculation_nesting, self.is_pair, self.window_id) = self._STRUCT.unpack_from(buf, offset)
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
        self.truncated, = U64.unpack_from(self._buf, off)
        off += U64.size

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

    Writes are scoped by speculation *window*, not by raw nesting depth. The live path from
    the architectural flow (window 0) to the current instruction is `_live`, one window id per
    depth. A read is masked only by writes made in windows on that live path; a write lands in
    the innermost live window. When the trace re-forks — including a same-depth re-fork after an
    unwind, which shares a depth but gets a fresh window id — the unwound sibling drops off the
    live path and its writes stop masking, so squashed speculation cannot hide an input read.

    Instantiate one _TaintTracker per input region (GPR, memory, SIMD, …).
    """

    def __init__(self) -> None:
        self._written: Dict[int, Set[int]] = {0: set()}  # window id -> offsets written in it
        self._live: List[int] = [0]                      # _live[d] = window id at depth d; 0 = architectural
        self._synth_id: int = 0                          # decreasing ids for skipped windows (never real)
        self.must_preserve: Set[int] = set()

    def enter(self, depth: int, window_id: int) -> None:
        """Move to the (depth, window_id) an instruction executed under, updating the live path."""
        # One instruction can open several windows at once (a bpas phase-B push plus a cond fork), so
        # nesting may rise by more than 1. The skipped intermediate levels ran no logged instruction and
        # hence hold no writes; fill them with empty placeholder windows (unique negative ids) so the
        # live-path union and discard bookkeeping stay well-formed.
        while len(self._live) < depth:
            self._synth_id -= 1
            self._live.append(self._synth_id)
            self._written[self._synth_id] = set()
        if depth < len(self._live) and self._live[depth] == window_id:
            self._discard(depth + 1)            # same window continues; drop only its unwound children
        else:
            self._discard(depth)                # a new window at this depth: fork or post-unwind sibling
            self._live.append(window_id)
            self._written[window_id] = set()    # window ids are never reused, so this is always fresh

    def _discard(self, from_depth: int) -> None:
        for wid in self._live[from_depth:]:
            del self._written[wid]              # every live window has an entry; a miss is a real bug
        del self._live[from_depth:]

    def on_read(self, offsets: Iterable[int]) -> None:
        written = set().union(*(self._written[w] for w in self._live))
        self.must_preserve.update(o for o in offsets if o not in written)

    def on_write(self, offsets: Iterable[int]) -> None:
        self._written[self._live[-1]].update(offsets)


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
        gpr_tracker.enter(ite.metadata.speculation_nesting, ite.metadata.window_id)
        mem_tracker.enter(ite.metadata.speculation_nesting, ite.metadata.window_id)

        for ma in ite.metadata.accesses():
            offsets = [o for o in (ma.effective_address + b - ite.cpu.gpr[_SANDBOX_BASE_GPR]
                                   for b in range(ma.element_size))
                       if 0 <= o < mem_size]
            if ma.is_atomic:
                # read-modify-write: the cell is read (a source) and then written
                mem_tracker.on_read(offsets)
                mem_tracker.on_write(offsets)
            elif ma.is_write:
                mem_tracker.on_write(offsets)
            else:
                mem_tracker.on_read(offsets)

        srcs, dests = decode_reg_accesses(ite.cpu.encoding, ite.cpu.pc)
        gpr_tracker.on_read(chain.from_iterable(map(map_register_to_offsets, srcs)))
        gpr_tracker.on_write(chain.from_iterable(map(map_register_to_offsets, dests)))

    for offset in gpr_tracker.must_preserve:
        gpr_u8[offset] = True
    for offset in mem_tracker.must_preserve:
        sandbox_u8[offset] = True

    return input_taint


_CACHE_LINE = 64
_NUM_SETS = 64


def _sandbox_base(ite) -> int:
    return ite.cpu.gpr[_SANDBOX_BASE_GPR]


def _code_start(cer: ContractExecutionResult) -> int:
    """The PC of the first traced instruction; PCs are normalized against it for reproducibility."""
    for ite in cer:
        return ite.cpu.pc
    return 0


def _initial_regs(cer: ContractExecutionResult) -> List[int]:
    """Register state at test-case entry (the CTR/ARCH observation clauses seed the trace with it)."""
    for ite in cer:
        return list(ite.cpu.gpr) + [ite.cpu.sp, ite.cpu.nzcv]
    return []


def _cache_sets(ma, base: int) -> Iterator[int]:
    for byte_idx in range(ma.element_size):
        yield ((ma.effective_address + byte_idx - base) // _CACHE_LINE) % _NUM_SETS


def _is_speculative(ite) -> bool:
    return ite.metadata.speculation_nesting != 0


def _ct_none(cer: ContractExecutionResult) -> CTrace:
    return CTrace.get_null()


def _ct_l1d(cer: ContractExecutionResult) -> CTrace:
    # Union bitmap of the L1D cache sets touched (arch and speculative merged), matching the single
    # unordered cache-set bitmap the hardware reports. Order has no observable counterpart here.
    bitmap = 0
    for ite in cer:
        base = _sandbox_base(ite)
        for ma in ite.metadata.accesses():
            for cache_set in _cache_sets(ma, base):
                bitmap |= 1 << cache_set
    ctrace = CTrace([bitmap])
    ctrace.hash_ = bitmap
    return ctrace


def _ct_pc(cer: ContractExecutionResult) -> CTrace:
    code_start = _code_start(cer)
    return CTrace([ite.cpu.pc - code_start for ite in cer])


def _ct_memory(cer: ContractExecutionResult) -> CTrace:
    trace: List[int] = []
    for ite in cer:
        base = _sandbox_base(ite)
        for ma in ite.metadata.accesses():
            trace.append(ma.effective_address - base)
    return CTrace(trace)


def _ct_ct(cer: ContractExecutionResult) -> CTrace:
    code_start = _code_start(cer)
    trace: List[int] = []
    for ite in cer:
        base = _sandbox_base(ite)
        trace.append(ite.cpu.pc - code_start)
        for ma in ite.metadata.accesses():
            trace.append(ma.effective_address - base)
    return CTrace(trace)


def _ct_nonspecstore(cer: ContractExecutionResult) -> CTrace:
    code_start = _code_start(cer)
    trace: List[int] = []
    for ite in cer:
        base = _sandbox_base(ite)
        spec = _is_speculative(ite)
        trace.append(ite.cpu.pc - code_start)
        for ma in ite.metadata.accesses():
            if not spec or not ma.is_write:   # speculative stores are not observed
                trace.append(ma.effective_address - base)
    return CTrace(trace)


def _ct_ctr(cer: ContractExecutionResult) -> CTrace:
    return CTrace(_initial_regs(cer) + _ct_ct(cer).raw)


def _ct_arch(cer: ContractExecutionResult) -> CTrace:
    code_start = _code_start(cer)
    trace: List[int] = _initial_regs(cer)
    for ite in cer:
        base = _sandbox_base(ite)
        trace.append(ite.cpu.pc - code_start)
        for ma in ite.metadata.accesses():
            if not ma.is_write or ma.is_atomic:
                trace.append(ma.before)       # value read from memory
            trace.append(ma.effective_address - base)
    return CTrace(trace)


def _line(addr: int) -> int:
    return (addr // _CACHE_LINE) * _CACHE_LINE


def _ct_tct(cer: ContractExecutionResult) -> CTrace:
    code_start = _code_start(cer)
    trace: List[int] = []
    for ite in cer:
        base = _sandbox_base(ite)
        trace.append(_line(ite.cpu.pc - code_start))
        for ma in ite.metadata.accesses():
            trace.append(_line(ma.effective_address - base))
    return CTrace(trace)


def _ct_tcto(cer: ContractExecutionResult) -> CTrace:
    code_start = _code_start(cer)
    trace: List[int] = []
    for ite in cer:
        base = _sandbox_base(ite)
        pc = ite.cpu.pc - code_start
        trace.append(_line(pc))
        if (pc + 4) // _CACHE_LINE != pc // _CACHE_LINE:   # instruction crosses a line
            trace.append(_line(pc + 4))
        for ma in ite.metadata.accesses():
            addr = ma.effective_address - base
            trace.append(_line(addr))
            if (addr + ma.element_size) // _CACHE_LINE != addr // _CACHE_LINE:
                trace.append(_line(addr + ma.element_size))
    return CTrace(trace)


_CTRACE_BUILDERS = {
    "none": _ct_none,
    "l1d": _ct_l1d,
    "pc": _ct_pc,
    "memory": _ct_memory,
    "ct": _ct_ct,
    "loads+stores+pc": _ct_ct,
    "ct-nonspecstore": _ct_nonspecstore,
    "ctr": _ct_ctr,
    "arch": _ct_arch,
    "tct": _ct_tct,
    "tcto": _ct_tcto,
}


def compute_ctrace(cer: ContractExecutionResult) -> CTrace:
    builder = _CTRACE_BUILDERS.get(CONF.contract_observation_clause)
    if builder is None:
        raise ConfigException(
            f"contract_observation_clause '{CONF.contract_observation_clause}' "
            f"is not implemented on AArch64")
    return builder(cer)


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
