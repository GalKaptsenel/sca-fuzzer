"""Device-free harness to compare two contract-executor traces field by field.

Drives the contract_executor binary directly (no /dev/executor) with a fixed sandbox base and a
fixed PAC profile, so a sealed test case's single resolution trace can be compared against a trace of
its relocated genuine binary. The one-run property — the resolution trace IS the genuine contract
trace — is asserted as equal ctraces and taints (and the genuine binary tracing without an FPAC).

Importable helper (no test cases here), reusable by the a-priori equivalence and safety tests.
"""
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.aarch64 import aarch64_qarma as qarma
from src.aarch64.aarch64_contract_executor import (
    ContractExecution, ContractExecutorService, SimArch, ExecutionClause, BranchPredictor)
from src.aarch64.aarch64_executor import _ce_memory_regs
from src.aarch64.aarch64_executor_input_encoder import MTE_TAG_COUNT
from src.aarch64.aarch64_kernel import PacKeys
from src.aarch64.aarch64_mte import MTE_INITIAL_DEFAULT_TAG
from src.aarch64.aarch64_printer import Aarch64Printer, Aarch64ASMLayout
from src.aarch64.aarch64_target_desc import Aarch64TargetDesc
from src.aarch64.aarch64_generator import Aarch64Generator
from src.aarch64.seal.pac import PacSigner
from src.aarch64.seal.sealer import make_sealer

CE_BIN = Path(__file__).resolve().parents[2] / "src/aarch64/contract_executor/contract_executor"
SANDBOX_BASE = 0xffff4000cc861000

# A fixed, valid PAC profile shared by the signer, the sealer and the CE, so the CE lowering and the
# relocated genuine trace are produced under identical signing (a consistent choice, not a machine).
PROFILE = qarma.profile(qarma_version=3, generic_qarma_version=0, auth_level=5,
                        va_size0=48, va_size1=48, tbi0=True, tbi1=True,
                        tbid0=False, tbid1=False, constpacfield=False, mtx0=False, mtx1=False)


def ce_available() -> bool:
    return CE_BIN.exists()


def make_signer() -> PacSigner:
    return PacSigner(lambda ptr, ctx, mn, keys: qarma.sign(ptr, ctx, mn, keys, PROFILE),
                     lambda ptr, mn: qarma.pac_field_mask(ptr, PROFILE, qarma.is_instr_key(mn)))


def _dispatch_data_size(tc) -> int:
    return 4 * sum(f.dispatch_table.size for f in tc.functions if f.dispatch_table)


def _assemble(tc) -> bytes:
    asm = Aarch64Printer(Aarch64TargetDesc()).print_layout(Aarch64ASMLayout(tc))
    return Aarch64Generator.in_memory_assemble(asm)


class CESession:
    """A single long-lived CE process plus the trace functions the sealer and tests call."""

    def __init__(self, primitives) -> None:
        self._primitives = frozenset(primitives)
        self._svc = ContractExecutorService(CE_BIN)
        self._data_size = 0   # set per sealed test case via set_data_size

    def stop(self) -> None:
        self._svc.stop()

    def __enter__(self) -> "CESession":
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def set_data_size(self, data_size: int) -> None:
        self._data_size = data_size

    def _mte_tags(self) -> Optional[List[int]]:
        return [MTE_INITIAL_DEFAULT_TAG] * MTE_TAG_COUNT if "mte" in self._primitives else None

    def _pac_keys(self, inp) -> Optional[PacKeys]:
        if "pac" not in self._primitives:
            return None
        if inp.pac_keys is None:
            raise AssertionError(f"PAC input {inp.seed} carries no keys")
        return PacKeys(*inp.pac_keys)

    def _pac_profile(self):
        return PROFILE if "pac" in self._primitives else None

    def trace_bytes(self, code: bytes, inp, nesting: int = 8, data_size: Optional[int] = None,
                    canonicalize_auth: bool = False):
        memory, regs = _ce_memory_regs(inp)
        execution = ContractExecution(
            code, memory, regs, SimArch.RVZR_ARCH_AARCH64, nesting, 0,
            req_mem_base_virt=SANDBOX_BASE, execution_clauses=ExecutionClause.COND,
            branch_predictor=BranchPredictor.NONE,
            mte_tags=self._mte_tags(), pac_keys=self._pac_keys(inp),
            pac_profile=self._pac_profile(),
            data_size=self._data_size if data_size is None else data_size,
            canonicalize_auth=canonicalize_auth)
        return self._svc.run(execution)

    def sealer(self, generator):
        return make_sealer(generator, _assemble, self._primitives,
                           make_signer() if "pac" in self._primitives else None,
                           trace_bytes_fn=self.trace_bytes)


def entry_fields(ite) -> Tuple:
    """The contract-observable state of one trace entry: PC-offset, encoding, all registers, flags,
    speculation nesting, and every memory access (address, size, direction, value, tag)."""
    c = ite.cpu
    accesses = tuple((a.effective_address, a.element_size, a.is_write, a.is_atomic, a.after,
                      a.allocation_tag) for a in ite.metadata.accesses())
    return (c.encoding, tuple(c.gpr), c.sp, c.nzcv, ite.metadata.speculation_nesting,
            ite.metadata.has_sandbox_memory_access, accesses)


def diff_traces(a, b) -> List[str]:
    """Human-readable per-entry differences between two traces (encoding, registers, flags,
    accesses); used to locate where two traces diverge."""
    out: List[str] = []
    ea, eb = list(a), list(b)
    if len(ea) != len(eb):
        out.append(f"entry count {len(ea)} != {len(eb)}")
    for i, (x, y) in enumerate(zip(ea, eb)):
        fx, fy = entry_fields(x), entry_fields(y)
        if fx != fy:
            out.append(f"entry {i} pc_off=0x{x.cpu.pc - ea[0].cpu.pc:x}: {fx} != {fy}")
    return out
