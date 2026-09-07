"""The rsb execution clause models Spectre-RSB / ret2spec: the hardware return-stack buffer is a fixed
depth, so a call chain deeper than it makes the deep RETs pop STALE addresses and speculate there. The
clause mirrors this with a finite shadow RSB (RSB_DEPTH) compared against the unbounded architectural
call stack.

This traces a call chain shallower than RSB_DEPTH (must stay inert — the RSB predicts every return
correctly) and one deeper than it (must speculate — the earliest returns mispredict a stale slot), and
checks the clause fires exactly in the deep case. Requires the CE binary and the executor's sandbox
base, so it is skipped where /sys/executor is absent."""
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")

# Matches execution_clause_rsb.c's RSB_DEPTH; anything deeper overflows the shadow RSB and mispredicts.
_RSB_DEPTH = 16


def _sandbox_base():
    try:
        return int(open("/sys/executor/print_sandbox_base").read(), 16)
    except Exception:
        return None


def _chain_asm(depth):
    """A `depth`-deep call chain f0 -> f1 -> ... -> f<depth>; each frame saves/restores X30 and returns,
    the deepest does one sandbox-masked load. Pre-sandboxed (masked) so it needs no generator pass."""
    out = [".test_case_enter:", ".section .data.main",
           ".function_0:", ".bb_0.0:", ".macro.measurement_start: NOP",
           "BL .function_1", ".macro.measurement_end: NOP", "B .test_case_exit"]
    for i in range(1, depth):
        out += [f".function_{i}:", f".bb_{i}.0:",
                "STR X30, [SP, #-16]!", f"BL .function_{i+1}", "LDR X30, [SP], #16", "RET"]
    out += [f".function_{depth}:", f".bb_{depth}.0:",
            "AND x0, x0, #0x1fff", "ADD x0, x0, x29", "LDR x1, [x0]", "RET",
            ".test_case_exit:"]
    return "\n".join(out) + "\n"


@unittest.skipUnless(os.path.exists("/dev/executor") and _sandbox_base() is not None,
                     "needs the executor sandbox base (/sys/executor)")
class RsbClauseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.chdir(_ROOT)
        from src.config import CONF
        CONF.load(os.path.join(_ROOT, "config.yml"))
        CONF.input_gen_seed = 0xB5B  # fixed => a deterministic input (and thus a stable sandbox SP)
        cls.CONF = CONF
        from src.aarch64.aarch64_generator import Aarch64Generator
        from src.aarch64.aarch64_contract_executor import ContractExecutorService
        from src.factory import get_input_generator
        cls.Gen = Aarch64Generator
        cls.base = _sandbox_base()
        cls.ce = ContractExecutorService(os.path.join(
            _ROOT, "src/aarch64/contract_executor/contract_executor"))
        cls.inp = get_input_generator(0).generate(1)[0]

    def _steps(self, depth, clause):
        from src.aarch64.aarch64_contract_executor import ContractExecution, ExecutionClause, SimArch
        from src.aarch64.aarch64_executor import _ce_memory_regs
        tc = self.Gen.in_memory_assemble(_chain_asm(depth))
        mem, regs = _ce_memory_regs(self.inp)
        nest = 0 if clause == ExecutionClause.SEQ else self.CONF.model_max_nesting
        ex = ContractExecution(tc, mem, regs, SimArch.RVZR_ARCH_AARCH64, nest,
                               self.CONF.model_max_spec_window, req_mem_base_virt=self.base,
                               execution_clauses=clause, data_size=0)
        return sum(1 for _ in self.ce.run(ex))

    def test_shallow_chain_is_inert(self):
        from src.aarch64.aarch64_contract_executor import ExecutionClause
        depth = 2                                    # << RSB_DEPTH: every return predicted correctly
        self.assertEqual(self._steps(depth, ExecutionClause.RSB),
                         self._steps(depth, ExecutionClause.SEQ),
                         "rsb clause must not speculate when the chain fits in the RSB")

    def test_deep_chain_speculates(self):
        from src.aarch64.aarch64_contract_executor import ExecutionClause
        depth = _RSB_DEPTH + 4                        # overflows the shadow RSB -> stale-return mispredict
        self.assertGreater(self._steps(depth, ExecutionClause.RSB),
                           self._steps(depth, ExecutionClause.SEQ),
                           "rsb clause must open a misprediction window past RSB_DEPTH")


if __name__ == "__main__":
    unittest.main()
