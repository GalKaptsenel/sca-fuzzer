"""Template test cases run through the contract executor (the CE) and produce the expected trace.

This needs only the CE binary, not the hardware executor (/dev/executor): a template test case is built,
sandboxed, assembled, and run through the CE directly. The architectural (seq) contract makes the trace
deterministic -- a fixed load is exactly one memory access.
"""
import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tests.conf_isolation import setUpModule, tearDownModule  # noqa: F401  (restores CONF + cwd)
_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
_CE_BIN = os.path.join(_ROOT, "src", "aarch64", "contract_executor", "contract_executor")
_SANDBOX_BASE = 0xffff4000cc861000     # a real sandbox base (from /sys/executor/print_sandbox_base)


def _ce_available():
    return os.path.exists(_CE_BIN)


@unittest.skipUnless(_ce_available(), "needs the contract-executor binary (build src/aarch64/contract_executor)")
class TemplateCETest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.chdir(_ROOT)
        from src.config import CONF
        CONF.load("config.yml")
        CONF.input_generator = "aarch64-nzcv"
        CONF.max_functions_per_test_case = 8
        CONF.max_calls_per_function = 1
        from src.isa_loader import InstructionSet
        cls.CONF = CONF
        cls.isa = InstructionSet("base.json", CONF.instruction_categories)

    def _ce_trace(self, template):
        from src.aarch64.template.runner import generate_test_case
        from src.aarch64.aarch64_generator import Aarch64SandboxPass
        from src.aarch64.aarch64_printer import Aarch64Printer, Aarch64ASMLayout
        from src.aarch64.aarch64_generator import Aarch64Generator
        from src.aarch64.aarch64_target_desc import Aarch64TargetDesc
        from src.aarch64.aarch64_executor import _ce_memory_regs
        from src.aarch64.aarch64_contract_executor import (
            ContractExecution, ContractExecutorService, SimArch, ExecutionClause, BranchPredictor)
        from src.factory import get_input_generator

        tc = generate_test_case(template, self.isa, 1, asm_file="/tmp/_tpl_ce.asm", assemble=False)
        patched = copy.deepcopy(tc)
        Aarch64SandboxPass().run_on_test_case(patched)         # run-time transform, applied here
        asm = Aarch64Printer(Aarch64TargetDesc()).print_layout(Aarch64ASMLayout(patched))
        code = Aarch64Generator.in_memory_assemble(asm)
        data_size = 4 * sum(f.dispatch_table.size for f in patched.functions if f.dispatch_table)

        inp = get_input_generator(1).generate(1)[0]
        memory, regs = _ce_memory_regs(inp)
        execution = ContractExecution(code, memory, regs, SimArch.RVZR_ARCH_AARCH64, 0,
                                      self.CONF.model_max_spec_window,
                                      req_mem_base_virt=_SANDBOX_BASE,
                                      execution_clauses=ExecutionClause.SEQ,
                                      branch_predictor=BranchPredictor.NONE, data_size=data_size)
        ce = ContractExecutorService(_CE_BIN)
        try:
            return list(ce.run(execution))
        finally:
            ce.stop()

    @staticmethod
    def _mem_count(trace):
        return sum(1 for ite in trace if ite.metadata.has_memory_access)

    def test_fixed_loads_are_the_only_memory_accesses(self):
        from src.aarch64.template import Template, Mem, specs
        from src.aarch64.template.objects import X0, X1, X2

        class T(Template):
            def build(self, b):
                b.instruction(specs.LDR, X0, Mem(X1))       # x0 = [x1]
                b.instruction(specs.LDR, X2, Mem(X0))       # x2 = [x0], chained

        self.assertEqual(self._mem_count(self._ce_trace(T())), 2)

    def test_branch_and_call_template_runs_and_traces_the_chained_loads(self):
        # the full shape (chained load, conditional branch, indirect call to a non-constrained callee):
        # it runs on the CE and the trace records at least the two chained loads
        from src.aarch64.template import Template, IndirectCall, Mem, specs
        from src.aarch64.template.objects import X0, X1, X2

        class T(Template):
            def build(self, b):
                b.instruction(specs.LDR, X0, Mem(X1))
                b.instruction(specs.LDR, X2, Mem(X0))
                with b.if_():
                    b.call(IndirectCall(targets=1))

        self.assertGreaterEqual(self._mem_count(self._ce_trace(T())), 2)


if __name__ == "__main__":
    unittest.main()
