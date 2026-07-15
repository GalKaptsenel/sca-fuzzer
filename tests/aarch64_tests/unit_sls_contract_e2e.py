"""End-to-end: the generator's unreachable flows (unreachable_bb_probability) combined with the sls
contract clause. A flow spliced after a non-last BB is architecturally unreachable, so the `seq`
contract never executes it (no speculation); the `sls` contract straight-line-speculates past the
parent's branch into the flow, executing its instructions — including loads — speculatively. Generated
programs are traced through the real contract executor under both contracts and compared.

Gated on the CE binary + the aarch64 cross-assembler being available (no /dev needed for contract
tracing). Run from any cwd."""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
_CE = os.path.join(_ROOT, "src/aarch64/contract_executor/contract_executor")
_BASE = 0x200000


def _runnable():
    return (os.path.exists(_CE) and shutil.which("aarch64-linux-gnu-as")
            and shutil.which("aarch64-linux-gnu-objcopy"))


@unittest.skipUnless(_runnable(), "needs the built CE binary + aarch64 cross-assembler")
class SlsContractE2ETest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.chdir(_ROOT)
        from src.config import CONF
        CONF.load("config.yml")
        from src.isa_loader import InstructionSet
        from src.factory import get_program_generator
        from src.aarch64.aarch64_generator import Aarch64SandboxPass
        from src.aarch64.aarch64_input_generator import AArch64InputGenerator
        from src.aarch64.aarch64_contract_executor import (ContractExecutorService, ContractExecution,
                                                           SimArch, ExecutionClause)
        from src.aarch64.aarch64_executor import _ce_memory_regs

        # Unreachable flows on every non-last BB; no main-flow conditional branches, so the ONLY
        # speculation source under sls is straight-line entry into a flow.
        CONF.__setattr__("unreachable_bb_probability", 1.0)
        CONF.__setattr__("max_unreachable_flow_length", 3)
        CONF.__setattr__("min_successors_per_bb", 1)
        CONF.__setattr__("max_successors_per_bb", 1)
        CONF.__setattr__("min_bb_per_function", 3)
        CONF.__setattr__("max_bb_per_function", 3)
        CONF.__setattr__("avg_mem_accesses", 16)

        isa = InstructionSet("base.json", CONF.instruction_categories)
        gen = get_program_generator(isa, 1)
        ce = ContractExecutorService(_CE)
        inputs = AArch64InputGenerator(0).generate(4)
        SEQ, SLS, ARCH = ExecutionClause.SEQ, ExecutionClause.SLS, SimArch.RVZR_ARCH_AARCH64

        def trace(tc_bytes, mem, regs, clause, nesting):
            ex = ContractExecution(tc_bytes, mem, regs, ARCH, nesting, 0,
                                   req_mem_base_virt=_BASE, execution_clauses=clause)
            return list(ce.run(ex))

        cls.seq_spec = cls.sls_spec = cls.sls_spec_mem = 0
        with tempfile.TemporaryDirectory() as d:
            asm, obj, binp = (os.path.join(d, f) for f in ("t.asm", "t.o", "t.bin"))
            for seed in range(8):
                gen._state = seed
                tc = gen.create_test_case(asm, disable_assembler=True)
                Aarch64SandboxPass().run_on_test_case(tc)
                gen.printer.print(tc, asm)
                subprocess.run(f"aarch64-linux-gnu-as -march=armv9-a+sve+memtag {asm} -o {obj}",
                               shell=True, check=True, capture_output=True)
                subprocess.run(f"aarch64-linux-gnu-objcopy -O binary {obj} {binp}",
                               shell=True, check=True, capture_output=True)
                tc_bytes = open(binp, "rb").read()
                for inp in inputs:
                    mem, regs = _ce_memory_regs(inp)
                    for e in trace(tc_bytes, mem, regs, SEQ, 0):
                        if e.metadata.speculation_nesting > 0:
                            cls.seq_spec += 1
                    for e in trace(tc_bytes, mem, regs, SLS, 4):
                        if e.metadata.speculation_nesting > 0:
                            cls.sls_spec += 1
                            if e.metadata.has_memory_access:
                                cls.sls_spec_mem += 1

    def test_seq_never_speculates(self):
        # The seq contract models no speculation, so the unreachable flow is entirely invisible to it.
        self.assertEqual(self.seq_spec, 0, "seq produced speculative trace entries")

    def test_sls_speculates_into_the_flow(self):
        self.assertGreater(self.sls_spec, 0, "sls executed no speculative instructions")

    def test_sls_leaks_flow_memory_accesses(self):
        # The detectable signal: the flow's loads run speculatively under sls (but never under seq).
        self.assertGreater(self.sls_spec_mem, 0, "sls saw no speculative memory access from the flow")


if __name__ == "__main__":
    unittest.main()
