"""Execution-based sandbox guarantee: run generated memory test cases through the contract executor
and assert EVERY memory access lands inside the sandbox (EA - x29 in [0, mask]) and that the accessed
offsets are spread across the region (not stuck at 0 / max). Gated on the CE being runnable (it needs
/dev/executor); skipped otherwise.

Run from any cwd (path bootstrap from __file__)."""
import os
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")

MASK = 0x1fff          # sandbox is main(4K)+faulty(4K); EA-x29 must lie in [0, 0x1fff]
BASE = 0x200000        # arbitrary page-aligned sandbox base passed to the CE


def _ce_runnable():
    """The CE opens /dev/executor at startup; if that fails it aborts. Probe cheaply."""
    return os.path.exists("/dev/executor")


@unittest.skipUnless(_ce_runnable(), "contract executor needs /dev/executor (kernel module not loaded)")
class SandboxStaysInRegionTest(unittest.TestCase):
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
        cls._ContractExecution = ContractExecution
        cls._SimArch, cls._SEQ, cls._mem_regs = SimArch, ExecutionClause.SEQ, _ce_memory_regs

        isa = InstructionSet("base.json", CONF.instruction_categories)
        gen = get_program_generator(isa, 1)
        ce = ContractExecutorService("src/aarch64/contract_executor/contract_executor")
        inputs = AArch64InputGenerator(0).generate(15)

        cls.offsets, cls.out_of_sandbox = [], []
        for seed in range(40):
            gen._state = seed
            tc = gen.create_test_case("/tmp/_sbce.asm", disable_assembler=True)
            Aarch64SandboxPass().run_on_test_case(tc)
            gen.printer.print(tc, "/tmp/_sbce.asm")
            subprocess.run("aarch64-linux-gnu-as -march=armv9-a+sve+memtag /tmp/_sbce.asm -o /tmp/_sbce.o",
                           shell=True, check=True, capture_output=True)
            subprocess.run("aarch64-linux-gnu-objcopy -O binary /tmp/_sbce.o /tmp/_sbce.bin",
                           shell=True, check=True, capture_output=True)
            tc_bytes = open("/tmp/_sbce.bin", "rb").read()
            for inp in inputs:
                mem, regs = cls._mem_regs(inp)
                ex = cls._ContractExecution(tc_bytes, mem, regs, cls._SimArch.RVZR_ARCH_AARCH64, 0, 0,
                                            req_mem_base_virt=BASE, execution_clauses=cls._SEQ)
                for ite in ce.run(ex):
                    if ite.metadata.has_memory_access:
                        off = ite.metadata.memory_access.effective_address - BASE
                        cls.offsets.append(off)
                        if not (0 <= off <= MASK):
                            cls.out_of_sandbox.append(off)

    def test_every_access_is_inside_the_sandbox(self):
        self.assertGreater(len(self.offsets), 0, "no memory accesses observed")
        self.assertEqual(self.out_of_sandbox, [], "memory accesses escaped the sandbox")

    def test_offsets_are_spread_not_stuck(self):
        # the masking must NOT collapse every access to a single offset (e.g. always 0 or always max)
        lines = {o // 64 for o in self.offsets}
        self.assertGreater(len(lines), 16, "accessed offsets are not spread across the sandbox")


@unittest.skipUnless(_ce_runnable(), "contract executor needs /dev/executor (kernel module not loaded)")
class SandboxOffsetInvariantTest(unittest.TestCase):
    """A masked access lands at exactly x29 + (base & mask): run a minimal `LDR X1, [X0]` (with the
    sandbox prologue) and check EA - x29 == X0 & mask for several base values, incl. out-of-range ones."""

    def test_effective_address_equals_masked_base(self):
        import struct
        os.chdir(_ROOT)
        from src.aarch64.aarch64_input_generator import AArch64InputGenerator
        from src.aarch64.aarch64_contract_executor import (ContractExecutorService, ContractExecution,
                                                           SimArch, ExecutionClause)
        from src.aarch64.aarch64_executor import _ce_memory_regs
        open("/tmp/_sbinv.s", "w").write("AND x0, x0, #0x1fff\nADD x0, x0, x29\nLDR x1, [x0]\nRET\n")
        subprocess.run("aarch64-linux-gnu-as -march=armv9-a /tmp/_sbinv.s -o /tmp/_sbinv.o",
                       shell=True, check=True, capture_output=True)
        subprocess.run("aarch64-linux-gnu-objcopy -O binary /tmp/_sbinv.o /tmp/_sbinv.bin",
                       shell=True, check=True, capture_output=True)
        tc = open("/tmp/_sbinv.bin", "rb").read()
        ce = ContractExecutorService("src/aarch64/contract_executor/contract_executor")
        mem, regs = _ce_memory_regs(AArch64InputGenerator(0).generate(1)[0])
        regs = bytearray(regs)
        for R in [0, 0x1234, 0x1fff, 0x2000, 0xdeadbeef, 0xffffffffffffffff]:
            struct.pack_into("<Q", regs, 0, R)        # x0 = R (first GPR slot)
            ex = ContractExecution(tc, mem, bytes(regs), SimArch.RVZR_ARCH_AARCH64, 0, 0,
                                   req_mem_base_virt=BASE, execution_clauses=ExecutionClause.SEQ)
            ea = next(ite.metadata.memory_access.effective_address
                      for ite in ce.run(ex) if ite.metadata.has_memory_access)
            self.assertEqual(ea - BASE, R & MASK, f"x0={R:#x}")


if __name__ == "__main__":
    unittest.main()
