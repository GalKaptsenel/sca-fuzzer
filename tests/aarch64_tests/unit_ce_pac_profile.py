"""The CE's PAC model end to end: the packed profile word reaches the C AddPAC intact (T0SZ != T1SZ,
TBID, non-canonical selbit case) and matches the Python model; a PAC instruction with no profile or no
keys in the request aborts the CE instead of signing with a guess. Needs only the CE binary."""
import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
_CE_BIN = os.path.join(_ROOT, "src/aarch64/contract_executor/contract_executor")
_BASE = 0xffff40004b9e1000   # any kernel-half sandbox base; no device needed


def _asm(value: int, inst: str) -> str:
    movs = [f"MOVZ x0, #0x{value & 0xffff:x}"] + \
           [f"MOVK x0, #0x{(value >> s) & 0xffff:x}, LSL #{s}" for s in (16, 32, 48)]
    return "\n".join([".test_case_enter:", ".section .data.main", ".function_0:", ".bb_0.0:",
                      ".macro.measurement_start: NOP", *movs, "MOVZ x1, #0x1234", inst, "NOP",
                      ".macro.measurement_end: NOP", "B .test_case_exit", ".test_case_exit:"]) + "\n"


@unittest.skipUnless(os.path.exists(_CE_BIN), "CE binary not built")
class CePacProfileTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.config import CONF
        cls._saved_cwd, cls._saved_conf = os.getcwd(), copy.deepcopy(CONF._borg_shared_state)
        os.chdir(_ROOT)
        CONF.load(os.path.join(_ROOT, "config.yml"))
        cls.CONF = CONF
        from src.aarch64 import aarch64_qarma as q
        from src.aarch64.aarch64_generator import Aarch64Generator
        from src.aarch64.aarch64_kernel import PacKeys
        from src.factory import get_input_generator
        cls.q, cls.Gen = q, Aarch64Generator
        cls.keys = PacKeys(*range(0x1111, 0x1111 + 10))
        cls.inp = get_input_generator(0).generate(1)[0]
        cls.profile = q.profile(3, 3, 3, 39, 48, True, True, False, True, False, False, False)

    @classmethod
    def tearDownClass(cls):
        cls.CONF._borg_shared_state.clear()
        cls.CONF._borg_shared_state.update(cls._saved_conf)
        os.chdir(cls._saved_cwd)

    def _run(self, value, inst, keys, profile):
        from src.aarch64.aarch64_contract_executor import (ContractExecution, ContractExecutorService,
                                                           ExecutionClause, SimArch)
        from src.aarch64.aarch64_executor import _ce_memory_regs
        from src.aarch64.seal.pac import _read_reg
        mem, regs = _ce_memory_regs(self.inp)
        ex = ContractExecution(self.Gen.in_memory_assemble(_asm(value, inst)), mem, regs,
                               SimArch.RVZR_ARCH_AARCH64, 0, self.CONF.model_max_spec_window,
                               req_mem_base_virt=_BASE, execution_clauses=ExecutionClause.SEQ,
                               pac_keys=keys, pac_profile=profile, data_size=0)
        ce = ContractExecutorService(_CE_BIN)
        trace = list(ce.run(ex))
        return _read_reg(trace[-1].cpu, "x0")

    def test_c_model_matches_python_through_the_wire(self):
        for value, pac in ((0x0ae500000ae50000, "PACIZB"),    # high half, bit63 != bit55, TBID1
                           (0x0000001234567000, "PACDZA"),    # low half: field [54:39] (T0SZ=25)
                           (0xffff400012345000, "PACIZA")):   # sandbox-like kernel pointer
            want = self.q.sign(value, 0, pac.lower(), self.keys.words(), self.profile)
            self.assertEqual(self._run(value, f"{pac} x0", self.keys, self.profile), want, f"{pac} {value:#x}")

    def test_pacga_uses_the_generic_algorithm(self):
        p = self.q.profile(3, 5, 3, 48, 48, True, True, False, True, False, False, False)   # APA3 + GPA
        want = self.q.computepac(0x0ae500000ae50000, 0x1234, self.keys.apga_lo, self.keys.apga_hi,
                                 4) & 0xFFFFFFFF00000000
        self.assertEqual(self._run(0x0ae500000ae50000, "PACGA x0, x0, x1", self.keys, p), want)

    def test_pacga_without_generic_algorithm_aborts(self):
        p = self.q.profile(3, 0, 3, 48, 48, True, True, False, True, False, False, False)
        with self.assertRaisesRegex(RuntimeError, r"contract_executor crashed \(exit code -6"):
            self._run(0x1000, "PACGA x0, x0, x1", self.keys, p)

    def test_pac_without_profile_or_keys_aborts(self):
        for keys, profile in ((self.keys, None), (None, self.profile)):
            with self.assertRaisesRegex(RuntimeError, r"contract_executor crashed \(exit code -6"):
                self._run(0x0000001234567000, "PACIZA x0", keys, profile)


if __name__ == "__main__":
    unittest.main()
