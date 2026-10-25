"""Regression: a tagged memory access in the contract executor must leave the base register carrying
the tag the preceding (merge) ADDG produced -- it must NOT stamp the accessed cell's tag onto the base
register.

The CE runs on a host without FEAT_MTE, so it masks a pointer's tag off to form the host address for the
access. It used to then write the *cell's* tag back onto the base register (a side effect a real
load/store never has). When a later instruction reads one of that register's tag bits (e.g. a
TBZ Xn,#58 branch), the stamped tag made the CE resolve the branch differently from hardware, so an
architecturally-reached block looked speculative and its accesses were wrongly marked decoy-eligible
(violation 072421). The fix preserves the register's real tag across the access.

Device-less: drives the contract-executor binary directly. The gadget retags x0 with a merge ADDG, then
does a register-offset tagged store; x0 must come out with the ADDG's result, independent of the cell."""
import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
_CE_BIN = os.path.join(_ROOT, "src/aarch64/contract_executor/contract_executor")
_BASE = 0x00ffff8873821000   # top byte 0x00 so the tag-masked address matches the tag-memory base
_CELL = 6                    # every granule's tag


def _gadget(x0_hi48_movs, x1_hi48_movs, addg_delta):
    """A merge ADDG that retags x0, then a register-offset tagged store through [x0, x1]."""
    return "\n".join([
        ".test_case_enter:", ".section .data.main", ".function_0:", ".bb_0.0:",
        ".macro.measurement_start: NOP",
        *x0_hi48_movs, *x1_hi48_movs,
        f"ADDG x0, x0, #0, #{addg_delta}",
        "STR w2, [x0, x1]",
        "NOP",
        ".macro.measurement_end: NOP", "B .test_case_exit", ".section .data.main",
        ".test_case_exit:"]) + "\n"


@unittest.skipUnless(os.path.exists(_CE_BIN), "CE binary not built")
class MteTagPreserveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.config import CONF
        cls._saved_cwd, cls._saved_conf = os.getcwd(), copy.deepcopy(CONF._borg_shared_state)
        os.chdir(_ROOT)
        CONF.load(os.path.join(_ROOT, "config.yml"))
        cls.CONF = CONF
        from src.aarch64.aarch64_contract_executor import (ContractExecution, ContractExecutorService,
                                                           ExecutionClause, SimArch, MteProfile)
        from src.aarch64.aarch64_executor import _ce_memory_regs
        from src.aarch64.aarch64_generator import Aarch64Generator
        from src.factory import get_input_generator
        cls._CE, cls._Exec, cls._Arch = ContractExecution, ExecutionClause, SimArch
        cls._MteProfile = MteProfile
        cls._mem_regs = staticmethod(_ce_memory_regs)
        cls._asm = staticmethod(Aarch64Generator.in_memory_assemble)
        cls.svc = ContractExecutorService(_CE_BIN)
        cls.inp = get_input_generator(0).generate(1)[0]

    @classmethod
    def tearDownClass(cls):
        cls.CONF._borg_shared_state.clear()
        cls.CONF._borg_shared_state.update(cls._saved_conf)
        os.chdir(cls._saved_cwd)

    def _x0_tag_after(self, asm):
        mem, regs = self._mem_regs(self.inp)
        ex = self._CE(self._asm(asm), mem, regs, self._Arch.RVZR_ARCH_AARCH64, 0,
                      self.CONF.model_max_spec_window, req_mem_base_virt=_BASE,
                      execution_clauses=self._Exec.SEQ, mte_tags=[_CELL] * 512,
                      mte_profile=self._MteProfile(native=False), data_size=0)
        x0 = list(self.svc.run(ex))[-1].cpu.gpr[0]
        return (x0 >> 56) & 0xF

    def test_access_leaves_addg_result_not_cell(self):
        """x0 tag 10, x1 tag 5, ADDG #7 -> x0 tag (10+7)%16 = 1; the store must not stamp the cell (6)."""
        x0 = ["MOVZ x0, #0x1000", "MOVK x0, #0x7382, LSL #16", "MOVK x0, #0xff88, LSL #32",
              "MOVK x0, #0x0aff, LSL #48"]   # _BASE | (tag 10 << 56)
        x1 = ["MOVZ x1, #0x0", "MOVK x1, #0x0500, LSL #48"]   # address 0 | (tag 5 << 56)
        tag = self._x0_tag_after(_gadget(x0, x1, 7))
        self.assertEqual(tag, 1, f"x0 must carry the ADDG result 1, got {tag} "
                                 f"({'cell-tag force-match regression' if tag == _CELL else 'unexpected'})")
        self.assertNotEqual(tag, _CELL, "the access must not stamp the accessed cell's tag onto x0")

    def test_base_only_access_also_preserves_addg_result(self):
        """x0 tag 3, ADDG #0 -> x0 tag 3; a base-only tagged store must leave x0 at 3, not the cell 6."""
        x0 = ["MOVZ x0, #0x1000", "MOVK x0, #0x7382, LSL #16", "MOVK x0, #0xff88, LSL #32",
              "MOVK x0, #0x03ff, LSL #48"]   # _BASE | (tag 3 << 56)
        x1 = ["MOVZ x1, #0x0"]
        tag = self._x0_tag_after(_gadget(x0, x1, 0))
        self.assertEqual(tag, 3, f"x0 must stay at its own tag 3, got {tag}")


if __name__ == "__main__":
    unittest.main()
