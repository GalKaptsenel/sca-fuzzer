"""Regression: the generator never emits a context AUT* (autia/autib/autda/autdb) whose context
register equals its pointer register. A self-context AUT* is an unsatisfiable PAC sequence that
FPAC-faults at the seal's genuine auth; the generator's patch pass (_patch_auth_context_collision)
must keep ctx != ptr in the program body."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")

from src.config import CONF
from src.isa_loader import InstructionSet
from src.aarch64.aarch64_generator import Aarch64RandomGenerator, _AUTH_CTX
from src.aarch64.aarch64_target_desc import Aarch64TargetDesc


class TestGeneratorAuthContext(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        CONF.load(os.path.join(_ROOT, "config_pac_mte_basic.yml"))
        cls.isa = InstructionSet(os.path.join(_ROOT, "base.json"), CONF.instruction_categories)
        cls.tmp = tempfile.mkdtemp()
        cls.norm = Aarch64TargetDesc.reg_normalized

    def test_body_auth_never_self_context(self):
        found = 0
        for seed in range(200):
            gen = Aarch64RandomGenerator(self.isa, seed)
            try:
                tc = gen.create_test_case(os.path.join(self.tmp, "t.asm"), disable_assembler=True)
            except Exception:
                continue
            for func in tc.functions:
                for bb in func:
                    for inst in bb:
                        if inst.name.lower() not in _AUTH_CTX or len(inst.operands) < 2:
                            continue
                        found += 1
                        ptr = self.norm.get(inst.operands[0].value, inst.operands[0].value)
                        ctx = self.norm.get(inst.operands[1].value, inst.operands[1].value)
                        with self.subTest(seed=seed, mnemonic=inst.name):
                            self.assertNotEqual(ptr, ctx, f"seed={seed} {inst.name}: ctx == ptr == {ptr}")
        self.assertGreater(found, 0, "no context AUT* generated across 200 seeds")


if __name__ == "__main__":
    unittest.main()
