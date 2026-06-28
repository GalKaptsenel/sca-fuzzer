"""Each Sealing decides for itself how to render its slot (the strip policy lives in the sealing, not
in ResolvedSealingTestCase). Verifies:
  - PacSealing.seal(sig) emits [MOVK, AUT*] most of the time and the arch-safe strip [NOP, XPAC*] with
    probability ~PacSealing._STRIP_PROB; seal(None) is ALWAYS the strip.
  - SandboxSealing.seal never strips (always [AND, ADD], value-independent).
  - MteSealing.seal never emits XPAC ([NOP] for 0/None, [ADDG] otherwise).
No kernel module needed — seal() only emits instructions.
"""
import os, sys, random, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config import CONF
from src.isa_loader import InstructionSet
from src.interfaces import Instruction, RegisterOperand
from src.aarch64.aarch64_generator import Aarch64RandomGenerator
from src.aarch64.aarch64_pac import PacSign, build_pac_specs
from src.aarch64.aarch64_sealer import PacSealing, SandboxSealing, MteSealing

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


def _names(insts):
    return [i.name.lower() for i in insts]


class SealingStripPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        CONF.load(os.path.join(_ROOT, "config_pac_mte_basic.yml"))
        isa = InstructionSet(os.path.join(_ROOT, "base.json"), CONF.instruction_categories)
        gen = Aarch64RandomGenerator(isa, 0x1234)
        _, auth_specs, xpac_specs = build_pac_specs(gen)
        cls.enc = PacSign(gen, auth_specs, xpac_specs)

    def _pac_sealing(self, value_reg="x0", ctx_reg="x1") -> PacSealing:
        inst = Instruction("autia", True, "", False)
        inst.operands = [RegisterOperand(value_reg, 64, True, True),
                         RegisterOperand(ctx_reg, 64, True, False)]
        return PacSealing(value_reg, inst, self.enc)

    def test_pac_seal_none_always_strips(self):
        ps = self._pac_sealing()
        for _ in range(64):
            self.assertEqual(_names(ps.seal(None)), ["nop", "xpaci"])

    def test_pac_seal_sig_strips_at_policy_rate(self):
        ps = self._pac_sealing()
        random.seed(0)
        n = 4000
        strips = sum(1 for _ in range(n) if _names(ps.seal(0x1234))[0] == "nop")
        rate = strips / n
        # both branches must occur, and the strip rate must track PacSealing._STRIP_PROB
        self.assertGreater(strips, 0)
        self.assertLess(strips, n)
        self.assertAlmostEqual(rate, PacSealing._STRIP_PROB, delta=0.05)

    def test_pac_seal_sig_nonstrip_is_movk_auth(self):
        ps = self._pac_sealing()
        random.seed(1)
        for _ in range(200):
            out = ps.seal(0x1234)
            if _names(out)[0] != "nop":
                self.assertEqual(_names(out), ["movk", "autia"])

    def test_sandbox_never_strips(self):
        sb = SandboxSealing("x0", "#0x1fff", "x29")
        for v in (None, 0, 0x1234):
            self.assertEqual(_names(sb.seal(v)), ["and", "add"])

    def test_mte_never_xpac(self):
        access = Instruction("ldr", True, "", False)
        mte = MteSealing("x0", access)
        self.assertEqual(_names(mte.seal(None)), ["nop"])
        self.assertEqual(_names(mte.seal(0)), ["nop"])
        self.assertEqual(_names(mte.seal(5)), ["addg"])


if __name__ == "__main__":
    unittest.main()
