"""Each Sealing decides for itself how to render its slot (the strip policy lives in the sealing, not
in ResolvedSealingTestCase). Verifies:
  - PacSealing.seal(sig) emits [MOVK, AUT*], or [MOVK, XPAC*] (strip) with probability
    ~CONF.pac_strip_prob; seal(None) is the all-NOP placeholder (no XPAC — the pointer stays the
    canonical clamped base), and no PAC* is ever emitted.
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
from src.aarch64.seal.pac import PacSign, build_pac_specs
from src.aarch64.seal.sealer import PacSealing, SandboxSealing, MteSealing
from tests.conf_isolation import setUpModule, tearDownModule  # noqa: F401  (restores CONF + cwd)

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


def _names(insts):
    return [i.name.lower() for i in insts]


@unittest.skip("PAC/MTE seals pending GlobalValueTable migration")
class SealingStripPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        CONF.load(os.path.join(_ROOT, "config_pac_mte_basic.yml"))
        isa = InstructionSet(os.path.join(_ROOT, "base.json"), CONF.instruction_categories)
        gen = Aarch64RandomGenerator(isa, 0x1234)
        _, auth_specs, xpac_specs = build_pac_specs(gen)
        cls.enc = PacSign(gen, auth_specs, xpac_specs, 0xFFFF << 48)

    def _pac_sealing(self, value_reg="x0", ctx_reg="x1") -> PacSealing:
        inst = Instruction("autia", True, "", False)
        inst.operands = [RegisterOperand(value_reg, 64, True, True),
                         RegisterOperand(ctx_reg, 64, True, False)]
        return PacSealing(value_reg, inst, self.enc)

    def test_pac_seal_none_is_inert(self):
        # The placeholder does nothing: it leaves the pointer the canonical clamped base, so it is pure
        # NOPs (no auth, no strip, no sign). The genuine render overwrites it via genuine() relocations.
        ps = self._pac_sealing()
        self.assertTrue(all(n == "nop" for n in _names(ps.seal(None, None))))

    def test_pac_never_emits_pac_star(self):
        # PAC* is a generation-only instruction; the instrumentation only loads a precomputed signature
        # (MOVK) and authenticates/strips it (AUT*/XPAC) — it never re-signs.
        ps = self._pac_sealing()
        rng = random.Random(7)
        for v in (None, 0x1234, 0xABCD):
            self.assertFalse(any(n.startswith("pac") for n in _names(ps.seal(v, rng))))

    def test_pac_seal_sig_strips_at_policy_rate(self):
        ps = self._pac_sealing()
        saved = CONF.pac_strip_prob
        try:
            CONF.pac_strip_prob = 0.1
            rng = random.Random(0)
            n = 4000
            strips = sum(1 for _ in range(n) if _names(ps.seal(0x1234, rng))[-1] in ("xpaci", "xpacd"))
            rate = strips / n
            # both branches must occur, and the strip rate must track CONF.pac_strip_prob
            self.assertGreater(strips, 0)
            self.assertLess(strips, n)
            self.assertAlmostEqual(rate, CONF.pac_strip_prob, delta=0.05)
        finally:
            CONF.pac_strip_prob = saved

    def test_pac_seal_sig_is_movk_then_auth_or_xpac(self):
        ps = self._pac_sealing()
        rng = random.Random(1)
        for _ in range(200):
            out = _names(ps.seal(0x1234, rng))
            self.assertEqual(out[0], "movk")             # the signature is always loaded
            if out[-1] not in ("xpaci", "xpacd"):        # not a strip -> a real auth
                self.assertEqual(out, ["movk", "autia"])

    def test_sandbox_never_strips(self):
        sb = SandboxSealing("x0", "#0x1fff", "x29")
        for v in (None, 0, 0x1234):
            self.assertEqual(_names(sb.seal(v, None)), ["and", "add"])

    def test_mte_retag_and_reconverge_is_addg_only(self):
        # MTE seal: ADDG the applied delta before the access, ADDG the complement back to the cell tag
        # after it (ADDG #0 for genuine, non-zero for a decoy). Placeholder is two NOPs; no XPAC, no LDG.
        access = Instruction("ldr", True, "", False)
        mte = MteSealing("x0", access, base_preserved=True)

        def deltas(insts):
            return [int(i.operands[3].value) for i in insts]  # the uimm4 tag delta of each ADDG

        self.assertEqual(_names(mte.seal(None, None)), ["nop", "nop"])
        self.assertEqual(_names(mte.seal(5, None, correct=5)), ["addg", "addg"])   # genuine
        self.assertEqual(deltas(mte.seal(5, None, correct=5)), [5, 0])             # before 5, back 0
        self.assertEqual(_names(mte.seal(8, None, correct=5)), ["addg", "addg"])   # decoy
        self.assertEqual(deltas(mte.seal(8, None, correct=5)), [8, 13])            # before 8, back 5-8 mod16


if __name__ == "__main__":
    unittest.main()
