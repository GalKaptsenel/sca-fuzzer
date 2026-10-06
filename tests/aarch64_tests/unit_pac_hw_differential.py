"""The software PAC model against the real CPU: random pointers (canonical and non-canonical, both VA
halves, tagged and untagged) signed by every AddPAC mnemonic and stripped by XPACI/XPACD through the
executor's REVISOR_PAC_SIGN / REVISOR_PAC_XPAC ioctls (the real instructions at EL1, under the device's
TCR_EL1), compared bit-for-bit with aarch64_qarma under the profile the device reports. Then a correctly
signed pointer must AUTH on hardware back to the model's Strip. Hardware-required."""
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.aarch64 import aarch64_qarma as q

_ADDPAC = ("pacia", "pacib", "pacda", "pacdb", "paciza", "pacizb", "pacdza", "pacdzb")
_N = 400


def _pointers(rng: random.Random):
    """Edge-dense pointer mix: canonical both halves, every bit63/bit55 combination, random tags."""
    for _ in range(_N):
        kind = rng.randrange(5)
        v = rng.randrange(1 << 64)
        if kind == 0:
            yield v & ((1 << 48) - 1)                              # canonical low
        elif kind == 1:
            yield v | ~((1 << 48) - 1) & ((1 << 64) - 1)            # canonical high
        elif kind == 2:
            yield (v & ~(1 << 63) | (1 << 55)) & ((1 << 64) - 1)    # bit63=0, bit55=1
        elif kind == 3:
            yield (v | (1 << 63)) & ~(1 << 55)                      # bit63=1, bit55=0
        else:
            yield v                                                 # anything


@unittest.skipUnless(os.path.exists("/dev/executor"), "kernel module not loaded")
class PacHwDifferentialTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.aarch64.aarch64_kernel import LocalHWExecutor, PacKeys
        cls.dev = LocalHWExecutor("/dev/executor", "/sys/executor")
        i = cls.dev.target_info()
        regs = q.decode_registers(i.tcr_el1, i.id_aa64isar1_el1, i.id_aa64isar2_el1)
        cls.profile = q.profile(**regs._asdict())
        rng = random.Random(0xC0FFEE)
        cls.keys = PacKeys(*(rng.randrange(1 << 64) for _ in range(10)))

    def setUp(self):
        self.rng = random.Random(0xBEEF)   # per test: its pointers don't depend on which tests ran before

    def test_addpac_matches_hardware(self):
        mismatches = []
        for mn in _ADDPAC:
            for ptr in _pointers(self.rng):
                ctx = 0 if "z" in mn[3:] else self.rng.randrange(1 << 64)
                hw = self.dev.pac_sign(ptr, ctx, mn, self.keys)
                sw = q.sign(ptr, ctx, mn, self.keys.words(), self.profile)
                if hw != sw:
                    mismatches.append(f"{mn} ptr={ptr:#018x} ctx={ctx:#018x} hw={hw:#018x} sw={sw:#018x}")
        self.assertEqual(mismatches, [], "\n".join(mismatches[:20]))

    def test_strip_matches_hardware(self):
        mismatches = []
        for mn, is_instr in (("xpaci", True), ("xpacd", False)):
            for ptr in _pointers(self.rng):
                hw = self.dev.pac_xpac(ptr, mn)
                sw = q.strip(ptr, self.profile, is_instr)
                if hw != sw:
                    mismatches.append(f"{mn} ptr={ptr:#018x} hw={hw:#018x} sw={sw:#018x}")
        self.assertEqual(mismatches, [], "\n".join(mismatches[:20]))

    def test_model_signed_pointer_verifies_on_hardware(self):
        # A real AUT* passes iff re-signing its hardware Strip reproduces the pointer (PAuth2 Auth).
        mismatches = []
        for mn in _ADDPAC:
            xpac = "xpaci" if q.is_instr_key(mn) else "xpacd"
            for ptr in _pointers(self.rng):
                ctx = 0 if "z" in mn[3:] else self.rng.randrange(1 << 64)
                signed = q.sign(q.strip(ptr, self.profile, q.is_instr_key(mn)), ctx, mn,
                                self.keys.words(), self.profile)
                resigned = self.dev.pac_sign(self.dev.pac_xpac(signed, xpac), ctx, mn, self.keys)
                if resigned != signed:
                    mismatches.append(f"{mn} signed={signed:#018x} hw-resigned={resigned:#018x}")
        self.assertEqual(mismatches, [], "\n".join(mismatches[:20]))


if __name__ == "__main__":
    unittest.main()
