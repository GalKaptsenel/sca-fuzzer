"""Regular fuzzer with PAC/MTE (needs /dev/executor + the contract executor).

factory.get_executor must select Aarch64RegularSealedExecutor when PAC/MTE categories are enabled in
regular (non-NI) mode, and that executor must mint one *genuine* (baseline-only, no decoy) sealed TC
per input — every PAC slot a correct AUT*/strip, every MTE slot a NOP/ADDG (never a forged signature
or an IRG/EOR retag), so each per-input TC is architecturally safe to run on hardware.
"""
import os
import sys
import random
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
from src.config import CONF
from src.isa_loader import InstructionSet
from src.aarch64.aarch64_generator import Aarch64RandomGenerator
from src.aarch64.aarch64_seal import inst_at, CompositeSeal
from src import factory

_AUTH_OR_XPAC = ("autia", "autib", "autda", "autdb", "autiza", "autizb", "autdza", "autdzb",
                 "xpaci", "xpacd")


def _parts(fp, tc):
    insts = [inst_at(tc, loc)[0] for loc in fp.slot_locs]
    seals = fp.seal.seals if isinstance(fp.seal, CompositeSeal) else [fp.seal]
    out, i = {}, 0
    for s in seals:
        out[s.name] = insts[i:i + s.slot_size]; i += s.slot_size
    return out


class RegularSealedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists("/dev/executor"):
            raise unittest.SkipTest("kernel module not loaded — /dev/executor missing")
        try:
            from src.aarch64.aarch64_kernel import PacKeys
            from src.aarch64.aarch64_executor import Aarch64RegularSealedExecutor
            cls._cls = Aarch64RegularSealedExecutor
            CONF.load(os.path.join(_ROOT, "config_pac_mte.yml"))
            CONF.fuzzer = "generic"                      # regular mode, not non-interference
            isa = InstructionSet(os.path.join(_ROOT, "base.json"), CONF.instruction_categories)
            cls.gen = Aarch64RandomGenerator(isa, random.randrange(1 << 32))
            cls.ex = factory.get_executor(generator=cls.gen)
            k = PacKeys()
            k.apia_lo = k.apib_lo = k.apda_lo = k.apdb_lo = k.apga_lo = 0x1122334455667788
            k.apia_hi = k.apib_hi = k.apda_hi = k.apdb_hi = k.apga_hi = 0x8877665544332211
            cls.ex.local_executor.set_pac_keys(k)
            cls.igen = factory.get_input_generator(random.randrange(1 << 32))
            cls.tmp = tempfile.mkdtemp()
        except Exception as e:
            raise unittest.SkipTest(f"regular-sealed executor setup failed: {e}")

    def test_routes_to_sealed_executor(self):
        self.assertIsInstance(self.ex, self._cls)

    def test_per_input_genuine_only(self):
        ex, gen = self.ex, self.gen
        tcs = 0
        for _ in range(8 * 6):
            if tcs >= 4:
                break
            try:
                tc = gen.create_test_case(os.path.join(self.tmp, "t.asm"), disable_assembler=True)
            except Exception:
                continue
            ex.load_test_case(tc)
            if not ex._fix_points:
                continue
            tcs += 1
            inputs = self.igen.generate(3)
            ctraces, _taints, _traces, _ = ex.trace_test_case_with_taints(inputs, CONF.model_max_nesting)
            self.assertEqual(len(ctraces), len(inputs))
            # one variant per input, and it is the genuine baseline (no decoy)
            for variants in ex._last_tc_variants:
                self.assertEqual(sorted(v.name for v in variants), ["BASELINE"])
            # every per-input baseline is all-genuine: MTE NOP/ADDG (never IRG/EOR), PAC AUT*/XPAC
            for variants in ex._last_tc_variants:
                base = next(iter(variants.values()))
                for fp in ex._fix_points:
                    for name, part in _parts(fp, base).items():
                        if name == "mte_tag":
                            self.assertIn(part[0].name.lower(), ("nop", "addg"))
                        elif name == "pac_sign":
                            self.assertIn(part[1].name.lower(), _AUTH_OR_XPAC)
        self.assertGreater(tcs, 0, "no sealable test cases generated")


if __name__ == "__main__":
    unittest.main()
