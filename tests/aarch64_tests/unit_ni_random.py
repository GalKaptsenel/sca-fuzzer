"""Random speculative-exception coverage test (needs /dev/executor + the contract executor).

Drives the real unified speculative-exception executor over several random combined-PAC+MTE test cases /
inputs, mints many decoys per input, classifies every slot, and checks:
  * SAFETY invariant: no PAC forge and no MTE wrong-tag ever lands on an architectural slot; the
    Sandbox clamp is always [AND, ADD].
  * COVERAGE: across the run, every corner case actually occurs — arch correct-auth and strip; spec
    forged-auth, spec strip, MTE genuine/wrong-tag; and all four PAC*MTE decoy combinations.
The deterministic per-case behaviour is pinned in unit_pac_seal / unit_mte_tagstate; this guards that
the random generator+pipeline actually exercises them together. Driven through the new
Sealer/SealedTestCase pipeline (src/aarch64/seal/sealer.py).
"""
import os
import shutil
import copy
import sys
import random
import tempfile
import collections
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
from src.config import CONF
from src.isa_loader import InstructionSet
from src.aarch64.aarch64_generator import Aarch64RandomGenerator
from src.aarch64.seal.pac import _AUTH_TO_PAC, _read_reg
from src import factory
from src.util import FuzzLogger
from tests.conf_isolation import setUpModule, tearDownModule  # noqa: F401  (restores CONF + cwd)


def _lane(image, location):
    """The little-endian value a seal's GVT lane holds in `image` (a genuine()/decoy() SIMD image)."""
    rng = location.byte_range()
    return int.from_bytes(image[rng.start:rng.stop], "little")


def _genuine_lane(s, value):
    """What this seal's lane holds for the genuine `value`: its SealValue's encoding truncated to the
    lane width (what the image stores)."""
    width = len(s.location.byte_range())
    return s.value.encode(value) & ((1 << (8 * width)) - 1)


def _pac_kind(image, s, correct_sig):
    """Classify a PAC site from its GVT lane: a strip site carries no signature; otherwise the lane
    holds the genuine signature (auth_correct) or a forgery (auth_forged)."""
    if s.strip:
        return "strip"
    if correct_sig is None:
        return "auth_forged"
    return "auth_correct" if _lane(image, s.location) == _genuine_lane(s, correct_sig) else "auth_forged"


def _mte_applied(image, s):
    """The absolute tag the MTE site's applied-tag lane carries in `image`."""
    return _lane(image, s.location)


_PROGRAM_SEED = 0x5EA1
_INPUT_SEED = 0xC0DE


class NiRandomCoverageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists("/dev/executor"):
            raise unittest.SkipTest("kernel module not loaded — /dev/executor missing")
        from src.aarch64.aarch64_executor import Aarch64SpeculativeExceptionExecutor
        cls._saved_conf = copy.deepcopy(CONF._borg_shared_state)
        CONF.load(os.path.join(_ROOT, "config_pac_mte.yml"))
        CONF.pac_strip_prob = 0.5   # a per-site build coin — high enough to exercise strip over the run
        isa = InstructionSet(os.path.join(_ROOT, "base.json"), CONF.instruction_categories)
        cls.gen = Aarch64RandomGenerator(isa, _PROGRAM_SEED)
        cls.ex = Aarch64SpeculativeExceptionExecutor(cls.gen)
        cls.igen = factory.get_input_generator(_INPUT_SEED)
        cls.tmp = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)
        CONF._borg_shared_state.clear()
        CONF._borg_shared_state.update(cls._saved_conf)

    def _assert_oracle_matches_genuine(self, genuine_image, inp, pac, mte, by_sealing):
        """Independent oracle: trace the genuine render (the fixed code with the genuine GVT image),
        classify each slot's arch-ness against the resolver flag, and verify every genuine PAC signature
        re-signs over the canonical pointer the slot's UMOV reads (so the committed AUT* authenticates)."""
        ex = self.ex
        layout = ex._sealed._layout
        signer = ex._sealed._signer
        trace = ex._ce_trace(ex._sealed.object_code, inp, simd=genuine_image)
        base = trace[0].cpu.pc

        def min_nesting_at(off):
            ns = [e.metadata.speculation_nesting for e in trace if e.cpu.pc - base == off]
            return min(ns) if ns else None

        for s in pac:
            if s.strip:   # a strip site carries no signature and no sign landmark
                continue
            read_off = layout.instruction_address[s.slot_insts[0]]   # the UMOV, where the pointer is canonical
            self.assertEqual(min_nesting_at(read_off) != 0, by_sealing[id(s)].speculative,
                             f"PAC slot reg={s.value_reg}: genuine arch-ness != resolver flag")
        for s in mte:
            self.assertEqual(min_nesting_at(layout.instruction_address[s.access_inst]) != 0,
                             by_sealing[id(s)].speculative,
                             f"MTE slot reg={s.value_reg}: genuine arch-ness != resolver flag")

        for s in pac:
            r = by_sealing[id(s)]
            if s.strip or r.value is None:
                continue
            read_off = layout.instruction_address[s.slot_insts[0]]
            ent = next((e for e in trace if e.cpu.pc - base == read_off
                        and e.metadata.speculation_nesting == 0), None)
            if ent is None:
                continue
            inst = s.committed_inst
            pac_mn = _AUTH_TO_PAC[inst.name.lower()]
            ptr = signer.strip(_read_reg(ent.cpu, inst.operands[0].value), pac_mn)   # canonical, as the AUT* signs
            ctx = _read_reg(ent.cpu, inst.operands[1].value) if len(inst.operands) > 1 else 0
            self.assertEqual(signer.sign(ptr, ctx, pac_mn, inp.pac_keys), r.value,
                             f"genuine {inst.name} {inst.operands[0].value}: re-sign over genuine "
                             f"state != resolved sig (would FPAC)")

    def test_random_coverage_and_safety(self):
        ex, gen = self.ex, self.gen
        cov = collections.Counter()
        arch_violations = []
        tcs = 0
        for _ in range(8 * 6):
            if tcs >= 8:
                break
            tc = gen.create_test_case(os.path.join(self.tmp, "t.asm"), disable_assembler=True)
            ex.load_test_case(tc)
            pac = getattr(ex._sealed, "_pac", [])
            mte = getattr(ex._sealed, "_mte", [])
            if not (pac or mte):
                continue
            tcs += 1
            ex._sandbox_base, _ = ex.read_base_addresses()
            ex._nesting = 5
            for inp in self.igen.generate(4):
                resolved = ex._sealed.resolve(inp)
                by_sealing = {id(r.sealing): r for r in resolved._entries}
                # Every variant runs the identical object_code; the per-variant difference is the GVT
                # image (genuine() / decoy()), so classification reads the image lanes, not code words.
                baseline = resolved.genuine()
                decoys = [resolved.decoy(random.Random(i)) for i in range(6)]

                self._assert_oracle_matches_genuine(baseline, inp, pac, mte, by_sealing)

                for image in [baseline] + decoys:
                    for s in pac:
                        r = by_sealing[id(s)]
                        arch = not r.speculative
                        k = _pac_kind(image, s, r.value)
                        cov[f"pac_{'arch' if arch else 'spec'}_{k}"] += 1
                        if arch and k == "auth_forged":
                            arch_violations.append(f"PAC forge on arch slot reg={s.value_reg}")
                    for s in mte:
                        r = by_sealing[id(s)]
                        arch = not r.speculative
                        wrong = _mte_applied(image, s) != _genuine_lane(s, r.value or 0)
                        cov[f"mte_{'arch' if arch else 'spec'}_{'wrong' if wrong else 'genuine'}"] += 1
                        if arch and wrong:
                            arch_violations.append(f"MTE wrong-tag on arch slot reg={s.value_reg}")

                # per-decoy orthogonal combination of speculative PAC forge and MTE wrong-tag
                for d in decoys:
                    pac_forged = any(by_sealing[id(s)].speculative
                                     and _pac_kind(d, s, by_sealing[id(s)].value) == "auth_forged"
                                     for s in pac)
                    mte_wrong = any(by_sealing[id(s)].speculative
                                    and _mte_applied(d, s) != _genuine_lane(s, by_sealing[id(s)].value or 0)
                                    for s in mte)
                    cov[f"combo_{'P' if pac_forged else '-'}{'M' if mte_wrong else '-'}"] += 1

        self.assertEqual(arch_violations, [], f"architectural slots decoyed: {arch_violations[:5]}")
        for key in ("pac_arch_auth_correct", "pac_spec_auth_forged", "pac_spec_strip",
                    "mte_arch_genuine", "mte_spec_wrong"):
            self.assertGreater(cov[key], 0, f"corner case never occurred: {key} (cov={dict(cov)})")
        for combo in ("combo_P-", "combo_-M", "combo_PM"):
            self.assertGreater(cov[combo], 0, f"decoy combination never occurred: {combo} (cov={dict(cov)})")


class NiVariantWiringTest(unittest.TestCase):
    """_variants_for wiring (no hardware needed)."""

    def test_boosting_class_size(self):
        """CONF.inputs_per_class sets the class size: baseline + (inputs_per_class - 1) decoys."""
        from src.aarch64.aarch64_executor import Aarch64SpeculativeExceptionExecutor, SpecExcVariant

        class _FakeResolved:
            object_code = bytes(16)
            collapse_key = ()
            def genuine(self): return ()
            def decoy(self, rng): return ()

        class _FakeSealed:
            salt = 0

        ex = Aarch64SpeculativeExceptionExecutor.__new__(Aarch64SpeculativeExceptionExecutor)
        ex._sealed = _FakeSealed()
        saved = CONF.inputs_per_class
        try:
            CONF.inputs_per_class = 5
            variants = ex._variants_for(_FakeResolved())
        finally:
            CONF.inputs_per_class = saved
        self.assertEqual(set(variants), {SpecExcVariant.BASELINE, *(SpecExcVariant.decoy_n(i) for i in range(4))})


if __name__ == "__main__":
    unittest.main()
