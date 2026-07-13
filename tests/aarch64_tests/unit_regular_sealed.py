"""Regular fuzzer with PAC/MTE (needs /dev/executor + the contract executor).

Verifies the routing (no PAC/MTE -> plain executor; PAC / MTE / both -> sealed executor with the
right primitives) and the per-input sealing of Aarch64RegularSealedExecutor, now driven through the
new Sealer/SealedTestCase pipeline (src/aarch64/seal/sealer.py): every architectural slot is
genuine (correct signature/tag — arch-safe); speculative slots are genuine or decoyed by a
per-sealing-class coin flip (consistent within a class, shared by every class member).
"""
import os
import sys
import random
import tempfile
import unittest
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
from src.config import CONF
from src.isa_loader import InstructionSet
from src.aarch64.aarch64_generator import Aarch64RandomGenerator
from src.aarch64.aarch64_relocations import (
    apply_relocations, read_word32, is_movk64, get_movk_imm16, is_addg, get_addg_tag, is_xpac,
    NOP_WORD)
from src import factory

_BASE = ["BASE-ARITH", "BASE-LOGICAL", "BASE-SHIFT", "BASE-BRANCH", "BASE-MEM-LOAD", "BASE-MEM-STORE"]


def _has_dev():
    return os.path.exists("/dev/executor")


def _value_entries(ex, resolved):
    """The resolved PAC/MTE entries (the value sealings, excluding the sandbox clamps)."""
    value_ids = {id(s) for s in getattr(ex._sealed, "_pac", [])} \
        | {id(s) for s in getattr(ex._sealed, "_mte", [])}
    return [r for r in resolved._entries if id(r.sealing) in value_ids]


def _slot_words(ex, variant_bytes, r):
    """The variant's machine words at this entry's slot positions (offsets from the placeholder
    layout; relocation rewrites words in place, so the offsets are variant-invariant)."""
    layout = ex._sealed._layout
    return [read_word32(variant_bytes, layout.instruction_address[i]) for i in r.sealing.slot_insts]


class RegularSealedRoutingTest(unittest.TestCase):
    """factory.get_executor selects the right executor for each category combination."""

    @classmethod
    def setUpClass(cls):
        if not _has_dev():
            raise unittest.SkipTest("kernel module not loaded — /dev/executor missing")
        from src.aarch64.aarch64_executor import (Aarch64LocalExecutor, Aarch64RegularSealedExecutor)
        cls.Local, cls.Sealed = Aarch64LocalExecutor, Aarch64RegularSealedExecutor
        CONF.load(os.path.join(_ROOT, "config_pac_mte_basic.yml"))   # fuzzer: basic

    def _executor_for(self, categories):
        CONF.instruction_categories = categories
        isa = InstructionSet(os.path.join(_ROOT, "base.json"), categories)
        gen = Aarch64RandomGenerator(isa, 0x1234)
        return factory.get_executor(generator=gen)

    def test_no_pac_mte_is_plain(self):
        ex = self._executor_for(list(_BASE))
        self.assertIs(type(ex), self.Local)

    def test_pac_only(self):
        ex = self._executor_for(["PAC"] + _BASE)
        self.assertIsInstance(ex, self.Sealed)
        self.assertEqual(ex._primitives, {"pac"})

    def test_mte_only(self):
        ex = self._executor_for(["MTE", "MTE-TAG-MEM"] + _BASE)
        self.assertIsInstance(ex, self.Sealed)
        self.assertEqual(ex._primitives, {"mte"})

    def test_pac_and_mte(self):
        ex = self._executor_for(["PAC", "MTE", "MTE-TAG-MEM"] + _BASE)
        self.assertIsInstance(ex, self.Sealed)
        self.assertEqual(ex._primitives, {"pac", "mte"})

    def test_mte_tags_fed_only_when_mte_active(self):
        """The sandbox is loaded with the uniform initial tag exactly when MTE is a live primitive;
        PAC-only never tags (so the kernel/CE leave tag memory untouched)."""
        from src.interfaces import Input
        from src.aarch64.aarch64_mte import MTE_INITIAL_DEFAULT_TAG
        from src.aarch64.aarch64_executor_input_encoder import MTE_TAG_COUNT
        inp = Input()
        self.assertIsNone(self._executor_for(["PAC"] + _BASE)._mte_tags_for(inp))
        tags = self._executor_for(["MTE", "MTE-TAG-MEM"] + _BASE)._mte_tags_for(inp)
        self.assertEqual(tags, [MTE_INITIAL_DEFAULT_TAG] * MTE_TAG_COUNT)
        self.assertEqual(self._executor_for(["PAC", "MTE", "MTE-TAG-MEM"] + _BASE)._mte_tags_for(inp),
                         [MTE_INITIAL_DEFAULT_TAG] * MTE_TAG_COUNT)


class RegularSealedSealingTest(unittest.TestCase):
    """Per-input sealing correctness, driving the real seal/resolve pipeline (no hardware step)."""

    @classmethod
    def setUpClass(cls):
        if not _has_dev():
            raise unittest.SkipTest("kernel module not loaded — /dev/executor missing")
        from src.aarch64.aarch64_kernel import PacKeys
        cls.PacKeys = PacKeys
        CONF.load(os.path.join(_ROOT, "config_pac_mte_basic.yml"))
        cls.isa = InstructionSet(os.path.join(_ROOT, "base.json"), CONF.instruction_categories)

    def _executor(self):
        gen = Aarch64RandomGenerator(self.isa, random.randrange(1 << 32))
        ex = factory.get_executor(generator=gen)
        type(self)._last_ex = ex
        return ex, gen

    def _has_value_slots(self, ex) -> bool:
        return bool(getattr(ex._sealed, "_pac", []) or getattr(ex._sealed, "_mte", []))

    def _assert_arch_genuine(self, ex, resolved, tc, violations):
        """Every architectural value slot in `tc` must carry the correct value (never a forgery)."""
        for r in _value_entries(ex, resolved):
            if r.speculative:
                continue
            words = _slot_words(ex, tc, r)
            if is_movk64(words[0]):                               # PAC: genuine sig or violation
                if get_movk_imm16(words[0]) != (r.value & 0xFFFF if r.value is not None else None):
                    violations.append(f"forged PAC on arch slot reg={r.sealing.value_reg}")
            elif is_addg(words[0]):                               # MTE: genuine delta or violation
                if get_addg_tag(words[0]) != ((r.value or 0) % 16):
                    violations.append(f"forged MTE retag on arch slot reg={r.sealing.value_reg}")
            # nop / xpac heads are arch-safe genuine encodings (strip / no-retag)

    def _accumulate_coverage(self, ex, resolved, tc, cov):
        for r in _value_entries(ex, resolved):
            words = _slot_words(ex, tc, r)
            if not r.speculative:
                cov["arch_genuine"] += 1
                continue
            if words[0] == NOP_WORD and is_xpac(words[-1]):
                cov["spec_strip"] += 1
            elif is_movk64(words[0]) and get_movk_imm16(words[0]) != (r.value & 0xFFFF if r.value is not None else -1):
                cov["spec_decoy"] += 1
            elif is_addg(words[0]) and get_addg_tag(words[0]) != ((r.value or 0) % 16):
                cov["spec_decoy"] += 1
            elif is_addg(words[0]) or words[0] == NOP_WORD:
                # a speculative MTE slot may legitimately resolve to an alt that equals the genuine
                # encoding (delta 0 -> nop); not counted as a distinct decoy.
                pass

    def test_arch_genuine_spec_decoy_with_corner_cases(self):
        ex, gen = self._executor()
        ig = factory.get_input_generator(random.randrange(1 << 32)); tmp = tempfile.mkdtemp()
        cov = {"arch_genuine": 0, "spec_decoy": 0, "spec_strip": 0}
        violations = []
        tcs = 0
        for _ in range(8 * 6):
            if tcs >= 5:
                break
            try:
                tc = gen.create_test_case(os.path.join(tmp, "t.asm"), disable_assembler=True)
            except Exception:
                continue
            ex.load_test_case(tc)
            if not self._has_value_slots(ex):
                continue
            tcs += 1
            ex._sandbox_base, _ = ex.read_base_addresses()
            for inp in ig.generate(3):
                resolved = ex._sealed.resolve(inp)               # fresh per-input resolve
                for i in range(8):                               # many minted decoys (distinct seeds)
                    decoy = apply_relocations(resolved.object_code, resolved.decoy(random.Random(i)))
                    self._assert_arch_genuine(ex, resolved, decoy, violations)
                    self._accumulate_coverage(ex, resolved, decoy, cov)
        self.assertEqual(violations, [], f"arch slots not genuine: {violations[:5]}")
        self.assertGreater(tcs, 0, "no sealable test cases generated")
        self.assertGreater(cov["arch_genuine"], 0, f"no arch slots seen ({cov})")
        self.assertGreater(cov["spec_decoy"], 0, f"no speculative decoy seen ({cov})")

    def _seal_a_tc(self, ex, gen, n_inputs=4):
        ig = factory.get_input_generator(random.randrange(1 << 32)); tmp = tempfile.mkdtemp()
        for _ in range(8 * 6):
            try:
                tc = gen.create_test_case(os.path.join(tmp, "t.asm"), disable_assembler=True)
            except Exception:
                continue
            ex.load_test_case(tc)
            if not self._has_value_slots(ex):
                continue
            inputs = ig.generate(n_inputs)
            ctraces, _t, _tr = ex.trace_test_case_with_taints(inputs, CONF.model_max_nesting)
            return inputs, ctraces
        self.skipTest("no sealable test case generated")

    def test_one_tc_per_input_and_merged_by_class(self):
        """caveat-4 merge: same-class inputs (equal collapse_key) get byte-identical kernel input
        files with no shared state — the determinism of _seal_input, not a cache, guarantees it."""
        ex, gen = self._executor()
        inputs, ctraces = self._seal_a_tc(ex, gen)
        self.assertEqual(len(ctraces), len(inputs))
        by_class = defaultdict(list)
        for inp in inputs:
            by_class[ex._sealed.resolve(inp).collapse_key].append(ex._seal_input(inp).code_reloc)
        for plans in by_class.values():
            self.assertTrue(all(p == plans[0] for p in plans),
                            "same-class inputs must run the identical program")

    def test_forced_genuine_or_decoy(self):
        """The per-class coin is _DECOY_PROB: 0 forces genuine, 1 forces the class decoy; both a pure
        function of the sealing class + salt, so _seal_input is stable across calls."""
        for prob in (0.0, 1.0):
            ex, gen = self._executor()
            ex._DECOY_PROB = prob                                     # override the per-class coin
            inputs, _ = self._seal_a_tc(ex, gen)
            for inp in inputs:
                resolved = ex._sealed.resolve(inp)
                plan = ex._seal_input(inp).code_reloc
                self.assertEqual(plan, ex._seal_input(inp).code_reloc, "must be deterministic")
                if prob == 0.0:
                    self.assertEqual(plan, resolved.genuine(), "prob 0 must be genuine everywhere")

    def test_pac_seal_prob_matches_seal_rate_and_stays_sandbox_safe(self):
        """pac_seal_prob is the probability that an eligible memory access is PAC-sealed (standalone
        AUT* seals are arch-safety, not gated). Two properties:
          * the observed seal rate over many data-access sites tracks pac_seal_prob;
          * a skipped access keeps its offset cancellation, so an unsealed (clamp-only) TC still
            traces on the CE without escaping the sandbox.
        For a TC, sites and standalone seals are salt-independent: p=1.0 seals every access
        (n_all = sites + standalone) and p=0.0 seals none (n_none = standalone), so sites = n_all -
        n_none and the sealed-data count at p is len(_pac) - n_none."""
        P = 0.5
        saved = CONF.pac_seal_prob
        saved_cats = CONF.instruction_categories
        try:
            CONF.instruction_categories = ["PAC"] + _BASE   # pure-PAC -> PacSealedTestCase
            isa = InstructionSet(os.path.join(_ROOT, "base.json"), CONF.instruction_categories)
            gen = Aarch64RandomGenerator(isa, random.randrange(1 << 32))
            ex = factory.get_executor(generator=gen)
            type(self)._last_ex = ex
            self.assertEqual(ex._primitives, {"pac"})
            ig = factory.get_input_generator(random.randrange(1 << 32)); tmp = tempfile.mkdtemp()

            total_sites = sealed = 0
            traced_unsealed = False
            for _ in range(8 * 12):
                if total_sites >= 120:
                    break
                try:
                    tc = gen.create_test_case(os.path.join(tmp, "t.asm"), disable_assembler=True)
                except Exception:
                    continue
                CONF.pac_seal_prob = 1.0; ex.load_test_case(tc); n_all = len(ex._sealed._pac)
                CONF.pac_seal_prob = 0.0; ex.load_test_case(tc); n_none = len(ex._sealed._pac)
                sites = n_all - n_none
                if sites == 0:
                    continue
                if not traced_unsealed:   # CONF is 0.0 -> fully unsealed; must still stay in-sandbox
                    ex.trace_test_case_with_taints(ig.generate(2), CONF.model_max_nesting)
                    traced_unsealed = True
                CONF.pac_seal_prob = P; ex.load_test_case(tc)
                sealed += len(ex._sealed._pac) - n_none
                total_sites += sites

            if total_sites < 80:
                self.skipTest(f"too few data-access sites ({total_sites}) for a rate check")
            rate = sealed / total_sites
            self.assertAlmostEqual(rate, P, delta=0.15,
                                   msg=f"seal rate {rate:.3f} over {total_sites} sites != {P}")
            self.assertTrue(traced_unsealed)
        finally:
            CONF.pac_seal_prob = saved
            CONF.instruction_categories = saved_cats

    def test_collapse_key_groups_identical_resolutions(self):
        """The same input resolved twice yields the same collapse_key (the sealing class); a class is
        minted once and shared. Sanity: a single input maps to exactly one class TC."""
        ex, gen = self._executor()
        ig = factory.get_input_generator(random.randrange(1 << 32)); tmp = tempfile.mkdtemp()
        for _ in range(8 * 6):
            try:
                tc = gen.create_test_case(os.path.join(tmp, "t.asm"), disable_assembler=True)
            except Exception:
                continue
            ex.load_test_case(tc)
            if not self._has_value_slots(ex):
                continue
            ex._sandbox_base, _ = ex.read_base_addresses()
            inp = ig.generate(1)[0]
            self.assertEqual(ex._sealed.resolve(inp).collapse_key,
                             ex._sealed.resolve(inp).collapse_key)
            return
        self.skipTest("no sealable test case generated")


if __name__ == "__main__":
    unittest.main()
