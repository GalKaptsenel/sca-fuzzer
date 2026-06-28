"""Regular fuzzer with PAC/MTE (needs /dev/executor + the contract executor).

Verifies the routing (no PAC/MTE -> plain executor; PAC / MTE / both -> sealed executor with the
right primitives) and the per-input sealing of Aarch64RegularSealedExecutor, now driven through the
new Sealer/SealedTestCase pipeline (src/aarch64/aarch64_sealer.py): every architectural slot is
genuine (correct signature/tag — arch-safe); speculative slots are genuine or decoyed by a
per-sealing-class coin flip (consistent within a class, shared by every class member).
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
from src.aarch64.aarch64_seal import inst_at
from src import factory

_BASE = ["BASE-ARITH", "BASE-LOGICAL", "BASE-SHIFT", "BASE-BRANCH", "BASE-MEM-LOAD", "BASE-MEM-STORE"]


def _has_dev():
    return os.path.exists("/dev/executor")


def _value_entries(ex, resolved):
    """The resolved PAC/MTE entries (the value sealings, excluding the sandbox clamps)."""
    value_ids = {id(s) for s in getattr(ex._sealed, "_pac", [])} \
        | {id(s) for s in getattr(ex._sealed, "_mte", [])}
    return [r for r in resolved._entries if id(r.sealing) in value_ids]


def _slot_insts(tc, r):
    return [inst_at(tc, loc)[0] for loc in r.sealing.slot_locs]


def _movk_imm(inst) -> int:
    return int(inst.template.split("#0x")[1].split(",")[0], 16)


def _addg_delta(inst) -> int:
    # template: "ADDG <reg>, <reg>, #0, #<delta>"
    return int(inst.template.rsplit("#", 1)[1])


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
        from src.aarch64.aarch64_mte import MTE_INITIAL_TAG
        from src.aarch64.aarch64_input_wire import MTE_TAG_COUNT
        inp = Input()
        self.assertIsNone(self._executor_for(["PAC"] + _BASE)._mte_tags_for(inp))
        tags = self._executor_for(["MTE", "MTE-TAG-MEM"] + _BASE)._mte_tags_for(inp)
        self.assertEqual(tags, [MTE_INITIAL_TAG] * MTE_TAG_COUNT)
        self.assertEqual(self._executor_for(["PAC", "MTE", "MTE-TAG-MEM"] + _BASE)._mte_tags_for(inp),
                         [MTE_INITIAL_TAG] * MTE_TAG_COUNT)


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
        k = self.PacKeys()
        k.apia_lo = k.apib_lo = k.apda_lo = k.apdb_lo = k.apga_lo = 0x1122334455667788
        k.apia_hi = k.apib_hi = k.apda_hi = k.apdb_hi = k.apga_hi = 0x8877665544332211
        ex.local_executor.set_pac_keys(k)
        return ex, gen

    def _has_value_slots(self, ex) -> bool:
        return bool(getattr(ex._sealed, "_pac", []) or getattr(ex._sealed, "_mte", []))

    def _assert_arch_genuine(self, ex, resolved, tc, violations):
        """Every architectural value slot in `tc` must carry the correct value (never a forgery)."""
        for r in _value_entries(ex, resolved):
            if r.speculative:
                continue
            insts = _slot_insts(tc, r)
            head = insts[0].name.lower()
            if head == "movk":                                    # PAC: genuine sig or violation
                if _movk_imm(insts[0]) != (r.value & 0xFFFF if r.value is not None else None):
                    violations.append(f"forged PAC on arch slot reg={r.sealing.value_reg}")
            elif head == "addg":                                  # MTE: genuine delta or violation
                if _addg_delta(insts[0]) != ((r.value or 0) % 16):
                    violations.append(f"forged MTE retag on arch slot reg={r.sealing.value_reg}")
            # nop / xpac heads are arch-safe genuine encodings (strip / no-retag)

    def _accumulate_coverage(self, ex, resolved, tc, cov):
        for r in _value_entries(ex, resolved):
            insts = _slot_insts(tc, r)
            head = insts[0].name.lower()
            if not r.speculative:
                cov["arch_genuine"] += 1
                continue
            if head == "nop" and insts[-1].name.lower() in ("xpaci", "xpacd"):
                cov["spec_strip"] += 1
            elif head == "movk" and _movk_imm(insts[0]) != (r.value & 0xFFFF if r.value is not None else -1):
                cov["spec_decoy"] += 1
            elif head == "addg" and _addg_delta(insts[0]) != ((r.value or 0) % 16):
                cov["spec_decoy"] += 1
            elif head == "addg" or head == "nop":
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
                for _ in range(8):                               # real rng over many minted decoys
                    decoy = resolved.decoy()
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
            ctraces, _t, _tr, _v = ex.trace_test_case_with_taints(inputs, CONF.model_max_nesting)
            return inputs, ctraces
        self.skipTest("no sealable test case generated")

    def test_one_tc_per_input_and_merged_by_class(self):
        ex, gen = self._executor()
        inputs, ctraces = self._seal_a_tc(ex, gen)
        self.assertEqual(len(ctraces), len(inputs))
        self.assertTrue(all(len(var) == 1 for var in ex._last_tc_variants))  # exactly one TC per input
        # inputs of the same sealing class share the one cached TC object (caveat-4 merge): the number
        # of distinct TC dicts equals the number of sealing classes minted.
        distinct_tcs = {id(var) for var in ex._last_tc_variants}
        self.assertEqual(len(distinct_tcs), len(ex._class_tc))

    def test_forced_genuine_or_decoy(self):
        from src.aarch64.aarch64_executor import NIVariant
        for prob, expected in ((0.0, NIVariant.BASELINE), (1.0, NIVariant.DECOY)):
            ex, gen = self._executor()
            ex._DECOY_PROB = prob                                     # override the per-class coin
            self._seal_a_tc(ex, gen)
            self.assertTrue(all(next(iter(var)) is expected for var in ex._last_tc_variants),
                            f"_DECOY_PROB={prob} should make every class {expected.name}")

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
