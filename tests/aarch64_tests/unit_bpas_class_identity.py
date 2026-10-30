"""ct-bpas sealing invariant (strict mode). Under a contract with the bpas (speculative store
bypass) clause, a seal slot that is reached only speculatively — e.g. bpas speculation crossing a
conditional branch into a block the architecture never takes — is decoy-eligible, while an
architecturally-reached slot is not. The detector is only sound if, across one equivalence class
(inputs that resolve to the same per-slot values and speculative statuses):
  * the architectural flow is ALWAYS genuine — every variant (baseline and decoys) carries the
    genuine value in every architectural slot's lane; and
  * the speculative flow is IDENTICAL across the class — for each variant the whole GVT image is
    byte-for-byte equal for every input in the class, so a speculative slot carries the exact same
    value (genuine, or a shared decoy) for all of them. A decoy may differ from genuine, but it is
    the same decoy for the whole class.
This mirrors boosting: architecturally-equivalent inputs are made indistinguishable except through
the microarchitectural channel. Needs /dev/executor (building the executor resolves seals).
"""
import copy
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import CONF
from src.isa_loader import InstructionSet
from src.aarch64.aarch64_generator import Aarch64RandomGenerator
from src.aarch64.aarch64_executor import SpecExcVariant
from src import factory

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
_PROGRAM_SEED = 0xB9A5
_INPUT_SEED = 0x5107
_MAX_TEST_CASES = 96
_BATCH = 48


class BpasClassIdentityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists("/dev/executor"):
            raise unittest.SkipTest("kernel module not loaded — /dev/executor missing")
        from src.aarch64.aarch64_executor import Aarch64SpeculativeExceptionExecutor
        cls._saved_conf = copy.deepcopy(CONF._borg_shared_state)
        CONF.load(os.path.join(_ROOT, "config_canon.yml"))   # canonicality seals, clause cond + bpas
        assert "bpas" in CONF.contract_execution_clause, "config must exercise the bpas clause"
        CONF.inputs_per_class = 3   # baseline + 2 decoys, so the across-class comparison is non-trivial
        isa = InstructionSet(os.path.join(_ROOT, "base.json"), CONF.instruction_categories)
        cls.gen = Aarch64RandomGenerator(isa, _PROGRAM_SEED)
        cls.ex = Aarch64SpeculativeExceptionExecutor(cls.gen)
        cls.tmp = tempfile.mkdtemp()
        cls._find_class()

    @classmethod
    def tearDownClass(cls):
        CONF._borg_shared_state.clear()
        CONF._borg_shared_state.update(cls._saved_conf)

    @classmethod
    def _find_class(cls):
        """Find a test case and two distinct same-class inputs whose class has a decoy-eligible
        speculative slot (so the invariant is exercised, not vacuous)."""
        igen = factory.get_input_generator(_INPUT_SEED)
        for _ in range(_MAX_TEST_CASES):
            tc = cls.gen.create_test_case(os.path.join(cls.tmp, "t.asm"), disable_assembler=True)
            cls.ex.load_test_case(tc)
            cls.ex._sandbox_base, _ = cls.ex.read_base_addresses()
            by_class = {}
            for inp in igen.generate(_BATCH):
                resolved = cls.ex._resolve(inp)
                if not resolved.has_decoy():
                    continue
                if not any(e.speculative for e in resolved._entries):
                    continue
                by_class.setdefault(resolved.collapse_key, []).append(inp)
            for members in by_class.values():
                if len(members) >= 2:
                    cls.tc, cls.inp_a, cls.inp_b = tc, members[0], members[1]
                    return
        raise AssertionError("no equivalence class with a speculative decoy-eligible slot found")

    def _arch_lanes(self, resolved):
        lanes = []
        for e in resolved._entries:
            if not e.speculative:
                for rng in e.sealing.frozen_ranges(e.speculative):
                    lanes.append(rng)
        return lanes

    def test_arch_slots_genuine_in_every_variant(self):
        self.ex.load_test_case(self.tc)
        self.ex._sandbox_base, _ = self.ex.read_base_addresses()
        resolved = self.ex._resolve(self.inp_a)
        genuine = resolved.genuine()
        arch_lanes = self._arch_lanes(resolved)
        self.assertGreater(len(arch_lanes), 0, "no architectural slot to check")
        variants = self.ex.variants_for_input(self.inp_a)
        for name, ei in variants.items():
            for rng in arch_lanes:
                self.assertEqual(ei.gvt[rng.start:rng.stop], genuine[rng.start:rng.stop],
                                 f"variant {name} perturbed an architectural lane {rng}")
        # And the decoys are not trivially genuine: at least one decoy image differs from the
        # baseline somewhere (necessarily in a speculative lane, since arch lanes are pinned above).
        decoys = [ei.gvt for n, ei in variants.items() if n != SpecExcVariant.BASELINE]
        self.assertTrue(any(d != genuine for d in decoys),
                        "every decoy equals genuine — no speculative slot was decoyed")

    def test_speculative_values_identical_across_class(self):
        self.ex.load_test_case(self.tc)
        self.ex._sandbox_base, _ = self.ex.read_base_addresses()
        self.assertEqual(self.ex.sealing_class_of(self.inp_a), self.ex.sealing_class_of(self.inp_b),
                         "test inputs are not in the same sealing class")
        va = self.ex.variants_for_input(self.inp_a)
        vb = self.ex.variants_for_input(self.inp_b)
        self.assertEqual(set(va), set(vb), "variant sets differ across the class")
        for name in va:
            self.assertEqual(va[name].gvt, vb[name].gvt,
                             f"variant {name} GVT image differs across the class — a speculative "
                             f"slot carries a different value for two architecturally-equivalent inputs")


if __name__ == "__main__":
    unittest.main()
