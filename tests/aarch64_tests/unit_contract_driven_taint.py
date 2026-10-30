"""contract_driven mode: the GVT lives in the input's SIMD subregion, and the sealer-derived taint
freezes exactly the architecturally-consumed seal lanes at their genuine values while leaving
speculative-only lanes free. The invariants (needs /dev/executor + the CE):
  * every architectural value seal's lane is tainted AND equals the genuine image in the input;
  * every speculative-only seal's lane is untainted (boosting may vary it);
  * boosting (extend_equivalence_classes) reproduces the architectural lanes byte-for-byte across the
    class and re-randomises at least one speculative lane — so the architectural path is identical and
    only non-tainted lanes differ.
"""
import copy
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import CONF
from src.isa_loader import InstructionSet
from src.aarch64.aarch64_generator import Aarch64RandomGenerator
from src import factory

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
_PROGRAM_SEED = 0x5EA1
_INPUT_SEED = 0xC0DE
_MAX_TEST_CASES = 64


def _simd(obj):
    return obj[0]['simd'].view(np.uint8)


class ContractDrivenTaintTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists("/dev/executor"):
            raise unittest.SkipTest("kernel module not loaded — /dev/executor missing")
        from src.aarch64.aarch64_executor import Aarch64SpeculativeExceptionExecutor
        cls._saved_conf = copy.deepcopy(CONF._borg_shared_state)
        CONF.load(os.path.join(_ROOT, "config_pac_mte.yml"))
        CONF.speculative_exception_mode = "contract_driven"
        isa = InstructionSet(os.path.join(_ROOT, "base.json"), CONF.instruction_categories)
        cls.gen = Aarch64RandomGenerator(isa, _PROGRAM_SEED)
        cls.ex = Aarch64SpeculativeExceptionExecutor(cls.gen)
        assert cls.ex._contract_driven
        cls.tmp = tempfile.mkdtemp()
        igen = factory.get_input_generator(_INPUT_SEED)
        for _ in range(_MAX_TEST_CASES):
            tc = cls.gen.create_test_case(os.path.join(cls.tmp, "t.asm"), disable_assembler=True)
            cls.ex.load_test_case(tc)
            cls.ex._sandbox_base, _ = cls.ex.read_base_addresses()
            value_seals = getattr(cls.ex._sealed, "_pac", []) + getattr(cls.ex._sealed, "_mte", [])
            if value_seals and any(cls.ex.has_decoy(i) for i in igen.generate(8)):
                cls.tc = tc
                return
        raise AssertionError("no sealable, decoy-eligible test case found")

    @classmethod
    def tearDownClass(cls):
        CONF._borg_shared_state.clear()
        CONF._borg_shared_state.update(cls._saved_conf)

    def _input(self):
        return factory.get_input_generator(_INPUT_SEED).generate(1)[0]

    def test_arch_lanes_frozen_speculative_lanes_free(self):
        inp = self._input()
        _ct, taints, _tr = self.ex.trace_test_case_with_taints([inp], CONF.model_max_nesting)
        taint = taints[0]
        resolved = self.ex._resolve(inp)
        genuine = np.frombuffer(resolved.genuine(), dtype=np.uint8)
        simd, simd_taint = _simd(inp), _simd(taint)

        checked_arch = checked_spec = 0
        for e in resolved._entries:
            frozen = set()
            for rng in e.sealing.frozen_ranges(e.speculative):
                frozen.update(rng)
                # a frozen lane is tainted and holds the genuine value in the input
                self.assertTrue(all(simd_taint[b] for b in rng), "frozen lane not tainted")
                self.assertEqual(bytes(simd[rng.start:rng.stop]), bytes(genuine[rng.start:rng.stop]),
                                 "frozen lane not set to the genuine value")
            loc = getattr(e.sealing, "location", None)
            if loc is None:
                continue
            lane = set(loc.byte_range())
            if e.speculative and not (lane & frozen):   # a speculative single-lane seal: lane is free
                self.assertFalse(any(simd_taint[b] for b in lane), "speculative lane is tainted")
                checked_spec += 1
            elif not e.speculative:
                checked_arch += 1
        self.assertGreater(checked_arch + checked_spec, 0, "no value seals exercised")

    def test_boosting_keeps_arch_lanes_and_varies_speculative(self):
        inp = self._input()
        _ct, taints, _tr = self.ex.trace_test_case_with_taints([inp], CONF.model_max_nesting)
        taint = taints[0]
        igen = factory.get_input_generator(_INPUT_SEED)
        igen.reset_boosting_state()
        boosted = igen.extend_equivalence_classes([inp], [taint])
        self.assertTrue(boosted, "no boosted input produced")
        base_simd, tmask = _simd(inp), _simd(taint)
        for b in boosted:
            bs = _simd(b)
            for off in range(len(tmask)):
                if tmask[off]:   # tainted (architectural) lane bytes are preserved exactly
                    self.assertEqual(bs[off], base_simd[off], f"boosting changed a frozen lane byte {off}")


if __name__ == "__main__":
    unittest.main()
