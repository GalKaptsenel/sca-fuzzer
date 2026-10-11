"""A-priori equivalence property: one CE trace of a sealed test case's CE lowering (real
instructions the CE emulates and concretizes) equals the genuine relocated trace — equal
ctrace, equal taint, and no divergence outside the seal slots. Device-free: drives the CE binary
directly. The redesign reuses this single trace instead of a separate genuine trace.
"""
import copy
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.config import CONF
from src.aarch64.aarch64_relocations import apply_relocations
from src.aarch64.aarch64_trace import compute_ctrace, compute_taint
from src.aarch64.seal.sealer import _slot_offsets
import tests.aarch64_tests.ce_equivalence as eq

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
_PROGRAM_SEED = 0x5EA1
_INPUT_SEED = 0xC0DE
_MAX_TEST_CASES = 64


@unittest.skipUnless(eq.ce_available(), "needs the contract-executor binary")
class TraceEquivalencePacTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from src.isa_loader import InstructionSet
        from src.aarch64.aarch64_generator import Aarch64RandomGenerator
        cls._saved_conf = copy.deepcopy(CONF._borg_shared_state)
        CONF.load(os.path.join(_ROOT, "config_pac.yml"))
        isa = InstructionSet(os.path.join(_ROOT, "base.json"), CONF.instruction_categories)
        cls.gen = Aarch64RandomGenerator(isa, _PROGRAM_SEED)
        cls.sess = eq.CESession({"pac"})
        cls.sealer = cls.sess.sealer(cls.gen)
        cls.tmp = tempfile.mkdtemp()
        cls._seal_a_pac_test_case()

    @classmethod
    def tearDownClass(cls):
        cls.sess.stop()
        CONF._borg_shared_state.clear()
        CONF._borg_shared_state.update(cls._saved_conf)

    @classmethod
    def _seal_a_pac_test_case(cls):
        for _ in range(_MAX_TEST_CASES):
            tc = cls.gen.create_test_case(os.path.join(cls.tmp, "t.asm"), disable_assembler=True)
            cls.sess.set_data_size(eq._dispatch_data_size(tc))
            sealed = cls.sealer.seal(tc)
            if sealed._pac:
                cls.sealed = sealed
                return
        raise AssertionError(f"no PAC-sealed test case in {_MAX_TEST_CASES} (seed {_PROGRAM_SEED})")

    def _slot_offsets(self):
        offs = set()
        for s, o in _slot_offsets(self.sealed._tc, self.sealed._layout, self.sealed._pac).items():
            offs |= set(o)
        return offs

    def test_ce_lowering_matches_genuine_on_consumed_fields(self):
        from src import factory
        ce_code = self.sealed.ce_object_code
        slot_offs = self._slot_offsets()
        inputs = factory.get_input_generator(_INPUT_SEED).generate(8)
        checked = 0
        for inp in inputs:
            resolved = self.sealed.resolve(inp)
            genuine_code = apply_relocations(resolved.object_code, resolved.genuine())
            cer_ce = self.sess.trace_bytes(ce_code, inp)
            cer_g = self.sess.trace_bytes(genuine_code, inp)

            self.assertEqual(len(list(cer_ce)), len(list(cer_g)), "entry counts differ")
            self.assertEqual(compute_ctrace(cer_ce), compute_ctrace(cer_g), "ctrace differs")
            self.assertEqual(compute_taint(cer_ce).tobytes(), compute_taint(cer_g).tobytes(),
                             "taint differs")

            for line in eq.diff_traces(cer_ce, cer_g):
                off = int(line.split("pc_off=0x")[1].split(":")[0], 16)
                self.assertIn(off, slot_offs, f"divergence outside a seal slot: {line}")
            checked += 1
        self.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main()
