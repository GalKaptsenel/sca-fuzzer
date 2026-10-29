"""Device-free precondition fuzzing for the PAC+SIGNEXT speculative-exception primitive: over many randomly-generated
test cases, drive the real sealer + contract executor (no /dev/executor) under config_pac_signext.yml
and assert the preconditions the primitive needs actually arise, and the genuine/decoy machinery is
coherent:
  * every TC seals as a PacSignExtSealedTestCase (every data access PAC-authenticated, each access
    optionally a sign-ext flip site);
  * some sign-ext flip sites are speculative-only, so decoys exist (the TC is speculative-exception-testable);
  * genuine() leaves every flip slot canonical (NOP); forced_noncanon() flips exactly bit 55
    (EOR #1<<55) at every speculative flip slot and never perturbs an architectural slot.
Self-contained: needs the CE binary but no device; restores CONF.
"""
import copy
import os
import random
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.config import CONF
from src.aarch64.seal.sealer import PacSignExtSealedTestCase, SignExtSealing, _SIGNEXT_BIT
import tests.aarch64_tests.ce_equivalence as eq


def _lane(image, location):
    rng = location.byte_range()
    return int.from_bytes(image[rng.start:rng.stop], "little")

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
_N_TCS = 40
_N_INPUTS = 3
_PROGRAM_SEED = 0xBEEF
_INPUT_SEED = 0xF00D


@unittest.skipUnless(eq.ce_available(), "needs the contract-executor binary")
class PacSignExtPreconditionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.isa_loader import InstructionSet
        from src import factory
        cls._saved_conf = copy.deepcopy(CONF._borg_shared_state)
        CONF.load(os.path.join(_ROOT, "config_pac_signext.yml"))
        cls._saved_profile = eq.PROFILE
        # Model generic auth (PACGA) so a randomly-generated pacga does not abort the CE.
        eq.PROFILE = eq.qarma.profile(qarma_version=3, generic_qarma_version=3, auth_level=5,
                                      va_size0=48, va_size1=48, tbi0=True, tbi1=True,
                                      tbid0=False, tbid1=False, constpacfield=False, mtx0=False, mtx1=False)
        isa = InstructionSet(os.path.join(_ROOT, "base.json"), CONF.instruction_categories)
        cls.gen = factory.get_program_generator(isa, _PROGRAM_SEED)
        cls.inputs = factory.get_input_generator(_INPUT_SEED).generate(_N_INPUTS)
        cls.tmp = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        eq.PROFILE = cls._saved_profile
        CONF._borg_shared_state.clear()
        CONF._borg_shared_state.update(cls._saved_conf)

    def test_preconditions_arise_and_decoys_are_coherent(self):
        dead = (RuntimeError, BrokenPipeError, OSError, ValueError)
        sess = eq.CESession({"pac", "signext"})
        sealer = sess.sealer(self.gen)
        sealed = sx_spec = ni_testable = flip_checks = 0
        try:
            for _ in range(_N_TCS):
                tc = self.gen.create_test_case(os.path.join(self.tmp, "t.asm"), disable_assembler=True)
                sess.set_data_size(eq._dispatch_data_size(tc))
                try:
                    s = sealer.seal(tc)
                except NotImplementedError as e:   # PAC/SIGNEXT not yet on the GST path: skip until migrated
                    self.skipTest(str(e))
                self.assertIsInstance(s, PacSignExtSealedTestCase)
                sealed += 1
                try:
                    s.resolve(self.inputs[0])   # traces the genuine render; a CE-rejected TC -> fresh session
                    for inp in self.inputs:
                        sx_spec += self._check_resolution(s.resolve(inp), s)
                        ni_testable += 1 if s.resolve(inp).has_decoy() else 0
                        flip_checks += self._flip_checks
                except dead:
                    sess.stop()
                    sess = eq.CESession({"pac", "signext"})
                    sealer = sess.sealer(self.gen)
        finally:
            sess.stop()
        self.assertGreater(sealed, 0, "no PAC+SIGNEXT test case sealed")
        self.assertGreater(sx_spec, 0, "no speculative sign-ext site ever arose (no precondition)")
        self.assertGreater(ni_testable, 0, "no speculative-exception-testable resolution (no decoy)")
        self.assertGreater(flip_checks, 0, "no sign-ext flip render was exercised")

    def _check_resolution(self, resolved, sealed) -> int:
        """Assert the genuine/decoy GST images at this resolution are coherent; return the speculative
        sign-ext site count it contributed. Every sign-ext lane is 0 in the genuine image (no flip); a
        decoy flips a speculative lane to the bit-55 mask and never an architectural one."""
        sx = [e for e in resolved._entries if isinstance(e.sealing, SignExtSealing)]
        spec = [e for e in sx if e.speculative]
        self._flip_checks = 0
        if not resolved.has_decoy():
            return len(spec)
        genuine = resolved.genuine()
        for e in sx:
            self.assertEqual(_lane(genuine, e.sealing.location), 0,
                             "genuine sign-ext lane carries a flip (should be canonical)")
        arch_ids = {id(e.sealing) for e in sx if not e.speculative}
        seen_flip = False
        for i in range(32):
            image = resolved.decoy(random.Random(i))
            for e in sx:
                lane = _lane(image, e.sealing.location)
                if id(e.sealing) in arch_ids:
                    self.assertEqual(lane, 0, "a decoy flipped an architectural sign-ext slot")
                elif lane != 0:
                    self.assertEqual(lane, e.sealing.value.encode(_SIGNEXT_BIT),
                                     "sign-ext decoy lane is not the bit-55 flip")
                    seen_flip = True
        if spec:
            self.assertTrue(seen_flip, "no speculative sign-ext flip across 32 decoys")
            self._flip_checks = 1
        return len(spec)


if __name__ == "__main__":
    unittest.main()
