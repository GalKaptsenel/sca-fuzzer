"""Device-free precondition fuzzing for the PAC+SIGNEXT NI primitive: over many randomly-generated
test cases, drive the real sealer + contract executor (no /dev/executor) under config_pac_signext.yml
and assert the preconditions the primitive needs actually arise, and the genuine/decoy machinery is
coherent:
  * every TC seals as a PacSignExtSealedTestCase (every data access PAC-authenticated, each access
    optionally a sign-ext flip site);
  * some sign-ext flip sites are speculative-only, so decoys exist (the TC is NI-testable);
  * genuine() leaves every flip slot canonical (NOP); forced_noncanon() flips exactly bit 55
    (EOR #1<<55) at every speculative flip slot and never perturbs an architectural slot.
Self-contained: needs the CE binary but no device; restores CONF.
"""
import copy
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.config import CONF
from src.aarch64.aarch64_relocations import NOP_WORD, eor_imm_word
from src.aarch64.seal.sealer import PacSignExtSealedTestCase, SignExtSealing, _SIGNEXT_BIT
import tests.aarch64_tests.ce_equivalence as eq

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
                s = sealer.seal(tc)
                self.assertIsInstance(s, PacSignExtSealedTestCase)
                sealed += 1
                try:
                    sess.trace_bytes(s.object_code, self.inputs[0])   # CE-rejected TC -> fresh session, skip
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
        self.assertGreater(ni_testable, 0, "no NI-testable resolution (no decoy)")
        self.assertGreater(flip_checks, 0, "no sign-ext flip render was exercised")

    def _check_resolution(self, resolved, sealed) -> int:
        """Assert the genuine/decoy renders at this resolution are coherent; return the speculative
        sign-ext site count it contributed."""
        sx = [e for e in resolved._entries if isinstance(e.sealing, SignExtSealing)]
        spec = [e for e in sx if e.speculative]
        sx_spec_offsets = {o for e in spec for o in resolved._offsets.get(id(e.sealing), ())}
        arch_offsets = {o for e in resolved._entries if not e.speculative
                        for o in resolved._offsets.get(id(e.sealing), ())}
        self._flip_checks = 0
        if resolved.has_decoy():
            g = {r.offset: r.value for r in resolved.genuine()}
            f = {r.offset: r.value for r in resolved.forced_noncanon()}   # perturb every speculative slot
            diffs = [o for o in g if g[o] != f.get(o)]
            self.assertTrue(diffs, "forced_noncanon() identical to genuine() despite has_decoy()")
            self.assertFalse([o for o in diffs if o in arch_offsets],
                             "a decoy perturbed an architectural slot")
            for off in sx_spec_offsets:
                self.assertEqual(g.get(off), NOP_WORD, f"genuine sign-ext slot @0x{off:x} not NOP")
                self.assertTrue(any(f.get(off) == eor_imm_word(r, r, _SIGNEXT_BIT) for r in range(31)),
                                f"forced sign-ext slot @0x{off:x} not EOR #1<<55")
                self._flip_checks += 1
        return len(spec)


if __name__ == "__main__":
    unittest.main()
