"""Sealing must be path-independent: `fuzz` (which generates a test case, consuming the global RNG)
and `reproduce` (which parses the same .asm and consumes nothing) must seal the SAME test case into
the SAME artifact — identical salt, object code, genuine GVT image, and decoy images — or a saved
violation cannot be reconstructed. Regression for the decoy salt being derived from the test-case
code (not the global RNG stream) and slot insertion being seeded from it. Needs /dev/executor.
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
from src.aarch64.aarch64_asm_parser import Aarch64AsmParser
from src import factory
from src.util import stable_rng

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
_SEED = 0x7A17
_INPUT_SEED = 0xC0FFEE


class SealReproducibleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists("/dev/executor"):
            raise unittest.SkipTest("kernel module not loaded — /dev/executor missing")
        from src.aarch64.aarch64_executor import Aarch64SpeculativeExceptionExecutor
        cls._Executor = Aarch64SpeculativeExceptionExecutor
        cls._saved_conf = copy.deepcopy(CONF._borg_shared_state)
        CONF.load(os.path.join(_ROOT, "config_pac_signext.yml"))
        cls.isa = InstructionSet(os.path.join(_ROOT, "base.json"), CONF.instruction_categories)
        cls.tmp = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        CONF._borg_shared_state.clear()
        CONF._borg_shared_state.update(cls._saved_conf)

    def _seal_generate(self, seed, asm_path):
        """The fuzz path: generate a test case (consumes the global RNG), then seal it."""
        gen = Aarch64RandomGenerator(self.isa, seed)
        tc = gen.create_test_case(asm_path, disable_assembler=True)
        ex = self._Executor(gen)
        ex.load_test_case(tc)
        ex._sandbox_base, _ = ex.read_base_addresses()
        return ex

    def _seal_reproduce(self, seed, asm_path):
        """The reproduce path: parse the SAME .asm (no generation), then seal it."""
        gen = Aarch64RandomGenerator(self.isa, seed)
        tc = Aarch64AsmParser(gen).parse_file(asm_path)
        ex = self._Executor(gen)
        ex.load_test_case(tc)
        ex._sandbox_base, _ = ex.read_base_addresses()
        return ex

    def test_fuzz_and_reproduce_seal_identically(self):
        asm = os.path.join(self.tmp, "t.asm")
        a = self._seal_generate(_SEED, asm)        # writes asm
        b = self._seal_reproduce(_SEED, asm)       # parses the same asm
        self.assertEqual(a._sealed.salt, b._sealed.salt, "salt is path-dependent")
        self.assertEqual(a._sealed.object_code, b._sealed.object_code, "object code is path-dependent")
        inp = factory.get_input_generator(_INPUT_SEED).generate(1)[0]
        ra, rb = a._resolve(inp), b._resolve(inp)
        self.assertEqual(ra.genuine(), rb.genuine(), "genuine image is path-dependent")
        for i in range(max(1, CONF.inputs_per_class - 1)):
            da = ra.decoy(stable_rng(ra.collapse_key, a._sealed.salt, i))
            db = rb.decoy(stable_rng(rb.collapse_key, b._sealed.salt, i))
            self.assertEqual(da, db, f"decoy {i} image is path-dependent")

    def test_salt_tracks_the_test_case(self):
        # Same test case -> same salt (determinism); a different program -> a different salt.
        a1 = self._seal_generate(_SEED, os.path.join(self.tmp, "a1.asm"))
        a2 = self._seal_generate(_SEED, os.path.join(self.tmp, "a2.asm"))
        self.assertEqual(a1._sealed.salt, a2._sealed.salt, "same seed/program gave different salts")
        other = self._seal_generate(_SEED ^ 0x1234, os.path.join(self.tmp, "b.asm"))
        self.assertNotEqual(a1._sealed.salt, other._sealed.salt, "different programs share a salt")


if __name__ == "__main__":
    unittest.main()
