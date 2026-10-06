"""PAC keys are a property of each input: drawn by the seeded input generator (per input, or one shared
set per campaign by CONF.pac_keys_per_input), reproducible from the seed, kept across copies, boosting
and save/load, and part of the input's identity. A PAC input file without keys is rejected."""
import copy
import os
import sys
import tempfile
import unittest

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, _ROOT)
from src.config import CONF
from src.aarch64.aarch64_input_generator import AArch64InputGenerator
from src.aarch64 import aarch64_executor_input_encoder as wire


class InputPacKeysTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._saved_conf = copy.deepcopy(CONF._borg_shared_state)
        CONF.load(os.path.join(_ROOT, "config_pac.yml"))

    @classmethod
    def tearDownClass(cls):
        CONF._borg_shared_state.clear()
        CONF._borg_shared_state.update(cls._saved_conf)

    def setUp(self):
        self._saved = (CONF.pac_keys_per_input, list(CONF.instruction_categories))

    def tearDown(self):
        CONF.pac_keys_per_input, CONF.instruction_categories = self._saved

    def _gen(self, n=6, seed=1234):
        return AArch64InputGenerator(seed).generate(n)

    def test_per_input_keys_are_distinct_and_reproducible(self):
        CONF.pac_keys_per_input = True
        keys = [i.pac_keys for i in self._gen()]
        self.assertTrue(all(k is not None and len(k) == 10 for k in keys))
        self.assertEqual(len(set(keys)), len(keys))
        self.assertEqual(keys, [i.pac_keys for i in self._gen()])
        self.assertNotEqual(keys, [i.pac_keys for i in self._gen(seed=999)])

    def test_shared_keys_are_one_set_and_reproducible(self):
        CONF.pac_keys_per_input = False
        gen = AArch64InputGenerator(1234)
        keys = {i.pac_keys for i in gen.generate(3) + gen.generate(3)}
        self.assertEqual(len(keys), 1)
        self.assertEqual(keys, {i.pac_keys for i in self._gen()})

    def test_no_pac_no_keys(self):
        CONF.instruction_categories = ["BASE-ARITH"]
        self.assertTrue(all(i.pac_keys is None for i in self._gen()))

    def test_keys_survive_copy_and_boosting(self):
        CONF.pac_keys_per_input = True
        gen = AArch64InputGenerator(1234)
        base = gen.generate(2)
        self.assertEqual([b.copy().pac_keys for b in base], [b.pac_keys for b in base])
        taints = [b.copy() for b in base]
        for t in taints:
            t.view("u1")[:] = 0
        boosted = gen.extend_equivalence_classes(base, taints)
        self.assertEqual([b.pac_keys for b in boosted], [b.pac_keys for b in base])

    def test_identity_includes_keys(self):
        CONF.pac_keys_per_input = True
        a = self._gen(1)[0]
        b = a.copy()
        b.pac_keys = tuple(reversed(a.pac_keys))
        self.assertEqual(a.tobytes(), b.tobytes())
        self.assertNotEqual(a.identity(), b.identity())
        self.assertNotEqual(hash(a), hash(b))

    def test_keys_round_trip_through_input_files(self):
        CONF.pac_keys_per_input = True
        inp = self._gen(1)[0]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "input.reif")
            with open(path, "wb") as f:
                f.write(wire.ExecutorInput(inp, pac_keys=list(inp.pac_keys)).serialize())
            loaded = AArch64InputGenerator(1).load([path])[0]
            self.assertEqual(loaded.identity(), inp.identity())
            with open(path, "wb") as f:
                f.write(wire.ExecutorInput(inp).serialize())
            with self.assertRaisesRegex(ValueError, "must carry its PAC keys"):
                AArch64InputGenerator(1).load([path])


if __name__ == "__main__":
    unittest.main()
