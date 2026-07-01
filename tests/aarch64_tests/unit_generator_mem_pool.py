"""Regression: the memory-access instruction picker must not crash when a config enables only one
direction (loads xor stores). A 50/50 store-vs-load draw that lands on an empty pool must fall
through to the non-empty pool instead of calling random.choice([])."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")

from src.config import CONF
from src.isa_loader import InstructionSet
from src.aarch64.aarch64_generator import Aarch64RandomGenerator


class MemPoolFallthroughTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        CONF.load(os.path.join(_ROOT, "config.yml"))
        isa = InstructionSet(os.path.join(_ROOT, "base.json"), CONF.instruction_categories)
        cls.gen = Aarch64RandomGenerator(isa, 0)
        cls.nomem = cls.gen.non_memory_access_instructions
        cls.load = cls.gen.load_instruction
        cls.store = cls.gen.store_instructions

    def test_store_only_pool_never_picks_empty_load(self):
        for _ in range(500):
            spec = self.gen._pick_random_instruction_spec(self.nomem, self.store, [], 1.0)
            self.assertIn(spec, self.store)

    def test_load_only_pool_never_picks_empty_store(self):
        for _ in range(500):
            spec = self.gen._pick_random_instruction_spec(self.nomem, [], self.load, 1.0)
            self.assertIn(spec, self.load)


if __name__ == "__main__":
    unittest.main()
