"""The sealed executor's resolution and enacted-input derivation against the real CE:
  * _resolve memoizes the pure resolution (one CE run per distinct input; reset per test case).
  * the cached resolution equals a fresh one.
  * variants_for_input / as_executor_input are deterministic (priming re-derives the exact slots).
  * reconstruct_enacted_code rebuilds a variant's exact machine code from its recorded relocations.
  * reproduce (arch input only) re-derives the identical ExecutorInput.
Needs /dev/executor + the CE.
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sealed_fixture import SealedExecutorFixture
from src.aarch64.aarch64_relocations import apply_relocations
from src.aarch64 import aarch64_executor_input_encoder as wire


class SealResolveTest(SealedExecutorFixture, unittest.TestCase):
    def test_resolve_is_memoized(self):
        inp, other = self._input(), self._input()
        with mock.patch.object(self.ex._sealed, "resolve",
                               wraps=self.ex._sealed.resolve) as spy:
            self.ex.variants_for_input(inp)
            self.ex.variants_for_input(inp)   # cache hit
            self.ex._resolve(inp)             # cache hit
            self.assertEqual(spy.call_count, 1, "one CE resolve per distinct input")
            self.ex.variants_for_input(other)
            self.assertEqual(spy.call_count, 2, "a different input resolves once more")

    def test_load_test_case_clears_cache(self):
        self.ex.variants_for_input(self._input())
        self.assertTrue(self.ex._resolve_cache)
        self.ex.load_test_case(self.tc)
        self.assertEqual(self.ex._resolve_cache, {}, "cache must reset per test case")

    def test_cached_resolve_matches_uncached(self):
        inp = self._input()
        cached = self.ex._resolve(inp)
        fresh = self.ex._sealed.resolve(inp)          # bypass the memo
        self.assertEqual(cached.collapse_key, fresh.collapse_key)
        self.assertEqual(cached.genuine(), fresh.genuine())

    def test_variants_for_input_is_deterministic(self):
        inp = self._input()
        a = self.ex.variants_for_input(inp)
        self.ex._resolve_cache = {}                    # force a genuinely fresh resolve
        b = self.ex.variants_for_input(inp)
        self.assertEqual(list(a), list(b), "same variant names, same order")
        for name in a:
            self.assertEqual(a[name].code_reloc, b[name].code_reloc,
                             f"variant {name} not reproducible — priming would diverge")

    def test_as_executor_input_seals_to_baseline(self):
        # the boost seam: an arch input becomes its (genuine) sealed kernel input file
        inp = self._input()
        ei = self.ex.as_executor_input(inp)
        self.assertIsInstance(ei, self.ExecutorInput)
        self.assertIs(ei.input_, inp)
        self.assertEqual(ei.code_reloc, self.ex._resolve(inp).genuine())

    def test_reproduce_rederives_identical_executor_input(self):
        # Reproduce saves the complete wire input but loads back only the arch input; the executor must
        # deterministically re-derive the identical ExecutorInput (all seal sections) from it alone.
        inp = self._input()
        ei1 = self.ex.as_executor_input(inp)
        inp2 = wire.deserialize(ei1.serialize()).input_          # exactly what the loader returns
        self.assertEqual(inp2.tobytes(), inp.tobytes(), "arch input round-trips exactly")
        self.ex._resolve_cache = {}                              # force a genuinely fresh derivation
        ei2 = self.ex.as_executor_input(inp2)
        self.assertEqual(ei2.serialize(), ei1.serialize(),
                         "re-derived enacted input matches the original bit-for-bit")

    def test_reconstruct_enacted_code(self):
        inp = self._input()
        _name, ei = next(iter(self.ex.variants_for_input(inp).items()))
        rebuilt = self.ex.reconstruct_enacted_code(ei)
        expected = apply_relocations(self.ex._sealed.object_code, list(ei.code_reloc))
        self.assertEqual(rebuilt, expected, "enacted code reproduces from the ExecutorInput")


if __name__ == "__main__":
    unittest.main()
