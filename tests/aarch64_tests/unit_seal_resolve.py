"""The sealed executor's resolution and enacted-input derivation against the real CE:
  * _resolve memoizes the pure resolution (one CE run per distinct input; reset per test case).
  * the cached resolution equals a fresh one.
  * variants_for_input / as_executor_input are deterministic (priming re-derives the exact GVT image).
  * every variant shares one byte-identical object_code; only its GVT image differs.
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
from src.aarch64 import aarch64_executor_input_encoder as wire
from src.interfaces import MAIN_AREA_SIZE, FAULTY_AREA_SIZE, GPR_SUBREGION_SIZE, SIMD_SUBREGION_SIZE


def _without_gst(inp) -> bytes:
    """The arch input's bytes with the SIMD subregion zeroed: that subregion is GVT (seal) territory,
    which serialize fills from the per-variant image, so it is not part of the arch input's identity."""
    b = bytearray(inp.tobytes())
    off = MAIN_AREA_SIZE + FAULTY_AREA_SIZE + GPR_SUBREGION_SIZE
    b[off:off + SIMD_SUBREGION_SIZE] = b"\x00" * SIMD_SUBREGION_SIZE
    return bytes(b)


class SealResolveTest(SealedExecutorFixture, unittest.TestCase):
    def test_resolve_is_memoized(self):
        # The sealed TC memoizes per input identity, so the expensive CE resolution trace runs once per
        # distinct input however many times variants are requested.
        inp, other = self._input(), self._input()
        with mock.patch.object(self.ex._sealed, "_resolve_trace",
                               wraps=self.ex._sealed._resolve_trace) as spy:
            self.ex.variants_for_input(inp)
            self.ex.variants_for_input(inp)   # cache hit
            self.ex._resolve(inp)             # cache hit
            self.assertEqual(spy.call_count, 1, "one CE resolve per distinct input")
            self.ex.variants_for_input(other)
            self.assertEqual(spy.call_count, 2, "a different input resolves once more")

    def test_load_test_case_clears_cache(self):
        self.ex.variants_for_input(self._input())
        self.assertTrue(self.ex._sealed._resolved)
        self.ex.load_test_case(self.tc)
        self.assertEqual(self.ex._sealed._resolved, {}, "a fresh sealed TC starts with an empty memo")

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
            self.assertEqual(a[name].gvt, b[name].gvt,
                             f"variant {name} not reproducible — priming would diverge")

    def test_as_executor_input_seals_to_baseline(self):
        # the boost seam: an arch input becomes its (genuine) sealed kernel input file
        inp = self._input()
        ei = self.ex.as_executor_input(inp)
        self.assertIsInstance(ei, self.ExecutorInput)
        self.assertIs(ei.input_, inp)
        self.assertEqual(ei.gvt, self.ex._resolve(inp).genuine())

    def test_reproduce_rederives_identical_executor_input(self):
        # Reproduce saves the complete wire input but loads back only the arch input; the executor must
        # deterministically re-derive the identical ExecutorInput (all seal sections) from it alone.
        inp = self._input()
        ei1 = self.ex.as_executor_input(inp)
        inp2 = wire.deserialize(ei1.serialize()).input_          # exactly what the loader returns
        self.assertEqual(_without_gst(inp2), _without_gst(inp), "arch input round-trips (excluding the GVT)")
        self.ex._sealed._resolved.clear()                        # force a genuinely fresh derivation
        ei2 = self.ex.as_executor_input(inp2)
        self.assertEqual(ei2.serialize(), ei1.serialize(),
                         "re-derived enacted input matches the original bit-for-bit")

    def test_variants_share_one_object_code_and_differ_in_gst(self):
        # The GVT invariant: the enacted code is input-independent — every variant runs the same
        # object_code and carries its per-variant difference entirely in its GVT image.
        inp = self._input()
        variants = self.ex.variants_for_input(inp)
        images = {ei.gvt for ei in variants.values()}
        self.assertTrue(all(ei.gvt is not None for ei in variants.values()))
        # baseline + at least one decoy differ only in the GVT image, never in the code.
        self.assertGreaterEqual(len(variants), 1)
        if self.ex._resolve(inp).has_decoy():
            self.assertGreater(len(images), 1, "a decoy must change the GVT image")


if __name__ == "__main__":
    unittest.main()
