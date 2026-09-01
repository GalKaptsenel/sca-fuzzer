"""The l1d contract observation must keep architectural and speculative cache-set footprints in
SEPARATE halves of the trace (bits [63:0]=arch, [127:64]=spec), mirroring x86's L1DTracer. Folding
them into one union bitmap groups phase-different inputs into one equivalence class, so P+P's
architectural-only persistence surfaces as a spurious violation (regression: an arch touch of set S
and a speculative touch of set S are DIFFERENT observations)."""
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.aarch64.aarch64_trace import _ct_l1d, _SANDBOX_BASE_GPR

_BASE = 0xffff40000ade1000


def _ite(sets, nesting):
    """A fake trace entry: one 1-byte access per set index, at the given speculation nesting."""
    gpr = [0] * 31
    gpr[_SANDBOX_BASE_GPR] = _BASE
    accesses = [types.SimpleNamespace(effective_address=_BASE + s * 64, element_size=1) for s in sets]
    meta = types.SimpleNamespace(speculation_nesting=nesting, accesses=lambda: accesses)
    return types.SimpleNamespace(cpu=types.SimpleNamespace(gpr=gpr), metadata=meta)


def _bitmap(cer):
    return _ct_l1d(cer).hash_


class CtL1dPhaseTest(unittest.TestCase):
    def test_architectural_set_goes_to_low_half(self):
        self.assertEqual(_bitmap([_ite([3], nesting=0)]), 1 << 3)

    def test_speculative_set_goes_to_high_half(self):
        self.assertEqual(_bitmap([_ite([3], nesting=5)]), 1 << (64 + 3))

    def test_arch_and_spec_touch_of_same_set_are_distinct(self):
        # THE regression: same set, different phase -> different contract observation
        arch = _bitmap([_ite([3], nesting=0)])
        spec = _bitmap([_ite([3], nesting=5)])
        self.assertNotEqual(arch, spec)

    def test_combined_halves(self):
        # arch sets {1,7} and speculative sets {3} -> low half {1,7}, high half {3}
        cer = [_ite([1, 7], nesting=0), _ite([3], nesting=2)]
        self.assertEqual(_bitmap(cer), (1 << (64 + 3)) | (1 << 1) | (1 << 7))

    def test_same_footprint_same_phase_agrees(self):
        a = _bitmap([_ite([2, 5], nesting=0), _ite([9], nesting=3)])
        b = _bitmap([_ite([5, 2], nesting=0), _ite([9], nesting=1)])   # order/depth irrelevant
        self.assertEqual(a, b)

    def test_multi_byte_access_spans_sets_per_phase(self):
        # a 128-byte access from set 3 covers sets 3 and 4 in whichever half its phase selects
        self.assertEqual(_bitmap([_ite_span(3, 128, nesting=5)]), (1 << (64 + 3)) | (1 << (64 + 4)))


def _ite_span(start_set, size, nesting):
    gpr = [0] * 31
    gpr[_SANDBOX_BASE_GPR] = _BASE
    acc = [types.SimpleNamespace(effective_address=_BASE + start_set * 64, element_size=size)]
    meta = types.SimpleNamespace(speculation_nesting=nesting, accesses=lambda: acc)
    return types.SimpleNamespace(cpu=types.SimpleNamespace(gpr=gpr), metadata=meta)


if __name__ == "__main__":
    unittest.main()
