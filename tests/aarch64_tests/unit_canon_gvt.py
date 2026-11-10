"""
GlobalValueTable register-indirection for the canon / branch-target (MaskFlip) GVT accesses. Device-free (no
/dev/executor, no CE): the code is byte-identical across variants, only the GVT image differs, and
the slot renders a fixed UMOV+EOR. The CE / HW genuine-vs-decoy differential needs a device.
    python -m unittest tests.aarch64_tests.unit_canon_gst
"""
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.interfaces import Instruction                                        # noqa: E402
from src.aarch64.gvt.builder import (                                         # noqa: E402
    CanonGvtAccess, BranchTargetGvtAccess, ResolvedGvtTestCase, _Resolved)
from src.aarch64.gvt.global_value_table import VectorGlobalValueTable          # noqa: E402

_ACCESS = Instruction("ldr", True, "", False, template="LDR x1, [x1]")   # dummy; render ignores it
_MASKS = [0x7F << 48, 0x3F << 49]                                        # non-canonical flip masks


class SlotRenderTest(unittest.TestCase):
    """UMOV (load the mask from a GVT vector lane) + shifted EOR (apply it): constant words, no
    immediate. The masks live in [54:48], so the lane is one byte (UMOV.b) and the EOR shifts it back
    into place (lsl #48) — the smallest access that covers the field."""

    def test_canon_pre_and_post_are_umov_eor(self):
        s = CanonGvtAccess("x1", _ACCESS, _MASKS, revert=True, gvt=VectorGlobalValueTable(256))
        self.assertEqual([i.template for i in s.slot_insts],
                         ["UMOV w27, v31.b[0]", "EOR x1, x1, x27, lsl #48",
                          "UMOV w27, v31.b[0]", "EOR x1, x1, x27, lsl #48"])
        self.assertEqual(s.n_before, 2)

    def test_canon_no_revert_is_pre_only(self):
        s = CanonGvtAccess("x1", _ACCESS, _MASKS, revert=False, gvt=VectorGlobalValueTable(256))
        self.assertEqual([i.template for i in s.slot_insts],
                         ["UMOV w27, v31.b[0]", "EOR x1, x1, x27, lsl #48"])

    def test_branch_target_reconverges(self):
        s = BranchTargetGvtAccess("x28", _ACCESS, _MASKS, gvt=VectorGlobalValueTable(256))
        self.assertEqual([i.template for i in s.slot_insts],
                         ["UMOV w27, v31.b[0]", "EOR x28, x28, x27, lsl #48",
                          "UMOV w27, v31.b[0]", "EOR x28, x28, x27, lsl #48"])

    def test_two_accesses_get_distinct_lanes(self):
        gvt = VectorGlobalValueTable(256)
        a = CanonGvtAccess("x1", _ACCESS, _MASKS, revert=True, gvt=gvt)
        b = CanonGvtAccess("x2", _ACCESS, _MASKS, revert=True, gvt=gvt)
        self.assertNotEqual(a.location.byte_range(), b.location.byte_range())


class ByteIdenticalCodeTest(unittest.TestCase):
    """Genuine and every decoy share one object_code; they differ only in the GVT image, and only in
    the instrumented slot's lane bytes."""

    def _resolved(self):
        gvt = VectorGlobalValueTable(256)
        access = CanonGvtAccess("x1", _ACCESS, _MASKS, revert=True, gvt=gvt)
        entry = _Resolved(access, value=None, alts=list(_MASKS), spec_nesting=1)   # genuine = no flip
        object_code = bytes(range(64))
        resolved = ResolvedGvtTestCase([entry], object_code, gvt, salt=0, cer=None)
        return access, object_code, resolved

    def test_object_code_shared_and_images_differ(self):
        access, object_code, resolved = self._resolved()
        self.assertIs(resolved.object_code, object_code)        # code never changes per variant
        genuine = resolved.genuine()
        decoys = {resolved.decoy(random.Random(s)) for s in range(8)}
        self.assertNotIn(genuine, decoys)                       # a decoy flips the lane
        lane = set(access.location.byte_range())
        for decoy in decoys:
            self.assertEqual(len(decoy), len(genuine))
            diff = [i for i in range(len(genuine)) if genuine[i] != decoy[i]]
            self.assertTrue(diff and set(diff) <= lane)         # differ only in the GVT access's lane

    def test_genuine_lane_is_zero(self):
        access, _oc, resolved = self._resolved()
        rng = access.location.byte_range()
        self.assertEqual(resolved.genuine()[rng.start:rng.stop], b"\x00" * len(rng))


if __name__ == "__main__":
    unittest.main()
