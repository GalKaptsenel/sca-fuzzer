"""
GlobalSealTable register-indirection for the canon / branch-target (MaskFlip) seals. Device-free (no
/dev/executor, no CE): the code is byte-identical across variants, only the GST image differs, and
the slot renders a fixed UMOV+EOR. The CE / HW genuine-vs-decoy differential needs a device.
    python -m unittest tests.aarch64_tests.unit_canon_gst
"""
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.interfaces import Instruction                                        # noqa: E402
from src.aarch64.seal.sealer import (                                         # noqa: E402
    CanonSealing, BranchTargetSealing, ResolvedSealingTestCase, _Resolved)
from src.aarch64.seal.global_seal_table import VectorGlobalSealTable          # noqa: E402

_ACCESS = Instruction("ldr", True, "", False, template="LDR x1, [x1]")   # dummy; render ignores it
_MASKS = [0x7F << 48, 0x3F << 49]                                        # non-canonical flip masks


class SlotRenderTest(unittest.TestCase):
    """UMOV (load the mask from GST register v31) + EOR (apply it): constant words, no immediate."""

    def test_canon_pre_and_post_are_umov_eor(self):
        s = CanonSealing("x1", _ACCESS, _MASKS, revert=True, gst=VectorGlobalSealTable(256))
        self.assertEqual([i.template for i in s.slot_insts],
                         ["UMOV x27, v31.d[0]", "EOR x1, x1, x27",
                          "UMOV x27, v31.d[0]", "EOR x1, x1, x27"])
        self.assertEqual(s.n_before, 2)

    def test_canon_no_revert_is_pre_only(self):
        s = CanonSealing("x1", _ACCESS, _MASKS, revert=False, gst=VectorGlobalSealTable(256))
        self.assertEqual([i.template for i in s.slot_insts],
                         ["UMOV x27, v31.d[0]", "EOR x1, x1, x27"])

    def test_branch_target_reconverges(self):
        s = BranchTargetSealing("x28", _ACCESS, _MASKS, gst=VectorGlobalSealTable(256))
        self.assertEqual([i.template for i in s.slot_insts],
                         ["UMOV x27, v31.d[0]", "EOR x28, x28, x27",
                          "UMOV x27, v31.d[0]", "EOR x28, x28, x27"])

    def test_two_seals_get_distinct_lanes(self):
        gst = VectorGlobalSealTable(256)
        a = CanonSealing("x1", _ACCESS, _MASKS, revert=True, gst=gst)
        b = CanonSealing("x2", _ACCESS, _MASKS, revert=True, gst=gst)
        self.assertNotEqual(a.location.byte_range(), b.location.byte_range())


class ByteIdenticalCodeTest(unittest.TestCase):
    """Genuine and every decoy share one object_code; they differ only in the GST image, and only in
    the sealed slot's lane bytes."""

    def _resolved(self):
        gst = VectorGlobalSealTable(256)
        seal = CanonSealing("x1", _ACCESS, _MASKS, revert=True, gst=gst)
        entry = _Resolved(seal, value=None, alts=list(_MASKS), spec_nesting=1)   # genuine = no flip
        object_code = bytes(range(64))
        resolved = ResolvedSealingTestCase([entry], object_code, gst, salt=0, cer=None)
        return seal, object_code, resolved

    def test_object_code_shared_and_images_differ(self):
        seal, object_code, resolved = self._resolved()
        self.assertIs(resolved.object_code, object_code)        # code never changes per variant
        genuine = resolved.genuine()
        decoys = {resolved.decoy(random.Random(s)) for s in range(8)}
        self.assertNotIn(genuine, decoys)                       # a decoy flips the lane
        lane = set(seal.location.byte_range())
        for decoy in decoys:
            self.assertEqual(len(decoy), len(genuine))
            diff = [i for i in range(len(genuine)) if genuine[i] != decoy[i]]
            self.assertTrue(diff and set(diff) <= lane)         # differ only in the seal's lane

    def test_genuine_lane_is_zero(self):
        seal, _oc, resolved = self._resolved()
        rng = seal.location.byte_range()
        self.assertEqual(resolved.genuine()[rng.start:rng.stop], b"\x00" * len(rng))


if __name__ == "__main__":
    unittest.main()
