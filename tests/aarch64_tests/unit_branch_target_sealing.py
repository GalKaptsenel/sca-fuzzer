"""Branch-target non-interference sealing (AArch64), self-contained (no /dev/executor, no CE):
  * _branch_target_mask_pool draws non-canonical high-bit runs within the guaranteed-fault range
    [54:VA] plus, when branch_target_seal_misalign is set, the low-bit misalignments {1,2,3};
    deterministic per salt; branch_target_canon_mask fixes the non-canonical run;
  * BranchTargetSealing flips the target register once, right before the BLR, with NO after-revert
    (the target is a dead scratch); genuine is a single NOP;
  * _resolve_branch_target classifies architectural vs speculative-only by the BLR's min nesting.
VA here is 48 (this box / Neoverse N3).
"""
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import CONF
from src.aarch64.aarch64_relocations import eor_imm_word, NOP_WORD
from src.aarch64.seal import sealer as S
from src.aarch64.aarch64_target_desc import INDIRECT_CALL_TARGET_REGISTER as TGT

_VA = 48
_TGT_N = int(TGT[1:])                       # x28 -> 28
_FAULT = (((1 << (55 - _VA)) - 1) << _VA)   # bits [54:48]


class _Ite:
    def __init__(self, pc, nesting):
        self.cpu = types.SimpleNamespace(pc=pc)
        self.metadata = types.SimpleNamespace(speculation_nesting=nesting)


class BranchTargetPoolTest(unittest.TestCase):
    def setUp(self):
        CONF.va_size = _VA
        CONF.branch_target_canon_mask = None
        CONF.branch_target_seal_misalign = True

    def test_pool_has_noncanonical_runs_and_misalignments(self):
        pool = S._branch_target_mask_pool(12345)
        canon = [m for m in pool if m >= 4]
        misalign = [m for m in pool if m < 4]
        self.assertEqual(sorted(misalign), [1, 2, 3], "misalign axis = low bits {1,2,3}")
        self.assertTrue(canon, "pool has non-canonical masks")
        for m in canon:
            self.assertEqual(m & ~_FAULT, 0, f"{m:#x} outside the guaranteed-fault range [54:{_VA}]")
            self.assertEqual(S._mask_runs(m), [m], f"{m:#x} not a single contiguous run")

    def test_misalign_toggle(self):
        CONF.branch_target_seal_misalign = False
        self.assertFalse([m for m in S._branch_target_mask_pool(1) if m < 4], "no misalign when disabled")

    def test_deterministic_per_salt(self):
        self.assertEqual(S._branch_target_mask_pool(7), S._branch_target_mask_pool(7))
        self.assertNotEqual(S._branch_target_mask_pool(7), S._branch_target_mask_pool(8))

    def test_fixed_canon_mask(self):
        CONF.branch_target_canon_mask = 1 << _VA
        self.assertEqual(S._branch_target_mask_pool(1), [1 << _VA, 1, 2, 3])

    def test_bad_fixed_mask_raises(self):
        CONF.branch_target_canon_mask = 1 << 20   # in the VA bits, not a fault
        with self.assertRaises(Exception):
            S._branch_target_mask_pool(1)


class BranchTargetRenderTest(unittest.TestCase):
    def test_genuine_is_single_nop_no_revert(self):
        s = S.BranchTargetSealing(TGT, None, [1 << _VA])
        self.assertEqual([i.name for i in s.seal(None, None)], ["nop"], "genuine = 1 NOP, no revert")

    def test_decoy_flips_once(self):
        s = S.BranchTargetSealing(TGT, None, [1 << _VA])
        for mask in (1 << _VA, 1, 3):
            decoy = s.seal(mask, None)
            self.assertEqual([i.name for i in decoy], ["eor"], "decoy = single flip, no revert")
            self.assertEqual(S._encode(decoy[0]), eor_imm_word(_TGT_N, _TGT_N, mask))

    def test_placeholder_encodes_to_nop(self):
        s = S.BranchTargetSealing(TGT, None, [1 << _VA])
        self.assertTrue(all(S._encode(i) == NOP_WORD for i in s.slot_insts))


class ResolveBranchTargetTest(unittest.TestCase):
    def setUp(self):
        self.pool = [1 << _VA, 1]
        self.blr = object()
        self.sealing = S.BranchTargetSealing(TGT, self.blr, self.pool)
        self.layout = types.SimpleNamespace(instruction_address={self.blr: 0x20})

    def _resolve(self, cer):
        return S._resolve_branch_target(self.sealing, cer, self.layout)

    def test_value_and_pool(self):
        value, alts, _ = self._resolve([_Ite(0x0, 0)])
        self.assertIsNone(value, "genuine keeps a valid target")
        self.assertEqual(alts, self.pool)

    # cer[0].pc is the code base; the BLR is at base + 0x20 (its layout offset).
    def test_architectural_not_speculative(self):
        _, _, spec = self._resolve([_Ite(0x100, 0), _Ite(0x120, 0)])   # BLR reached at nesting 0
        self.assertEqual(spec, 0)
        self.assertFalse(S._Resolved(self.sealing, None, self.pool, spec).speculative)

    def test_speculative_only(self):
        _, _, spec = self._resolve([_Ite(0x100, 0), _Ite(0x120, 3)])
        self.assertEqual(spec, 3)
        self.assertTrue(S._Resolved(self.sealing, None, self.pool, spec).speculative)

    def test_min_nesting_wins(self):
        _, _, spec = self._resolve([_Ite(0x100, 0), _Ite(0x120, 3), _Ite(0x120, 0)])
        self.assertEqual(spec, 0)


if __name__ == "__main__":
    unittest.main()
