"""Canonicality non-interference sealing (AArch64), self-contained (no /dev/executor, no CE):
  * the EOR / bitmask-immediate encoder is byte-exact against the repo assembler;
  * _canon_mask_pool draws RANDOM contiguous flip runs within the guaranteed-fault range [54:VA]
    (a random subset per decoy, like PAC's forgery pool), deterministic per salt; config mask must be
    one such run;
  * CanonSealing flips the base before the access and, when the base is preserved, flips it back
    after (EOR self-inverse) so the decoy re-converges with the baseline — no downstream cascade;
  * _resolve_canon offers the pool and classifies architectural vs speculative-only;
  * end-to-end, genuine keeps every slot canonical (NOPs) and decoy flips+reverts only the
    speculative-only slot with a single chosen mask.
"""
import os
import sys
import types
import random
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import CONF
from src.aarch64.aarch64_relocations import (encode_bitmask_imm64, eor_imm_word, NOP_WORD, Relocation)
from src.aarch64.aarch64_generator import Aarch64Generator
from src.aarch64.seal import sealer as S

_FAULT_39 = (((1 << (55 - 39)) - 1) << 39)   # bits [54:39]


def _asm_word(text: str) -> int:
    return int.from_bytes(Aarch64Generator.in_memory_assemble(text)[:4], "little")


class _Ite:
    def __init__(self, pc, nesting, has_mem):
        self.cpu = types.SimpleNamespace(pc=pc)
        self.metadata = types.SimpleNamespace(speculation_nesting=nesting, has_memory_access=has_mem)


class BitmaskEncoderTest(unittest.TestCase):
    def test_eor_word_matches_assembler(self):
        for mask in (1 << 47, 1 << 54, 1 << 39, 0x007FFF8000000000):
            with self.subTest(mask=hex(mask)):
                self.assertEqual(eor_imm_word(0, 0, mask), _asm_word(f"EOR X0, X0, #{mask}"))
        self.assertEqual(eor_imm_word(3, 3, 1 << 47), _asm_word("EOR X3, X3, #0x800000000000"))

    def test_non_encodable_masks_raise(self):
        for bad in (0, (1 << 64) - 1, (1 << 47) | (1 << 40)):
            with self.subTest(bad=hex(bad)):
                with self.assertRaises(ValueError):
                    encode_bitmask_imm64(bad)


class MaskRunsTest(unittest.TestCase):
    def test_single_run(self):
        self.assertEqual(S._mask_runs(1 << 47), [1 << 47])
        self.assertEqual(S._mask_runs(0x007FFF8000000000), [0x007FFF8000000000])

    def test_gapped_mask_splits(self):
        self.assertEqual(S._mask_runs((1 << 47) | (1 << 40)), [1 << 40, 1 << 47])

    def test_empty(self):
        self.assertEqual(S._mask_runs(0), [])


class CanonPoolTest(unittest.TestCase):
    def setUp(self):
        CONF.instruction_set = "aarch64"
        CONF.set_to_arch_defaults()
        CONF.va_size = 39
        CONF.canonicality_mask = None

    def tearDown(self):
        CONF.set_to_arch_defaults()

    def test_pool_masks_are_single_faulting_runs(self):
        pool = S._canon_mask_pool(12345)
        self.assertTrue(pool)
        for m in pool:
            self.assertNotEqual(m, 0)
            self.assertEqual(S._mask_runs(m), [m], f"{m:#x} not a single run")
            self.assertFalse(m & ~_FAULT_39, f"{m:#x} outside [54:39]")

    def test_pool_is_random_not_all_the_same(self):
        self.assertGreater(len(set(S._canon_mask_pool(12345))), 1, "pool should vary (random subset)")

    def test_pool_deterministic_per_salt(self):
        self.assertEqual(S._canon_mask_pool(7), S._canon_mask_pool(7))
        self.assertNotEqual(S._canon_mask_pool(7), S._canon_mask_pool(8))

    def test_config_mask_single_run_passes(self):
        CONF.canonicality_mask = 1 << 47
        self.assertEqual(S._canon_mask_pool(1), [1 << 47])

    def test_config_mask_multi_run_rejected(self):
        CONF.canonicality_mask = (1 << 47) | (1 << 40)
        with self.assertRaisesRegex(Exception, "single contiguous run"):
            S._canon_mask_pool(1)

    def test_config_mask_outside_fault_range_rejected(self):
        CONF.canonicality_mask = 1 << 60   # top byte, not in [54:39]
        with self.assertRaisesRegex(Exception, "single contiguous run"):
            S._canon_mask_pool(1)

    def test_missing_va_rejected(self):
        CONF.va_size = None
        with self.assertRaisesRegex(Exception, "va_size"):
            S._canon_mask_pool(1)

    def test_va_out_of_range_rejected(self):
        CONF.va_size = 55
        with self.assertRaisesRegex(Exception, "out of range"):
            S._canon_mask_pool(1)


class CanonSealingRenderTest(unittest.TestCase):
    def setUp(self):
        self.pool = [1 << 47]

    def test_base_preserved_flips_then_reverts(self):
        s = S.CanonSealing("x3", None, self.pool, revert=True)
        self.assertEqual([i.name for i in s.seal(None, None)], ["nop", "nop"], "genuine = 2 NOPs")
        decoy = s.seal(1 << 47, None)
        self.assertEqual([i.name for i in decoy], ["eor", "eor"], "decoy = flip + revert")
        w0, w1 = S._encode(decoy[0]), S._encode(decoy[1])
        self.assertEqual(w0, eor_imm_word(3, 3, 1 << 47))
        self.assertEqual(w0, w1, "same mask before and after -> EOR self-inverse -> base restored")

    def test_base_not_preserved_flips_only(self):
        s = S.CanonSealing("x3", None, self.pool, revert=False)
        self.assertEqual([i.name for i in s.seal(None, None)], ["nop"], "genuine = 1 NOP")
        self.assertEqual([i.name for i in s.seal(1 << 47, None)], ["eor"], "decoy = flip only")

    def test_placeholder_encodes_to_nop(self):
        s = S.CanonSealing("x5", None, self.pool, revert=True)
        self.assertTrue(all(S._encode(i) == NOP_WORD for i in s.slot_insts))


class ResolveCanonTest(unittest.TestCase):
    def setUp(self):
        self.pool = [1 << 47, 1 << 48]
        self.access = object()
        self.sealing = S.CanonSealing("x1", self.access, self.pool, revert=True)
        self.layout = types.SimpleNamespace(instruction_address={self.access: 0x20})

    def _resolve(self, cer):
        return S._resolve_canon(self.sealing, cer, self.layout)

    def test_value_and_pool(self):
        value, alts, _ = self._resolve([_Ite(0x0, 0, False)])
        self.assertIsNone(value, "genuine is canonical")
        self.assertEqual(alts, self.pool, "alternatives are the sealing's mask pool")

    def test_architectural_not_speculative(self):
        _, _, spec = self._resolve([_Ite(0x100, 0, False), _Ite(0x120, 0, True)])
        self.assertEqual(spec, 0)
        self.assertFalse(S._Resolved(self.sealing, None, self.pool, spec).speculative)

    def test_speculative_only(self):
        _, _, spec = self._resolve([_Ite(0x100, 0, False), _Ite(0x120, 3, True)])
        self.assertEqual(spec, 3)
        self.assertTrue(S._Resolved(self.sealing, None, self.pool, spec).speculative)

    def test_min_nesting_wins(self):
        _, _, spec = self._resolve([_Ite(0x100, 0, False), _Ite(0x120, 3, True), _Ite(0x120, 0, True)])
        self.assertEqual(spec, 0)

    def test_unreached_is_speculative(self):
        _, _, spec = self._resolve([_Ite(0x100, 0, False)])
        self.assertIsNone(spec)
        self.assertTrue(S._Resolved(self.sealing, None, self.pool, spec).speculative)


class GenuineDecoyTest(unittest.TestCase):
    """genuine keeps the slot canonical (NOPs); decoy flips+reverts a speculative-only slot with one
    chosen pool mask; an architectural slot is never perturbed."""

    def setUp(self):
        self.pool = [1 << 47, 1 << 48, 1 << 49]
        self.object_code = b"\x00" * 8

    def _resolved(self, spec_nesting):
        s = S.CanonSealing("x1", object(), self.pool, revert=True)   # 2-word slot (flip + revert)
        entry = S._Resolved(s, None, self.pool, spec_nesting)
        offsets = {id(s): (0, 4)}
        return S.ResolvedSealingTestCase([entry], self.object_code, offsets, salt=123)

    def test_genuine_all_nop(self):
        r = self._resolved(spec_nesting=3)
        self.assertEqual(r.genuine(), (Relocation(0, NOP_WORD), Relocation(4, NOP_WORD)))

    def test_decoy_flips_and_reverts_speculative_slot(self):
        r = self._resolved(spec_nesting=3)
        relocs = r.decoy(random.Random(1))
        self.assertEqual(len(relocs), 2)
        (o0, w0), (o1, w1) = (relocs[0].offset, relocs[0].value), (relocs[1].offset, relocs[1].value)
        self.assertEqual((o0, o1), (0, 4))
        self.assertNotEqual(w0, NOP_WORD, "decoy flips (EOR), not NOP")
        self.assertEqual(w0, w1, "flip and revert use the SAME mask -> base restored (re-converge)")
        chosen = [eor_imm_word(1, 1, m) for m in self.pool]
        self.assertIn(w0, chosen, "the flip is one of the pool masks")

    def test_decoy_leaves_architectural_slot_canonical(self):
        r = self._resolved(spec_nesting=0)   # architectural -> not decoy-eligible
        self.assertEqual(r.decoy(random.Random(1)), (Relocation(0, NOP_WORD), Relocation(4, NOP_WORD)))


if __name__ == "__main__":
    unittest.main()
