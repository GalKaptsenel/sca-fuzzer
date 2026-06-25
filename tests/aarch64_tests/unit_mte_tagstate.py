"""
MTE tag-state model tests.

  1. TestMteTagState        — the speculative address->tag map: uniform default, granule keying,
                              stores, and depth push/pop (speculation copy + unwind revert).
  2. TestCorrectTagFromTrace — _classify_mte_slots populates fp.correct_tag from the trace, tracking
                              architectural and speculative tag stores with revert on unwind.
"""
import types
import unittest

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.aarch64.aarch64_mte import MteTagState, MTE_GRANULE, mte_tag_store_effect, MTEFixPoint
import src.aarch64.aarch64_executor as ex

_classify = ex.Aarch64MteNonInterferenceExecutor._classify_mte_slots


# ===========================================================================
# 1. MteTagState
# ===========================================================================

class TestMteTagState(unittest.TestCase):

    def test_uniform_default(self):
        s = MteTagState(default_tag=5)
        self.assertEqual(s.tag_at(0x4000_0000), 5)

    def test_set_overrides_one_granule(self):
        s = MteTagState(default_tag=5)
        s.set(0x4000_1234, tag=9)
        self.assertEqual(s.tag_at(0x4000_1234), 9)       # same byte
        self.assertEqual(s.tag_at(0x4000_1230), 9)       # same granule
        self.assertEqual(s.tag_at(0x4000_1240), 5)       # next granule -> default

    def test_granule_ignores_tag_byte(self):
        tagged, untagged = 0x0500_0000_4000_1234, 0x0000_0000_4000_1234
        self.assertEqual(MteTagState.granule(tagged), MteTagState.granule(untagged))
        s = MteTagState(default_tag=0)
        s.set(tagged, tag=7)
        self.assertEqual(s.tag_at(untagged), 7)

    def test_set_two_granules(self):
        s = MteTagState(default_tag=0)
        s.set(0x4000_0000, tag=3, n_granules=2)
        self.assertEqual(s.tag_at(0x4000_0000), 3)
        self.assertEqual(s.tag_at(0x4000_0000 + MTE_GRANULE), 3)
        self.assertEqual(s.tag_at(0x4000_0000 + 2 * MTE_GRANULE), 0)

    # ── speculation: copy on enter, revert on unwind ─────────────────────
    def test_speculation_inherits_architectural(self):
        s = MteTagState(default_tag=0)
        s.set(0x4000_0000, tag=2)        # architectural
        s.to_depth(1)                    # enter speculation -> copies the arch layer
        self.assertEqual(s.tag_at(0x4000_0000), 2)

    def test_speculative_store_reverted_on_unwind(self):
        s = MteTagState(default_tag=0)
        s.to_depth(1)
        s.set(0x4000_0000, tag=9)        # speculative store
        self.assertEqual(s.tag_at(0x4000_0000), 9)   # visible within the window
        s.to_depth(0)                    # unwind
        self.assertEqual(s.tag_at(0x4000_0000), 0)   # reverted

    def test_nested_unwind_keeps_shallower_store(self):
        s = MteTagState(default_tag=0)
        s.to_depth(1); s.set(0x4000_0000, tag=5)     # depth-1 speculative store
        s.to_depth(2); s.set(0x4000_0000, tag=7)     # depth-2 speculative store
        self.assertEqual(s.tag_at(0x4000_0000), 7)
        s.to_depth(1)                                 # unwind depth 2 only
        self.assertEqual(s.tag_at(0x4000_0000), 5)   # depth-1 store survives
        s.to_depth(0)
        self.assertEqual(s.tag_at(0x4000_0000), 0)   # all speculative gone


# ===========================================================================
# 2. correct_tag populated from the trace
# ===========================================================================

_CODE_BASE = 0x1000

def _ite(pc_off, nesting, ea, mem=True, store=None):
    ns = types.SimpleNamespace
    md = ns(has_memory_access=mem, speculation_nesting=nesting,
            memory_access=ns(effective_address=ea))
    it = ns(cpu=ns(pc=_CODE_BASE + pc_off, encoding=0), metadata=md)
    it._store = store  # (addr, tag, n_granules) or None
    return it


class TestCorrectTagFromTrace(unittest.TestCase):

    def setUp(self):
        # Stub the disassembly-based STG detection: an entry stores iff it carries ._store.
        self._orig = ex.mte_tag_store_effect
        ex.mte_tag_store_effect = lambda ite: getattr(ite, "_store", None)

    def tearDown(self):
        ex.mte_tag_store_effect = self._orig

    def _fp(self, slot_id):
        return MTEFixPoint(slot_id=slot_id, value_reg="x5")

    def test_default_tag_when_no_stores(self):
        fp = self._fp(0)
        _classify([_ite(0, 0, ea=0x4000_0000)], {0: fp}, default_tag=6)
        self.assertEqual(fp.correct_tag, 6)

    def test_ptr_tag_is_ea_top_byte(self):
        fp = self._fp(0)
        _classify([_ite(0, 0, ea=0x0500_0000_4000_0000)], {0: fp}, default_tag=6)
        self.assertEqual(fp.ptr_tag, 5)          # pointer's own tag (EA bits [59:56])
        self.assertEqual(fp.correct_tag, 6)      # cell's allocation tag (region default)

    def test_architectural_store_updates_tag(self):
        fp = self._fp(0)
        cer = [
            _ite(0, 0, ea=0x4000_0000, store=(0x4000_0000, 9, 1)),  # STG (arch) tags the cell 9
            _ite(8, 0, ea=0x4000_0000),                             # later access to that cell
        ]
        _classify(cer, {8: fp}, default_tag=6)
        self.assertEqual(fp.correct_tag, 9)

    def test_speculative_store_reverted_for_arch_access(self):
        fp = self._fp(0)
        cer = [
            _ite(0, 2, ea=0x4000_0000, store=(0x4000_0000, 9, 1)),  # STG under speculation
            _ite(8, 0, ea=0x4000_0000),                             # arch access after unwind
        ]
        _classify(cer, {8: fp}, default_tag=6)
        self.assertEqual(fp.correct_tag, 6)                         # reverted -> default

    def test_speculative_access_sees_speculative_store(self):
        fp = self._fp(0)
        cer = [
            _ite(0, 1, ea=0x4000_0000, store=(0x4000_0000, 9, 1)),  # speculative STG
            _ite(8, 1, ea=0x4000_0000),                             # speculative access, same window
        ]
        _classify(cer, {8: fp}, default_tag=6)
        self.assertEqual(fp.spec_nesting, 1)
        self.assertEqual(fp.correct_tag, 9)                         # sees the live speculative tag

    def test_store_to_other_granule_does_not_affect(self):
        fp = self._fp(0)
        cer = [
            _ite(0, 0, ea=0x4000_0100, store=(0x4000_0100, 9, 1)),
            _ite(8, 0, ea=0x4000_0000),
        ]
        _classify(cer, {8: fp}, default_tag=6)
        self.assertEqual(fp.correct_tag, 6)

    def test_st2g_tags_two_granules(self):
        fp = self._fp(0)
        cer = [
            _ite(0, 0, ea=0x4000_0000, store=(0x4000_0000, 4, 2)),
            _ite(8, 0, ea=0x4000_0000 + MTE_GRANULE),
        ]
        _classify(cer, {8: fp}, default_tag=6)
        self.assertEqual(fp.correct_tag, 4)


# ===========================================================================
# 3. mte_tag_store_effect — STG detection + the written tag is Xt's tag
# ===========================================================================

class TestMteTagStoreEffect(unittest.TestCase):

    def _ite(self, encoding, gpr):
        ns = types.SimpleNamespace
        regs = list(gpr) + [0] * (31 - len(gpr))
        return ns(cpu=ns(pc=0, encoding=encoding, gpr=regs, sp=0),
                  metadata=ns(has_memory_access=True,
                              memory_access=ns(effective_address=0x4000_0000)))

    def test_non_store_is_none(self):
        # an LDR (not a tag store) -> None
        self.assertIsNone(mte_tag_store_effect(self._ite(0xf9400020, [0])))

    def test_stg_uses_xt_tag_not_address_tag(self):
        # stg x0, [x1]: tag written is x0's tag (5), independent of x1/the address tag
        ite = self._ite(0xd9200820, gpr=[5 << 56])   # x0 carries tag 5
        eff = mte_tag_store_effect(ite)
        self.assertIsNotNone(eff)
        addr, tag, n = eff
        self.assertEqual(tag, 5)              # Xt's (x0) tag, not the EA's
        self.assertEqual(n, 1)

    def test_st2g_two_granules(self):
        addr, tag, n = mte_tag_store_effect(self._ite(0xd9a00862, gpr=[0, 0, 7 << 56]))  # x2 tag 7
        self.assertEqual((tag, n), (7, 2))


# ===========================================================================
# 4. Deep speculation nesting — copy-on-enter, pop-on-unwind across many levels
# ===========================================================================

class TestMteTagStateDeepNesting(unittest.TestCase):

    def test_deep_push_then_partial_unwind(self):
        s = MteTagState(default_tag=0)
        s.set(0x4000_0000, tag=1)                      # architectural (layer 0)
        for depth in range(1, 6):                      # push to depth 5; each layer tags distinctly
            s.to_depth(depth)
            s.set(0x4000_0000, tag=depth + 1)          # depth d -> tag d+1
        self.assertEqual(s.tag_at(0x4000_0000), 6)     # deepest live store (depth 5 -> tag 6)
        s.to_depth(2)                                  # unwind 5,4,3
        self.assertEqual(s.tag_at(0x4000_0000), 3)     # depth-2 store survives (tag 3)
        s.to_depth(0)                                  # unwind everything speculative
        self.assertEqual(s.tag_at(0x4000_0000), 1)     # back to the architectural tag

    def test_reentering_a_depth_starts_from_shallower_state(self):
        s = MteTagState(default_tag=0)
        s.to_depth(1); s.set(0x4000_0000, tag=4)       # depth-1 speculative store
        s.to_depth(0)                                  # window collapses
        s.to_depth(1)                                  # a *new* depth-1 window
        self.assertEqual(s.tag_at(0x4000_0000), 0)     # inherits arch (0), not the old tag 4


if __name__ == "__main__":
    unittest.main()
