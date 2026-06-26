"""
MTE tag-state model tests.

  1. TestMteTagState        — the speculative address->tag map: uniform default, granule keying,
                              stores, and depth push/pop (speculation copy + unwind revert).
  2. TestCorrectTagFromTrace — _classify_mte_slots populates fp.correct_tag from the trace, tracking
                              architectural and speculative tag stores with revert on unwind.
"""
import types
import random
import unittest

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.aarch64.aarch64_mte import MteTagState, MTE_GRANULE, mte_tag_store_effect, MTEFixPoint, MteTag
import src.aarch64.aarch64_mte as mte_mod


def _classify(cer, offset_to_fp, default_tag):
    """Resolve each MTEFixPoint from the trace by its guarded access's byte offset — the per-fp form
    of the former _classify_mte_slots pass. x29 (in cer[0]) carries the region's default tag."""
    if cer:
        cer[0].cpu.gpr = [0] * 29 + [default_tag << 56] + [0, 0]
    for off, fp in offset_to_fp.items():
        fp.trigger = object()    # stands in for the guarded access instruction
        fp.resolve(cer, types.SimpleNamespace(instruction_address={fp.trigger: off}))


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
        self._orig = mte_mod.mte_tag_store_effect
        mte_mod.mte_tag_store_effect = lambda ite: getattr(ite, "_store", None)

    def tearDown(self):
        mte_mod.mte_tag_store_effect = self._orig

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


class TestMteArchTagFix(unittest.TestCase):
    """The architectural genuine fill (MteTag.genuine): NOP when the pointer's tag already matches
    the cell's allocation tag, ADDG #delta when it doesn't — including after an STG retag. Standalone
    (its own STG stub) so it neither inherits nor leaks into other test classes."""

    _CELL = 0x4000_0000

    def setUp(self):
        self._orig = mte_mod.mte_tag_store_effect
        mte_mod.mte_tag_store_effect = lambda ite: getattr(ite, "_store", None)

    def tearDown(self):
        mte_mod.mte_tag_store_effect = self._orig

    def _genuine(self, stores, ptr_tag, default_tag):
        """stores: (addr, tag, n) applied architecturally before an access to _CELL whose pointer
        carries ptr_tag. Returns (genuine mnemonics, fp). An anchor entry pins code_base so the
        access's offset is stable regardless of how many stores precede it."""
        cer = [_ite(0, 0, ea=0x5000_0000)]                                   # anchor (not a fix point)
        cer += [_ite(0x10, 0, ea=s[0], store=s) for s in stores]
        cer.append(_ite(0x40, 0, ea=(ptr_tag << 56) | self._CELL))
        fp = MTEFixPoint(slot_id=0, value_reg="x5")
        _classify(cer, {0x40: fp}, default_tag=default_tag)
        return [i.name for i in MteTag().genuine(fp, random.Random(0))], fp

    def test_match_no_store_is_nop(self):
        out, _ = self._genuine([], ptr_tag=6, default_tag=6)                 # ptr 6 == region 6
        self.assertEqual(out, ["nop"])

    def test_mismatch_no_store_fixes_with_addg(self):
        out, fp = self._genuine([], ptr_tag=3, default_tag=6)                # ptr 3 != cell 6
        self.assertEqual(out, ["addg"])
        self.assertEqual((fp.correct_tag, fp.ptr_tag), (6, 3))

    def test_stg_changes_tag_then_mismatch_fixes(self):
        out, fp = self._genuine([(self._CELL, 9, 1)], ptr_tag=6, default_tag=6)  # cell 9, ptr 6
        self.assertEqual(fp.correct_tag, 9)
        self.assertEqual(out, ["addg"])

    def test_stg_changes_tag_but_ptr_matches_is_nop(self):
        # STG retags the cell to 9 AND the access pointer carries tag 9: changed, yet correct -> NOP.
        out, fp = self._genuine([(self._CELL, 9, 1)], ptr_tag=9, default_tag=6)
        self.assertEqual(fp.correct_tag, 9)
        self.assertEqual(out, ["nop"])                                       # no tag-fix needed

    def test_stg_changes_then_code_flow_restores_is_nop(self):
        # STG retags to 9, a later arch STG restores the region tag 6; ptr tag 6 -> matches -> NOP.
        out, fp = self._genuine([(self._CELL, 9, 1), (self._CELL, 6, 1)], ptr_tag=6, default_tag=6)
        self.assertEqual(fp.correct_tag, 6)
        self.assertEqual(out, ["nop"])


class _Rng:
    """Deterministic stand-in: random() returns a fixed value; choice picks a fixed index."""
    def __init__(self, r, idx=0): self._r, self._idx = r, idx
    def random(self): return self._r
    def choice(self, seq): return seq[self._idx]


class TestMteDecoy(unittest.TestCase):
    """The speculative decoy fill (MteTag.decoy): a retag — IRG (random tag) or EOR (flip tag bits)."""
    def _fp(self):
        fp = MTEFixPoint(slot_id=0, value_reg="x5"); fp.correct_tag, fp.ptr_tag = 6, 6
        return fp

    def test_decoy_irg(self):
        out = MteTag().decoy(self._fp(), _Rng(0.1))          # random() < 0.5 -> IRG
        self.assertEqual([i.name for i in out], ["irg"])

    def test_decoy_eor_flips_tag_bits_only(self):
        out = MteTag().decoy(self._fp(), _Rng(0.9, idx=0))   # random() >= 0.5 -> EOR a tag mask
        self.assertEqual([i.name for i in out], ["eor"])
        # the EOR mask must touch only the tag field [59:56]
        mask = int(out[0].template.split("#0x")[1], 16)
        self.assertEqual(mask & ~(0xF << 56), 0)
        self.assertNotEqual(mask, 0)


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
