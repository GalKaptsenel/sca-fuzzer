"""
BPU model correctness tests.

Every test is grounded in one of:
  (RE)   A specific claim from the Neoverse N3 reverse-engineering findings
  (TAGE) A fundamental property of the TAGE predictor (Seznec & Michaud 2006)

Tests do NOT inspect raw counter state (_state).  They drive observable
predictions through the public API and check that the BEHAVIOUR matches
the claim — not that the arithmetic produces a particular internal value.

Run with:  python tests/unit_saturating_bp.py
"""

import os
import sys
import random
import unittest

# Import the BPU model exactly as the contract executor loads it — as a top-level sibling module
# from its own directory — rather than through the `src` package (whose __init__ pulls in the heavy
# revizor stack, scipy included). Keeps this unit test fast and dependency-light.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "aarch64", "contract_executor"))

from saturating_bp import (
    SaturatingCounterBP,
    SaturatingCounterBPCommon,
    TAGEPHT,
    BimodalBP,
    PHR,
    Aarch64NeoverseN3BPU,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fresh_bpu():
    # Explicit fold width: the model has no default; the tests pin the 11-bit fold they assert on.
    return Aarch64NeoverseN3BPU(fold_group_width=11, tag_width=8)

def step(predictor, address, taken, target):
    """A full non-speculative branch step: update the prediction tables, then advance the history.
    The model has no all-in-one method by design (the driver/engine composes the two), so the tests
    compose them here."""
    predictor.update(address, taken, target)
    predictor.advance(address, taken, target)

def make_pht(n_bits=3):
    """Minimal TAGEPHT with identity index/tag functions for isolation."""
    return TAGEPHT(
        counter_bit_width=n_bits, num_sets=1, assoc=1,
        index_fn=lambda pc: 0,
        tag_fn=lambda pc: pc,
    )

FLUSH_A, FLUSH_B = 0xF000, 0xF100   # canonical flush branch pair

def flush_phr(bpu, n_taken_branches):
    """Simulate n_taken_branches taken-branch updates to shift the PHR."""
    for i in range(n_taken_branches):
        addr   = FLUSH_A if i % 2 == 0 else FLUSH_B
        target = FLUSH_B if i % 2 == 0 else FLUSH_A
        step(bpu,addr, taken=True, target=target)


# ---------------------------------------------------------------------------
# (TAGE) Saturating counter semantics
# ---------------------------------------------------------------------------

class TestSaturatingCounterBehaviour(unittest.TestCase):
    """
    Properties every N-bit saturating counter must have regardless of
    bit-width or initial state.  No internal state (_state) is read.
    """

    def _drive_to_saturation(self, bp, addr, taken, n=16):
        """Push counter far past the threshold so it is fully saturated."""
        for _ in range(n):
            bp.update(addr, taken=taken)

    def test_default_predicts_not_taken(self):
        """(TAGE) Base-table entries default to weakly-not-taken.
        A cold (never-updated) counter must predict False."""
        bp = SaturatingCounterBP(counter_bit_width=3, num_sets=1, assoc=1)
        bp.update(0x1000, taken=False)   # allocates (miss → insert with default)
        # one not-taken update from weakly-not-taken should keep it not-taken
        self.assertFalse(bp.predict(0x1000))

    def test_weakly_correct_allocation_flips_in_exactly_one_update(self):
        """(TAGE) Newly allocated entries must be *weakly* correct.
        'Weakly' means a single update in the opposite direction crosses
        the decision boundary — not two, not zero.

        This is the defining property that TAGE allocation initialises to
        (threshold) for taken and (threshold-1) for not-taken.
        """
        for n_bits in [2, 3, 4]:
            with self.subTest(n_bits=n_bits):
                bp = SaturatingCounterBP(counter_bit_width=n_bits, num_sets=1, assoc=1)

                # Weakly-taken allocation: one not-taken must flip prediction
                bp.allocate(0xAAAA, taken=True)
                self.assertTrue(bp.predict(0xAAAA), "just allocated taken → must predict taken")
                bp.update(0xAAAA, taken=False)
                self.assertFalse(bp.predict(0xAAAA), "one not-taken must flip weakly-taken")

                # Weakly-not-taken allocation: one taken must flip prediction
                bp.allocate(0xBBBB, taken=False)
                self.assertFalse(bp.predict(0xBBBB), "just allocated not-taken → must predict not-taken")
                bp.update(0xBBBB, taken=True)
                self.assertTrue(bp.predict(0xBBBB), "one taken must flip weakly-not-taken")

    def test_saturated_counter_requires_multiple_flips(self):
        """(TAGE) A strongly saturated counter must withstand more than one
        opposite update before changing its prediction.

        This is what 'strongly taken/not-taken' means: the counter has
        been reinforced and needs to be worn down before it flips.
        """
        for n_bits in [2, 3, 4]:
            with self.subTest(n_bits=n_bits):
                bp = SaturatingCounterBP(counter_bit_width=n_bits, assoc=1, num_sets=1)
                addr = 0xCAFE

                # Saturate towards taken
                self._drive_to_saturation(bp, addr, taken=True)
                self.assertTrue(bp.predict(addr))

                # A single opposite update must NOT flip a saturated counter
                bp.update(addr, taken=False)
                self.assertTrue(bp.predict(addr),
                    "a single not-taken update must not flip a strongly-taken counter")

    def test_lru_eviction_with_assoc2(self):
        """(TAGE) LRU replacement: the least-recently-used entry is evicted.

        With assoc=2, three sequential inserts (with a touch on the first
        after the second) must evict the second, not the first.
        """
        bp = SaturatingCounterBP(
            counter_bit_width=3, num_sets=1, assoc=2,
            index_fn=lambda pc: 0,
            tag_fn=lambda pc: pc,
        )
        # Insert A, then B.  Touch A (making B the LRU).  Insert C → B evicted.
        bp.allocate(0xAAAA, taken=True)
        bp.allocate(0xBBBB, taken=True)
        bp.predict(0xAAAA, touch_lru=True)    # A is now MRU; B is LRU
        bp.allocate(0xCCCC, taken=True)       # C inserted → B evicted

        self.assertIsNotNone(bp.predict(0xAAAA, touch_lru=False, allocate_on_miss=False), "A must survive")
        self.assertIsNone(   bp.predict(0xBBBB, touch_lru=False, allocate_on_miss=False), "B must be evicted")
        self.assertIsNotNone(bp.predict(0xCCCC, touch_lru=False, allocate_on_miss=False), "C must survive")

    def test_different_addresses_same_set_independently_trained(self):
        """(TAGE) Entries with different tags in the same set are independent.

        With assoc=2, training address A taken and B not-taken must produce
        correct independent predictions for both — they do not interfere.
        """
        bp = SaturatingCounterBP(
            counter_bit_width=3, num_sets=1, assoc=2,
            index_fn=lambda pc: 0,
            tag_fn=lambda pc: pc,
        )
        bp.allocate(0xAAAA, taken=True)
        bp.allocate(0xBBBB, taken=False)
        self.assertTrue( bp.predict(0xAAAA), "A trained taken → must predict taken")
        self.assertFalse(bp.predict(0xBBBB), "B trained not-taken → must predict not-taken")


# ---------------------------------------------------------------------------
# (RE) PHR behaviour
# ---------------------------------------------------------------------------

class TestPHRBehaviour(unittest.TestCase):
    """
    RE findings on the Neoverse N3 PHR:
      - 300 bits = 75 records × 4 bits/record
      - Update rule: PHR_new = (PHR_old << 4) XOR footprint(pc, target)
      - Updated ONLY on taken branches
      - Flush: K=38 iterations × 2 branches = 76 taken updates × 4 bits = 304 bits > 300
    """

    def test_not_taken_branches_never_update_phr(self):
        """(RE) PHR must be unchanged by any number of not-taken branches."""
        bpu = fresh_bpu()
        phr_start = bpu._phr.read()

        for _ in range(100):
            step(bpu,0x1000, taken=False, target=0x1100)

        self.assertEqual(bpu._phr.read(), phr_start,
            "PHR must not change on not-taken branches")

    def test_taken_branch_updates_phr(self):
        """(RE) A single taken branch must change PHR from its prior value.
        Use pc=0x4 (bit 2 set) so footprint = b2^...= 1 ≠ 0.
        Addresses like 0x1000/0x2000 sit in bits [12:13] which are outside
        the footprint formula range [2:9] for PC and [3:10] for target, giving
        footprint=0 and a spurious pass.
        """
        bpu = fresh_bpu()
        phr_before = bpu._phr.read()
        step(bpu,0x4, taken=True, target=0x0)  # footprint(0x4, 0x0) = pc[2]=1 → fp=1
        self.assertNotEqual(bpu._phr.read(), phr_before)

    def test_phr_footprint_target_bits_matter(self):
        """(RE) The footprint formula mixes target bits [3:10].
        Two branches at the same PC but with targets that differ in bit 3
        (the lowest target bit in the formula) must produce different PHR states.
        """
        bpu_a = fresh_bpu()
        bpu_b = fresh_bpu()

        pc = 0x1000
        target_a = 0b0000_0000_0000  # bit 3 = 0
        target_b = 0b0000_0000_1000  # bit 3 = 1

        step(bpu_a,pc, taken=True, target=target_a)
        step(bpu_b,pc, taken=True, target=target_b)

        self.assertNotEqual(bpu_a._phr.read(), bpu_b._phr.read(),
            "target bit 3 is in the footprint formula — different targets must produce different PHR")

    def test_phr_footprint_target_low_bits_irrelevant(self):
        """(RE) Target bits [0:2] are NOT in the footprint formula.
        Two branches at the same PC whose targets differ ONLY in bits [0:2]
        must produce the same PHR state.
        """
        bpu_a = fresh_bpu()
        bpu_b = fresh_bpu()

        pc = 0x1000
        target_base = 0xABC0
        target_a = target_base | 0b000   # bits [0:2] = 0
        target_b = target_base | 0b111   # bits [0:2] = 1

        step(bpu_a,pc, taken=True, target=target_a)
        step(bpu_b,pc, taken=True, target=target_b)

        self.assertEqual(bpu_a._phr.read(), bpu_b._phr.read(),
            "target bits [0:2] are not in the footprint — PHR must be identical")

    def test_phr_footprint_pc_bits_matter(self):
        """(RE) The footprint formula includes PC bits [2:9].
        Two branches at PCs differing in bit 2 (lowest PC bit in formula)
        must produce different PHR states (given same target).
        """
        bpu_a = fresh_bpu()
        bpu_b = fresh_bpu()

        target = 0x5678
        pc_a = 0x1000  # bit 2 = 0
        pc_b = 0x1004  # bit 2 = 1

        step(bpu_a,pc_a, taken=True, target=target)
        step(bpu_b,pc_b, taken=True, target=target)

        self.assertNotEqual(bpu_a._phr.read(), bpu_b._phr.read(),
            "PC bit 2 is in the footprint formula — different PCs must produce different PHR")

    def test_phr_left_shift_semantics_oldest_record_shifts_out(self):
        """(RE) PHR update is a left-shift by 4.  After 75 records (300 bits)
        the oldest record is exactly at the MSB.  After 76 records it is shifted
        out and lost — the PHR only retains the latest 75 branch records.

        Uses pc=0x4 (bit 2 set) so footprint=1 (nonzero).
        Flush branches use addresses whose footprints are all 0 (bits [2:9] of PC
        and [3:10] of target are zero for the 0x100*(i%8) pattern), so they do
        not contaminate bits [296:299] during the 74-update window.
        """
        bpu = fresh_bpu()
        # One branch: footprint(0x4, 0x0) = pc[2]=1 → PHR = 1, in bits [0:3]
        step(bpu,0x4, taken=True, target=0x0)
        footprint_0 = bpu._phr.read() & 0xF   # = 1

        # 74 more taken branches — all have footprint=0 (verified: 0x100*(i%8)
        # have bits [2:9] = 0 so the formula produces 0 for all 8 cycling addresses)
        for i in range(74):
            step(bpu,0x100 * (i % 8), taken=True, target=0x200 * (i % 8))

        # After 75 total: footprint_0 has been shifted left by 74×4=296 bits
        # and sits exactly in bits [296:299].  The 74 flush branches contribute
        # nothing there (their highest contribution is at bit 4×73=292).
        phr_75 = bpu._phr.read()
        self.assertEqual((phr_75 >> 296) & 0xF, footprint_0,
            "after 75 taken branches, the first branch's footprint must occupy bits [296:299]")

        # 76th branch: footprint(0xABC, 0xDEF) = 7 (nonzero).
        # This shifts footprint_0 to bit 300 — outside the 300-bit mask, so it
        # disappears.  Bits [296:299] now contain the first flush branch's footprint
        # shifted up to position 4×74=296, which ≠ footprint_0.
        step(bpu,0xABC, taken=True, target=0xDEF)   # footprint = 7
        self.assertNotEqual((bpu._phr.read() >> 296) & 0xF, footprint_0,
            "after 76 taken branches, the original footprint must have been shifted out")

    def test_76_taken_branch_flush_makes_phr_deterministic(self):
        """(RE) The ARM Spectre-BHB flush sequence uses K=38 iterations × 2 branches
        = 76 taken updates × 4 bits = 304 bits > 300 bits PHR length.

        After this flush, PHR must be identical for any two BPUs regardless
        of their starting state, because the shift erases all original bits.
        """
        bpu_cold  = fresh_bpu()
        bpu_dirty = fresh_bpu()

        # Contaminate bpu_dirty with 60 random taken branches
        rng = random.Random(0xDEADBEEF)
        for _ in range(60):
            step(bpu_dirty,rng.randint(0, 0xFFFF) & ~3, taken=True,
                             target=rng.randint(0, 0xFFFF) & ~3)

        self.assertNotEqual(bpu_cold._phr.read(), bpu_dirty._phr.read(),
            "sanity: PHR values must differ before flush")

        # Apply identical 76-branch flush to both
        flush_phr(bpu_cold,  76)
        flush_phr(bpu_dirty, 76)

        self.assertEqual(bpu_cold._phr.read(), bpu_dirty._phr.read(),
            "76 taken branches (304 bits > 300-bit PHR) must fully overwrite all prior state")

    def test_74_taken_branch_flush_is_insufficient(self):
        """(RE) 74 taken branches × 4 bits = 296 bits < 300 bits.
        The initial PHR's lowest 4 bits survive in positions [296:299].
        Two BPUs whose starting PHR differs only in bits [0:3] must still
        differ after a 74-branch flush.
        """
        bpu_a = fresh_bpu()
        bpu_b = fresh_bpu()

        # Differ only in the lowest 4 PHR bits — these will survive
        bpu_a._phr._value = 0x0
        bpu_b._phr._value = 0xF   # bits [0:3] all set

        flush_phr(bpu_a, 74)
        flush_phr(bpu_b, 74)

        self.assertNotEqual(bpu_a._phr.read(), bpu_b._phr.read(),
            "74-branch flush covers only 296 bits; the initial 4-bit footprint survives in bits [296:299]")


# ---------------------------------------------------------------------------
# (RE) Index function: PC bit usage
# ---------------------------------------------------------------------------

class TestIndexFunctionFromRE(unittest.TestCase):
    """
    RE finding: for tagged PHTs, the index uses PC bits [8:18] (10 bits) XOR-ed
    with the lowest bits of PHR.  PC bits [0:7] are NOT part of the index.

    This is tested by directly calling the index function (the RE characterised
    which PC bits influence the PHT set — verifying that is not reimplementing
    the code; it is checking the RE claim holds).
    """

    def setUp(self):
        self.bpu = fresh_bpu()
        self.index_fn_t1 = self.bpu._phts[1]._bp._index_fn   # table 1 index
        self.index_fn_t2 = self.bpu._phts[2]._bp._index_fn   # table 2 index
        # Tests run with PHR=0 (fresh BPU) so PHR contribution is zero

    def test_pc_bits_0_to_7_do_not_affect_table1_index(self):
        """(RE) PC bits [0:7] not used in index → addresses differing ONLY in
        bits [0:7] must map to the SAME PHT set."""
        base = 0x12300   # bits [8+] are fixed
        for bit in range(0, 8):
            addr_with_bit = base | (1 << bit)
            self.assertEqual(
                self.index_fn_t1(base),
                self.index_fn_t1(addr_with_bit),
                f"PC bit {bit} (in [0:7]) must NOT affect table-1 index"
            )

    def test_pc_bit_8_affects_table1_index(self):
        """(RE) PC bit 8 IS used in the index.
        Two addresses differing in bit 8 must map to different PHT sets.
        Use 0x0000 vs 0x0100 — bit 8 is unambiguously 0 vs 1.
        (0x12300 already has bit 8 set, so 0x12300 | 0x100 is a no-op.)
        """
        addr_a = 0x0000   # bit 8 = 0
        addr_b = 0x0100   # bit 8 = 1
        self.assertNotEqual(
            self.index_fn_t1(addr_a),
            self.index_fn_t1(addr_b),
            "PC bit 8 IS in the index — addresses must map to different sets"
        )

    def test_same_index_fn_for_both_tagged_tables(self):
        """(RE) Both tagged tables use PC[8:18] XOR PHR as index.
        Two addresses differing only in PC[0:7] must share the same index
        in BOTH table 1 and table 2.
        """
        base = 0xABC00
        addr_with_low_bits = base | 0xFF  # bits [0:7] all set
        for fn, name in [(self.index_fn_t1, "table1"), (self.index_fn_t2, "table2")]:
            self.assertEqual(fn(base), fn(addr_with_low_bits),
                f"PC[0:7] must not affect {name} index")

    def test_index_uses_folded_history_not_just_low_bits(self):
        """(RE) Index = (PC>>8) XOR fold(PHR). The history is XOR-folded in group_width(=11)-bit
        chunks, so high PHR bits are NOT lost: PHR bit 11 folds onto the same index bit as PHR
        bit 0, and the two XOR-cancel. This is the folded-history behaviour, not a low-bits mask.
        """
        addr = 0x12300
        bpu = fresh_bpu()
        index_fn = bpu._phts[1]._bp._index_fn

        baseline = index_fn(addr)             # PHR = 0

        bpu._phr._value = 1 << 0
        idx_bit0 = index_fn(addr)
        self.assertNotEqual(idx_bit0, baseline, "PHR bit 0 must affect the index")

        bpu._phr._value = 1 << 11             # one group up -> folds onto bit 0
        self.assertEqual(index_fn(addr), idx_bit0,
            "PHR bit 11 folds onto the same index bit as PHR bit 0 (11-bit fold)")

        bpu._phr._value = (1 << 0) | (1 << 11)   # both set -> XOR-cancel in the fold
        self.assertEqual(index_fn(addr), baseline,
            "PHR bits 0 and 11 must XOR-cancel under the fold")


# ---------------------------------------------------------------------------
# (TAGE) Allocation policy
# ---------------------------------------------------------------------------

class TestTAGEAllocationPolicy(unittest.TestCase):
    """
    TAGE allocation rules (Seznec & Michaud 2006, section 3):
      - Only update the *provider* (longest matching table) on each branch
      - On misprediction, allocate in the *first* longer-history table with a tag miss
      - Allocation initialises the entry to weakly-correct direction
      - No allocation on a correct prediction
    """

    def test_no_allocation_in_longer_tables_on_correct_prediction(self):
        """(TAGE) Allocation happens only on misprediction.
        If the base predictor correctly predicts not-taken from the start,
        tables 1 and 2 must remain completely empty.
        """
        bpu = fresh_bpu()
        snap_t1_before = bpu._phts[1].snapshot()
        snap_t2_before = bpu._phts[2].snapshot()

        # Default prediction is not-taken, so all not-taken updates are correct
        for _ in range(20):
            step(bpu,0x1000, taken=False, target=0x1100)

        self.assertEqual(bpu._phts[1].snapshot(), snap_t1_before,
            "no misprediction → table 1 must stay empty")
        self.assertEqual(bpu._phts[2].snapshot(), snap_t2_before,
            "no misprediction → table 2 must stay empty")

    def test_misprediction_allocates_in_longer_table_not_taken(self):
        """(TAGE) On misprediction, the longer-history table overrides the base.

        Base is driven strongly taken.  One not-taken branch mispredicts.
        Since not-taken doesn't update PHR, the NEXT prediction uses the same
        PHR state as the allocation — the longer-table entry must be found and
        override the strongly-taken base prediction.
        """
        bpu = fresh_bpu()
        addr, target = 0x2000, 0x2100

        # Drive base to strongly taken (correct predictions, PHR grows each time)
        for _ in range(8):
            step(bpu,addr, taken=True, target=target)
        self.assertTrue(bpu.predict(addr), "base must be strongly taken after 8 updates")

        # One not-taken: misprediction → allocation in table 1 or 2 (weakly not-taken)
        # PHR does NOT change on not-taken, so the subsequent predict uses the same index
        step(bpu,addr, taken=False, target=target)
        self.assertFalse(bpu.predict(addr),
            "longer-table entry (weakly not-taken) must override strongly-taken base")

    def test_longer_table_entry_persists_across_one_base_update(self):
        """(TAGE) Once a longer-table entry exists, it remains the provider
        (higher priority than base) even after the base is updated.

        After the above misprediction scenario, the next correct not-taken
        update trains the longer-table entry further (weakly → more strongly
        not-taken).  Prediction should still be not-taken.
        """
        bpu = fresh_bpu()
        addr, target = 0x3000, 0x3100

        for _ in range(8):
            step(bpu,addr, taken=True, target=target)
        step(bpu,addr, taken=False, target=target)   # misprediction + allocation
        step(bpu,addr, taken=False, target=target)   # correct not-taken in longer table
        self.assertFalse(bpu.predict(addr),
            "longer-table entry must still dominate after a correct training update")

    def test_allocation_only_in_first_longer_table_with_tag_miss(self):
        """(TAGE) When misprediction occurs at the base, search tables 1 then 2
        for the first tag miss and allocate only there — NOT in both tables.

        After the first misprediction, table 1 gets an entry.
        We drive a second misprediction (with PHR changed so the table-1 entry
        is at a *different* index — simulating a tag hit situation by verifying
        only one new table-2 entry appears).

        Here we directly check that exactly one of the longer tables changed
        after a misprediction when the other already has an entry.
        """
        bpu = fresh_bpu()
        addr, target = 0x4000, 0x4100

        snap_t1_before = bpu._phts[1].snapshot()
        snap_t2_before = bpu._phts[2].snapshot()

        # First misprediction: table 1 miss → allocate in table 1
        for _ in range(8):
            step(bpu,addr, taken=True, target=target)
        step(bpu,addr, taken=False, target=target)

        snap_t1_after  = bpu._phts[1].snapshot()
        snap_t2_after  = bpu._phts[2].snapshot()

        t1_changed = snap_t1_after != snap_t1_before
        t2_changed = snap_t2_after != snap_t2_before

        # Exactly one table should have a new entry (the first with a tag miss)
        self.assertTrue(t1_changed or t2_changed,
            "at least one longer table must receive an entry on misprediction")
        self.assertFalse(t1_changed and t2_changed,
            "only the first tag-miss table must receive an entry — not both")

    def test_tag_miss_returns_none_before_any_training(self):
        """(TAGE) A tagged PHT that has never seen a branch must return None
        (tag miss) — it cannot provide a prediction, so the base is used.
        """
        pht = make_pht()
        self.assertIsNone(pht.predict(0xDEAD),
            "cold PHT must return None (tag miss) for any unseen address")

    def test_different_pcs_have_independent_entries(self):
        """(TAGE) Entries for different branch PCs must not interfere.
        Training PC_A must not affect predictions for PC_B (in a separate set or tag).
        """
        pht = make_pht()
        pht.allocate(0x1000, taken=True)    # PC_A: taken
        # PC_B has a different tag — must still be a miss
        self.assertIsNone(pht.predict(0x2000),
            "training PC_A must not create an entry for PC_B")

    def test_update_does_not_change_flag1_invariant_under_diverse_inputs(self):
        """(TAGE) Every call to update() must be handled by exactly one provider.
        The flag==1 assertion inside update() captures this.  Drive 1000 updates
        with mixed taken/not-taken across multiple PCs and verify no AssertionError.
        """
        bpu = fresh_bpu()
        addrs   = [0x1004, 0x2008, 0x300C, 0x4010, 0x5014]
        targets = [0x1104, 0x2108, 0x310C, 0x4110, 0x5114]
        rng = random.Random(0x1234)
        for _ in range(1000):
            i     = rng.randrange(len(addrs))
            taken = rng.choice([True, False])
            step(bpu,addrs[i], taken=taken, target=targets[i])
            # AssertionError inside update() would propagate here


# ---------------------------------------------------------------------------
# (RE+TAGE) Full-BPU reset behaviour
# ---------------------------------------------------------------------------

class TestBPUResetBehaviour(unittest.TestCase):

    def test_reset_makes_bpu_predict_like_cold_start(self):
        """(TAGE+RE) After reset(), a trained BPU must behave identically to a
        freshly constructed one.  Any address that was strongly taken before
        reset must predict not-taken afterwards (default cold state).
        """
        bpu_trained = fresh_bpu()
        addr, target = 0xA000, 0xA100

        for _ in range(10):
            step(bpu_trained,addr, taken=True, target=target)
        self.assertTrue(bpu_trained.predict(addr))

        bpu_trained.reset()
        bpu_cold = fresh_bpu()

        self.assertEqual(bpu_trained.predict(addr), bpu_cold.predict(addr),
            "reset() must make predictions indistinguishable from a cold BPU")

    def test_reset_clears_phr(self):
        """(RE) reset() must zero the PHR.  Any taken history is discarded."""
        bpu = fresh_bpu()
        for _ in range(20):
            step(bpu,0xB000, taken=True, target=0xB100)
        self.assertNotEqual(bpu._phr.read(), 0, "sanity: PHR must be nonzero before reset")

        bpu.reset()
        self.assertEqual(bpu._phr.read(), 0, "PHR must be 0 after reset()")

    def test_snapshot_is_all_empty_after_reset(self):
        """(TAGE) After reset(), every PHT set must be empty — no entries survive."""
        bpu = fresh_bpu()
        for _ in range(30):
            step(bpu,0xC000, taken=True, target=0xC100)
        bpu.reset()

        for set_snap in bpu.snapshot():
            self.assertEqual(set_snap, (), "all PHT entries must be evicted by reset()")


# ===========================================================================
# (BPU machinery) Speculative PHR: checkpoint / rollback / commit / advance,
# and the misprediction drive sequence. These exercise the speculation
# mechanism in isolation from the TAGE tables and from the N3 specifics.
# ===========================================================================

def make_phr(n_bits=300):
    """PHR with a simple deterministic, always-nonzero fold, to test the mechanism
    independently of the real N3 footprint."""
    return PHR(n_bits, lambda v, pc, target: (v << 4) ^ (((pc ^ target) & 0xF) | 1))


class TestPHRSpeculationPrimitives(unittest.TestCase):
    """The PHR is pure mechanism: advance + a LIFO checkpoint stack."""

    def setUp(self):
        self.phr = make_phr()

    def test_advance_value_is_pure(self):
        self.phr._value = 0x123
        out = self.phr.advance_value(0x123, 0x40, 0x80)
        self.assertEqual(self.phr._value, 0x123, "advance_value must not mutate live state")
        self.assertNotEqual(out, 0x123, "advance_value must fold to a new value")

    def test_advance_not_taken_is_noop(self):
        self.phr._value = 0xABC
        self.phr.advance(0x40, False, 0x80)
        self.assertEqual(self.phr._value, 0xABC, "a not-taken branch must not change the PHR")

    def test_advance_taken_folds(self):
        self.phr._value = 0xABC
        expected = self.phr.advance_value(0xABC, 0x40, 0x80)
        self.phr.advance(0x40, True, 0x80)
        self.assertEqual(self.phr._value, expected, "a taken branch must fold (pc,target) in")

    def test_fold_is_xor_of_group_width_chunks(self):
        A, B, C = 0x123, 0x456, 0x7        # 11-bit chunks (C is the partial top chunk)
        self.phr._value = (C << 22) | (B << 11) | A
        self.assertEqual(self.phr.fold(33, 11), A ^ B ^ C, "fold = XOR of 11-bit chunks")
        self.assertEqual(self.phr.fold(11, 11), A, "folding only the low group keeps chunk 0")
        self.assertEqual(self.phr.fold(300, 11), A ^ B ^ C, "zero high chunks do not change the fold")

    def test_checkpoint_rollback_round_trip(self):
        self.phr._value = 0x111
        self.phr.checkpoint()
        self.phr.advance(0x40, True, 0x80)
        self.assertNotEqual(self.phr._value, 0x111)
        self.phr.rollback()
        self.assertEqual(self.phr._value, 0x111, "rollback must restore the checkpointed value")

    def test_commit_keeps_live_value_and_drops_snapshot(self):
        self.phr._value = 0x111
        self.phr.checkpoint()
        self.phr.advance(0x40, True, 0x80)
        live = self.phr._value
        self.phr.commit()
        self.assertEqual(self.phr._value, live, "commit must keep the live (advanced) value")
        self.assertEqual(self.phr._stack, [], "commit must discard the snapshot")

    def test_nested_checkpoints_rollback_lifo(self):
        self.phr._value = 1
        self.phr.checkpoint()           # [1]
        self.phr._value = 2
        self.phr.checkpoint()           # [1, 2]
        self.phr._value = 3
        self.phr.rollback()             # -> 2
        self.assertEqual(self.phr._value, 2)
        self.phr.rollback()             # -> 1
        self.assertEqual(self.phr._value, 1)

    def test_commit_inner_then_rollback_targets_outer(self):
        self.phr._value = 1
        self.phr.checkpoint()           # [1]
        self.phr._value = 2
        self.phr.checkpoint()           # [1, 2]
        self.phr._value = 3
        self.phr.commit()               # drop inner (2); live stays 3 -> [1]
        self.assertEqual(self.phr._value, 3)
        self.phr.rollback()             # -> outer (1)
        self.assertEqual(self.phr._value, 1)

    def test_checkpoint_preserves_full_300_bit_value(self):
        big = (1 << 300) - 1            # full-width history value
        self.phr._value = big
        self.phr.checkpoint()
        self.phr.advance(0x40, True, 0x80)
        self.phr.rollback()
        self.assertEqual(self.phr._value, big,
            "a 300-bit value must survive checkpoint/rollback losslessly (kept as a Python int)")

    def test_reset_clears_value_and_stack(self):
        self.phr._value = 0x555
        self.phr.checkpoint()
        self.phr.checkpoint()
        self.phr.reset()
        self.assertEqual(self.phr._value, 0)
        self.assertEqual(self.phr._stack, [], "reset must drop outstanding checkpoints")

    def test_rollback_without_checkpoint_raises(self):
        """checkpoint/rollback must be balanced; an unmatched rollback is a programming error."""
        with self.assertRaises(IndexError):
            self.phr.rollback()


class TestMispredictionDriveSequence(unittest.TestCase):
    """The driver (execution clause) composes the PHR primitives into the textbook recovery.
    These tests pin the exact sequence the C driver must mirror:
        forward at a mispredict -> checkpoint(); advance(predicted)
        misprediction recovery  -> rollback();   advance(arch_taken)
    """

    def setUp(self):
        self.phr = make_phr()

    def _open_window(self, pc, predicted, target):
        self.phr.checkpoint()
        self.phr.advance(pc, predicted, target)

    def _recover(self, pc, arch_taken, target):
        self.phr.rollback()
        self.phr.advance(pc, arch_taken, target)

    def test_predicted_taken_arch_not_taken(self):
        pc, target = 0x40, 0x80
        before = self.phr._value = 0x100
        self._open_window(pc, predicted=True, target=target)
        self.assertEqual(self.phr._value, self.phr.advance_value(before, pc, target),
            "during the window the wrong path sees history advanced with the prediction (taken)")
        self._recover(pc, arch_taken=False, target=target)
        self.assertEqual(self.phr._value, before,
            "after recovery the correct path sees history NOT advanced (arch not-taken)")

    def test_predicted_not_taken_arch_taken(self):
        pc, target = 0x44, 0x88
        before = self.phr._value = 0x100
        self._open_window(pc, predicted=False, target=target)
        self.assertEqual(self.phr._value, before,
            "during the window the wrong path sees history NOT advanced (predicted not-taken)")
        self._recover(pc, arch_taken=True, target=target)
        self.assertEqual(self.phr._value, self.phr.advance_value(before, pc, target),
            "after recovery the correct path sees history advanced (arch taken)")

    def test_nested_misprediction_recovers_each_level(self):
        pcB, tgtB = 0x40, 0x80
        pcC, tgtC = 0x60, 0xC0
        before_B = self.phr._value = 0x10
        self._open_window(pcB, predicted=True, target=tgtB)     # B mispredicts
        wrong_B = self.phr._value
        self.assertEqual(wrong_B, self.phr.advance_value(before_B, pcB, tgtB))
        self._open_window(pcC, predicted=True, target=tgtC)     # nested C on B's wrong path
        self.assertEqual(self.phr._value, self.phr.advance_value(wrong_B, pcC, tgtC))
        self._recover(pcC, arch_taken=False, target=tgtC)       # C recovers first (LIFO)
        self.assertEqual(self.phr._value, wrong_B, "C recovery returns to B's wrong-path history")
        self._recover(pcB, arch_taken=False, target=tgtB)       # then B recovers
        self.assertEqual(self.phr._value, before_B, "B recovery returns to pre-B history")
        self.assertEqual(self.phr._stack, [], "all windows closed -> empty checkpoint stack")


class TestBPUSpeculationDelegation(unittest.TestCase):
    """The BPU separates table update (counters) from history advance, and exposes the speculation
    mechanism as thin pass-throughs to its PHR."""

    def test_update_is_counters_only(self):
        p = fresh_bpu()
        before = p._phr.read()
        p.update(0x4, taken=True, target=0x0)
        self.assertEqual(p._phr.read(), before, "update() touches counters only; the PHR must not move")

    def test_advance_moves_history_on_taken(self):
        p = fresh_bpu()
        before = p._phr.read()
        p.advance(0x4, taken=True, target=0x0)
        self.assertNotEqual(p._phr.read(), before, "advance() advances the PHR on a taken branch")

    def test_advance_not_taken_is_noop(self):
        p = fresh_bpu()
        before = p._phr.read()
        p.advance(0x4, taken=False, target=0x0)
        self.assertEqual(p._phr.read(), before, "advance() must not move the PHR on a not-taken branch")

    def test_update_trains_the_tables(self):
        p = fresh_bpu()
        addr = 0x2000
        for _ in range(8):
            p.update(addr, taken=True)          # target omitted: optional, ignored by the direction model
        self.assertTrue(p.predict(addr), "counters trained taken via update() must predict taken")

    def test_checkpoint_advance_rollback_delegate_to_phr(self):
        p = fresh_bpu()
        p._phr._value = 0x123
        p.checkpoint()
        p.advance(0x40, True, 0x80)
        self.assertNotEqual(p._phr.read(), 0x123)
        p.rollback()
        self.assertEqual(p._phr.read(), 0x123, "BPU.rollback must delegate to the PHR")


# ===========================================================================
# (TAGE mechanics) Generic predictor behaviour — provider selection, the base
# predictor, weakly-correct allocation, and the layered reset delegation.
# Independent of the N3 footprint / geometry.
# ===========================================================================

class TestTAGEMechanics(unittest.TestCase):

    def test_base_predictor_always_predicts_purely(self):
        """The base is the always-available provider: predict() returns a concrete direction for any
        PC (weakly-not-taken default on a cold index), so the provider scan never falls through — and
        predict() is a PURE read that must not allocate/mutate."""
        base = BimodalBP(counter_bit_width=3, num_sets=64)
        for addr in (0x0, 0x40, 0x4000, 0xDEAD0):
            self.assertIn(base.predict(addr), (True, False),
                "base predictor must always return a concrete prediction")
        self.assertEqual(base.snapshot(), (), "predict() must not allocate base entries (pure read)")

    def test_tagged_table_allocation_is_weakly_correct(self):
        """A freshly allocated tagged entry predicts its allocation direction, and a single
        opposite update flips it (weakly correct)."""
        pht = make_pht()
        pht.allocate(0x1000, taken=True)
        self.assertTrue(pht.predict(0x1000), "just-allocated taken entry predicts taken")
        pht.update(0x1000, taken=False)
        self.assertFalse(pht.predict(0x1000), "one not-taken flips a weakly-taken entry")

    def test_saturating_counter_bp_reset_evicts(self):
        bp = SaturatingCounterBP(counter_bit_width=3, num_sets=1, assoc=1)
        bp.update(0x1000, taken=True)
        self.assertIsNotNone(bp.predict(0x1000, allocate_on_miss=False), "entry present before reset")
        bp.reset()
        self.assertIsNone(bp.predict(0x1000, allocate_on_miss=False), "reset() must evict all entries")

    def test_table_reset_clears_entries(self):
        """BimodalBP / TAGEPHT reset() must clear their tables."""
        base = BimodalBP(counter_bit_width=3, num_sets=8)
        base.update(0x40, taken=True)
        self.assertNotEqual(base.snapshot(), (), "a trained base has an entry")
        base.reset()
        self.assertEqual(base.snapshot(), (), "BimodalBP.reset() must clear all entries")

        pht = make_pht()
        pht.allocate(0x1000, taken=True)
        pht.reset()
        for set_snap in pht.snapshot():
            self.assertEqual(set_snap, (), "TAGEPHT.reset() must evict all entries")

    def test_longer_table_overrides_base_after_allocation(self):
        """When a tagged (longer-history) entry exists, it is the provider over the base."""
        bpu = fresh_bpu()
        addr, target = 0x5000, 0x5100
        for _ in range(8):                       # base -> strongly taken
            step(bpu,addr, taken=True, target=target)
        self.assertTrue(bpu.predict(addr))
        step(bpu,addr, taken=False, target=target)   # mispredict -> allocate weakly-not-taken longer
        self.assertFalse(bpu.predict(addr), "longer-history provider must override the base")

    def test_predict_is_side_effect_free(self):
        """Regression: predict() must not mutate predictor state, so wrong-path (speculative)
        predicts cannot pollute the tables — only the PHR is speculative. Cold predicts allocate
        nothing."""
        bpu = fresh_bpu()
        snap_before = bpu.snapshot()
        for pc in (0x4, 0x40, 0x4000, 0xDEAD0, 0x1234):
            bpu.predict(pc)
        self.assertEqual(bpu.snapshot(), snap_before, "predict() must not allocate/mutate any table")


# ===========================================================================
# (Neoverse N3 specifics) The reverse-engineered configuration — NOT generic
# TAGE. Source: our Neoverse-N3 reverse-engineering efforts (kb-bpu-re).
# ===========================================================================

class TestNeoverseN3Specifics(unittest.TestCase):
    """Pins facts particular to the Arm Neoverse-N3 model, per our reverse-engineering efforts
    (see kb-bpu-re) — NOT generic TAGE/counter properties (those live in
    TestSaturatingCounterBehaviour / TestTAGEMechanics / TestTAGEAllocationPolicy).

    Two deliberately distinct kinds of assertion live here:
      * RE'd ground truth — history lengths [0, 172, 300]; the 300-bit PHR advanced by a 4-bit
        record per taken branch; the exact PHR footprint bit-formula. These change only if the RE
        is revised.
      * NOT-yet-RE'd placeholders, configured at the construction seam (bootstrap_director): the PHR
        fold (algorithm + group width, 11) and the tag width (8) are NOT based on any RE result and
        must still be fully reverse-engineered. Their tests pin ONLY the wiring and the fold/mask
        mechanism at the current placeholder value — they must never be read as hardware facts.
    """

    def setUp(self):
        self.bpu = fresh_bpu()

    def test_history_lengths(self):
        self.assertEqual(self.bpu._histlen, [0, 172, 300],
            "N3 model uses history lengths 0 (base), 172, 300")

    def test_three_tables_base_plus_two_tagged(self):
        self.assertEqual(len(self.bpu._phts), 3, "one base + two tagged tables")
        self.assertIsInstance(self.bpu._phts[0], BimodalBP)
        self.assertIsInstance(self.bpu._phts[1], TAGEPHT)
        self.assertIsInstance(self.bpu._phts[2], TAGEPHT)

    def test_table_geometry(self):
        base = self.bpu._phts[0]
        self.assertEqual(base._num_sets, 1 << 12, "base bimodal has 4096 indexes")
        self.assertEqual(base._counter_bit_width, 3)
        for pht in self.bpu._phts[1:]:               # the two tagged tables
            self.assertEqual(len(pht._bp._table), 1 << 11, "tagged tables have 2048 sets")
            self.assertEqual(pht._bp._counter_bit_width, 3, "3-bit saturating counters")
        self.assertEqual(self.bpu._phts[1]._bp._table[0]._lru_cache.capacity, 4, "table 1 is 4-way")
        self.assertEqual(self.bpu._phts[2]._bp._table[0]._lru_cache.capacity, 2, "table 2 is 2-way")

    def test_phr_is_300_bits(self):
        self.assertEqual(self.bpu._phr._n_bits, 300)
        self.assertEqual(self.bpu._phr._mask, (1 << 300) - 1)

    def test_footprint_exact_value_from_pc_bits(self):
        """footprint b0..b3 = (pc[2..5]^pc[6..9]^target[3..6]^target[7..10]); pc=0xC sets b0,b1."""
        step(self.bpu,0xC, taken=True, target=0x0)   # pc bits 2,3 set -> b0=b1=1 -> fp=0b0011
        self.assertEqual(self.bpu._phr.read(), 0b0011)

    def test_footprint_exact_value_from_target_bit(self):
        step(self.bpu,0x0, taken=True, target=0x8)   # target bit 3 -> b0=1 -> fp=0b0001
        self.assertEqual(self.bpu._phr.read(), 0b0001)

    def test_phr_shifts_left_by_four_per_taken_branch(self):
        """Each taken branch shifts the PHR left by 4 and XORs in the new 4-bit footprint."""
        step(self.bpu,0xC, taken=True, target=0x0)   # PHR = 0x3
        step(self.bpu,0x0, taken=True, target=0x8)   # PHR = (0x3 << 4) ^ 0x1
        self.assertEqual(self.bpu._phr.read(), (0x3 << 4) ^ 0x1)

    def test_fold_group_width(self):
        self.assertEqual(self.bpu._fold_group_width, 11,
            "the test fixture constructs the model with fold width 11 (the working hypothesis)")

    def test_fold_group_width_configurable(self):
        """The fold width is a constructor knob so the pending microbench can settle 10 vs 11."""
        bpu10 = Aarch64NeoverseN3BPU(fold_group_width=10, tag_width=8)
        self.assertEqual(bpu10._fold_group_width, 10)
        index_fn = bpu10._phts[1]._bp._index_fn
        bpu10._phr._value = 1 << 0
        idx0 = index_fn(0x12300)
        bpu10._phr._value = 1 << 10          # one group up at width 10 -> folds onto bit 0
        self.assertEqual(index_fn(0x12300), idx0,
            "with fold width 10, PHR bit 10 folds onto the same index bit as bit 0")

    def test_tag_is_masked_to_tag_width(self):
        self.assertEqual(self.bpu._tag_width, 8)
        tag_fn = self.bpu._phts[1]._bp._table[0]._tag_fn
        for addr in (0x0, 0x1234, 0xFFFFFFFF, 0xDEADBEEF):
            self.assertLess(tag_fn(addr), 1 << 8, "tag must fit in tag_width (8) bits")

    def test_tag_width_configurable(self):
        """Tag width is a constructor knob (RE-pending); masking lets distinct branches tag-alias."""
        bpu4 = Aarch64NeoverseN3BPU(fold_group_width=11, tag_width=4)
        tag_fn = bpu4._phts[1]._bp._table[0]._tag_fn
        self.assertLess(tag_fn(0xFFFFFFFF), 1 << 4, "tag must fit in the configured 4 bits")


if __name__ == "__main__":
    unittest.main(verbosity=2)
