"""Unit tests for the PTE sealing model layers (no hardware, no capstone):
    * pagetable_model -- descriptor bit-field layouts (the single source of bit truth)
    * environment     -- page roles/partition, per-variant plan, fuzz policy

Run from the repo root:
    python -m unittest tests.aarch64_tests.unit_pte_model
"""
import os
import sys
import unittest
from random import Random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.aarch64.seal.pagetable_model import (BitField, DescriptorLayout,          # noqa: E402
                                              PageTableDescriptor, LEAF_LAYOUT, TABLE_LAYOUT)
from src.interfaces import PAGE_SIZE, MAIN_AREA_SIZE                                # noqa: E402
from src.aarch64.seal.environment import (                                         # noqa: E402
    SandboxPage, SandboxPageMap, PteOverride, EnvironmentPlan, PteFuzzPolicy,
    serialize_pte_overrides, deserialize_pte_overrides, default_sandbox_page_map, LEVEL_LEAF)


class BitFieldTest(unittest.TestCase):
    def test_mask_extract_insert(self):
        f = BitField("ap", 6, 2)
        self.assertEqual(f.mask, 0b11 << 6)
        self.assertEqual(f.max_value, 3)
        self.assertEqual(f.extract(0b10 << 6), 0b10)
        self.assertEqual(f.insert(0, 3), 0b11 << 6)
        # insert clears the old field bits first
        self.assertEqual(f.insert(0b11 << 6, 1), 0b01 << 6)

    def test_insert_rejects_overflow(self):
        with self.assertRaises(ValueError):
            BitField("valid", 0, 1).insert(0, 2)

    def test_values_enumerates_all(self):
        self.assertEqual(list(BitField("sh", 8, 2).values()), [0, 1, 2, 3])


class DescriptorLayoutTest(unittest.TestCase):
    def test_duplicate_and_overlap_rejected(self):
        with self.assertRaises(ValueError):
            DescriptorLayout("d", [BitField("a", 0), BitField("a", 1)])
        with self.assertRaises(ValueError):
            DescriptorLayout("d", [BitField("a", 0, 2), BitField("b", 1)])

    def test_unknown_field_is_loud(self):
        with self.assertRaises(KeyError):
            LEAF_LAYOUT.field("nonexistent")

    def test_mask_for_combines_and_validates(self):
        m = LEAF_LAYOUT.mask_for(["valid", "ap"])
        self.assertEqual(m, LEAF_LAYOUT.field("valid").mask | LEAF_LAYOUT.field("ap").mask)
        with self.assertRaises(KeyError):
            LEAF_LAYOUT.mask_for(["valid", "bogus"])

    def test_output_address_and_type_never_fuzzable(self):
        for layout, addr in ((LEAF_LAYOUT, "oa"), (TABLE_LAYOUT, "next_addr")):
            self.assertNotIn(addr, layout.default_fuzzable_field_names)
            self.assertNotIn("type", layout.default_fuzzable_field_names)

    def test_leaf_has_expected_fuzzable_attributes(self):
        names = set(LEAF_LAYOUT.default_fuzzable_field_names)
        self.assertTrue({"valid", "attr_indx", "ap", "sh", "af", "uxn"} <= names)

    def test_layouts_have_no_bit_overlap(self):
        # constructor already checks; assert the whole descriptor is consistent by rebuilding
        for layout in (LEAF_LAYOUT, TABLE_LAYOUT):
            DescriptorLayout(layout.name, layout.fields)   # would raise on overlap


class PageTableDescriptorTest(unittest.TestCase):
    def test_get_with_field_roundtrip(self):
        d = PageTableDescriptor(0, LEAF_LAYOUT).with_field("valid", 1).with_field("ap", 2)
        self.assertEqual(d.get("valid"), 1)
        self.assertEqual(d.get("ap"), 2)

    def test_differs_only_in(self):
        base = PageTableDescriptor(0xDEAD_0000_0000_0FFF, LEAF_LAYOUT)
        flipped = base.with_field("ap", (base.get("ap") ^ 0b11))
        self.assertTrue(base.differs_only_in(flipped, ["ap"]))
        self.assertFalse(base.differs_only_in(flipped, ["sh"]))


class SandboxPageMapTest(unittest.TestCase):
    def setUp(self):
        self.m = default_sandbox_page_map()

    def test_index_mismatch_and_dup_name_rejected(self):
        with self.assertRaises(ValueError):
            SandboxPageMap([SandboxPage(1, "a", 0)])
        with self.assertRaises(ValueError):
            SandboxPageMap([SandboxPage(0, "a", 0), SandboxPage(1, "a", PAGE_SIZE)])

    def test_role_partition(self):
        self.assertEqual([p.name for p in self.m.arch_pages], ["main"])
        self.assertEqual([p.name for p in self.m.spec_only_pages], ["faulty"])

    def test_spill_clamp_only_before_spec_only_or_end(self):
        # main precedes the spec-only faulty page -> a full-width arch access there needs a clamp.
        self.assertTrue(self.m.arch_access_needs_size_clamp(self.m.page("main")))
        # a spec-only page is never an arch base, so it never reports needing a clamp.
        self.assertFalse(self.m.arch_access_needs_size_clamp(self.m.page("faulty")))

    def test_spec_only_containing_maps_addresses_to_pages(self):
        base = 0x1_0000
        # an address in the faulty page is flagged; one in main (arch) is not.
        self.assertEqual(self.m.spec_only_containing(base + MAIN_AREA_SIZE + 8, base).name, "faulty")
        self.assertIsNone(self.m.spec_only_containing(base + 8, base))

    def test_require_spec_only(self):
        self.m.require_spec_only()   # ok
        with self.assertRaises(ValueError):
            SandboxPageMap([SandboxPage(0, "only", 0)]).require_spec_only()


class EnvironmentPlanTest(unittest.TestCase):
    def test_override_rejects_bits_outside_mask(self):
        with self.assertRaises(ValueError):
            PteOverride(2, LEVEL_LEAF, mask=0b10, value=0b01)

    def test_pte_codec_roundtrip(self):
        overrides = (PteOverride(2, LEVEL_LEAF, 0b1, 0b0), PteOverride(2, LEVEL_LEAF, 0xF0, 0xA0))
        self.assertEqual(deserialize_pte_overrides(serialize_pte_overrides(overrides)), overrides)
        self.assertEqual(deserialize_pte_overrides(serialize_pte_overrides(())), ())

    def test_empty_plan(self):
        self.assertTrue(EnvironmentPlan().is_empty)
        self.assertFalse(EnvironmentPlan(pte_overrides=(PteOverride(2, LEVEL_LEAF, 1, 0),)).is_empty)

    def test_pte_for_page(self):
        plan = EnvironmentPlan(pte_overrides=(PteOverride(2, LEVEL_LEAF, 0b1, 0b0),
                                              PteOverride(3, LEVEL_LEAF, 0b1, 0b0)))
        self.assertEqual(len(plan.pte_for_page(2)), 1)


class PteFuzzPolicyTest(unittest.TestCase):
    def setUp(self):
        self.m = default_sandbox_page_map()
        self.policy = PteFuzzPolicy(leaf_fields=["valid", "ap", "attr_indx"])

    def test_unknown_field_is_loud(self):
        with self.assertRaises(KeyError):
            PteFuzzPolicy(leaf_fields=["bogus"])

    def test_genuine_plan_is_empty(self):
        self.assertTrue(self.policy.genuine_plan().is_empty)

    def test_decoy_only_touches_spec_only_pages_within_allowed_bits(self):
        allowed = LEAF_LAYOUT.mask_for(["valid", "ap", "attr_indx"])
        spec_indices = {p.index for p in self.m.spec_only_pages}
        plan = self.policy.decoy_plan(self.m, Random(1234))
        self.assertFalse(plan.is_empty)
        for o in plan.pte_overrides:
            self.assertIn(o.page_index, spec_indices)
            self.assertEqual(o.mask & ~allowed, 0)      # never touches disallowed bits (e.g. oa)
            self.assertEqual(o.value & ~o.mask, 0)

    def test_decoy_requires_a_spec_only_page(self):
        empty = SandboxPageMap([SandboxPage(0, "only", 0)])
        with self.assertRaises(ValueError):
            self.policy.decoy_plan(empty, Random(0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
