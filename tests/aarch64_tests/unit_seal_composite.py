"""
Unit tests for the Sandbox seal, CompositeSeal, the decoy policy, and the engine driving a composite.

Hardware-free and ISA-free: the seals are pure Python, so the fix points and the sealed test case
are built by hand.
"""
import random
import unittest
from typing import List, Optional

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.interfaces import TestCase, Function, BasicBlock, Instruction, GeneratorException
from src.aarch64.aarch64_seal import (
    Sandbox, CompositeSeal, SealedNIInstrumentation,
    index_instructions, inst_at, fill_slot_at, make_nop, is_speculative, _SANDBOX_MASK,
)
from src.aarch64.aarch64_mte import MteTag, MTEFixPoint

_RETAG = ("irg", "eor")  # MteTag decoy instruction names

# A policy that decoys the speculative slots but never the sandbox clamp.
NO_SANDBOX = lambda fp, seal: seal.name != "sandbox" and is_speculative(fp)


def _fp(slot_id=0, reg="x5", spec_nesting=None) -> MTEFixPoint:
    return MTEFixPoint(slot_id=slot_id, value_reg=reg, spec_nesting=spec_nesting)

def _build_sealed_tc(seal, fps: List[MTEFixPoint]) -> TestCase:
    tc = TestCase(seed=0)
    actor = list(tc.actors.values())[0]
    func = Function(".function_main_0", actor)
    for fp in fps:
        bb = BasicBlock(f".bb_slot_{fp.slot_id}")
        fp.slot_insts = seal.placeholder(fp)
        for inst in fp.slot_insts:
            bb.insert_after(bb.end, inst)
        func.append(bb)
    tc.functions.append(func)
    locs = index_instructions(tc)
    for fp in fps:
        fp.slot_locs = [locs[id(si)] for si in fp.slot_insts]
    return tc

def _slot(tc: TestCase, fp: MTEFixPoint) -> List[Instruction]:
    return [inst_at(tc, loc)[0] for loc in fp.slot_locs]

def _names(insts) -> List[str]:
    return [i.name.lower() for i in insts]


# ===========================================================================
# Sandbox seal — always in-bounds, never decoyed
# ===========================================================================

class TestSandboxSeal(unittest.TestCase):

    def setUp(self):
        self.seal = Sandbox(_SANDBOX_MASK)

    def test_slot_size_is_two(self):
        self.assertEqual(self.seal.slot_size, 2)

    def test_genuine_clamps_in_bounds(self):
        s = self.seal.genuine(_fp(reg="x5"))
        self.assertEqual(_names(s), ["and", "add"])
        self.assertIn("x5", s[0].template)
        self.assertIn("x29", s[1].template)  # rebased on the sandbox base register

    def test_placeholder_is_genuine(self):
        self.assertEqual(_names(self.seal.placeholder(_fp())), ["and", "add"])

    def test_decoy_raises(self):
        with self.assertRaises(NotImplementedError):
            self.seal.decoy(_fp(), random.Random(0))


# ===========================================================================
# CompositeSeal
# ===========================================================================

class TestCompositeSeal(unittest.TestCase):

    def setUp(self):
        self.seal = CompositeSeal([Sandbox(_SANDBOX_MASK), MteTag()])

    def test_name_and_slot_size(self):
        self.assertEqual(self.seal.name, "sandbox+mte_tag")
        self.assertEqual(self.seal.slot_size, 2 + 1)

    def test_placeholder_concatenates_members(self):
        self.assertEqual(_names(self.seal.placeholder(_fp())), ["and", "add", "nop"])

    def test_genuine_concatenates_members(self):
        self.assertEqual(_names(self.seal.genuine(_fp())), ["and", "add", "nop"])

    def test_empty_raises(self):
        with self.assertRaises(AssertionError):
            CompositeSeal([])

    def test_fill_policy_protects_sandbox(self):
        """With NO_SANDBOX, the sandbox stays [AND, ADD]; only mte_tag is decoyed."""
        rng = random.Random(0)
        for _ in range(64):
            n = _names(self.seal.fill(_fp(spec_nesting=1), rng, NO_SANDBOX))
            self.assertEqual(n[:2], ["and", "add"])  # sandbox genuine always
            self.assertIn(n[2], _RETAG)              # mte_tag (only eligible member) decoyed

    def test_fill_genuine_when_policy_denies(self):
        rng = random.Random(0)
        n = _names(self.seal.fill(_fp(spec_nesting=0), rng, NO_SANDBOX))  # arch slot
        self.assertEqual(n, ["and", "add", "nop"])


class TestCompositeSubsets(unittest.TestCase):
    """Two decoyable members → decoy() explores every non-empty subset."""

    def setUp(self):
        self.seal = CompositeSeal([MteTag(), MteTag()])

    def test_genuine_is_two_nops(self):
        self.assertEqual(_names(self.seal.genuine(_fp())), ["nop", "nop"])

    def test_decoy_explores_every_subset(self):
        rng = random.Random(1)
        subsets = set()
        for _ in range(256):
            n = _names(self.seal.decoy(_fp(), rng))
            subsets.add((n[0] in _RETAG, n[1] in _RETAG))
        self.assertEqual(subsets, {(True, False), (False, True), (True, True)})


# ===========================================================================
# Engine + decoy policy
# ===========================================================================

class TestEngineWithComposite(unittest.TestCase):

    def setUp(self):
        self.seal = CompositeSeal([Sandbox(_SANDBOX_MASK), MteTag()])

    def _engine(self, seal, fps, should_decoy=None):
        prep = _build_sealed_tc(seal, fps)
        eng = SealedNIInstrumentation(seal, should_decoy)
        eng.set_sealed(prep, fps)
        return eng

    def test_baseline_all_genuine(self):
        for sn in (0, 1, None):
            fp = _fp(spec_nesting=sn)
            base = self._engine(self.seal, [fp], NO_SANDBOX).baseline()
            with self.subTest(spec_nesting=sn):
                self.assertEqual(_names(_slot(base, fp)), ["and", "add", "nop"])

    def test_decoy_arch_slot_stays_genuine(self):
        fp = _fp(spec_nesting=0)
        decoy = next(self._engine(self.seal, [fp], NO_SANDBOX).decoys(random.Random(0)))
        self.assertEqual(_names(_slot(decoy, fp)), ["and", "add", "nop"])

    def test_decoy_spec_slot_keeps_sandbox_decoys_tag(self):
        fp = _fp(spec_nesting=1)
        decoy = next(self._engine(self.seal, [fp], NO_SANDBOX).decoys(random.Random(0)))
        n = _names(_slot(decoy, fp))
        self.assertEqual(n[:2], ["and", "add"])  # sandbox always applied
        self.assertIn(n[2], _RETAG)              # tag decoyed on the speculative slot

    def test_policy_can_protect_sandbox_in_mix(self):
        fps = [_fp(slot_id=0, spec_nesting=0), _fp(slot_id=1, spec_nesting=1)]
        decoy = next(self._engine(self.seal, fps, NO_SANDBOX).decoys(random.Random(2)))
        self.assertEqual(_names(_slot(decoy, fps[0])), ["and", "add", "nop"])  # arch: genuine
        n1 = _names(_slot(decoy, fps[1]))
        self.assertEqual(n1[:2], ["and", "add"])
        self.assertIn(n1[2], _RETAG)

    def test_never_policy_yields_all_genuine(self):
        """A policy that never allows decoy: baseline == decoy, and a raising decoy() is never hit."""
        fp = _fp(spec_nesting=1)
        eng = self._engine(Sandbox(_SANDBOX_MASK), [fp], should_decoy=lambda fp, seal: False)
        decoy = next(eng.decoys(random.Random(0)))
        self.assertEqual(_names(_slot(decoy, fp)), ["and", "add"])

    def test_custom_policy_can_decoy_arch_slots(self):
        """The policy is general: a 'decoy everywhere' lambda decoys even architectural slots."""
        fp = _fp(spec_nesting=0)
        eng = self._engine(MteTag(), [fp], should_decoy=lambda fp, seal: True)
        decoy = next(eng.decoys(random.Random(0)))
        self.assertIn(_names(_slot(decoy, fp))[0], _RETAG)


# ===========================================================================
# Slot mechanics (fill_slot_at / inst_at)
# ===========================================================================

class TestSlotMechanics(unittest.TestCase):

    def _tc(self):
        fp = _fp()                                       # Sandbox slot is 2 instructions
        return _build_sealed_tc(Sandbox(_SANDBOX_MASK), [fp]), fp

    def test_short_fill_is_padded_with_nops(self):
        tc, fp = self._tc()
        fill_slot_at(tc, fp.slot_locs, [make_nop()])     # 1 inst for a 2-slot
        self.assertEqual(_names(_slot(tc, fp)), ["nop", "nop"])  # padded one-for-one

    def test_over_long_fill_is_rejected(self):
        tc, fp = self._tc()
        with self.assertRaises(AssertionError):          # silent truncation would hide a seal bug
            fill_slot_at(tc, fp.slot_locs, [make_nop(), make_nop(), make_nop()])

    def test_inst_at_out_of_range_raises(self):
        tc, _ = self._tc()
        with self.assertRaises(GeneratorException):
            inst_at(tc, (0, 0, 999))


# ===========================================================================
# is_speculative — the NI gate (None / 0 / >0)
# ===========================================================================

class TestIsSpeculative(unittest.TestCase):

    def test_architectural_slot_is_not_speculative(self):
        self.assertFalse(is_speculative(_fp(spec_nesting=0)))     # arch-reached → must stay genuine

    def test_speculative_slot(self):
        self.assertTrue(is_speculative(_fp(spec_nesting=1)))
        self.assertTrue(is_speculative(_fp(spec_nesting=4)))

    def test_unreached_slot_is_speculative(self):
        # never reached architecturally → decoying it cannot perturb the arch path, so it is safe
        self.assertTrue(is_speculative(_fp(spec_nesting=None)))


# ===========================================================================
# CompositeSeal — subset exhaustiveness + the never-decoyed-member contract
# ===========================================================================

class TestCompositeContracts(unittest.TestCase):

    def test_three_members_decoy_explores_all_seven_subsets(self):
        seal = CompositeSeal([MteTag(), MteTag(), MteTag()])
        rng, seen = random.Random(3), set()
        for _ in range(400):
            n = _names(seal.decoy(_fp(), rng))
            seen.add(tuple(x in _RETAG for x in n))
        self.assertEqual(len(seen), 7)                         # 2^3 - 1 non-empty subsets
        self.assertNotIn((False, False, False), seen)          # never the empty subset

    def test_decoy_raises_if_a_member_cannot_be_decoyed(self):
        # decoy() decoys a subset of ALL members; a Sandbox can't be decoyed, so a composite holding
        # one must be driven via fill(policy) (which excludes it), never decoy() wholesale.
        with self.assertRaises(NotImplementedError):
            CompositeSeal([Sandbox(_SANDBOX_MASK)]).decoy(_fp(), random.Random(0))


# ===========================================================================
# Value-commitment semantics — a seal commits to a value, not to a location
# ===========================================================================

class TestValueCommitment(unittest.TestCase):
    """A seal acts only on fp.value_reg (wherever the committed value currently lives); it carries
    no notion of how the value got there. The same commitment in a different register seals the
    same way, on that register."""

    def test_seal_acts_only_on_value_reg(self):
        for reg in ("x5", "x9", "x0", "x23"):
            clamp = Sandbox(_SANDBOX_MASK).genuine(_fp(reg=reg))
            self.assertTrue(all(reg in i.template for i in clamp))
            retag = MteTag().decoy(_fp(reg=reg), random.Random(0))
            self.assertIn(reg, retag[0].template)

    def test_same_commitment_different_register(self):
        # the "same value" observed in x1 at one use site and x7 at another (after it moved) seals
        # identically — same op, each on its own register, no shared/historical state.
        a = MTEFixPoint(slot_id=0, value_reg="x1"); a.ptr_tag, a.correct_tag = 2, 5
        b = MTEFixPoint(slot_id=1, value_reg="x7"); b.ptr_tag, b.correct_tag = 2, 5
        ga, gb = MteTag().genuine(a), MteTag().genuine(b)
        self.assertEqual(ga[0].name, gb[0].name)               # same op (delta 3)
        self.assertIn("x1", ga[0].template)
        self.assertIn("x7", gb[0].template)


if __name__ == "__main__":
    unittest.main()
