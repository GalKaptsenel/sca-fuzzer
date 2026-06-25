"""
Random generation tests verifying the MTE seal/engine promises across many scenarios.

For each generated TC we assign every fix point a random spec_nesting ∈ {0, 1, None}, then mint
the engine's baseline and a decoy instance, and verify, per slot:

  baseline (all genuine):
    always NOP — the correct (architectural) tag is already on the sandboxed pointer.

  decoy:
    spec_nesting == 0 (architectural): NOP (genuine — preserves the NI invariant)
    spec_nesting != 0 / None (speculative): retagged — IRG Xd,Xd (random tag) or
                                            EOR Xd,Xd,#<tag-bit> (a different tag), on fp.value_reg

Reachability: every sealed slot survives in both the baseline and the decoy.
"""
import os
import random
import sys
import tempfile
import unittest
from typing import List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")  # repo root: config.yml / base.json

from src.config import CONF
from src.isa_loader import InstructionSet
from src.aarch64.aarch64_generator import Aarch64RandomGenerator
from src.aarch64.aarch64_seal import inst_at, is_speculative
from src.aarch64.aarch64_mte import MTEInstrumentation, MTEFixPoint, MTE_SLOT_SIZE

_RETAG_NAMES = ("irg", "eor")   # the two decoy kinds MteTag can emit
# The pass now seals untainted accesses with sandbox+tag; never decoy the sandbox clamp.
_NO_SANDBOX = lambda fp, seal: seal.name != "sandbox" and is_speculative(fp)

_ISA: Optional[InstructionSet] = None
_TMPDIR = None


def setUpModule():
    # Pure generator tests — no kernel module needed.
    global _ISA, _TMPDIR
    CONF.load(os.path.join(_ROOT, "config.yml"))
    _ISA = InstructionSet(os.path.join(_ROOT, "base.json"), CONF.instruction_categories)
    _TMPDIR = tempfile.mkdtemp()


def tearDownModule():
    import shutil
    if _TMPDIR:
        shutil.rmtree(_TMPDIR, ignore_errors=True)


# ===========================================================================
# Helpers
# ===========================================================================

def _slot_inst(tc, fp: MTEFixPoint):
    return inst_at(tc, fp.slot_locs[-1])[0]   # the tag is the last slot instruction


def _gen_sealed(seed: int):
    """Return (fix_points, sealed_tc, mte) or None if the TC has no memory accesses."""
    gen = Aarch64RandomGenerator(_ISA, seed)
    mte = MTEInstrumentation(gen)
    asm_path = os.path.join(_TMPDIR, f"mte_{seed}.asm")
    try:
        tc = gen.create_test_case(asm_path, disable_assembler=True)
        sealed_tc, fix_points = mte.seal_test_case(tc)
    except Exception:
        return None
    if not fix_points:
        return None
    return fix_points, sealed_tc, mte


def _baseline_decoy(mte, sealed_tc, fix_points, seed=0):
    eng = mte.make_engine(should_decoy=_NO_SANDBOX)
    eng.set_sealed(sealed_tc, fix_points)
    return eng.baseline(), next(eng.decoys(random.Random(seed)))


# ===========================================================================
# 1. TestMteRandomPromises — baseline/decoy promises, random spec_nesting
# ===========================================================================

class TestMteRandomPromises(unittest.TestCase):

    _SCENARIOS = None  # list of (fix_points, baseline, decoy, error|None)

    @classmethod
    def setUpClass(cls):
        cls._SCENARIOS = []
        rng = random.Random(54321)
        tc_count = 0

        for seed in range(500):
            result = _gen_sealed(seed)
            if result is None:
                continue
            fix_points, sealed_tc, mte = result
            for fp in fix_points:
                fp.spec_nesting = [0, 1, None][rng.randint(0, 2)]
            try:
                base, decoy = _baseline_decoy(mte, sealed_tc, fix_points, seed=seed)
            except Exception as e:  # pragma: no cover
                cls._SCENARIOS.append((fix_points, None, None, str(e)))
                continue
            cls._SCENARIOS.append((fix_points, base, decoy, None))
            tc_count += 1
            if tc_count >= 100:
                break

        assert tc_count >= 20, f"Too few valid TCs: {tc_count}"

    def _iter_slots(self):
        """Yield (fp, baseline_inst, decoy_inst) for every valid slot."""
        for fix_points, base, decoy, err in self._SCENARIOS:
            if base is None:
                continue
            for fp in fix_points:
                yield fp, _slot_inst(base, fp), _slot_inst(decoy, fp)

    def test_minting_never_crashes(self):
        errors = [e for _, _, _, e in self._SCENARIOS if e is not None]
        self.assertEqual(errors, [],
                         f"{len(errors)} unexpected crashes:\n" + "\n".join(errors[:5]))

    def test_every_slot_in_baseline_and_decoy(self):
        # _slot_inst raises if a slot location is missing, so simply touching each is the check.
        for fp, ib, idd in self._iter_slots():
            self.assertIsNotNone(ib)
            self.assertIsNotNone(idd)

    def test_baseline_always_nop(self):
        for fp, ib, _ in self._iter_slots():
            with self.subTest(slot=fp.slot_id, spec=fp.spec_nesting):
                self.assertEqual(ib.name.lower(), "nop")

    def test_decoy_arch_slot_is_nop(self):
        for fp, _, idd in self._iter_slots():
            if fp.spec_nesting != 0:
                continue
            with self.subTest(slot=fp.slot_id):
                self.assertEqual(idd.name.lower(), "nop")

    def test_decoy_spec_slot_is_retagged(self):
        checked = 0
        for fp, _, idd in self._iter_slots():
            if fp.spec_nesting == 0:
                continue
            with self.subTest(slot=fp.slot_id, spec=fp.spec_nesting):
                self.assertIn(idd.name.lower(), _RETAG_NAMES)
            checked += 1
        self.assertGreater(checked, 0, "No speculative slots encountered")

    def test_decoy_spec_slot_retags_fp_reg(self):
        for fp, _, idd in self._iter_slots():
            if fp.spec_nesting == 0:
                continue
            with self.subTest(slot=fp.slot_id, reg=fp.value_reg):
                self.assertIn(fp.value_reg.lower(), idd.template.lower())

    def test_spec_nesting_none_is_retagged(self):
        checked = 0
        for fp, _, idd in self._iter_slots():
            if fp.spec_nesting is not None:
                continue
            with self.subTest(slot=fp.slot_id):
                self.assertIn(idd.name.lower(), _RETAG_NAMES)
            checked += 1
        self.assertGreater(checked, 0, "No spec_nesting=None slots encountered")


# ===========================================================================
# 2. TestMteRandomReachability — sealed slot positions are valid in variants
# ===========================================================================

class TestMteRandomReachability(unittest.TestCase):

    def test_slot_locs_match_fix_points(self):
        """Every fix point records its seal's slot_size positions, all unique slot_ids."""
        for seed in range(100):
            result = _gen_sealed(seed)
            if result is None:
                continue
            fix_points, _, _ = result
            with self.subTest(seed=seed):
                self.assertEqual(len({fp.slot_id for fp in fix_points}), len(fix_points))
                for fp in fix_points:
                    self.assertEqual(len(fp.slot_locs), fp.seal.slot_size)


# ===========================================================================
# 3. TestMteRandomDecoyKinds — both retag kinds appear; arch path stays genuine
# ===========================================================================

class TestMteRandomDecoyKinds(unittest.TestCase):

    def test_both_retag_kinds_appear_over_many_decoys(self):
        """Across many decoy instances of all-speculative TCs, both IRG and EOR are emitted."""
        seen = set()
        for seed in range(60):
            result = _gen_sealed(seed)
            if result is None:
                continue
            fix_points, sealed_tc, mte = result
            for fp in fix_points:
                fp.spec_nesting = 1
            eng = mte.make_engine(should_decoy=_NO_SANDBOX)
            eng.set_sealed(sealed_tc, fix_points)
            gen = eng.decoys(random.Random(seed))
            for _ in range(4):
                decoy = next(gen)
                for fp in fix_points:
                    seen.add(_slot_inst(decoy, fp).name.lower())
            if _RETAG_NAMES[0] in seen and _RETAG_NAMES[1] in seen:
                break
        self.assertEqual(seen, set(_RETAG_NAMES),
                         f"expected both retag kinds {set(_RETAG_NAMES)}, saw {seen}")

    def test_spec_nesting_none_and_one_both_retag(self):
        """spec_nesting=None and spec_nesting=1 are both treated as speculative (retagged)."""
        for seed in range(30):
            result = _gen_sealed(seed)
            if result is None:
                continue
            fix_points, sealed_tc, mte = result
            for value in (1, None):
                for fp in fix_points:
                    fp.spec_nesting = value
                _, decoy = _baseline_decoy(mte, sealed_tc, fix_points, seed=seed)
                for fp in fix_points:
                    with self.subTest(seed=seed, spec=value, slot=fp.slot_id):
                        self.assertIn(_slot_inst(decoy, fp).name.lower(), _RETAG_NAMES)


if __name__ == "__main__":
    unittest.main()
