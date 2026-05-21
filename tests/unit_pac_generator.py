"""
Structural unit tests for PAC TC1/TC2/TC3 slot generation.

Tests are grouped by layer:
  1. TestMakeLoadImmInsts  — MOVZ+MOVK×3 encoding for arbitrary 64-bit values
  2. TestMakeSlotInsts     — slot structure for every include_ctx/include_ptr/include_auth combo
  3. TestMakeTc1SlotInsts  — TC1-specific slot (ctx_restore + ptr_restore + XPAC)
  4. TestSlotTagIntegrity  — _pac_slot_id / _pac_slot_pos set on every produced instruction
  5. TestInstrumentStage2  — end-to-end: structural correctness of TC1/TC2/TC3
  6. TestTc3EqualsTc2      — TC3==TC2 invariant when spec_nesting==0 (always with current CE)
  7. TestErrorCases        — RuntimeError raised loudly on missing captured values

Design principle: tests verify external structure (instruction names, templates, slot tags) —
not internal implementation choices that may validly change.
"""
import re
import copy
import random
import unittest
from typing import Dict, List, Optional

from src.interfaces import (
    TestCase, Function, BasicBlock, Instruction, Actor, ActorMode, ActorPL
)
from src.aarch64.aarch64_generator import (
    PACInstrumentation, FixPoint, SignedRegInfo,
    SLOT_SIZE, FIX_COUNT_CTX, FIX_COUNT_PTR,
    CTX_SLOT_START, PTR_SLOT_START, AUTH_SLOT_POS,
)


# ===========================================================================
# Shared test infrastructure
# ===========================================================================

class _TestPACI(PACInstrumentation):
    """Minimal PACInstrumentation that bypasses the ISA loader.

    Overrides only the two methods that need a real generator/spec-database.
    All pure instruction-building helpers (_make_movz, _make_movk, …) are
    inherited unchanged, so the tests exercise the real encoding logic.
    """
    def __init__(self):
        pass  # skip __init__ — no generator/ISA needed

    def _make_auth_inst(self, mnemonic: str, reg: str, ctx_reg: Optional[str]) -> Instruction:
        t = f"{mnemonic.upper()} {reg}" + (f", {ctx_reg}" if ctx_reg else "")
        return Instruction(mnemonic, is_instrumentation=True, template=t)

    def _make_xpac_inst(self, mnemonic: str, reg: str, slot_id: int, pos: int) -> Instruction:
        i = Instruction(mnemonic, is_instrumentation=True, template=f"{mnemonic.upper()} {reg}")
        i._pac_slot_id = slot_id
        i._pac_slot_pos = pos
        return i


_PACI = _TestPACI()  # one shared instance — all methods are stateless


def _info(reg="x0", ctx_reg="x1", pac_mn="pacia", auth_mn="autia", xpac_mn="xpaci") -> SignedRegInfo:
    return SignedRegInfo(reg=reg, ctx_reg=ctx_reg,
                        pac_mnemonic=pac_mn, auth_mnemonic=auth_mn, xpac_mnemonic=xpac_mn)


def _fp(slot_id=0, reg="x0", ctx_reg="x1",
        signed_value=None, ctx_value=None, spec_nesting=None) -> FixPoint:
    fp = FixPoint(slot_id=slot_id, bb=BasicBlock(".bb"), mem_inst=None,
                  info=_info(reg=reg, ctx_reg=ctx_reg))
    fp.signed_value = signed_value
    fp.ctx_value    = ctx_value
    fp.spec_nesting = spec_nesting
    return fp


def _build_prep_tc(fps: List[FixPoint]) -> TestCase:
    """Build a minimal stage-1 prep TestCase: one BB per FixPoint with SLOT_SIZE tagged NOPs."""
    tc = TestCase(seed=0)
    actor = list(tc.actors.values())[0]
    func = Function(".function_main_0", actor)
    for fp in fps:
        bb = BasicBlock(f".bb_slot_{fp.slot_id}")
        for pos in range(SLOT_SIZE):
            nop = Instruction("nop", is_instrumentation=True, template="NOP")
            nop._pac_slot_id  = fp.slot_id
            nop._pac_slot_pos = pos
            bb.insert_after(bb.end, nop)
        fp.bb = bb
        func.append(bb)
    tc.functions.append(func)
    return tc


def _slot_insts(tc: TestCase, slot_id: int) -> List[Instruction]:
    """Return the 9 instructions in a slot, in position order."""
    pos_map: Dict[int, Instruction] = {}
    for func in tc.functions:
        for bb in func:
            for inst in bb:
                if getattr(inst, "_pac_slot_id", None) == slot_id:
                    pos_map[inst._pac_slot_pos] = inst
    return [pos_map[p] for p in range(SLOT_SIZE)]


def _decode_value(insts: List[Instruction]) -> int:
    """Reconstruct the 64-bit integer loaded by a MOVZ + MOVK× sequence."""
    val = 0
    for inst in insts:
        m_imm = re.search(r'#0x([0-9a-fA-F]+)', inst.template)
        m_lsl = re.search(r'LSL #(\d+)', inst.template)
        assert m_imm, f"No immediate in: {inst.template!r}"
        imm = int(m_imm.group(1), 16)
        lsl = int(m_lsl.group(1)) if m_lsl else 0
        val |= (imm << lsl)
    return val


# ===========================================================================
# 1. TestMakeLoadImmInsts
# ===========================================================================

class TestMakeLoadImmInsts(unittest.TestCase):
    """_make_load_imm_insts: MOVZ + 3×MOVK encodes exact 64-bit values."""

    def _load(self, value, reg="x0", start=0):
        return _PACI._make_load_imm_insts(0, reg, value, start)

    # ── length and instruction sequence ─────────────────────────────────

    def test_returns_four_instructions(self):
        self.assertEqual(len(self._load(0)), 4)

    def test_first_is_movz(self):
        self.assertEqual(self._load(0)[0].name, "movz")

    def test_rest_are_movk(self):
        for i, inst in enumerate(self._load(0)[1:], start=1):
            with self.subTest(pos=i):
                self.assertEqual(inst.name, "movk")

    # ── encoding correctness ─────────────────────────────────────────────

    def test_zero_value(self):
        self.assertEqual(_decode_value(self._load(0)), 0)

    def test_all_ones(self):
        v = 0xFFFF_FFFF_FFFF_FFFF
        self.assertEqual(_decode_value(self._load(v)), v)

    def test_lowest_16_bits_only(self):
        v = 0x0000_0000_0000_ABCD
        self.assertEqual(_decode_value(self._load(v)), v)

    def test_second_chunk_only(self):
        v = 0x0000_0000_CAFE_0000
        self.assertEqual(_decode_value(self._load(v)), v)

    def test_third_chunk_only(self):
        v = 0x0000_BEEF_0000_0000
        self.assertEqual(_decode_value(self._load(v)), v)

    def test_top_chunk_only(self):
        v = 0xDEAD_0000_0000_0000
        self.assertEqual(_decode_value(self._load(v)), v)

    def test_all_chunks_different(self):
        v = 0xDEAD_BEEF_CAFE_1234
        self.assertEqual(_decode_value(self._load(v)), v)

    def test_full_roundtrip_many_values(self):
        vals = [0, 1, 0x8000, 0xFFFF,
                0xDEAD_BEEF_CAFE_1234, 0x0001_0002_0003_0004,
                0xFFFF_FFFF_FFFF_FFFF, 0x0000_0000_FFFF_FFFF]
        for v in vals:
            with self.subTest(v=hex(v)):
                self.assertEqual(_decode_value(self._load(v)), v)

    # ── MOVK LSL shifts ──────────────────────────────────────────────────

    def test_movk_lsl_shifts(self):
        insts = self._load(0xDEAD_BEEF_CAFE_1234)
        expected_lsls = [0, 16, 32, 48]  # MOVZ has no LSL, MOVKs 16/32/48
        for i, inst in enumerate(insts):
            m = re.search(r'LSL #(\d+)', inst.template)
            got_lsl = int(m.group(1)) if m else 0
            self.assertEqual(got_lsl, expected_lsls[i],
                             f"pos {i}: expected LSL #{expected_lsls[i]}, got #{got_lsl}")

    # ── register name propagated ─────────────────────────────────────────

    def test_register_in_templates(self):
        for reg in ("x0", "x5", "x29", "x15"):
            with self.subTest(reg=reg):
                for inst in self._load(0xABCD_1234, reg=reg):
                    self.assertIn(reg, inst.template)

    # ── slot_id and start_pos tags ───────────────────────────────────────

    def test_slot_id_tagged(self):
        for sid in (0, 3, 7):
            insts = _PACI._make_load_imm_insts(sid, "x0", 0x1234, 0)
            for inst in insts:
                self.assertEqual(inst._pac_slot_id, sid)

    def test_start_pos_tagged(self):
        for start in (CTX_SLOT_START, PTR_SLOT_START):
            insts = _PACI._make_load_imm_insts(0, "x0", 0x1234, start)
            for offset, inst in enumerate(insts):
                self.assertEqual(inst._pac_slot_pos, start + offset,
                                 f"start={start} offset={offset}")

    # ── 16-bit chunk in each position ────────────────────────────────────

    def test_each_immediate_is_16bit(self):
        v = 0xDEAD_BEEF_CAFE_1234
        for inst in self._load(v):
            m = re.search(r'#0x([0-9a-fA-F]+)', inst.template)
            imm = int(m.group(1), 16)
            self.assertLessEqual(imm, 0xFFFF, f"immediate {imm:#x} exceeds 16 bits")


# ===========================================================================
# 2. TestMakeSlotInsts
# ===========================================================================

SIGNED_VAL = 0xDEAD_BEEF_CAFE_1234
CTX_VAL    = 0x0000_0000_1234_ABCD


class TestMakeSlotInsts(unittest.TestCase):
    """_make_slot_insts: slot structure for all include_ctx/ptr/auth combinations."""

    def _slot(self, fp, include_ctx=True, include_ptr=True, include_auth=True):
        return _PACI._make_slot_insts(fp, include_ctx, include_ptr, include_auth)

    # ── length ───────────────────────────────────────────────────────────

    def test_always_slot_size(self):
        fp = _fp(signed_value=SIGNED_VAL, ctx_value=CTX_VAL, spec_nesting=0)
        for ctx in (True, False):
            for ptr in (True, False):
                for auth in (True, False):
                    with self.subTest(ctx=ctx, ptr=ptr, auth=auth):
                        # skip the raise case
                        if ptr and fp.signed_value is None:
                            continue
                        result = self._slot(fp, ctx, ptr, auth)
                        self.assertEqual(len(result), SLOT_SIZE)

    # ── auth position ─────────────────────────────────────────────────────

    def test_auth_at_position_8(self):
        fp = _fp(signed_value=SIGNED_VAL, ctx_value=CTX_VAL, spec_nesting=0)
        insts = self._slot(fp)
        self.assertEqual(insts[AUTH_SLOT_POS].name, "autia")

    def test_nop_at_position_8_when_no_auth(self):
        fp = _fp(signed_value=SIGNED_VAL, ctx_value=CTX_VAL, spec_nesting=0)
        insts = self._slot(fp, include_auth=False)
        self.assertEqual(insts[AUTH_SLOT_POS].name, "nop")

    # ── ctx restore positions 0–3 ─────────────────────────────────────────

    def test_ctx_restore_pattern_movz_movk(self):
        fp = _fp(signed_value=SIGNED_VAL, ctx_value=CTX_VAL, spec_nesting=0)
        insts = self._slot(fp)
        self.assertEqual(insts[0].name, "movz")
        for i in range(1, FIX_COUNT_CTX):
            self.assertEqual(insts[i].name, "movk")

    def test_ctx_positions_nop_when_include_ctx_false(self):
        fp = _fp(signed_value=SIGNED_VAL, ctx_value=CTX_VAL, spec_nesting=0)
        insts = self._slot(fp, include_ctx=False)
        for i in range(FIX_COUNT_CTX):
            with self.subTest(pos=i):
                self.assertEqual(insts[i].name, "nop")

    def test_ctx_positions_nop_when_ctx_reg_is_none(self):
        info = SignedRegInfo(reg="x0", ctx_reg=None, pac_mnemonic="paciza",
                             auth_mnemonic="autiza", xpac_mnemonic="xpaci")
        fp = FixPoint(slot_id=0, bb=BasicBlock(".bb"), mem_inst=None, info=info)
        fp.signed_value = SIGNED_VAL
        fp.ctx_value    = None   # legitimate: zero-context variant
        fp.spec_nesting = 0
        insts = self._slot(fp)
        for i in range(FIX_COUNT_CTX):
            with self.subTest(pos=i):
                self.assertEqual(insts[i].name, "nop")

    def test_ctx_value_correctly_encoded(self):
        fp = _fp(signed_value=SIGNED_VAL, ctx_value=CTX_VAL, spec_nesting=0)
        insts = self._slot(fp)
        self.assertEqual(_decode_value(insts[CTX_SLOT_START:CTX_SLOT_START + FIX_COUNT_CTX]),
                         CTX_VAL)

    def test_ctx_register_in_ctx_templates(self):
        fp = _fp(reg="x0", ctx_reg="x3", signed_value=SIGNED_VAL, ctx_value=CTX_VAL)
        insts = self._slot(fp)
        for i in range(FIX_COUNT_CTX):
            self.assertIn("x3", insts[i].template,
                          f"ctx reg 'x3' not in ctx position {i}: {insts[i].template!r}")

    # ── ptr restore positions 4–7 ─────────────────────────────────────────

    def test_ptr_restore_pattern_movz_movk(self):
        fp = _fp(signed_value=SIGNED_VAL, ctx_value=CTX_VAL, spec_nesting=0)
        insts = self._slot(fp)
        self.assertEqual(insts[PTR_SLOT_START].name, "movz")
        for i in range(1, FIX_COUNT_PTR):
            self.assertEqual(insts[PTR_SLOT_START + i].name, "movk")

    def test_ptr_positions_nop_when_include_ptr_false(self):
        fp = _fp(signed_value=SIGNED_VAL, ctx_value=CTX_VAL, spec_nesting=0)
        insts = self._slot(fp, include_ptr=False)
        for i in range(FIX_COUNT_PTR):
            with self.subTest(pos=PTR_SLOT_START + i):
                self.assertEqual(insts[PTR_SLOT_START + i].name, "nop")

    def test_ptr_value_correctly_encoded(self):
        fp = _fp(signed_value=SIGNED_VAL, ctx_value=CTX_VAL, spec_nesting=0)
        insts = self._slot(fp)
        self.assertEqual(_decode_value(insts[PTR_SLOT_START:PTR_SLOT_START + FIX_COUNT_PTR]),
                         SIGNED_VAL)

    def test_ptr_register_in_ptr_templates(self):
        fp = _fp(reg="x7", ctx_reg="x2", signed_value=SIGNED_VAL, ctx_value=CTX_VAL)
        insts = self._slot(fp)
        for i in range(FIX_COUNT_PTR):
            self.assertIn("x7", insts[PTR_SLOT_START + i].template,
                          f"ptr reg 'x7' missing at ptr pos {i}")

    # ── all mnemonics ────────────────────────────────────────────────────

    def test_all_auth_mnemonics(self):
        for pac_mn, auth_mn, xpac_mn in [
            ("pacia",  "autia",  "xpaci"),
            ("pacib",  "autib",  "xpaci"),
            ("pacda",  "autda",  "xpacd"),
            ("pacdb",  "autdb",  "xpacd"),
            ("paciza", "autiza", "xpaci"),
            ("pacdza", "autdza", "xpacd"),
        ]:
            with self.subTest(auth_mn=auth_mn):
                info = SignedRegInfo(reg="x0", ctx_reg=None if auth_mn.endswith("za") else "x1",
                                    pac_mnemonic=pac_mn, auth_mnemonic=auth_mn, xpac_mnemonic=xpac_mn)
                fp2 = FixPoint(slot_id=0, bb=BasicBlock(".bb"), mem_inst=None, info=info)
                fp2.signed_value = SIGNED_VAL
                fp2.ctx_value    = None if info.ctx_reg is None else CTX_VAL
                fp2.spec_nesting = 0
                insts = self._slot(fp2)
                self.assertEqual(insts[AUTH_SLOT_POS].name, auth_mn)

    # ── ctx and ptr encode different registers / values ───────────────────

    def test_ctx_and_ptr_use_different_registers(self):
        fp = _fp(reg="x5", ctx_reg="x6", signed_value=0x0000_0000_0000_FFFF,
                 ctx_value=0x0000_0000_FFFF_0000, spec_nesting=0)
        insts = self._slot(fp)
        for i in range(FIX_COUNT_CTX):
            self.assertIn("x6", insts[i].template)
        for i in range(FIX_COUNT_PTR):
            self.assertIn("x5", insts[PTR_SLOT_START + i].template)


# ===========================================================================
# 3. TestMakeTc1SlotInsts
# ===========================================================================

class TestMakeTc1SlotInsts(unittest.TestCase):
    """_make_tc1_slot_insts: ctx_restore + ptr_restore(signed_value) + XPAC."""

    def _tc1(self, fp):
        return _PACI._make_tc1_slot_insts(fp)

    # ── length ───────────────────────────────────────────────────────────

    def test_returns_slot_size(self):
        fp = _fp(signed_value=SIGNED_VAL, ctx_value=CTX_VAL, spec_nesting=0)
        self.assertEqual(len(self._tc1(fp)), SLOT_SIZE)

    # ── position 8 is always XPAC, never AUTH ────────────────────────────

    def test_position_8_is_xpac(self):
        fp = _fp(signed_value=SIGNED_VAL, ctx_value=CTX_VAL)
        insts = self._tc1(fp)
        self.assertEqual(insts[AUTH_SLOT_POS].name, "xpaci")

    def test_xpacd_for_data_variants(self):
        info = SignedRegInfo(reg="x0", ctx_reg="x1", pac_mnemonic="pacda",
                             auth_mnemonic="autda", xpac_mnemonic="xpacd")
        fp2 = FixPoint(slot_id=0, bb=BasicBlock(".bb"), mem_inst=None, info=info)
        fp2.signed_value = SIGNED_VAL
        fp2.ctx_value    = CTX_VAL
        insts = self._tc1(fp2)
        self.assertEqual(insts[AUTH_SLOT_POS].name, "xpacd")

    def test_no_auth_instruction_anywhere(self):
        fp = _fp(signed_value=SIGNED_VAL, ctx_value=CTX_VAL)
        insts = self._tc1(fp)
        for inst in insts:
            self.assertNotIn(inst.name, ("autia", "autib", "autda", "autdb",
                                         "autiza", "autizb", "autdza", "autdzb"),
                             f"AUTH found in TC1 at pos {inst._pac_slot_pos}")

    # ── ptr restore encodes signed_value (not the raw pointer) ───────────

    def test_ptr_encodes_signed_value(self):
        fp = _fp(signed_value=SIGNED_VAL, ctx_value=CTX_VAL)
        insts = self._tc1(fp)
        encoded = _decode_value(insts[PTR_SLOT_START:PTR_SLOT_START + FIX_COUNT_PTR])
        self.assertEqual(encoded, SIGNED_VAL)

    def test_ptr_restore_pattern_movz_movk(self):
        fp = _fp(signed_value=SIGNED_VAL, ctx_value=CTX_VAL)
        insts = self._tc1(fp)
        self.assertEqual(insts[PTR_SLOT_START].name, "movz")
        for i in range(1, FIX_COUNT_PTR):
            self.assertEqual(insts[PTR_SLOT_START + i].name, "movk")

    # ── ctx restore ───────────────────────────────────────────────────────

    def test_ctx_encodes_ctx_value(self):
        fp = _fp(signed_value=SIGNED_VAL, ctx_value=CTX_VAL)
        insts = self._tc1(fp)
        encoded = _decode_value(insts[CTX_SLOT_START:CTX_SLOT_START + FIX_COUNT_CTX])
        self.assertEqual(encoded, CTX_VAL)

    def test_ctx_nops_when_no_ctx_reg(self):
        info = SignedRegInfo(reg="x0", ctx_reg=None, pac_mnemonic="paciza",
                             auth_mnemonic="autiza", xpac_mnemonic="xpaci")
        fp2 = FixPoint(slot_id=0, bb=BasicBlock(".bb"), mem_inst=None, info=info)
        fp2.signed_value = SIGNED_VAL
        fp2.ctx_value    = None
        insts = self._tc1(fp2)
        for i in range(FIX_COUNT_CTX):
            with self.subTest(pos=i):
                self.assertEqual(insts[i].name, "nop")

    # ── structure summary ────────────────────────────────────────────────

    def test_full_name_sequence(self):
        fp = _fp(signed_value=SIGNED_VAL, ctx_value=CTX_VAL)
        names = [i.name for i in self._tc1(fp)]
        self.assertEqual(names, ["movz", "movk", "movk", "movk",
                                  "movz", "movk", "movk", "movk",
                                  "xpaci"])

    def test_full_name_sequence_zero_ctx(self):
        info = SignedRegInfo(reg="x0", ctx_reg=None, pac_mnemonic="paciza",
                             auth_mnemonic="autiza", xpac_mnemonic="xpaci")
        fp2 = FixPoint(slot_id=0, bb=BasicBlock(".bb"), mem_inst=None, info=info)
        fp2.signed_value = SIGNED_VAL
        names = [i.name for i in self._tc1(fp2)]
        self.assertEqual(names, ["nop", "nop", "nop", "nop",
                                  "movz", "movk", "movk", "movk",
                                  "xpaci"])


# ===========================================================================
# 4. TestSlotTagIntegrity
# ===========================================================================

class TestSlotTagIntegrity(unittest.TestCase):
    """Every instruction produced by the slot builders has the correct slot tags."""

    def _check_tags(self, insts: List[Instruction], expected_sid: int,
                    expected_start_pos: int = 0):
        for offset, inst in enumerate(insts):
            pos = expected_start_pos + offset
            with self.subTest(pos=pos):
                self.assertEqual(getattr(inst, "_pac_slot_id", "MISSING"), expected_sid,
                                 f"_pac_slot_id wrong on '{inst.name}' at pos {pos}")
                self.assertEqual(getattr(inst, "_pac_slot_pos", "MISSING"), pos,
                                 f"_pac_slot_pos wrong on '{inst.name}' at pos {pos}")

    def test_make_load_imm_tags(self):
        for sid in (0, 5):
            for start in (CTX_SLOT_START, PTR_SLOT_START):
                insts = _PACI._make_load_imm_insts(sid, "x0", 0x1234, start)
                self._check_tags(insts, sid, start)

    def test_make_slot_insts_all_tags(self):
        fp = _fp(slot_id=3, signed_value=SIGNED_VAL, ctx_value=CTX_VAL)
        insts = _PACI._make_slot_insts(fp, True, True, True)
        self._check_tags(insts, expected_sid=3, expected_start_pos=0)

    def test_make_tc1_slot_insts_all_tags(self):
        fp = _fp(slot_id=7, signed_value=SIGNED_VAL, ctx_value=CTX_VAL)
        insts = _PACI._make_tc1_slot_insts(fp)
        self._check_tags(insts, expected_sid=7, expected_start_pos=0)

    def test_instrument_stage2_tags_preserved(self):
        fp = _fp(slot_id=2, signed_value=SIGNED_VAL, ctx_value=CTX_VAL, spec_nesting=0)
        prep = _build_prep_tc([fp])
        _tc1, tc2, _tc3 = _PACI.instrument_stage2(prep, [fp])
        for pos, inst in enumerate(_slot_insts(tc2, slot_id=2)):
            with self.subTest(pos=pos):
                self.assertEqual(inst._pac_slot_id, 2)
                self.assertEqual(inst._pac_slot_pos, pos)


# ===========================================================================
# 5. TestInstrumentStage2
# ===========================================================================

class TestInstrumentStage2(unittest.TestCase):
    """instrument_stage2: structural correctness of TC1/TC2/TC3."""

    def _run(self, fps, prep=None):
        if prep is None:
            prep = _build_prep_tc(fps)
        return _PACI.instrument_stage2(prep, fps)

    # ── TC1 structure ─────────────────────────────────────────────────────

    def test_tc1_position_8_is_xpac(self):
        fp = _fp(signed_value=SIGNED_VAL, ctx_value=CTX_VAL, spec_nesting=0)
        tc1, _, _ = self._run([fp])
        self.assertEqual(_slot_insts(tc1, 0)[AUTH_SLOT_POS].name, "xpaci")

    def test_tc1_ptr_encodes_signed_value(self):
        fp = _fp(signed_value=SIGNED_VAL, ctx_value=CTX_VAL, spec_nesting=0)
        tc1, _, _ = self._run([fp])
        s = _slot_insts(tc1, 0)
        self.assertEqual(_decode_value(s[PTR_SLOT_START:PTR_SLOT_START + FIX_COUNT_PTR]),
                         SIGNED_VAL)

    def test_tc1_ctx_encodes_ctx_value(self):
        fp = _fp(signed_value=SIGNED_VAL, ctx_value=CTX_VAL, spec_nesting=0)
        tc1, _, _ = self._run([fp])
        s = _slot_insts(tc1, 0)
        self.assertEqual(_decode_value(s[CTX_SLOT_START:CTX_SLOT_START + FIX_COUNT_CTX]),
                         CTX_VAL)

    def test_tc1_full_name_sequence(self):
        fp = _fp(signed_value=SIGNED_VAL, ctx_value=CTX_VAL, spec_nesting=0)
        tc1, _, _ = self._run([fp])
        names = [i.name for i in _slot_insts(tc1, 0)]
        self.assertEqual(names, ["movz", "movk", "movk", "movk",
                                  "movz", "movk", "movk", "movk",
                                  "xpaci"])

    # ── TC2 structure ─────────────────────────────────────────────────────

    def test_tc2_position_8_is_auth(self):
        fp = _fp(signed_value=SIGNED_VAL, ctx_value=CTX_VAL, spec_nesting=0)
        _, tc2, _ = self._run([fp])
        self.assertEqual(_slot_insts(tc2, 0)[AUTH_SLOT_POS].name, "autia")

    def test_tc2_ptr_encodes_signed_value(self):
        fp = _fp(signed_value=SIGNED_VAL, ctx_value=CTX_VAL, spec_nesting=0)
        _, tc2, _ = self._run([fp])
        s = _slot_insts(tc2, 0)
        self.assertEqual(_decode_value(s[PTR_SLOT_START:PTR_SLOT_START + FIX_COUNT_PTR]),
                         SIGNED_VAL)

    def test_tc2_ctx_encodes_ctx_value(self):
        fp = _fp(signed_value=SIGNED_VAL, ctx_value=CTX_VAL, spec_nesting=0)
        _, tc2, _ = self._run([fp])
        s = _slot_insts(tc2, 0)
        self.assertEqual(_decode_value(s[CTX_SLOT_START:CTX_SLOT_START + FIX_COUNT_CTX]),
                         CTX_VAL)

    def test_tc2_full_name_sequence(self):
        fp = _fp(signed_value=SIGNED_VAL, ctx_value=CTX_VAL, spec_nesting=0)
        _, tc2, _ = self._run([fp])
        names = [i.name for i in _slot_insts(tc2, 0)]
        self.assertEqual(names, ["movz", "movk", "movk", "movk",
                                  "movz", "movk", "movk", "movk",
                                  "autia"])

    # ── zero-context variant ──────────────────────────────────────────────

    def test_zero_ctx_tc1_positions_0_3_are_nops(self):
        info = SignedRegInfo(reg="x0", ctx_reg=None, pac_mnemonic="paciza",
                             auth_mnemonic="autiza", xpac_mnemonic="xpaci")
        fp2 = FixPoint(slot_id=0, bb=None, mem_inst=None, info=info)
        fp2.signed_value = SIGNED_VAL
        fp2.ctx_value    = None
        fp2.spec_nesting = 0
        prep = _build_prep_tc([fp2])
        tc1, tc2, tc3 = _PACI.instrument_stage2(prep, [fp2])
        for tc, label in ((tc1, "TC1"), (tc2, "TC2"), (tc3, "TC3")):
            s = _slot_insts(tc, 0)
            for i in range(FIX_COUNT_CTX):
                with self.subTest(variant=label, pos=i):
                    self.assertEqual(s[i].name, "nop")

    # ── multi-slot independence ───────────────────────────────────────────

    def test_two_slots_independently_filled(self):
        fp0 = _fp(slot_id=0, reg="x0", ctx_reg="x1",
                  signed_value=0xAAAA_AAAA_AAAA_AAAA,
                  ctx_value=0x1111_1111_1111_1111, spec_nesting=0)
        fp1 = _fp(slot_id=1, reg="x5", ctx_reg="x6",
                  signed_value=0xBBBB_BBBB_BBBB_BBBB,
                  ctx_value=0x2222_2222_2222_2222, spec_nesting=0)
        _, tc2, _ = self._run([fp0, fp1])
        s0 = _slot_insts(tc2, 0)
        s1 = _slot_insts(tc2, 1)
        self.assertEqual(_decode_value(s0[PTR_SLOT_START:PTR_SLOT_START+4]),
                         0xAAAA_AAAA_AAAA_AAAA)
        self.assertEqual(_decode_value(s1[PTR_SLOT_START:PTR_SLOT_START+4]),
                         0xBBBB_BBBB_BBBB_BBBB)
        self.assertEqual(_decode_value(s0[CTX_SLOT_START:CTX_SLOT_START+4]),
                         0x1111_1111_1111_1111)
        self.assertEqual(_decode_value(s1[CTX_SLOT_START:CTX_SLOT_START+4]),
                         0x2222_2222_2222_2222)

    def test_tc1_tc2_tc3_are_independent_objects(self):
        """Modifying instructions in tc1 must not affect tc2 or tc3."""
        fp = _fp(signed_value=SIGNED_VAL, ctx_value=CTX_VAL, spec_nesting=0)
        tc1, tc2, tc3 = self._run([fp])
        tc1_s = _slot_insts(tc1, 0)
        tc2_s = _slot_insts(tc2, 0)
        tc3_s = _slot_insts(tc3, 0)
        for i in range(SLOT_SIZE):
            self.assertIsNot(tc1_s[i], tc2_s[i],
                             f"TC1 and TC2 share instruction object at pos {i}")
            self.assertIsNot(tc2_s[i], tc3_s[i],
                             f"TC2 and TC3 share instruction object at pos {i}")

    # ── all PAC mnemonic pairs ────────────────────────────────────────────

    def test_all_mnemonic_pairs(self):
        pairs = [
            ("pacia", "autia", "xpaci"),
            ("pacib", "autib", "xpaci"),
            ("pacda", "autda", "xpacd"),
            ("pacdb", "autdb", "xpacd"),
        ]
        for pac_mn, auth_mn, xpac_mn in pairs:
            with self.subTest(auth=auth_mn, xpac=xpac_mn):
                info = SignedRegInfo(reg="x0", ctx_reg="x1",
                                    pac_mnemonic=pac_mn, auth_mnemonic=auth_mn,
                                    xpac_mnemonic=xpac_mn)
                fp2 = FixPoint(slot_id=0, bb=None, mem_inst=None, info=info)
                fp2.signed_value = SIGNED_VAL
                fp2.ctx_value    = CTX_VAL
                fp2.spec_nesting = 0
                prep = _build_prep_tc([fp2])
                tc1, tc2, tc3 = _PACI.instrument_stage2(prep, [fp2])
                self.assertEqual(_slot_insts(tc1, 0)[AUTH_SLOT_POS].name, xpac_mn)
                self.assertEqual(_slot_insts(tc2, 0)[AUTH_SLOT_POS].name, auth_mn)
                self.assertEqual(_slot_insts(tc3, 0)[AUTH_SLOT_POS].name, auth_mn)


# ===========================================================================
# 6. TestTc3EqualsTc2
# ===========================================================================

class TestTc3EqualsTc2(unittest.TestCase):
    """TC3 == TC2 invariant: when spec_nesting==0 (always with current CE) the contract
    executor never speculates so all fix-points are arch-path, and TC3 and TC2 receive
    the identical slot content (both call _make_slot_insts(fp, True, True, True))."""

    def _compare_slots(self, tc_a: TestCase, tc_b: TestCase, slot_id: int) -> None:
        sa = _slot_insts(tc_a, slot_id)
        sb = _slot_insts(tc_b, slot_id)
        for pos, (ia, ib) in enumerate(zip(sa, sb)):
            with self.subTest(pos=pos):
                self.assertEqual(ia.name, ib.name,
                                 f"name mismatch at pos {pos}: {ia.name!r} vs {ib.name!r}")
                self.assertEqual(ia.template, ib.template,
                                 f"template mismatch at pos {pos}: {ia.template!r} vs {ib.template!r}")

    def test_tc3_equals_tc2_spec_nesting_zero(self):
        fp = _fp(signed_value=SIGNED_VAL, ctx_value=CTX_VAL, spec_nesting=0)
        prep = _build_prep_tc([fp])
        _, tc2, tc3 = _PACI.instrument_stage2(prep, [fp])
        self._compare_slots(tc2, tc3, 0)

    def test_tc3_equals_tc2_spec_nesting_none(self):
        fp = _fp(signed_value=SIGNED_VAL, ctx_value=CTX_VAL, spec_nesting=None)
        prep = _build_prep_tc([fp])
        _, tc2, tc3 = _PACI.instrument_stage2(prep, [fp])
        self._compare_slots(tc2, tc3, 0)

    def test_tc3_equals_tc2_multi_slot(self):
        fps = [
            _fp(slot_id=0, signed_value=0x1111_2222_3333_4444,
                ctx_value=0xAAAA_BBBB_CCCC_DDDD, spec_nesting=0),
            _fp(slot_id=1, reg="x5", ctx_reg="x6",
                signed_value=0x5555_6666_7777_8888,
                ctx_value=0xEEEE_FFFF_0000_1111, spec_nesting=0),
        ]
        prep = _build_prep_tc(fps)
        _, tc2, tc3 = _PACI.instrument_stage2(prep, fps)
        self._compare_slots(tc2, tc3, 0)
        self._compare_slots(tc2, tc3, 1)

    def test_tc3_differs_tc1(self):
        """TC3/TC2 have AUTH at pos 8, TC1 has XPAC — they are not identical."""
        fp = _fp(signed_value=SIGNED_VAL, ctx_value=CTX_VAL, spec_nesting=0)
        prep = _build_prep_tc([fp])
        tc1, _, tc3 = _PACI.instrument_stage2(prep, [fp])
        s1 = _slot_insts(tc1, 0)
        s3 = _slot_insts(tc3, 0)
        # position 8: TC1=XPAC, TC3=AUTH
        self.assertNotEqual(s1[AUTH_SLOT_POS].name, s3[AUTH_SLOT_POS].name)

    def test_tc3_spec_path_has_auth_at_8(self):
        """Even when spec_nesting>0 (spec slot), position 8 always has AUTH."""
        fp = _fp(signed_value=SIGNED_VAL, ctx_value=CTX_VAL, spec_nesting=1)
        prep = _build_prep_tc([fp])
        # Pick a fixed seed so random.choice is deterministic
        random.seed(42)
        _, _, tc3 = _PACI.instrument_stage2(prep, [fp])
        # Position 8 must be AUTH regardless of which combo was chosen
        self.assertEqual(_slot_insts(tc3, 0)[AUTH_SLOT_POS].name, "autia")


# ===========================================================================
# 7. TestErrorCases
# ===========================================================================

class TestErrorCases(unittest.TestCase):
    """RuntimeError raised loudly on missing captured values — no silent fallbacks."""

    # ── _make_slot_insts ─────────────────────────────────────────────────

    def test_signed_none_with_include_ptr_raises(self):
        fp = _fp(signed_value=None, ctx_value=CTX_VAL)
        with self.assertRaises(RuntimeError) as cm:
            _PACI._make_slot_insts(fp, True, True, True)
        self.assertIn("signed_value", str(cm.exception))
        self.assertIn("include_ptr", str(cm.exception))

    def test_ctx_none_with_include_ctx_and_ctx_reg_raises(self):
        fp = _fp(signed_value=SIGNED_VAL, ctx_value=None)  # ctx_reg defaults to "x1"
        with self.assertRaises(RuntimeError) as cm:
            _PACI._make_slot_insts(fp, True, True, True)
        self.assertIn("ctx_value", str(cm.exception))
        self.assertIn("include_ctx", str(cm.exception))

    def test_signed_none_include_ptr_false_ok(self):
        """include_ptr=False: signed_value=None is fine (NOPs fill that range)."""
        fp = _fp(signed_value=None, ctx_value=CTX_VAL)
        result = _PACI._make_slot_insts(fp, True, False, True)
        self.assertEqual(len(result), SLOT_SIZE)
        for i in range(FIX_COUNT_PTR):
            self.assertEqual(result[PTR_SLOT_START + i].name, "nop")

    def test_ctx_none_with_ctx_reg_none_ok(self):
        """Zero-context variant: ctx_reg=None, ctx_value=None is fine."""
        info = SignedRegInfo(reg="x0", ctx_reg=None, pac_mnemonic="paciza",
                             auth_mnemonic="autiza", xpac_mnemonic="xpaci")
        fp2 = FixPoint(slot_id=0, bb=BasicBlock(".bb"), mem_inst=None, info=info)
        fp2.signed_value = SIGNED_VAL
        fp2.ctx_value    = None
        result = _PACI._make_slot_insts(fp2, True, True, True)
        self.assertEqual(len(result), SLOT_SIZE)

    def test_slot_id_in_error_message(self):
        fp = _fp(slot_id=5, signed_value=None, ctx_value=CTX_VAL)
        with self.assertRaises(RuntimeError) as cm:
            _PACI._make_slot_insts(fp, True, True, True)
        self.assertIn("5", str(cm.exception))

    # ── _make_tc1_slot_insts ─────────────────────────────────────────────

    def test_tc1_signed_none_raises(self):
        fp = _fp(signed_value=None, ctx_value=CTX_VAL)
        with self.assertRaises(RuntimeError) as cm:
            _PACI._make_tc1_slot_insts(fp)
        self.assertIn("signed_value", str(cm.exception))
        self.assertIn("TC1", str(cm.exception))

    def test_tc1_ctx_none_with_ctx_reg_raises(self):
        fp = _fp(signed_value=SIGNED_VAL, ctx_value=None)  # ctx_reg="x1"
        with self.assertRaises(RuntimeError) as cm:
            _PACI._make_tc1_slot_insts(fp)
        self.assertIn("ctx_value", str(cm.exception))
        self.assertIn("TC1", str(cm.exception))

    def test_tc1_ctx_none_with_no_ctx_reg_ok(self):
        info = SignedRegInfo(reg="x0", ctx_reg=None, pac_mnemonic="paciza",
                             auth_mnemonic="autiza", xpac_mnemonic="xpaci")
        fp2 = FixPoint(slot_id=0, bb=BasicBlock(".bb"), mem_inst=None, info=info)
        fp2.signed_value = SIGNED_VAL
        fp2.ctx_value    = None
        result = _PACI._make_tc1_slot_insts(fp2)
        self.assertEqual(len(result), SLOT_SIZE)

    # ── instrument_stage2 propagates errors ──────────────────────────────

    def test_stage2_propagates_signed_none_error(self):
        fp = _fp(signed_value=None, ctx_value=CTX_VAL, spec_nesting=0)
        prep = _build_prep_tc([fp])
        with self.assertRaises(RuntimeError):
            _PACI.instrument_stage2(prep, [fp])

    def test_stage2_propagates_ctx_none_error(self):
        fp = _fp(signed_value=SIGNED_VAL, ctx_value=None, spec_nesting=0)
        prep = _build_prep_tc([fp])
        with self.assertRaises(RuntimeError):
            _PACI.instrument_stage2(prep, [fp])

    def test_stage2_one_bad_fp_among_good_raises(self):
        """Any single bad FixPoint aborts the whole instrument_stage2 call."""
        good = _fp(slot_id=0, signed_value=SIGNED_VAL, ctx_value=CTX_VAL, spec_nesting=0)
        bad  = _fp(slot_id=1, signed_value=None,       ctx_value=CTX_VAL, spec_nesting=0)
        prep = _build_prep_tc([good, bad])
        with self.assertRaises(RuntimeError):
            _PACI.instrument_stage2(prep, [good, bad])


if __name__ == "__main__":
    unittest.main(verbosity=2)
