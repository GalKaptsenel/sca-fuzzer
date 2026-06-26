"""
Unit tests for the MTE seal (MteTag) + the sealing pass (MTEInstrumentation) + the engine.

Groups:
  1. TestMteTagSeal      — placeholder/genuine/decoy slot encodings (NOP / IRG / EOR-retag)
  2. TestMteEngine       — baseline (all genuine) and decoy (retag speculative slots) variants
  3. TestMteTaintTracking — sealing-pass taint decisions including ADDG/SUBG propagation
  4. TestMteFixPoint     — reset() semantics
  5. TestMteSealE2E      — seal_test_case structural invariants
"""
import random
import unittest
from typing import Dict, List, Optional, Tuple

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # run from any cwd
from src.interfaces import (
    TestCase, Function, BasicBlock, Instruction, Actor, ActorMode, ActorPL,
    OT, RegisterOperand, MemoryOperand, ImmediateOperand,
)
from src.aarch64.aarch64_seal import (
    Sandbox, CompositeSeal, SealedNIInstrumentation,
    _SandboxInstrumentationBase,
    index_instructions, inst_at, is_speculative, _SANDBOX_MASK,
)
from src.aarch64.aarch64_mte import SealInstrumentation, MTEFixPoint, MteTag, MTE_SLOT_SIZE
from src.aarch64.aarch64_target_desc import AArch64MemRole


# ===========================================================================
# Shared test infrastructure
# ===========================================================================

_ACTOR = Actor(ActorMode.HOST, ActorPL.KERNEL, 0, "main")

_NORM: Dict[str, str] = {f"x{i}": f"x{i}" for i in range(31)}
_NORM["sp"] = "sp"
for i in range(31):
    _NORM[f"w{i}"] = f"x{i}"


class _MTE(SealInstrumentation):
    """SealInstrumentation with the ISA bypassed — no real generator needed."""
    def __init__(self, norm=None):
        self._norm           = norm if norm is not None else dict(_NORM)
        self._sandbox_mask   = "#0x3fffff"
        self._sandbox_base_reg = "x29"
        self._fixpoint_cls   = MTEFixPoint
        self._sandbox        = Sandbox(_SANDBOX_MASK)
        value_seals          = [MteTag()]
        self._value_composite = value_seals[0]
        self._composite      = CompositeSeal([self._sandbox] + value_seals)
        self._stg_seal       = Sandbox(_SANDBOX_MASK & ~0xF)
        self.last_taint_log: List[str] = []


def _mem_base(reg: str) -> MemoryOperand:
    base = RegisterOperand(reg, 64, src=True, dest=False)
    base.mem_role = AArch64MemRole.BASE
    return MemoryOperand(reg, 64, src=True, dest=False, inner=[base])

def _reg_dest(reg: str) -> RegisterOperand:
    return RegisterOperand(reg, 64, src=False, dest=True)

def _reg_src(reg: str) -> RegisterOperand:
    return RegisterOperand(reg, 64, src=True, dest=False)

def _reg_both(reg: str) -> RegisterOperand:
    return RegisterOperand(reg, 64, src=True, dest=True)

def _imm(value: str, name: str = "n/a") -> ImmediateOperand:
    op = ImmediateOperand(value, 12)
    op.name = name
    return op

def _mem_inst(base_reg: str, name: str = "ldr") -> Instruction:
    inst = Instruction(name, has_memory_access=True)
    inst.operands.append(_reg_dest("x0"))
    inst.operands.append(_mem_base(base_reg))
    return inst

def _addg_inst(dest: str, src: str, imm6: int, imm4: int) -> Instruction:
    inst = Instruction("addg", is_instrumentation=False)
    inst.operands.append(RegisterOperand(dest, 64, src=True, dest=True))
    inst.operands.append(RegisterOperand(src, 64, src=True, dest=False))
    inst.operands.append(ImmediateOperand(str(imm6), 6))
    inst.operands.append(ImmediateOperand(str(imm4), 4))
    return inst

def _subg_inst(dest: str, src: str, imm6: int, imm4: int) -> Instruction:
    inst = Instruction("subg", is_instrumentation=False)
    inst.operands.append(RegisterOperand(dest, 64, src=True, dest=True))
    inst.operands.append(RegisterOperand(src, 64, src=True, dest=False))
    inst.operands.append(ImmediateOperand(str(imm6), 6))
    inst.operands.append(ImmediateOperand(str(imm4), 4))
    return inst

def _irg_inst(dest: str, src: str) -> Instruction:
    inst = Instruction("irg", is_instrumentation=False)
    inst.operands.append(RegisterOperand(dest, 64, src=False, dest=True))
    inst.operands.append(RegisterOperand(src, 64, src=True, dest=False))
    return inst

def _simple_function(*instructions: Instruction) -> Function:
    func = Function(".function_test", _ACTOR)
    bb = BasicBlock(".bb_0")
    for inst in instructions:
        bb.insert_after(bb.end, inst)
    func.append(bb)
    return func


def _fp(slot_id=0, reg="x5", spec_nesting=None) -> MTEFixPoint:
    return MTEFixPoint(slot_id=slot_id, value_reg=reg, spec_nesting=spec_nesting)

def _build_prep_tc(fps: List[MTEFixPoint]) -> TestCase:
    """Build a sealed TestCase: one BB per fix point holding its NOP placeholder; records slot_locs."""
    tc = TestCase(seed=0)
    actor = list(tc.actors.values())[0]
    func = Function(".function_main_0", actor)
    for fp in fps:
        bb = BasicBlock(f".bb_slot_{fp.slot_id}")
        nop = Instruction("nop", is_instrumentation=True, template="NOP")
        bb.insert_after(bb.end, nop)
        fp.slot_insts = [nop]
        func.append(bb)
    tc.functions.append(func)
    locs = index_instructions(tc)
    for fp in fps:
        fp.slot_locs = [locs[id(si)] for si in fp.slot_insts]
    return tc

def _slot_inst(tc: TestCase, fp: MTEFixPoint) -> Instruction:
    """The fix point's tag instruction in tc — the last slot position (after any sandbox)."""
    return inst_at(tc, fp.slot_locs[-1])[0]

def _engine(fps: List[MTEFixPoint], prep: Optional[TestCase] = None):
    """An NI engine sealed with prep (built from fps if not given). Each fix point carries its own
    seal; default any bare hand-built fix point to a fresh MteTag."""
    seal = MteTag()
    for fp in fps:
        if fp.seal is None:
            fp.seal = seal
    if prep is None:
        prep = _build_prep_tc(fps)
    eng = SealedNIInstrumentation()
    eng.set_sealed(prep, fps)
    return eng, prep

def _call_build_mte_slots(mte: _MTE, func: Function):
    fix_points: List[MTEFixPoint] = []
    insertions: List = []
    taint_log: List[str] = []
    slot_counter = mte._build_mte_slots(func, 0, fix_points, insertions, taint_log)
    return slot_counter, fix_points, insertions, taint_log


# ===========================================================================
# 1. TestMteTagSeal — slot encodings
# ===========================================================================

class TestMteTagSeal(unittest.TestCase):

    def setUp(self):
        self.seal = MteTag()

    def test_placeholder_is_single_nop(self):
        s = self.seal.placeholder(_fp())
        self.assertEqual(len(s), MTE_SLOT_SIZE)
        self.assertEqual(s[0].name, "nop")
        self.assertTrue(s[0].is_instrumentation)

    def test_genuine_is_nop(self):
        # The correct tag is already on the sandboxed pointer, so genuine keeps it (NOP).
        self.assertEqual(self.seal.genuine(_fp(), random.Random(0))[0].name, "nop")

    def test_decoy_is_irg_or_retag(self):
        rng = random.Random(0)
        names = {self.seal.decoy(_fp(reg="x5"), rng)[0].name for _ in range(64)}
        self.assertTrue(names <= {"irg", "eor"})
        self.assertEqual(names, {"irg", "eor"})  # both decoy kinds appear over many draws

    def test_decoy_uses_fp_reg(self):
        rng = random.Random(0)
        for _ in range(32):
            self.assertIn("x7", self.seal.decoy(_fp(reg="x7"), rng)[0].template)

    def test_retag_masks_touch_only_tag_field(self):
        # Every EOR mask flips bits only in the tag field [59:56]; address bits stay intact.
        for mask in MteTag._TAG_FLIP_MASKS:
            self.assertNotEqual(mask, 0)
            self.assertEqual(mask & ~(0xF << 56), 0)


class TestMteGenuineTagFix(unittest.TestCase):
    """genuine sets the pointer's tag to the cell's tag, but only when they mismatch (ADDG delta)."""

    def setUp(self):
        self.seal = MteTag()

    def _fp(self, ptr_tag=None, correct_tag=None, reg="x5") -> MTEFixPoint:
        fp = MTEFixPoint(slot_id=0, value_reg=reg)
        fp.ptr_tag, fp.correct_tag = ptr_tag, correct_tag
        return fp

    def test_no_tag_info_is_nop(self):
        self.assertEqual(self.seal.genuine(self._fp(), random.Random(0))[0].name, "nop")

    def test_matching_tags_no_fix(self):
        self.assertEqual(self.seal.genuine(self._fp(ptr_tag=3, correct_tag=3), random.Random(0))[0].name, "nop")

    def test_mismatch_fixes_with_addg(self):
        s = self.seal.genuine(self._fp(ptr_tag=3, correct_tag=7, reg="x9"), random.Random(0))
        self.assertEqual(s[0].name, "addg")
        self.assertIn("x9", s[0].template)
        self.assertIn("#0, #4", s[0].template)   # delta = (7 - 3) % 16

    def test_mismatch_wraps_mod16(self):
        s = self.seal.genuine(self._fp(ptr_tag=15, correct_tag=1), random.Random(0))
        self.assertEqual(s[0].name, "addg")
        self.assertIn("#0, #2", s[0].template)   # delta = (1 - 15) % 16 = 2

    def test_all_16x16_tag_combinations(self):
        """Exhaustive: every (ptr_tag, correct_tag) → NOP when equal, ADDG #0,#delta otherwise."""
        for ptr in range(16):
            for cell in range(16):
                s = self.seal.genuine(self._fp(ptr_tag=ptr, correct_tag=cell), random.Random(0))
                delta = (cell - ptr) % 16
                with self.subTest(ptr=ptr, cell=cell):
                    if delta == 0:
                        self.assertEqual(s[0].name, "nop")
                    else:
                        self.assertEqual(s[0].name, "addg")
                        self.assertIn(f"#0, #{delta}", s[0].template)
                        self.assertTrue(1 <= delta <= 15)   # encodable 4-bit tag offset


# ===========================================================================
# 2. TestMteEngine — baseline / decoy variants
# ===========================================================================

class TestMteEngine(unittest.TestCase):

    def _baseline_decoy(self, fps, seed=0):
        eng, _ = _engine(fps)
        return eng.baseline(random.Random(0)), next(eng.decoys(random.Random(seed)))

    # ── baseline: every slot genuine (NOP) ───────────────────────────────
    def test_baseline_all_nop(self):
        for sn in (0, 1, None):
            fp = _fp(spec_nesting=sn)
            base, _ = self._baseline_decoy([fp])
            with self.subTest(spec_nesting=sn):
                self.assertEqual(_slot_inst(base, fp).name, "nop")

    # ── decoy: genuine on arch path, retag on speculative path ────────────
    def test_decoy_arch_slot_is_nop(self):
        fp = _fp(spec_nesting=0)
        _, decoy = self._baseline_decoy([fp])
        self.assertEqual(_slot_inst(decoy, fp).name, "nop")

    def test_decoy_spec_slot_is_retagged(self):
        fp = _fp(spec_nesting=1)
        _, decoy = self._baseline_decoy([fp])
        self.assertIn(_slot_inst(decoy, fp).name, ("irg", "eor"))

    def test_decoy_spec_none_is_retagged(self):
        # spec_nesting=None (CE never reached this access) → treated as speculative.
        fp = _fp(spec_nesting=None)
        _, decoy = self._baseline_decoy([fp])
        self.assertIn(_slot_inst(decoy, fp).name, ("irg", "eor"))

    def test_decoy_retag_uses_fp_reg(self):
        fp = _fp(reg="x9", spec_nesting=1)
        eng, _ = _engine([fp])
        rng = random.Random(1)
        for _ in range(32):
            self.assertIn("x9", _slot_inst(next(eng.decoys(rng)), fp).template)

    def test_mixed_arch_and_spec(self):
        fps = [_fp(slot_id=0, spec_nesting=0), _fp(slot_id=1, spec_nesting=1)]
        _, decoy = self._baseline_decoy(fps)
        self.assertEqual(_slot_inst(decoy, fps[0]).name, "nop")
        self.assertIn(_slot_inst(decoy, fps[1]).name, ("irg", "eor"))

    def test_multiple_slots_all_filled(self):
        fps = [_fp(slot_id=i, reg=f"x{i+1}", spec_nesting=i) for i in range(4)]
        base, decoy = self._baseline_decoy(fps)
        for fp in fps:
            with self.subTest(slot=fp.slot_id):
                self.assertIsNotNone(_slot_inst(base, fp))
                self.assertIsNotNone(_slot_inst(decoy, fp))

    def test_decoys_is_unbounded(self):
        fp = _fp(spec_nesting=1)
        eng, _ = _engine([fp])
        gen = eng.decoys(random.Random(0))
        self.assertEqual(len([next(gen) for _ in range(5)]), 5)


# ===========================================================================
# 3. TestMteTaintTracking
# ===========================================================================

class TestMteTaintTracking(unittest.TestCase):

    def setUp(self):
        self.mte = _MTE()

    # ── basic decisions ───────────────────────────────────────────────────

    def test_first_access_gets_sandbox_nop(self):
        mem = _mem_inst("x5")
        func = _simple_function(mem)
        _, fix_points, insertions, _ = _call_build_mte_slots(self.mte, func)
        self.assertEqual(len(fix_points), 1)
        _, _, sandbox, offset_subs, nop_insts = insertions[0]
        self.assertGreater(len(sandbox), 0)  # sandbox instructions present
        self.assertGreater(len(nop_insts), 0)
        self.assertEqual(nop_insts[0].name, "nop")

    def test_second_access_same_reg_no_sandbox(self):
        """Second access on same register: already tainted, no AND+ADD."""
        mem1 = _mem_inst("x5")
        mem2 = _mem_inst("x5")
        func = _simple_function(mem1, mem2)
        _, _, insertions, _ = _call_build_mte_slots(self.mte, func)
        _, _, sandbox1, _, _ = insertions[0]
        _, _, sandbox2, _, _ = insertions[1]
        self.assertGreater(len(sandbox1), 0)  # first access: sandboxed
        self.assertEqual(len(sandbox2), 0)    # second access: already tainted, no sandbox

    def test_write_clears_taint_triggers_sandbox_again(self):
        """Write to x5 between two accesses: second access is re-sandboxed."""
        mem1 = _mem_inst("x5")
        write = Instruction("mov")
        write.operands.append(_reg_dest("x5"))
        mem2 = _mem_inst("x5")
        func = _simple_function(mem1, write, mem2)
        _, _, insertions, _ = _call_build_mte_slots(self.mte, func)
        _, _, sandbox2, _, _ = insertions[1]
        self.assertGreater(len(sandbox2), 0)  # re-sandboxed after taint cleared

    def test_fix_point_reg_matches_base(self):
        mem = _mem_inst("x3")
        func = _simple_function(mem)
        _, fix_points, _, _ = _call_build_mte_slots(self.mte, func)
        self.assertEqual(fix_points[0].value_reg, "x3")

    def test_implicit_mem_skipped(self):
        """Instruction with no MEM operand in operands/implicit_operands → skip."""
        inst = Instruction("ldr", has_memory_access=True)  # no MEM operand added
        func = _simple_function(inst)
        _, fix_points, insertions, _ = _call_build_mte_slots(self.mte, func)
        self.assertEqual(len(fix_points), 0)
        self.assertEqual(len(insertions), 0)

    # ── STG-family tag stores: 16B-aligned clamp, no tag slot / fix point ────
    def test_stg_aligned_clamp_no_fixpoint(self):
        func = _simple_function(_mem_inst("x5", name="stg"))
        _, fix_points, insertions, _ = _call_build_mte_slots(self.mte, func)
        self.assertEqual(fix_points, [])                 # STG is a tag store, not a sealed use
        _, _, sandbox, _, tag = insertions[0]
        self.assertEqual([i.name for i in sandbox], ["and", "add"])  # clamp
        self.assertEqual(tag, [])                                    # no tag slot
        self.assertIn("0x1ff0", sandbox[0].template)                # 16B-aligned mask (mask & ~0xF)

    def test_stg_then_data_access_reclamps(self):
        # STG clamps+rewrites x5, so a following data access to x5 is a fresh first-use (sandbox+tag)
        func = _simple_function(_mem_inst("x5", name="stg"), _mem_inst("x5"))
        _, fix_points, _, _ = _call_build_mte_slots(self.mte, func)
        self.assertEqual(len(fix_points), 1)             # only the data access is a fix point
        self.assertEqual(fix_points[0].seal.name, "sandbox+mte_tag")

    # ── LDG: a tag LOAD — clamp the base, but not tag-checked, so no tag fix point ────────
    def test_ldg_clamps_without_fixpoint_or_alignment(self):
        func = _simple_function(_mem_inst("x5", name="ldg"))
        _, fix_points, insertions, _ = _call_build_mte_slots(self.mte, func)
        self.assertEqual(fix_points, [])                 # LDG is not tag-checked → no fix point
        _, _, sandbox, _, tag = insertions[0]
        self.assertEqual([i.name for i in sandbox], ["and", "add"])   # clamped in-region
        self.assertIn("0x1fff", sandbox[0].template)     # full mask, NOT the 16B-aligned 0x1ff0
        self.assertEqual(tag, [])                        # no tag slot

    def test_ldg_after_clamp_reuses_taint(self):
        # First access clamps x5 (tainted); a following LDG on x5 needs no re-clamp.
        func = _simple_function(_mem_inst("x5"), _mem_inst("x5", name="ldg"))
        _, _, insertions, _ = _call_build_mte_slots(self.mte, func)
        self.assertEqual(insertions[1][2], [])           # LDG emits no clamp (x5 already in-region)

    # ── address-preserving tag ops propagate taint (IRG, ADDG/SUBG with imm6==0) ──
    # Taint means "in the sandbox region". An op may change the tag freely (the tag is corrected at
    # the access); what gates propagation is whether the ADDRESS is preserved, i.e. imm6==0.

    def test_addg_imm6_zero_propagates_despite_tag_change(self):
        """ADDG x6, x5, #0, #4: imm6==0 keeps the address, so a tainted x5 propagates to x6 even
        though imm4 changes the tag (the tag is fixed at the access, not by the taint)."""
        mem1 = _mem_inst("x5")               # taints x5
        addg = _addg_inst("x6", "x5", 0, 4)  # address preserved (imm6==0), tag offset 4
        mem2 = _mem_inst("x6")               # x6 tainted → no re-clamp
        func = _simple_function(mem1, addg, mem2)
        _, _, insertions, taint_log = _call_build_mte_slots(self.mte, func)
        _, _, sandbox2, _, _ = insertions[1]
        self.assertEqual(len(sandbox2), 0, "imm6==0 must propagate taint:\n" + "\n".join(taint_log))

    def test_addg_imm6_nonzero_clears_taint(self):
        """ADDG x6, x5, #8, #0: imm6!=0 moves the address (up to 1008B) so x6 may leave the region;
        taint must NOT propagate and the next access re-clamps."""
        mem1 = _mem_inst("x5")
        addg = _addg_inst("x6", "x5", 8, 0)  # address moved → clear
        mem2 = _mem_inst("x6")
        func = _simple_function(mem1, addg, mem2)
        _, _, insertions, _ = _call_build_mte_slots(self.mte, func)
        _, _, sandbox2, _, _ = insertions[1]
        self.assertGreater(len(sandbox2), 0)  # x6 untainted → re-clamp

    def test_subg_imm6_zero_propagates(self):
        """SUBG x6, x5, #0, #2 (imm6==0) with x5 tainted → x6 tainted."""
        func = _simple_function(_mem_inst("x5"), _subg_inst("x6", "x5", 0, 2), _mem_inst("x6"))
        _, _, insertions, _ = _call_build_mte_slots(self.mte, func)
        self.assertEqual(len(insertions[1][2]), 0)

    def test_subg_imm6_nonzero_clears(self):
        """SUBG x6, x5, #16, #0 (imm6!=0) → address moved → taint cleared."""
        func = _simple_function(_mem_inst("x5"), _subg_inst("x6", "x5", 16, 0), _mem_inst("x6"))
        _, _, insertions, _ = _call_build_mte_slots(self.mte, func)
        self.assertGreater(len(insertions[1][2]), 0)

    def test_irg_preserves_address_propagates_taint(self):
        """IRG x6, x5 re-tags but keeps the address, so a tainted x5 propagates to x6."""
        mem1 = _mem_inst("x5")
        irg = _irg_inst("x6", "x5")
        mem2 = _mem_inst("x6")
        func = _simple_function(mem1, irg, mem2)
        _, _, insertions, taint_log = _call_build_mte_slots(self.mte, func)
        _, _, sandbox2, _, _ = insertions[1]
        self.assertEqual(len(sandbox2), 0, "IRG preserves the address:\n" + "\n".join(taint_log))

    def test_addg_propagates_from_untainted_stays_untainted(self):
        """ADDG x6, x5, #0, #0 with x5 untainted → x6 also untainted (propagating 'not in region')."""
        addg = _addg_inst("x6", "x5", 0, 0)  # x5 not tainted
        mem = _mem_inst("x6")
        func = _simple_function(addg, mem)
        _, _, insertions, _ = _call_build_mte_slots(self.mte, func)
        _, _, sandbox, _, _ = insertions[0]
        self.assertGreater(len(sandbox), 0)  # x6 not tainted → gets sandbox

    def test_addg_same_reg_self_preserving(self):
        """ADDG x5, x5, #0, #0 with x5 tainted → x5 stays tainted."""
        mem1 = _mem_inst("x5")
        addg = _addg_inst("x5", "x5", 0, 0)
        mem2 = _mem_inst("x5")
        func = _simple_function(mem1, addg, mem2)
        _, _, insertions, _ = _call_build_mte_slots(self.mte, func)
        _, _, sandbox2, _, _ = insertions[1]
        self.assertEqual(len(sandbox2), 0)  # x5 still tainted

    # ── CFG join intersection ─────────────────────────────────────────────

    def test_join_both_tainted_no_sandbox(self):
        """x5 tainted on both branches of a diamond → join sees x5 as tainted."""
        func = Function(".function_test", _ACTOR)
        head = BasicBlock(".head")
        left = BasicBlock(".left")
        right = BasicBlock(".right")
        tail = BasicBlock(".tail")

        head.successors = [left, right]
        left.successors = [tail]
        right.successors = [tail]
        func.extend([head, left, right, tail])

        # Sandbox x5 in head (both branches see it tainted)
        mem_head = _mem_inst("x5")
        head.insert_after(head.end, mem_head)

        mem_tail = _mem_inst("x5")
        tail.insert_after(tail.end, mem_tail)

        _, _, insertions, _ = _call_build_mte_slots(self.mte, func)
        self.assertEqual(len(insertions), 2)
        # head access: gets sandbox (first time)
        _, _, sandbox_head, _, _ = insertions[0]
        self.assertGreater(len(sandbox_head), 0)  # first access: sandboxed
        # tail access: x5 tainted on both paths → no sandbox
        _, _, sandbox_tail, _, _ = insertions[1]
        self.assertEqual(len(sandbox_tail), 0)    # both paths tainted → no re-sandbox

    def test_join_one_path_untainted_gets_sandbox(self):
        """x5 tainted only on LEFT branch → join intersection drops it → sandbox in tail."""
        func = Function(".function_test", _ACTOR)
        head = BasicBlock(".head")
        left = BasicBlock(".left")
        right = BasicBlock(".right")
        tail = BasicBlock(".tail")

        head.successors = [left, right]
        left.successors = [tail]
        right.successors = [tail]
        func.extend([head, left, right, tail])

        # x5 sandboxed only on left branch
        mem_left = _mem_inst("x5")
        left.insert_after(left.end, mem_left)

        mem_tail = _mem_inst("x5")
        tail.insert_after(tail.end, mem_tail)

        _, _, insertions, _ = _call_build_mte_slots(self.mte, func)
        # tail insertion must include sandbox (x5 untainted on right path)
        tail_ins = [ins for ins in insertions if ins[0] is mem_tail]
        self.assertEqual(len(tail_ins), 1)
        _, _, sandbox_tail, _, _ = tail_ins[0]
        self.assertGreater(len(sandbox_tail), 0)  # untainted on right path → re-sandboxed


# ===========================================================================
# 4. TestMteFixPoint
# ===========================================================================

class TestMteFixPoint(unittest.TestCase):

    def test_reset_clears_spec_nesting(self):
        fp = _fp(spec_nesting=5)
        fp.reset()
        self.assertIsNone(fp.spec_nesting)

    def test_reset_keeps_structural_fields(self):
        fp = _fp(reg="x7", spec_nesting=3)
        fp.slot_locs = [(0, 0, 0)]
        fp.reset()
        self.assertEqual(fp.value_reg, "x7")       # structural — survives reset
        self.assertEqual(fp.slot_locs, [(0, 0, 0)])


# ===========================================================================
# 5. TestMteSealE2E
# ===========================================================================

class TestMteSealE2E(unittest.TestCase):

    def setUp(self):
        self.mte = _MTE()

    def _tc_with_mem(self, base_reg: str = "x5") -> TestCase:
        tc = TestCase(seed=0)
        actor = list(tc.actors.values())[0]
        func = Function(".function_main_0", actor)
        bb = BasicBlock(".bb_0")
        mem = _mem_inst(base_reg)
        bb.insert_after(bb.end, mem)
        func.append(bb)
        tc.functions.append(func)
        return tc

    def test_seal_returns_fix_points(self):
        tc = self._tc_with_mem()
        _, fix_points = self.mte.seal_test_case(tc)
        self.assertEqual(len(fix_points), 1)

    def test_seal_records_slot_locs(self):
        tc = self._tc_with_mem("x5")
        prep, fix_points = self.mte.seal_test_case(tc)
        fp = fix_points[0]
        self.assertEqual(len(fp.slot_locs), fp.seal.slot_size)  # composite (sandbox+tag) on first use
        self.assertEqual(_slot_inst(prep, fp).name, "nop")      # tag placeholder is a NOP

    def test_seal_nop_before_memory_access(self):
        tc = self._tc_with_mem("x5")
        prep, _ = self.mte.seal_test_case(tc)
        names = [i.name for func in prep.functions for bb in func for i in bb]
        ldr_idx = next(i for i, n in enumerate(names) if n == "ldr")
        self.assertIn("nop", names[:ldr_idx])  # placeholder appears before the access

    def test_seal_sandbox_before_nop_placeholder(self):
        """Sandbox instrumentation must appear before the tag (NOP) placeholder."""
        tc = self._tc_with_mem("x5")
        prep, fix_points = self.mte.seal_test_case(tc)
        all_insts = [i for func in prep.functions for bb in func for i in bb]
        nop = fix_points[0].slot_insts[-1]   # tag placeholder is last; sandbox AND/ADD precede it
        nop_idx = all_insts.index(nop)
        before = [inst for inst in all_insts[:nop_idx] if inst.is_instrumentation]
        self.assertGreater(len(before), 0, "No sandbox instrumentation found before NOP placeholder")

    def test_seal_fix_point_reg_correct(self):
        tc = self._tc_with_mem("x7")
        _, fix_points = self.mte.seal_test_case(tc)
        self.assertEqual(fix_points[0].value_reg, "x7")

    def test_seal_to_engine_round_trip(self):
        """Sealing output feeds the engine without error; baseline + decoy mint test cases."""
        tc = self._tc_with_mem("x5")
        prep, fix_points = self.mte.seal_test_case(tc)
        for fp in fix_points:
            fp.spec_nesting = 0
        eng = self.mte.make_engine()
        eng.set_sealed(prep, fix_points)
        self.assertIsInstance(eng.baseline(random.Random(0)), TestCase)
        self.assertIsInstance(next(eng.decoys(random.Random(0))), TestCase)

    def test_seal_baseline_slot_is_nop(self):
        tc = self._tc_with_mem("x5")
        prep, fix_points = self.mte.seal_test_case(tc)
        for fp in fix_points:
            fp.spec_nesting = 0
        eng = self.mte.make_engine()
        eng.set_sealed(prep, fix_points)
        self.assertEqual(_slot_inst(eng.baseline(random.Random(0)), fix_points[0]).name, "nop")

    def test_seal_multiple_accesses_separate_slots(self):
        tc = TestCase(seed=0)
        actor = list(tc.actors.values())[0]
        func = Function(".function_main_0", actor)
        bb = BasicBlock(".bb_0")
        for reg in ("x3", "x4", "x5"):
            bb.insert_after(bb.end, _mem_inst(reg))
        func.append(bb)
        tc.functions.append(func)

        prep, fix_points = self.mte.seal_test_case(tc)
        self.assertEqual(len(fix_points), 3)
        self.assertEqual(len({fp.slot_id for fp in fix_points}), 3)


# ===========================================================================
# 6. TestMteCompositeWiring — the pass seals with CompositeSeal([Sandbox, MteTag])
# ===========================================================================

class TestMteCompositeWiring(unittest.TestCase):
    """First use of a base -> sandbox+tag (composite); an already-clamped base -> tag only; and the
    engine (sandbox-protecting policy) keeps the clamp on every variant, decoying only the tag."""

    def setUp(self):
        self.mte = _MTE()
        self.policy = lambda fp, s: s.name != "sandbox" and is_speculative(fp)

    def _tc(self, *regs) -> TestCase:
        tc = TestCase(seed=0)
        actor = list(tc.actors.values())[0]
        func = Function(".function_main_0", actor)
        bb = BasicBlock(".bb_0")
        for r in regs:
            bb.insert_after(bb.end, _mem_inst(r))
        func.append(bb)
        tc.functions.append(func)
        return tc

    def _slot_names(self, tc, fp):
        return [inst_at(tc, loc)[0].name for loc in fp.slot_locs]

    def test_first_use_composite_repeat_tag_only(self):
        _, fps = self.mte.seal_test_case(self._tc("x5", "x5"))   # same base, no write between
        self.assertEqual(fps[0].seal.name, "sandbox+mte_tag")    # first use -> clamp + tag
        self.assertEqual(fps[1].seal.name, "mte_tag")            # already clamped -> tag only

    def test_write_between_reclamps(self):
        tc = TestCase(seed=0)
        actor = list(tc.actors.values())[0]
        func = Function(".f", actor)
        bb = BasicBlock(".bb")
        bb.insert_after(bb.end, _mem_inst("x5"))
        w = Instruction("mov"); w.operands.append(_reg_dest("x5"))
        bb.insert_after(bb.end, w)                                # writes x5 -> clears taint
        bb.insert_after(bb.end, _mem_inst("x5"))
        func.append(bb); tc.functions.append(func)
        _, fps = self.mte.seal_test_case(tc)
        self.assertEqual([fp.seal.name for fp in fps],
                         ["sandbox+mte_tag", "sandbox+mte_tag"])  # both are first-uses

    def test_composite_slot_is_clamp_then_tag(self):
        prep, fps = self.mte.seal_test_case(self._tc("x5"))
        self.assertEqual(self._slot_names(prep, fps[0]), ["and", "add", "nop"])

    def test_tag_only_slot_has_no_clamp(self):
        prep, fps = self.mte.seal_test_case(self._tc("x5", "x5"))
        self.assertEqual(self._slot_names(prep, fps[1]), ["nop"])

    def test_baseline_keeps_clamp_genuine(self):
        prep, fps = self.mte.seal_test_case(self._tc("x5"))
        for fp in fps:
            fp.spec_nesting = 1
        eng = self.mte.make_engine(should_decoy=self.policy)
        eng.set_sealed(prep, fps)
        self.assertEqual(self._slot_names(eng.baseline(random.Random(0)), fps[0]), ["and", "add", "nop"])

    def test_decoy_keeps_clamp_retags_tag(self):
        prep, fps = self.mte.seal_test_case(self._tc("x5"))
        for fp in fps:
            fp.spec_nesting = 1
        eng = self.mte.make_engine(should_decoy=self.policy)
        eng.set_sealed(prep, fps)
        names = self._slot_names(next(eng.decoys(random.Random(0))), fps[0])
        self.assertEqual(names[:2], ["and", "add"])              # clamp never decoyed
        self.assertIn(names[2], ("irg", "eor"))                  # tag decoyed on the spec slot

    def test_decoy_arch_slot_all_genuine(self):
        prep, fps = self.mte.seal_test_case(self._tc("x5"))
        for fp in fps:
            fp.spec_nesting = 0                                  # architectural
        eng = self.mte.make_engine(should_decoy=self.policy)
        eng.set_sealed(prep, fps)
        self.assertEqual(self._slot_names(next(eng.decoys(random.Random(0))), fps[0]),
                         ["and", "add", "nop"])


if __name__ == "__main__":
    unittest.main()
