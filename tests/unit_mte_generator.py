"""
Unit tests for MTEInstrumentation and MTEFixPoint.

Groups:
  1. TestMteInstructionBuilders  — NOP/IRG/MOVK template and _mte_slot_id correctness
  2. TestMteWrongTagComputation  — wrong_upper16 for various sandbox_base values
  3. TestMteFillSlot             — _fill_slot replaces placeholder correctly
  4. TestMteInstrumentStage2     — TC1/TC2/TC3 structure (NOP/IRG/MOVK per arch/spec)
  5. TestMteTaintTracking        — taint decisions including ADDG/SUBG propagation
  6. TestMteErrorCases           — slot_map assertion, spec_nesting None vs 0 vs >0
  7. TestMteStage1E2E            — instrument_stage1 structural invariants
"""
import copy
import unittest
from typing import Dict, List, Optional, Tuple

from src.interfaces import (
    TestCase, Function, BasicBlock, Instruction, Actor, ActorMode, ActorPL,
    OT, RegisterOperand, MemoryOperand, ImmediateOperand,
)
from src.aarch64.aarch64_generator import (
    MTEInstrumentation, MTEFixPoint, MTEVariant,
    _SandboxInstrumentationBase,
    MTE_SLOT_SIZE,
)


# ===========================================================================
# Shared test infrastructure
# ===========================================================================

_ACTOR = Actor(ActorMode.HOST, ActorPL.KERNEL, 0, "main")

_NORM: Dict[str, str] = {f"x{i}": f"x{i}" for i in range(31)}
_NORM["sp"] = "sp"
for i in range(31):
    _NORM[f"w{i}"] = f"x{i}"


class _MTE(MTEInstrumentation):
    """MTEInstrumentation with the ISA bypassed — no real generator needed."""
    def __init__(self, norm=None):
        self._norm           = norm if norm is not None else dict(_NORM)
        self._sandbox_mask   = "#0x3fffff"
        self._sandbox_base_reg = "x29"
        self.last_taint_log: List[str] = []


_MTE_INST = _MTE()  # shared stateless instance


def _mem_base(reg: str) -> MemoryOperand:
    return MemoryOperand(reg, 64, src=True, dest=False)

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

def _simple_function(*instructions: Instruction) -> Function:
    func = Function(".function_test", _ACTOR)
    bb = BasicBlock(".bb_0")
    for inst in instructions:
        bb.insert_after(bb.end, inst)
    func.append(bb)
    return func

def _unpack_mte(d: dict) -> tuple:
    """Extract (baseline, randomize_tag, wrong_tag) TestCases from instrument_stage2 dict."""
    return d[MTEVariant.BASELINE], d[MTEVariant.RANDOMIZE_TAG], d[MTEVariant.WRONG_TAG]


def _fp(slot_id=0, reg="x5", spec_nesting=None) -> MTEFixPoint:
    return MTEFixPoint(slot_id=slot_id, bb=BasicBlock(".bb"),
                       mem_inst=None, reg=reg, slot_insts=[],
                       spec_nesting=spec_nesting)

def _build_prep_tc(fps: List[MTEFixPoint]) -> TestCase:
    """Build a stage-1 TestCase: one BB per FixPoint with a tagged NOP."""
    tc = TestCase(seed=0)
    actor = list(tc.actors.values())[0]
    func = Function(".function_main_0", actor)
    for fp in fps:
        bb = BasicBlock(f".bb_slot_{fp.slot_id}")
        nop = Instruction("nop", is_instrumentation=True, template="NOP")
        nop._mte_slot_id = fp.slot_id
        bb.insert_after(bb.end, nop)
        fp.bb = bb
        func.append(bb)
    tc.functions.append(func)
    return tc

def _slot_inst(tc: TestCase, slot_id: int) -> Optional[Instruction]:
    """Return the single instruction tagged with _mte_slot_id == slot_id."""
    for func in tc.functions:
        for bb in func:
            for inst in bb:
                if getattr(inst, '_mte_slot_id', None) == slot_id:
                    return inst
    return None

def _call_build_mte_slots(mte: _MTE, func: Function):
    fix_points: List[MTEFixPoint] = []
    insertions: List = []
    taint_log: List[str] = []
    slot_counter = mte._build_mte_slots(func, 0, fix_points, insertions, taint_log)
    return slot_counter, fix_points, insertions, taint_log


# ===========================================================================
# 1. TestMteInstructionBuilders
# ===========================================================================

class TestMteInstructionBuilders(unittest.TestCase):

    # ── NOP ───────────────────────────────────────────────────────────────

    def test_nop_name(self):
        self.assertEqual(_MTE_INST._make_mte_nop(0).name, "nop")

    def test_nop_template(self):
        self.assertEqual(_MTE_INST._make_mte_nop(0).template, "NOP")

    def test_nop_is_instrumentation(self):
        self.assertTrue(_MTE_INST._make_mte_nop(0).is_instrumentation)

    # ── IRG ───────────────────────────────────────────────────────────────

    def test_irg_name(self):
        self.assertEqual(_MTE_INST._make_mte_irg("x5", 0).name, "irg")

    def test_irg_template_format(self):
        inst = _MTE_INST._make_mte_irg("x5", 0)
        self.assertIn("x5", inst.template)
        self.assertIn("IRG", inst.template)

    # ── MOVK wrong tag ───────────────────────────────────────────────────

    def test_movk_name(self):
        self.assertEqual(_MTE_INST._make_mte_movk_wrong_tag("x5", 0x1234, 0).name, "movk")

    def test_movk_lsl_48(self):
        inst = _MTE_INST._make_mte_movk_wrong_tag("x5", 0x1234, 0)
        self.assertIn("LSL #48", inst.template)

    def test_movk_immediate_16bit_clamp(self):
        # wrong_upper16 must be truncated to 16 bits
        inst = _MTE_INST._make_mte_movk_wrong_tag("x5", 0x1_0000, 0)
        self.assertIn("#0x0000", inst.template)

    def test_movk_immediate_encoded(self):
        inst = _MTE_INST._make_mte_movk_wrong_tag("x5", 0xABCD, 0)
        self.assertIn("0xabcd", inst.template.lower())

    def test_movk_register_in_template(self):
        inst = _MTE_INST._make_mte_movk_wrong_tag("x3", 0x1234, 0)
        self.assertIn("x3", inst.template)



# ===========================================================================
# 2. TestMteWrongTagComputation
# ===========================================================================

class TestMteWrongTagComputation(unittest.TestCase):
    """Verify wrong_upper16 formula for a range of sandbox_base values."""

    def _wrong_upper16(self, sandbox_base: int) -> int:
        sandbox_upper16 = (sandbox_base >> 48) & 0xFFFF
        arch_tag = (sandbox_base >> 56) & 0xF
        wrong_tag = arch_tag ^ 1
        return (sandbox_upper16 & ~(0xF << 8)) | (wrong_tag << 8)

    def _run_stage2(self, sandbox_base: int, spec_nesting: int) -> Tuple[TestCase, TestCase, TestCase]:
        from src.aarch64.aarch64_generator import MTEVariant
        mte = _MTE()
        fp = _fp(slot_id=0, reg="x5", spec_nesting=spec_nesting)
        prep = _build_prep_tc([fp])
        d = mte.instrument_stage2(prep, [fp], sandbox_base)
        return d[MTEVariant.BASELINE], d[MTEVariant.RANDOMIZE_TAG], d[MTEVariant.WRONG_TAG]

    def test_tc3_spec_slot_has_movk(self):
        _, _, tc3 = self._run_stage2(sandbox_base=0x0600000000000000, spec_nesting=1)
        inst = _slot_inst(tc3, 0)
        self.assertIsNotNone(inst)
        self.assertEqual(inst.name, "movk")

    def test_tc3_arch_slot_has_nop(self):
        _, _, tc3 = self._run_stage2(sandbox_base=0x0600000000000000, spec_nesting=0)
        inst = _slot_inst(tc3, 0)
        self.assertEqual(inst.name, "nop")

    def test_movk_wrong_upper16_in_tc3(self):
        sandbox_base = 0x06AB000000000000
        expected_wu16 = self._wrong_upper16(sandbox_base)
        _, _, tc3 = self._run_stage2(sandbox_base, spec_nesting=1)
        inst = _slot_inst(tc3, 0)
        self.assertIn(f"0x{expected_wu16 & 0xFFFF:04x}", inst.template.lower())


# ===========================================================================
# 3. TestMteFillSlot
# ===========================================================================

class TestMteFillSlot(unittest.TestCase):

    def setUp(self):
        self.mte = _MTE()

    def _prep_with_nop(self, slot_id: int) -> Tuple[TestCase, Dict]:
        fp = _fp(slot_id=slot_id)
        tc = _build_prep_tc([fp])
        slot_map = self.mte._find_slot_insts(tc)
        return tc, slot_map, fp

    def test_fill_slot_replaces_nop(self):
        tc, slot_map, fp = self._prep_with_nop(0)
        irg = self.mte._make_mte_irg("x5", 0)
        self.mte._fill_slot(slot_map, fp, irg)
        inst = _slot_inst(tc, 0)
        self.assertEqual(inst.name, "irg")

    def test_fill_slot_old_instruction_removed(self):
        tc, slot_map, fp = self._prep_with_nop(0)
        old_nop = slot_map[0][0]
        irg = self.mte._make_mte_irg("x5", 0)
        self.mte._fill_slot(slot_map, fp, irg)
        # old_nop must not be reachable in any bb
        all_insts = [i for func in tc.functions for bb in func for i in bb]
        self.assertNotIn(old_nop, all_insts)

    def test_fill_slot_missing_id_raises(self):
        tc, slot_map, fp = self._prep_with_nop(0)
        fp_bad = _fp(slot_id=99)
        with self.assertRaises(AssertionError):
            self.mte._fill_slot(slot_map, fp_bad, self.mte._make_mte_nop(99))



# ===========================================================================
# 4. TestMteInstrumentStage2
# ===========================================================================

class TestMteInstrumentStage2(unittest.TestCase):

    def _run(self, fps, sandbox_base=0x0600000000000000, prep=None):
        from src.aarch64.aarch64_generator import MTEVariant
        mte = _MTE()
        if prep is None:
            prep = _build_prep_tc(fps)
        d = mte.instrument_stage2(prep, fps, sandbox_base)
        return d[MTEVariant.BASELINE], d[MTEVariant.RANDOMIZE_TAG], d[MTEVariant.WRONG_TAG]

    # ── TC1: all placeholders → NOP ───────────────────────────────────────

    def test_tc1_arch_slot_is_nop(self):
        fp = _fp(spec_nesting=0)
        tc1, _, _ = self._run([fp])
        self.assertEqual(_slot_inst(tc1, 0).name, "nop")

    def test_tc1_spec_slot_is_nop(self):
        fp = _fp(spec_nesting=1)
        tc1, _, _ = self._run([fp])
        self.assertEqual(_slot_inst(tc1, 0).name, "nop")

    def test_tc1_spec_nesting_none_is_nop(self):
        fp = _fp(spec_nesting=None)
        tc1, _, _ = self._run([fp])
        self.assertEqual(_slot_inst(tc1, 0).name, "nop")

    # ── TC2: arch→NOP, spec→IRG ──────────────────────────────────────────

    def test_tc2_arch_slot_is_nop(self):
        fp = _fp(spec_nesting=0)
        _, tc2, _ = self._run([fp])
        self.assertEqual(_slot_inst(tc2, 0).name, "nop")

    def test_tc2_spec_slot_is_irg(self):
        fp = _fp(spec_nesting=1)
        _, tc2, _ = self._run([fp])
        self.assertEqual(_slot_inst(tc2, 0).name, "irg")

    def test_tc2_spec_irg_uses_fp_reg(self):
        fp = _fp(reg="x7", spec_nesting=1)
        _, tc2, _ = self._run([fp])
        self.assertIn("x7", _slot_inst(tc2, 0).template)

    def test_tc2_spec_none_is_irg(self):
        """spec_nesting=None (CE never reached this access) → treated as spec → IRG."""
        fp = _fp(spec_nesting=None)
        _, tc2, _ = self._run([fp])
        self.assertEqual(_slot_inst(tc2, 0).name, "irg")

    # ── TC3: arch→NOP, spec→MOVK wrong tag ───────────────────────────────

    def test_tc3_arch_slot_is_nop(self):
        fp = _fp(spec_nesting=0)
        _, _, tc3 = self._run([fp])
        self.assertEqual(_slot_inst(tc3, 0).name, "nop")

    def test_tc3_spec_slot_is_movk(self):
        fp = _fp(spec_nesting=1)
        _, _, tc3 = self._run([fp])
        self.assertEqual(_slot_inst(tc3, 0).name, "movk")

    def test_tc3_spec_movk_has_lsl48(self):
        fp = _fp(spec_nesting=1)
        _, _, tc3 = self._run([fp])
        self.assertIn("LSL #48", _slot_inst(tc3, 0).template)

    def test_tc3_spec_none_is_movk(self):
        """spec_nesting=None → treated as spec → MOVK wrong tag."""
        fp = _fp(spec_nesting=None)
        _, _, tc3 = self._run([fp])
        self.assertEqual(_slot_inst(tc3, 0).name, "movk")

    # ── multiple slots ────────────────────────────────────────────────────

    def test_multiple_slots_all_filled(self):
        fps = [_fp(slot_id=i, reg=f"x{i+1}", spec_nesting=i) for i in range(4)]
        tc1, tc2, tc3 = self._run(fps)
        for i in range(4):
            with self.subTest(slot=i):
                self.assertIsNotNone(_slot_inst(tc1, i))
                self.assertIsNotNone(_slot_inst(tc2, i))
                self.assertIsNotNone(_slot_inst(tc3, i))

    def test_arch_and_spec_slots_differ_in_tc2(self):
        """At least one slot must be IRG (spec_nesting=1) and one NOP (spec_nesting=0)."""
        fps = [_fp(slot_id=0, spec_nesting=0), _fp(slot_id=1, spec_nesting=1)]
        _, tc2, _ = self._run(fps)
        self.assertEqual(_slot_inst(tc2, 0).name, "nop")
        self.assertEqual(_slot_inst(tc2, 1).name, "irg")


# ===========================================================================
# 5. TestMteTaintTracking
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
        self.assertEqual(fix_points[0].reg, "x3")

    def test_implicit_mem_skipped(self):
        """Instruction with no MEM operand in operands/implicit_operands → skip."""
        inst = Instruction("ldr", has_memory_access=True)  # no MEM operand added
        func = _simple_function(inst)
        _, fix_points, insertions, _ = _call_build_mte_slots(self.mte, func)
        self.assertEqual(len(fix_points), 0)
        self.assertEqual(len(insertions), 0)

    # ── ADDG tag-preserving propagation ──────────────────────────────────

    def test_addg_imm4_zero_propagates_taint(self):
        """ADDG x6, x5, #8, #0 with x5 tainted → x6 becomes tainted."""
        mem1 = _mem_inst("x5")           # taints x5
        addg = _addg_inst("x6", "x5", 8, 0)  # propagates x5's tag to x6
        mem2 = _mem_inst("x6")          # x6 should be tainted → no sandbox
        func = _simple_function(mem1, addg, mem2)
        _, _, insertions, taint_log = _call_build_mte_slots(self.mte, func)
        _, _, sandbox2, _, _ = insertions[1]
        self.assertEqual(len(sandbox2), 0, f"Expected no sandbox for mem2 (taint should propagate). Log:\n" + "\n".join(taint_log))

    def test_addg_imm4_nonzero_clears_taint(self):
        """ADDG x6, x5, #8, #4 (imm4≠0) does NOT propagate tag → x6 untainted."""
        mem1 = _mem_inst("x5")
        addg = _addg_inst("x6", "x5", 8, 4)  # imm4=4 ≠ 0 → no propagation
        mem2 = _mem_inst("x6")
        func = _simple_function(mem1, addg, mem2)
        _, _, insertions, _ = _call_build_mte_slots(self.mte, func)
        _, _, sandbox2, _, _ = insertions[1]
        self.assertGreater(len(sandbox2), 0)  # x6 untainted → gets sandbox

    def test_subg_imm4_zero_propagates_taint(self):
        """SUBG x6, x5, #8, #0 with x5 tainted → x6 becomes tainted."""
        mem1 = _mem_inst("x5")
        subg = _subg_inst("x6", "x5", 8, 0)
        mem2 = _mem_inst("x6")
        func = _simple_function(mem1, subg, mem2)
        _, _, insertions, _ = _call_build_mte_slots(self.mte, func)
        _, _, sandbox2, _, _ = insertions[1]
        self.assertEqual(len(sandbox2), 0)

    def test_addg_propagates_from_untainted_clears_dest(self):
        """ADDG x6, x5, #0, #0 with x5 untainted → x6 also untainted."""
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
# 6. TestMteErrorCases
# ===========================================================================

class TestMteErrorCases(unittest.TestCase):

    def setUp(self):
        self.mte = _MTE()

    def test_fill_slot_missing_slot_id_asserts(self):
        fp = _fp(slot_id=42)
        tc = _build_prep_tc([fp])
        slot_map = self.mte._find_slot_insts(tc)
        fp_bad = _fp(slot_id=99)
        with self.assertRaises(AssertionError) as cm:
            self.mte._fill_slot(slot_map, fp_bad, self.mte._make_mte_nop(99))
        self.assertIn("99", str(cm.exception))

    def test_spec_nesting_none_treated_as_spec(self):
        """spec_nesting=None (CE never reached this access) → treated as spec → IRG/MOVK."""
        fp = _fp(spec_nesting=None)
        tc1, tc2, tc3 = _unpack_mte(self.mte.instrument_stage2(_build_prep_tc([fp]), [fp], 0))
        self.assertEqual(_slot_inst(tc1, 0).name, "nop")   # TC1 always NOP
        self.assertEqual(_slot_inst(tc2, 0).name, "irg")   # spec → IRG
        self.assertEqual(_slot_inst(tc3, 0).name, "movk")  # spec → MOVK wrong tag

    def test_spec_nesting_zero_treated_as_arch(self):
        """spec_nesting=0 → not speculative → NOP in TC2 and TC3."""
        fp = _fp(spec_nesting=0)
        tc1, tc2, tc3 = _unpack_mte(self.mte.instrument_stage2(_build_prep_tc([fp]), [fp], 0))
        self.assertEqual(_slot_inst(tc2, 0).name, "nop")
        self.assertEqual(_slot_inst(tc3, 0).name, "nop")

    def test_spec_nesting_positive_treated_as_spec(self):
        """spec_nesting=1 (or any >0) → speculative → IRG/MOVK in TC2/TC3."""
        fp = _fp(spec_nesting=1)
        _, tc2, tc3 = _unpack_mte(self.mte.instrument_stage2(_build_prep_tc([fp]), [fp], 0))
        self.assertEqual(_slot_inst(tc2, 0).name, "irg")
        self.assertEqual(_slot_inst(tc3, 0).name, "movk")

    def test_reset_clears_spec_nesting(self):
        fp = _fp(spec_nesting=5)
        fp.reset()
        self.assertIsNone(fp.spec_nesting)


# ===========================================================================
# 7. TestMteStage1E2E
# ===========================================================================

class TestMteStage1E2E(unittest.TestCase):

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

    def test_stage1_returns_fix_points(self):
        tc = self._tc_with_mem()
        _, fix_points = self.mte.instrument_stage1(tc)
        self.assertEqual(len(fix_points), 1)

    def test_stage1_nop_placeholder_appears_in_tc(self):
        tc = self._tc_with_mem("x5")
        prep, fix_points = self.mte.instrument_stage1(tc)
        sid = fix_points[0].slot_id
        inst = _slot_inst(prep, sid)
        self.assertIsNotNone(inst)
        self.assertEqual(inst.name, "nop")

    def test_stage1_nop_before_memory_access(self):
        tc = self._tc_with_mem("x5")
        prep, _ = self.mte.instrument_stage1(tc)
        all_insts = [i for func in prep.functions for bb in func for i in bb]
        names = [i.name for i in all_insts]
        ldr_idx = next(i for i, n in enumerate(names) if n == "ldr")
        # NOP placeholder must appear before LDR (either directly or after AND+ADD)
        before = names[:ldr_idx]
        self.assertIn("nop", before)

    def test_stage1_instrumentation_before_nop_placeholder(self):
        """Sandbox instrumentation must appear before the NOP placeholder."""
        tc = self._tc_with_mem("x5")
        prep, fix_points = self.mte.instrument_stage1(tc)
        all_insts = [i for func in prep.functions for bb in func for i in bb]
        nop_idx = next((i for i, inst in enumerate(all_insts)
                        if inst.name == "nop" and inst.is_instrumentation
                        and fix_points and inst in fix_points[0].slot_insts), None)
        if nop_idx is not None:
            before = [inst for inst in all_insts[:nop_idx] if inst.is_instrumentation]
            self.assertGreater(len(before), 0, "No sandbox instrumentation found before NOP placeholder")

    def test_stage1_fix_point_reg_correct(self):
        tc = self._tc_with_mem("x7")
        _, fix_points = self.mte.instrument_stage1(tc)
        self.assertEqual(fix_points[0].reg, "x7")

    def test_stage1_to_stage2_round_trip(self):
        """stage1 output feeds stage2 without error; all slots get filled."""
        tc = self._tc_with_mem("x5")
        prep, fix_points = self.mte.instrument_stage1(tc)
        for fp in fix_points:
            fp.spec_nesting = 0
        tc1, tc2, tc3 = _unpack_mte(self.mte.instrument_stage2(prep, fix_points, 0x0600000000000000))
        self.assertIsInstance(tc1, TestCase)
        self.assertIsInstance(tc2, TestCase)
        self.assertIsInstance(tc3, TestCase)

    def test_stage2_tc1_slot_is_nop(self):
        tc = self._tc_with_mem("x5")
        prep, fix_points = self.mte.instrument_stage1(tc)
        for fp in fix_points:
            fp.spec_nesting = 0
        tc1, _, _ = _unpack_mte(self.mte.instrument_stage2(prep, fix_points, 0))
        inst = _slot_inst(tc1, fix_points[0].slot_id)
        self.assertEqual(inst.name, "nop")

    def test_stage1_multiple_accesses_separate_slots(self):
        tc = TestCase(seed=0)
        actor = list(tc.actors.values())[0]
        func = Function(".function_main_0", actor)
        bb = BasicBlock(".bb_0")
        for reg in ("x3", "x4", "x5"):
            bb.insert_after(bb.end, _mem_inst(reg))
        func.append(bb)
        tc.functions.append(func)

        prep, fix_points = self.mte.instrument_stage1(tc)
        self.assertEqual(len(fix_points), 3)
        slot_ids = {fp.slot_id for fp in fix_points}
        self.assertEqual(len(slot_ids), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
