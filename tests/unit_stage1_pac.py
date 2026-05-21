"""
Stage-1 unit and E2E tests for PACInstrumentation and _SandboxInstrumentationBase.

Groups:
  1. TestSandboxBaseNormReg     — _norm_reg: alias resolution and identity fallback
  2. TestSandboxBaseSandboxInsts — _make_sandbox_insts: AND + ADD templates
  3. TestSandboxBaseOffsetSubs  — _make_offset_sub_insts: pimm splitting / reg forwarding
  4. TestSandboxBaseGetMemBase  — _get_mem_base_reg: MEM operand extraction
  5. TestSandboxBaseDestRegs    — _dest_regs: dest registers + writeback detection
  6. TestTopoSort               — _topo_sort: topological ordering of CFGs
  7. TestBuildXpacSlotsTaint    — _build_xpac_slots: standalone vs XPAC slot decisions
  8. TestStage1E2E              — instrument_stage1 → instrument_stage2 structural E2E
"""
import unittest
from typing import Dict, List, Optional

from src.interfaces import (
    TestCase, Function, BasicBlock, Instruction, Actor, ActorMode, ActorPL,
    OT, RegisterOperand, MemoryOperand, ImmediateOperand,
)
from src.aarch64.aarch64_generator import (
    PACInstrumentation, FixPoint, SignedRegInfo,
    _SandboxInstrumentationBase,
    SLOT_SIZE, AUTH_SLOT_POS, CTX_SLOT_START, PTR_SLOT_START,
    FIX_COUNT_CTX, FIX_COUNT_PTR,
)


# ===========================================================================
# Shared test infrastructure
# ===========================================================================

_ACTOR = Actor(ActorMode.HOST, ActorPL.KERNEL, 0, "main")

# Normalization map used in all taint tests: xN maps to itself; also covers
# a few aliases (w0→x0) so writeback tests can verify alias resolution.
_NORM: Dict[str, str] = {f"x{i}": f"x{i}" for i in range(31)}
_NORM["sp"] = "sp"
# alias: w-registers → same physical x-register
for i in range(31):
    _NORM[f"w{i}"] = f"x{i}"


class _MinBase(_SandboxInstrumentationBase):
    """Minimal concrete subclass — only sets the three required attributes."""
    def __init__(self, norm=None, mask="#0x3fffff", base="x29"):
        self._norm = norm if norm is not None else dict(_NORM)
        self._sandbox_mask = mask
        self._sandbox_base_reg = base


class _StagePACI(PACInstrumentation):
    """PACInstrumentation with ISA bypassed for stage-1 unit tests.

    _sign_prob=0 / _auth_prob=0 means instrument_stage1 inserts no PAC/AUT
    and every memory access gets standalone sandbox (AND+ADD).
    _make_xpac_inst and _make_auth_inst are overridden to avoid needing specs.
    """
    def __init__(self, norm=None):
        self._sign_prob  = 0.0
        self._auth_prob  = 0.0
        self._xpac_prob  = 1.0
        self._norm       = norm if norm is not None else dict(_NORM)
        self._sandbox_mask    = "#0x3fffff"
        self._sandbox_base_reg = "x29"
        self.last_taint_log: List[str] = []

    def _make_auth_inst(self, mnemonic, reg, ctx_reg):
        t = f"{mnemonic.upper()} {reg}" + (f", {ctx_reg}" if ctx_reg else "")
        return Instruction(mnemonic, is_instrumentation=True, template=t)

    def _make_xpac_inst(self, mnemonic, reg, slot_id, pos):
        i = Instruction(mnemonic, is_instrumentation=True,
                        template=f"{mnemonic.upper()} {reg}")
        i._pac_slot_id  = slot_id
        i._pac_slot_pos = pos
        return i


def _reg_dest(reg: str) -> RegisterOperand:
    return RegisterOperand(reg, 64, src=False, dest=True)

def _reg_src(reg: str) -> RegisterOperand:
    return RegisterOperand(reg, 64, src=True, dest=False)

def _mem_base(reg: str) -> MemoryOperand:
    return MemoryOperand(reg, 64, src=True, dest=False)

def _imm_named(value: str, name: str) -> ImmediateOperand:
    op = ImmediateOperand(value, 12)
    op.name = name
    return op

def _mem_named(value: str, name: str) -> MemoryOperand:
    """MemoryOperand whose .name is set to `name` (used for pimm/simm encoding)."""
    op = MemoryOperand(value, 0, src=True, dest=False)
    op.name = name
    return op

def _simple_function(*instructions: Instruction) -> Function:
    """One-BB function containing the given instructions in order."""
    func = Function(".function_test", _ACTOR)
    bb = BasicBlock(".bb_0")
    for inst in instructions:
        bb.insert_after(bb.end, inst)
    func.append(bb)
    return func


def _pac_inst(reg: str, ctx_reg: Optional[str] = None, mnemonic: str = "pacia") -> Instruction:
    inst = Instruction(mnemonic, is_instrumentation=True,
                       template=f"{mnemonic.upper()} {reg}")
    inst.operands.append(RegisterOperand(reg, 64, src=True, dest=True))
    if ctx_reg:
        inst.operands.append(RegisterOperand(ctx_reg, 64, src=True, dest=False))
    return inst


def _mem_inst(base_reg: str, name: str = "ldr") -> Instruction:
    """Memory-accessing instruction: ldr x_dest, [base_reg]."""
    inst = Instruction(name, has_memory_access=True)
    inst.operands.append(_reg_dest("x0"))
    inst.operands.append(_mem_base(base_reg))
    return inst


def _call_build_xpac_slots(paci: _StagePACI, func: Function, sign_ins: Dict):
    """Thin helper: call _build_xpac_slots and return all five outputs."""
    fix_points: List[FixPoint] = []
    xpac_ins: List = []
    standalone_ins: List = []
    all_pac_at_auth: Dict = {}
    taint_log: List[str] = []
    slot_counter = paci._build_xpac_slots(
        func, sign_ins, {}, 0,
        fix_points, xpac_ins, standalone_ins, all_pac_at_auth, taint_log)
    return slot_counter, fix_points, xpac_ins, standalone_ins, all_pac_at_auth


# ===========================================================================
# 1. TestSandboxBaseNormReg
# ===========================================================================

class TestSandboxBaseNormReg(unittest.TestCase):

    def setUp(self):
        norm = {"x0": "x0", "w0": "x0", "x1": "x1"}
        self.base = _MinBase(norm=norm)

    def test_known_alias_resolves(self):
        self.assertEqual(self.base._norm_reg("w0"), "x0")

    def test_canonical_name_returns_itself(self):
        self.assertEqual(self.base._norm_reg("x0"), "x0")

    def test_unknown_register_returns_itself(self):
        self.assertEqual(self.base._norm_reg("x99"), "x99")

    def test_empty_norm_always_identity(self):
        base = _MinBase(norm={})
        self.assertEqual(base._norm_reg("x5"), "x5")

    def test_all_xN_registers_resolve_to_themselves(self):
        base = _MinBase()
        for i in range(31):
            with self.subTest(i=i):
                self.assertEqual(base._norm_reg(f"x{i}"), f"x{i}")

    def test_wN_resolves_to_xN(self):
        base = _MinBase()
        for i in range(31):
            with self.subTest(i=i):
                self.assertEqual(base._norm_reg(f"w{i}"), f"x{i}")


# ===========================================================================
# 2. TestSandboxBaseSandboxInsts
# ===========================================================================

class TestSandboxBaseSandboxInsts(unittest.TestCase):

    def setUp(self):
        self.base = _MinBase(mask="#0x3fffff", base="x29")

    def _insts(self, reg="x5"):
        return self.base._make_sandbox_insts(reg)

    def test_returns_two_instructions(self):
        self.assertEqual(len(self._insts()), 2)

    def test_first_is_and(self):
        self.assertEqual(self._insts()[0].name, "and")

    def test_second_is_add(self):
        self.assertEqual(self._insts()[1].name, "add")

    def test_and_uses_mask(self):
        insts = self._insts("x7")
        self.assertIn("#0x3fffff", insts[0].template)

    def test_and_uses_register(self):
        insts = self._insts("x7")
        self.assertIn("x7", insts[0].template)

    def test_add_uses_base_reg(self):
        insts = self._insts("x7")
        self.assertIn("x29", insts[1].template)

    def test_add_uses_register(self):
        insts = self._insts("x7")
        self.assertIn("x7", insts[1].template)

    def test_both_are_instrumentation(self):
        for inst in self._insts():
            self.assertTrue(inst.is_instrumentation)


# ===========================================================================
# 3. TestSandboxBaseOffsetSubs
# ===========================================================================

class TestSandboxBaseOffsetSubs(unittest.TestCase):

    def setUp(self):
        self.base = _MinBase()

    def _mem_no_offset(self) -> Instruction:
        """LDR [x1] — one MemoryOperand (base only), no extra."""
        inst = Instruction("ldr", has_memory_access=True)
        inst.operands.append(_mem_base("x1"))
        return inst

    def _mem_pimm(self, value: int) -> Instruction:
        """LDR [x1, #value] — base MemoryOperand + pimm MemoryOperand."""
        inst = Instruction("ldr", has_memory_access=True)
        inst.operands.append(_mem_base("x1"))
        inst.operands.append(_mem_named(str(value), "pimm"))
        return inst

    def _mem_reg_offset(self, offset_reg: str) -> Instruction:
        """LDR [x1, x2] — base + register offset MemoryOperand."""
        inst = Instruction("ldr", has_memory_access=True)
        inst.operands.append(_mem_base("x1"))
        inst.operands.append(_mem_named(offset_reg, "simm"))
        return inst

    def test_no_offset_returns_empty(self):
        result = self.base._make_offset_sub_insts(self._mem_no_offset(), "x1")
        self.assertEqual(result, [])

    def test_pimm_small_single_sub(self):
        result = self.base._make_offset_sub_insts(self._mem_pimm(8), "x1")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "sub")
        self.assertIn("#8", result[0].template)

    def test_pimm_large_splits_into_chunks(self):
        # 5000 > 4095, so needs two SUBs: 4095 + 905
        result = self.base._make_offset_sub_insts(self._mem_pimm(5000), "x1")
        self.assertEqual(len(result), 2)
        self.assertIn("#4095", result[0].template)
        self.assertIn("#905", result[1].template)

    def test_pimm_zero_returns_empty(self):
        result = self.base._make_offset_sub_insts(self._mem_pimm(0), "x1")
        self.assertEqual(result, [])

    def test_reg_offset_single_sub_with_reg(self):
        result = self.base._make_offset_sub_insts(self._mem_reg_offset("x2"), "x1")
        self.assertEqual(len(result), 1)
        self.assertIn("x2", result[0].template)
        self.assertIn("x1", result[0].template)

    def test_base_reg_appears_in_all_subs(self):
        result = self.base._make_offset_sub_insts(self._mem_pimm(5000), "x3")
        for sub in result:
            self.assertIn("x3", sub.template)

    def test_all_subs_are_instrumentation(self):
        result = self.base._make_offset_sub_insts(self._mem_pimm(8), "x1")
        for inst in result:
            self.assertTrue(inst.is_instrumentation)


# ===========================================================================
# 4. TestSandboxBaseGetMemBase
# ===========================================================================

class TestSandboxBaseGetMemBase(unittest.TestCase):

    def setUp(self):
        self.base = _MinBase()

    def test_returns_mem_operand_value(self):
        inst = Instruction("ldr", has_memory_access=True)
        inst.operands.append(_mem_base("x5"))
        self.assertEqual(self.base._get_mem_base_reg(inst), "x5")

    def test_no_mem_operand_returns_none(self):
        inst = Instruction("add")
        inst.operands.append(_reg_dest("x0"))
        inst.operands.append(_reg_src("x1"))
        self.assertIsNone(self.base._get_mem_base_reg(inst))

    def test_mem_in_implicit_operands(self):
        inst = Instruction("str", has_memory_access=True)
        inst.implicit_operands.append(_mem_base("x7"))
        self.assertEqual(self.base._get_mem_base_reg(inst), "x7")

    def test_first_mem_operand_returned_when_multiple(self):
        inst = Instruction("ldp", has_memory_access=True)
        inst.operands.append(_mem_base("x1"))
        inst.operands.append(_mem_named("8", "pimm"))
        # _get_mem_base_reg returns the first OT.MEM value found
        self.assertEqual(self.base._get_mem_base_reg(inst), "x1")


# ===========================================================================
# 5. TestSandboxBaseDestRegs
# ===========================================================================

class TestSandboxBaseDestRegs(unittest.TestCase):

    def setUp(self):
        self.base = _MinBase()

    def test_single_dest_register(self):
        inst = Instruction("mov")
        inst.operands.append(_reg_dest("x3"))
        self.assertEqual(self.base._dest_regs(inst), frozenset({"x3"}))

    def test_no_dest_returns_empty(self):
        inst = Instruction("cmp")
        inst.operands.append(_reg_src("x0"))
        inst.operands.append(_reg_src("x1"))
        self.assertEqual(self.base._dest_regs(inst), frozenset())

    def test_register_not_in_norm_excluded(self):
        base = _MinBase(norm={"x0": "x0"})  # only x0 tracked
        inst = Instruction("mov")
        inst.operands.append(_reg_dest("x5"))  # x5 not in norm
        self.assertEqual(base._dest_regs(inst), frozenset())

    def test_alias_normalized(self):
        inst = Instruction("mov")
        inst.operands.append(_reg_dest("w3"))  # w3 → x3 via norm
        self.assertIn("x3", self.base._dest_regs(inst))

    def test_simm_writeback_adds_base(self):
        """Pre/post-index: base register is also written (writeback)."""
        inst = Instruction("ldr", has_memory_access=True)
        inst.operands.append(_reg_dest("x0"))        # load dest
        inst.operands.append(_mem_base("x1"))         # base
        op_simm = ImmediateOperand("8", 9)
        op_simm.name = "simm"
        inst.operands.append(op_simm)
        result = self.base._dest_regs(inst)
        self.assertIn("x1", result)

    def test_no_simm_no_writeback(self):
        """Unsigned-offset addressing (pimm) has no writeback."""
        inst = Instruction("ldr", has_memory_access=True)
        inst.operands.append(_reg_dest("x0"))
        inst.operands.append(_mem_base("x1"))
        op_pimm = ImmediateOperand("8", 12)
        op_pimm.name = "pimm"
        inst.operands.append(op_pimm)
        result = self.base._dest_regs(inst)
        # x0 is written (load dest), x1 is not (no writeback on pimm)
        self.assertIn("x0", result)
        self.assertNotIn("x1", result)

    def test_implicit_dest_register(self):
        inst = Instruction("pop")
        inst.implicit_operands.append(_reg_dest("x0"))
        self.assertIn("x0", self.base._dest_regs(inst))


# ===========================================================================
# 6. TestTopoSort
# ===========================================================================

class TestTopoSort(unittest.TestCase):

    def _sort(self, func: Function):
        return _MinBase._topo_sort(func)

    def _func_with_bbs(self, *bbs: BasicBlock) -> Function:
        func = Function(".function_test", _ACTOR)
        for bb in bbs:
            func.append(bb)
        return func

    def test_single_block(self):
        bb = BasicBlock(".bb_0")
        func = self._func_with_bbs(bb)
        preds, topo = self._sort(func)
        self.assertEqual(topo, [bb])
        self.assertEqual(preds[bb], [])

    def test_linear_chain(self):
        a, b, c = BasicBlock(".a"), BasicBlock(".b"), BasicBlock(".c")
        a.successors = [b]
        b.successors = [c]
        func = self._func_with_bbs(a, b, c)
        preds, topo = self._sort(func)
        self.assertEqual(topo, [a, b, c])
        self.assertEqual(preds[a], [])
        self.assertEqual(preds[b], [a])
        self.assertEqual(preds[c], [b])

    def test_diamond_join(self):
        a, b, c, d = BasicBlock(".a"), BasicBlock(".b"), BasicBlock(".c"), BasicBlock(".d")
        a.successors = [b, c]
        b.successors = [d]
        c.successors = [d]
        func = self._func_with_bbs(a, b, c, d)
        preds, topo = self._sort(func)
        # a must come first, d must come last
        self.assertEqual(topo[0], a)
        self.assertEqual(topo[-1], d)
        # b and c before d
        self.assertLess(topo.index(b), topo.index(d))
        self.assertLess(topo.index(c), topo.index(d))
        # d has both b and c as predecessors
        self.assertIn(b, preds[d])
        self.assertIn(c, preds[d])

    def test_predecessor_map_correct(self):
        a, b, c = BasicBlock(".a"), BasicBlock(".b"), BasicBlock(".c")
        a.successors = [b, c]
        func = self._func_with_bbs(a, b, c)
        preds, _ = self._sort(func)
        self.assertIn(a, preds[b])
        self.assertIn(a, preds[c])

    def test_all_bbs_in_topo(self):
        a, b, c = BasicBlock(".a"), BasicBlock(".b"), BasicBlock(".c")
        a.successors = [b]
        b.successors = [c]
        func = self._func_with_bbs(a, b, c)
        _, topo = self._sort(func)
        self.assertSetEqual(set(topo), {a, b, c})


# ===========================================================================
# 7. TestBuildXpacSlotsTaint
# ===========================================================================

class TestBuildXpacSlotsTaint(unittest.TestCase):
    """Tests for _build_xpac_slots taint tracking without a real ISA.

    sign_ins maps an anchor instruction to (pac_inst, bb), meaning
    a PACIA for pac_inst's register is notionally inserted before anchor.
    auth_ins is always empty (we don't test auth slots here).
    """

    def setUp(self):
        self.paci = _StagePACI()

    # ── untainted base → standalone ───────────────────────────────────────

    def test_untainted_base_yields_standalone(self):
        mem = _mem_inst("x5")
        func = _simple_function(mem)
        _, fix_points, xpac_ins, standalone_ins, _ = \
            _call_build_xpac_slots(self.paci, func, {})
        self.assertEqual(len(fix_points), 0)
        self.assertEqual(len(xpac_ins), 0)
        self.assertEqual(len(standalone_ins), 1)

    def test_standalone_prepends_sandbox_insts(self):
        """Standalone insertion includes AND+ADD sandbox before the memory access."""
        mem = _mem_inst("x5")
        func = _simple_function(mem)
        _, _, _, standalone_ins, _ = _call_build_xpac_slots(self.paci, func, {})
        inst, bb, insts = standalone_ins[0]
        names = [i.name for i in insts]
        self.assertIn("and", names)
        self.assertIn("add", names)

    # ── tainted base → XPAC slot ──────────────────────────────────────────

    def test_tainted_base_yields_xpac_slot(self):
        anchor = Instruction("nop")   # sign_ins key — PACIA notionally before this
        mem    = _mem_inst("x3")
        func   = _simple_function(anchor, mem)
        sign_ins = {anchor: (_pac_inst("x3", "x4"), func[0])}
        _, fix_points, xpac_ins, standalone_ins, _ = \
            _call_build_xpac_slots(self.paci, func, sign_ins)
        self.assertEqual(len(fix_points), 1)
        self.assertEqual(len(xpac_ins), 1)
        self.assertEqual(len(standalone_ins), 0)

    def test_xpac_slot_has_correct_reg(self):
        anchor = Instruction("nop")
        mem    = _mem_inst("x3")
        func   = _simple_function(anchor, mem)
        sign_ins = {anchor: (_pac_inst("x3", "x4"), func[0])}
        _, fix_points, _, _, _ = _call_build_xpac_slots(self.paci, func, sign_ins)
        fp = fix_points[0]
        self.assertEqual(fp.info.reg, "x3")

    def test_xpac_slot_instructions_tagged(self):
        anchor = Instruction("nop")
        mem    = _mem_inst("x3")
        func   = _simple_function(anchor, mem)
        sign_ins = {anchor: (_pac_inst("x3", "x4"), func[0])}
        _, fix_points, xpac_ins, _, _ = _call_build_xpac_slots(self.paci, func, sign_ins)
        _, _, slot_insts, _ = xpac_ins[0]
        sid = fix_points[0].slot_id
        for inst in slot_insts:
            self.assertEqual(inst._pac_slot_id, sid)

    # ── write clears taint → subsequent access is standalone ─────────────

    def test_write_clears_taint(self):
        """PACIA taints x3; then a write to x3 clears it; memory access → standalone."""
        anchor  = Instruction("nop")
        # MOV x3, x0 — writes to x3, clears taint
        write   = Instruction("mov")
        write.operands.append(_reg_dest("x3"))
        mem     = _mem_inst("x3")
        func    = _simple_function(anchor, write, mem)
        sign_ins = {anchor: (_pac_inst("x3", "x4"), func[0])}
        _, fix_points, xpac_ins, standalone_ins, _ = \
            _call_build_xpac_slots(self.paci, func, sign_ins)
        # taint was cleared before the memory access
        self.assertEqual(len(fix_points), 0)
        self.assertEqual(len(standalone_ins), 1)

    def test_second_memory_access_after_xpac_is_standalone(self):
        """After the XPAC slot consumes the taint, a second access on same reg is standalone."""
        anchor = Instruction("nop")
        mem1   = _mem_inst("x3")
        mem2   = _mem_inst("x3")
        func   = _simple_function(anchor, mem1, mem2)
        sign_ins = {anchor: (_pac_inst("x3", "x4"), func[0])}
        _, fix_points, xpac_ins, standalone_ins, _ = \
            _call_build_xpac_slots(self.paci, func, sign_ins)
        # First access: XPAC slot (tainted). Second access: standalone (taint consumed).
        self.assertEqual(len(fix_points), 1)
        self.assertEqual(len(xpac_ins), 1)
        self.assertEqual(len(standalone_ins), 1)

    # ── CFG join: intersection ─────────────────────────────────────────────

    def test_join_node_intersection_both_tainted(self):
        """If x3 is tainted on BOTH incoming paths, the join node sees it as tainted."""
        actor = _ACTOR
        func  = Function(".function_test", actor)
        head  = BasicBlock(".head")
        left  = BasicBlock(".left")
        right = BasicBlock(".right")
        tail  = BasicBlock(".tail")

        # head → left → tail and head → right → tail (diamond)
        head.successors  = [left, right]
        left.successors  = [tail]
        right.successors = [tail]
        func.extend([head, left, right, tail])

        # PACIA for x3 in head (seen on both paths)
        anchor = Instruction("nop_anchor")
        head.insert_after(head.end, anchor)

        # memory access in tail
        mem = _mem_inst("x3")
        tail.insert_after(tail.end, mem)

        sign_ins = {anchor: (_pac_inst("x3", "x4"), head)}
        _, fix_points, xpac_ins, standalone_ins, _ = \
            _call_build_xpac_slots(self.paci, func, sign_ins)

        # Taint flows through both branches → join sees x3 tainted → XPAC slot
        self.assertEqual(len(fix_points), 1)
        self.assertEqual(len(xpac_ins), 1)
        self.assertEqual(len(standalone_ins), 0)


# ===========================================================================
# 8. TestStage1E2E — instrument_stage1 → instrument_stage2 round-trip
# ===========================================================================

class TestStage1E2E(unittest.TestCase):
    """End-to-end: create a minimal TC, run both stages, verify structural invariants.

    With _sign_prob=0 / _auth_prob=0 there are no FixPoints, so we test the
    pure-standalone path through stage1 and the trivial (no-slot) stage2.
    The critical structural invariants verified:
      - stage1 returns a valid TC (no exception)
      - every memory-access instruction has an AND+ADD sandbox prefix
      - stage2 returns three independent TCs from a zero-fix-point stage1 result
    """

    def setUp(self):
        self.paci = _StagePACI()

    def _tc_with_mem(self, base_reg: str = "x5") -> TestCase:
        tc = TestCase(seed=0)
        actor = list(tc.actors.values())[0]
        func = Function(".function_main_0", actor)
        bb = BasicBlock(".bb_0")
        mem = _mem_inst(base_reg, name="ldr")
        bb.insert_after(bb.end, mem)
        func.append(bb)
        tc.functions.append(func)
        return tc

    def test_stage1_returns_without_error(self):
        tc = self._tc_with_mem()
        prep, fix_points = self.paci.instrument_stage1(tc)
        self.assertIsInstance(prep, TestCase)

    def test_stage1_no_fix_points_when_no_signing(self):
        tc = self._tc_with_mem()
        _, fix_points = self.paci.instrument_stage1(tc)
        self.assertEqual(fix_points, [])

    def test_stage1_inserts_and_add_before_memory_access(self):
        """AND+ADD sandbox must appear before the LDR in the output TC."""
        tc = self._tc_with_mem("x5")
        prep, _ = self.paci.instrument_stage1(tc)
        all_insts = [i for func in prep.functions
                       for bb in func
                       for i in bb]
        names = [i.name for i in all_insts]
        ldr_idx = next(i for i, n in enumerate(names) if n == "ldr")
        # At least an AND and ADD must appear before the LDR
        before = names[:ldr_idx]
        self.assertIn("and", before)
        self.assertIn("add", before)

    def test_stage1_sandbox_targets_correct_register(self):
        """The AND+ADD templates must reference the LDR base register."""
        base_reg = "x7"
        tc = self._tc_with_mem(base_reg)
        prep, _ = self.paci.instrument_stage1(tc)
        sandbox_insts = [i for func in prep.functions
                           for bb in func
                           for i in bb
                           if i.name in ("and", "add") and i.is_instrumentation]
        for inst in sandbox_insts:
            self.assertIn(base_reg, inst.template,
                          f"Expected {base_reg!r} in sandbox inst: {inst.template!r}")

    def test_stage2_returns_three_tcs_from_empty_fix_points(self):
        tc = self._tc_with_mem()
        prep, fix_points = self.paci.instrument_stage1(tc)
        tc1, tc2, tc3 = self.paci.instrument_stage2(prep, fix_points)
        self.assertIsInstance(tc1, TestCase)
        self.assertIsInstance(tc2, TestCase)
        self.assertIsInstance(tc3, TestCase)

    def test_stage2_three_tcs_are_independent_objects(self):
        tc = self._tc_with_mem()
        prep, fix_points = self.paci.instrument_stage1(tc)
        tc1, tc2, tc3 = self.paci.instrument_stage2(prep, fix_points)
        self.assertIsNot(tc1, tc2)
        self.assertIsNot(tc2, tc3)

    def test_stage1_multiple_memory_accesses_each_get_sandbox(self):
        """Every memory access — regardless of register — gets AND+ADD."""
        tc = TestCase(seed=0)
        actor = list(tc.actors.values())[0]
        func = Function(".function_main_0", actor)
        bb = BasicBlock(".bb_0")
        for reg in ("x5", "x6", "x7"):
            mem = _mem_inst(reg)
            bb.insert_after(bb.end, mem)
        func.append(bb)
        tc.functions.append(func)

        prep, fix_points = self.paci.instrument_stage1(tc)
        self.assertEqual(fix_points, [])
        # Count AND instructions — expect one per memory access
        and_count = sum(1 for func in prep.functions
                          for bb in func
                          for i in bb
                          if i.name == "and" and i.is_instrumentation)
        self.assertEqual(and_count, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
