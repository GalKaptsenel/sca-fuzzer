"""Generation of multiple functions with direct calls and returns (Phase 1a).

Invariants checked:
  * the call graph is acyclic — a function only ever calls a HIGHER-indexed function (forward-only
    layout), so there is no direct or indirect recursion / loop;
  * index 0 is the inline entry: it exits the test case (never returns) and gets no stack frame;
  * every other function is a callee that ends in a return;
  * a non-leaf callee (one that makes calls) gets a prologue that spills the link register and an
    epilogue that restores it before the return; a leaf callee gets neither;
  * the default configuration (one function) reproduces the original single-function behaviour;
  * generated test cases still assemble.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # run from any cwd


class FunctionCallsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from src.config import CONF
        CONF.load("config.yml")
        from src.isa_loader import InstructionSet
        from src.factory import get_program_generator
        cls.CONF = CONF
        cls.isa = InstructionSet("base.json", CONF.instruction_categories)
        cls.get_gen = staticmethod(get_program_generator)

    # CONF is a process-wide Borg singleton; _gen mutates several generation options in place. Snapshot
    # them before each test and restore after, so this suite leaks no state into any other (or itself).
    _MUTATED = ("min_functions_per_test_case", "max_functions_per_test_case", "function_call_probability",
                "program_size", "max_calls_per_function", "function_size_shrink",
                "indirect_call_probability", "dispatch_call_probability")

    def setUp(self):
        self._saved = {k: getattr(self.CONF, k) for k in self._MUTATED}

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(self.CONF, k, v)

    def _gen(self, n, min_f, max_f, call_prob=0.5, program_size=8, assemble=False,
             max_calls=1000, shrink=1.0, indirect=0.0, dispatch=0.0):
        C = self.CONF
        C.min_functions_per_test_case = min_f
        C.max_functions_per_test_case = max_f
        C.function_call_probability = call_prob
        C.program_size = program_size
        C.max_calls_per_function = max_calls
        C.function_size_shrink = shrink
        C.indirect_call_probability = indirect
        C.dispatch_call_probability = dispatch
        gen = self.get_gen(self.isa, 1)
        out = []
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.asm")
            for s in range(n):
                gen._state = s
                out.append(gen.create_test_case(path, disable_assembler=not assemble))
        return out

    @staticmethod
    def _func_index(name):
        return int(name.rsplit(".", 1)[1].split("_")[-1]) if "function_" in name else -1

    def _calls(self, func):
        return [i for bb in func for i in list(bb) + bb.terminators if i.is_call]

    # ---- forward-only call graph (no loops) -----------------------------------------------
    def test_calls_are_forward_only(self):
        for tc in self._gen(30, 3, 5, call_prob=0.6):
            index = {f.name: i for i, f in enumerate(tc.functions)}
            for i, f in enumerate(tc.functions):
                for call in self._calls(f):
                    target = call.operands[0].value
                    self.assertIn(target, index, f"call to unknown function {target}")
                    self.assertGreater(index[target], i,
                                       f"{f.name} calls {target}: not forward-only (loop risk)")

    def test_no_self_calls(self):
        for tc in self._gen(20, 2, 4, call_prob=0.8):
            for f in tc.functions:
                for call in self._calls(f):
                    self.assertNotEqual(call.operands[0].value, f.name, "self-recursive call")

    # ---- entry vs callee terminators ------------------------------------------------------
    def test_entry_exits_callees_return(self):
        for tc in self._gen(20, 2, 4):
            entry = tc.functions[0]
            self.assertNotEqual(entry.exit.terminators[0].name, "ret",
                                "entry function must not return")
            for callee in tc.functions[1:]:
                self.assertEqual(callee.exit.terminators[0].name, "ret",
                                 f"{callee.name} (callee) must end in ret")

    # ---- prologue / epilogue for non-leaf callees -----------------------------------------
    @staticmethod
    def _has_prologue(func):
        return any(i.name == "str" and i.is_instrumentation for bb in func for i in bb)

    @staticmethod
    def _has_epilogue(func):
        return any(i.name == "ldr" and i.is_instrumentation for i in func.exit)

    def test_frames_only_for_nonleaf_callees(self):
        seen_nonleaf_callee = False
        seen_leaf_callee = False
        for tc in self._gen(40, 3, 5, call_prob=0.5):
            entry = tc.functions[0]
            # the entry never returns -> no frame even when it makes calls
            self.assertFalse(self._has_prologue(entry), "entry must not get a frame")
            for callee in tc.functions[1:]:
                if callee.is_leaf:
                    seen_leaf_callee = True
                    self.assertFalse(self._has_prologue(callee), f"{callee.name} leaf: no prologue")
                    self.assertFalse(self._has_epilogue(callee), f"{callee.name} leaf: no epilogue")
                else:
                    seen_nonleaf_callee = True
                    self.assertTrue(self._has_prologue(callee), f"{callee.name} non-leaf: prologue")
                    self.assertTrue(self._has_epilogue(callee), f"{callee.name} non-leaf: epilogue")
        self.assertTrue(seen_nonleaf_callee, "no non-leaf callee exercised")
        self.assertTrue(seen_leaf_callee, "no leaf callee exercised")

    # ---- default (single function) is unchanged -------------------------------------------
    def test_default_single_function(self):
        for tc in self._gen(10, 1, 1):
            self.assertEqual(len(tc.functions), 1)
            self.assertFalse(self._calls(tc.functions[0]), "single function must make no calls")
            self.assertFalse(self._has_prologue(tc.functions[0]), "single function must have no frame")
            self.assertNotEqual(tc.functions[0].exit.terminators[0].name, "ret")

    # ---- call fan-out is capped -----------------------------------------------------------
    def test_max_calls_per_function(self):
        cap = 2
        for tc in self._gen(30, 3, 5, call_prob=0.9, program_size=20, max_calls=cap):
            for f in tc.functions:
                self.assertLessEqual(len(self._calls(f)), cap,
                                     f"{f.name} emitted more than {cap} calls")

    # ---- successive functions shrink in length --------------------------------------------
    def test_function_size_shrink(self):
        # With shrink=0.5, each function should be no longer than the previous (monotone non-increasing
        # instruction count by index); a strict drop appears once sizes exceed the 1-instruction floor.
        saw_drop = False
        for tc in self._gen(20, 4, 4, call_prob=0.0, program_size=40, shrink=0.5):
            sizes = [sum(1 for bb in f for _ in bb) for f in tc.functions]
            for a, b in zip(sizes, sizes[1:]):
                self.assertLessEqual(b, a, f"function sizes not non-increasing: {sizes}")
            if sizes[0] > sizes[-1]:
                saw_drop = True
        self.assertTrue(saw_drop, "shrink produced no size reduction across functions")

    # ---- still assembles ------------------------------------------------------------------
    def test_multi_function_assembles(self):
        # create_test_case with disable_assembler=False raises if the assembler rejects the output
        self._gen(5, 3, 3, call_prob=0.4, assemble=True)

    # ---- single-target indirect calls (BLR materialized by ADR) ---------------------------
    def test_indirect_calls_are_blr_materialized_by_adr(self):
        # indirect_call_probability=1.0: every call is a BLR whose target register is loaded by an
        # immediately-preceding ADR to the same (forward) function label.
        from src.interfaces import OT
        saw = 0
        for tc in self._gen(30, 3, 5, call_prob=0.8, program_size=20, indirect=1.0):
            index = {f.name: i for i, f in enumerate(tc.functions)}
            for fi, f in enumerate(tc.functions):
                for bb in f:
                    insts = list(bb) + bb.terminators
                    for j, ins in enumerate(insts):
                        if not ins.is_call:
                            continue
                        self.assertEqual(ins.name, "blr", "indirect call must be BLR")
                        target = ins.operands[0].value
                        self.assertGreater(index[target], fi, "indirect call not forward-only")
                        reg = next(o.value for o in ins.operands if o.type == OT.REG)
                        from src.aarch64.aarch64_target_desc import INDIRECT_CALL_TARGET_REGISTER
                        self.assertEqual(reg, INDIRECT_CALL_TARGET_REGISTER,  # never a data register
                                         f"indirect target must be the dedicated reg, got {reg}")
                        adr = insts[j - 1]
                        self.assertEqual(adr.name, "adr", "BLR must be preceded by an ADR")
                        self.assertEqual(adr.operands[0].value, reg, "ADR must load the BLR's register")
                        self.assertEqual(adr.operands[-1].value, target, "ADR must target the callee")
                        saw += 1
        self.assertGreater(saw, 0, "no indirect calls exercised")

    def test_indirect_calls_assemble(self):
        self._gen(5, 3, 3, call_prob=0.6, program_size=16, indirect=1.0, assemble=True)

    # ---- multi-target dispatch calls (BLR target loaded from a per-function jump table) --------
    @staticmethod
    def _next_pow2(m):
        size = 1
        while size < m:
            size <<= 1
        return size

    def test_dispatch_calls_use_table_sequence(self):
        # indirect=dispatch=1.0: in a function with >= 2 forward callees every BLR is a dispatch, so its
        # target x28 is materialized by AND/ADR/LDRSW/ADD off the function's own local (.L) jump table.
        from src.interfaces import OT
        from src.aarch64.aarch64_target_desc import INDIRECT_CALL_TARGET_REGISTER as XD
        saw = 0
        for tc in self._gen(30, 4, 5, call_prob=0.8, program_size=16, indirect=1.0, dispatch=1.0):
            for fi, f in enumerate(tc.functions):
                if len(tc.functions[fi + 1:]) < 2:
                    continue   # too few callees to dispatch -> single-target (checked separately)
                for bb in f:
                    insts = list(bb) + bb.terminators
                    for j, ins in enumerate(insts):
                        if ins.name != "blr":
                            continue
                        self.assertIsNotNone(f.dispatch_table, f"{f.name} must own a dispatch table")
                        seq = [insts[j - 4].name, insts[j - 3].name, insts[j - 2].name, insts[j - 1].name]
                        self.assertEqual(seq, ["and", "adr", "ldrsw", "add"],
                                         f"{f.name}: dispatch BLR must be preceded by AND/ADR/LDRSW/ADD")
                        adr = insts[j - 3]
                        self.assertTrue(f.dispatch_table.label.startswith(".L"),
                                        "dispatch table must be a local (.L) label")
                        self.assertEqual(adr.operands[-1].value, f.dispatch_table.label,
                                         "ADR must load the function's dispatch table base")
                        self.assertEqual(adr.operands[0].value, XD, "table base loaded into the target reg")
                        blr_reg = next(o.value for o in ins.operands if o.type == OT.REG)
                        self.assertEqual(blr_reg, XD, "only the dedicated register holds the target")
                        saw += 1
        self.assertGreater(saw, 0, "no dispatch calls exercised")

    def test_dispatch_table_is_forward_and_pow2(self):
        for tc in self._gen(30, 4, 5, call_prob=0.8, program_size=16, indirect=1.0, dispatch=1.0):
            index = {f.name: i for i, f in enumerate(tc.functions)}
            for fi, f in enumerate(tc.functions):
                table = f.dispatch_table
                if table is None:
                    continue
                self.assertGreaterEqual(table.size, 2, "dispatch table must have >= 2 entries")
                self.assertEqual(table.size & (table.size - 1), 0, "table size must be a power of two")
                forward = tc.functions[fi + 1:]
                self.assertEqual(table.size, self._next_pow2(len(forward)))
                for k, callee in enumerate(table.entries):
                    self.assertGreater(index[callee.name], fi, "table entry must be a forward callee")
                    self.assertIs(callee, forward[k % len(forward)], "entries cycle the forward callees")

    def test_dispatch_falls_back_when_too_few_callees(self):
        # A function with a single forward callee cannot dispatch -> single-target ADR even at prob 1.0.
        for tc in self._gen(30, 3, 4, call_prob=0.9, program_size=16, indirect=1.0, dispatch=1.0):
            for fi, f in enumerate(tc.functions):
                if len(tc.functions[fi + 1:]) >= 2:
                    continue
                self.assertIsNone(f.dispatch_table, f"{f.name} has < 2 callees: no dispatch table")
                for bb in f:
                    insts = list(bb) + bb.terminators
                    for j, ins in enumerate(insts):
                        if ins.name == "blr":
                            self.assertEqual(insts[j - 1].name, "adr",
                                             "single-target BLR must be preceded by a lone ADR")

    def test_dispatch_calls_assemble(self):
        self._gen(6, 4, 4, call_prob=0.7, program_size=16, indirect=1.0, dispatch=1.0, assemble=True)


if __name__ == "__main__":
    unittest.main()
