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

    def _gen(self, n, min_f, max_f, call_prob=0.5, program_size=8, assemble=False):
        C = self.CONF
        C.__setattr__("min_functions_per_test_case", min_f)
        C.__setattr__("max_functions_per_test_case", max_f)
        C.__setattr__("function_call_probability", call_prob)
        C.__setattr__("program_size", program_size)
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

    # ---- still assembles ------------------------------------------------------------------
    def test_multi_function_assembles(self):
        # create_test_case with disable_assembler=False raises if the assembler rejects the output
        self._gen(5, 3, 3, call_prob=0.4, assemble=True)


if __name__ == "__main__":
    unittest.main()
