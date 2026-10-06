"""Object-based Python template builder (src/aarch64/template).

Invariants checked:
  * a template builds a TestCase with the fixed instructions it emitted, in order, marked from-template;
  * a Hole expands into exactly `n` generated instructions, gone as a marker afterwards;
  * a `kind`-constrained hole yields only that instruction class (LOAD / ALU);
  * a `regs`-constrained hole uses only whitelisted data registers;
  * the whole thing assembles through the generator's normal printer/assembler path;
  * an unconstrained hole coexists with the legacy behaviour (count-only fill).
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # run from any cwd
from tests.conf_isolation import setUpModule, tearDownModule  # noqa: F401  (restores CONF + cwd)


class TemplateBuilderTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from src.config import CONF
        CONF.load("config.yml")
        from src.isa_loader import InstructionSet
        cls.CONF = CONF
        cls.isa = InstructionSet("base.json", CONF.instruction_categories)

    def setUp(self):
        # CONF is a process-wide singleton; reset it before each test so tests stay independent
        # regardless of order or of other test files that mutate it.
        self.CONF.load("config.yml")
        # templates create callees and calls explicitly; give the limits room (else they raise)
        self.CONF.max_functions_per_test_case = 16
        self.CONF.max_calls_per_function = 8

    def _run(self, template, seed=1, assemble=False):
        from src.aarch64.template.runner import generate_test_case
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.asm")
            return generate_test_case(template, self.isa, seed, asm_file=path, assemble=assemble)

    @staticmethod
    def _body(tc):
        # the template's own instructions, excluding the harness's .measurement_start macro
        return [i for i in tc.functions[0].get_first_bb() if i.name != "macro"]

    # ---- holes -----------------------------------------------------------------------------------
    def test_hole_expands_to_n_instructions(self):
        from src.aarch64.template import Template, Hole

        class T(Template):
            def build(self, b):
                b.hole(Hole(n=5))

        body = self._body(self._run(T()))
        self.assertEqual(len(body), 5)
        self.assertTrue(all(getattr(i, "template_hole", None) is None for i in body))

    def test_load_hole_only_loads(self):
        from src.aarch64.template import Template, Hole, Kind

        class T(Template):
            def build(self, b):
                b.hole(Hole(n=4, kind=Kind.LOAD))

        body = self._body(self._run(T()))
        self.assertEqual(len(body), 4)
        self.assertTrue(all(i.has_memory_access and not i.name.lower().startswith("st") for i in body))

    def test_alu_hole_has_no_memory_or_branch(self):
        from src.aarch64.template import Template, Hole, Kind

        class T(Template):
            def build(self, b):
                b.hole(Hole(n=6, kind=Kind.ALU))

        body = self._body(self._run(T()))
        self.assertEqual(len(body), 6)
        self.assertTrue(all(not i.has_memory_access and not i.control_flow for i in body))

    def test_reg_whitelist(self):
        from src.aarch64.template import Template, Hole, Kind
        from src.aarch64.template.objects import X0, X1, X2

        allowed = {"x0", "x1", "x2", "w0", "w1", "w2"}

        class T(Template):
            def build(self, b):
                b.hole(Hole(n=8, kind=Kind.ALU, regs=[X0, X1, X2]))

        from src.interfaces import RegisterOperand
        for inst in self._body(self._run(T())):
            for op in inst.operands:
                if isinstance(op, RegisterOperand):
                    self.assertIn(op.value, allowed, f"{inst.name} used {op.value}")

    # ---- fixed ops + ordering --------------------------------------------------------------------
    def test_fixed_op_sets_exact_operands_and_order(self):
        from src.aarch64.template import Template, Hole, Kind
        from src.aarch64.template.objects import X0, X1, X2
        from src.interfaces import OT

        # ISA-agnostic: find a 3x 64-bit-register, non-memory, non-branch instruction
        cand = next((s for s in self.isa.instructions
                     if not s.control_flow and not s.has_mem_operand
                     and len(s.operands) == 3
                     and all(o.type == OT.REG and o.width == 64 for o in s.operands)), None)
        if cand is None:
            self.skipTest("no 3x64-bit-register ALU instruction in the loaded set")

        class T(Template):
            def build(self, b):
                b.instruction(cand.name, X0, X1, X2)
                b.hole(Hole(n=2, kind=Kind.ALU))

        body = self._body(self._run(T()))
        self.assertEqual(body[0].name, cand.name)
        self.assertTrue(body[0].is_from_template)
        self.assertEqual([o.value for o in body[0].operands], ["x0", "x1", "x2"])
        self.assertEqual(len(body), 3)

    # ---- assembles end-to-end --------------------------------------------------------------------
    def test_assembles(self):
        from src.aarch64.template import Template, Hole, Kind
        from src.aarch64.template.runner import generate_test_case

        class T(Template):
            def build(self, b):
                b.hole(Hole(n=3, kind=Kind.LOAD))
                b.hole(Hole(n=3, kind=Kind.ALU))

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.asm")
            tc = generate_test_case(T(), self.isa, 1, asm_file=path, assemble=True)
            self.assertTrue(os.path.exists(tc.asm_path))
            self.assertTrue(os.path.exists(tc.bin_path))


    # ---- seeds / determinism (integrates with Revizor's seeding) ---------------------------------
    def test_deterministic_for_a_seed(self):
        from src.aarch64.template import Template, Hole, Kind
        from src.aarch64.template.runner import generate_test_case

        class T(Template):
            def build(self, b):
                b.hole(Hole(4, kind=Kind.LOAD))
                b.hole(Hole(4, kind=Kind.ALU))

        def seq(seed):
            tc = generate_test_case(T(), self.isa, seed, assemble=False)
            return [(i.name, tuple(o.value for o in i.operands))
                    for i in tc.functions[0].get_first_bb()]

        self.assertEqual(seq(42), seq(42))      # same seed -> identical program
        self.assertNotEqual(seq(42), seq(43))   # different seed -> different program

    def test_fuzzer_loop_advances_seed(self):
        # mirrors the fuzzer: one generator reused across iterations, seed advancing -> fresh programs
        from src.aarch64.template import Template, Hole, Kind
        from src.aarch64.template.runner import build_test_case
        from src.factory import get_program_generator

        class T(Template):
            def build(self, b):
                b.hole(Hole(5, kind=Kind.ALU))

        gen = get_program_generator(self.isa, 7)
        gen.set_seed(7)
        first = [i.name for i in build_test_case(gen, T(), assemble=False).functions[0].get_first_bb()]
        second = [i.name for i in build_test_case(gen, T(), assemble=False).functions[0].get_first_bb()]
        self.assertNotEqual(first, second)

    # ---- interface: required vs optional + loud validation ---------------------------------------
    def test_argument_enforcement(self):
        from src.aarch64.template import Hole, Mem, IndirectCall
        self.assertEqual(Hole().n, 1)          # n defaults to a single instruction
        with self.assertRaises(TypeError):
            Mem()                              # base is required
        with self.assertRaises(TypeError):
            Hole(3, 2)                         # optional params are keyword-only
        with self.assertRaises(ValueError):
            Hole(0)                            # n >= 1
        with self.assertRaises(ValueError):
            Hole((5, 2))                       # range must be (min <= max)
        with self.assertRaises(TypeError):
            Hole(kind="alu")                   # wrong type, not a Kind
        self.assertIsNone(IndirectCall().targets)
        with self.assertRaises(ValueError):
            IndirectCall(targets=[])           # empty target list
        with self.assertRaises(TypeError):
            IndirectCall(populate="x")         # populate must be True/False/Hole
        self.assertEqual(Hole(n_bb=3).n_bb, 3)
        with self.assertRaises(ValueError):
            Hole(n_bb=0)                       # n_bb follows the same rules as n
        with self.assertRaises(ValueError):
            Hole(n_bb=(3, 2))

    def test_instruction_form_selected_by_operand_signature(self):
        from src.aarch64.template import Template, Mem, specs
        from src.aarch64.template.objects import X0, X1

        class T(Template):
            def build(self, b):
                b.instruction(specs.LDR, X0, Mem(X1))          # 64-bit register + memory form

        ld = self._body(self._run(T()))[0]
        self.assertEqual(ld.name, "ldr")
        self.assertEqual(ld.operands[0].value, "x0")           # the 64-bit form was chosen, not w0

        class Bad(Template):
            def build(self, b):
                b.instruction(specs.LDR, X0)                   # no LDR form takes a single register

        with self.assertRaises(ValueError):
            self._run(Bad())

    def test_hole_count_flexibility(self):
        from src.aarch64.template import Template, Hole, Kind

        def count(n):
            class T(Template):
                def build(self, b):
                    b.hole(Hole(n, kind=Kind.ALU))
            return len(self._body(self._run(T())))

        self.assertEqual(count(1), 1)              # single
        self.assertEqual(count(7), 7)              # exact
        self.assertTrue(3 <= count((3, 6)) <= 6)   # range
        self.assertGreaterEqual(count(None), 1)    # unconstrained random count

    # ---- indirect-call gadget --------------------------------------------------------------------
    def test_single_target_call_is_adr_blr(self):
        from src.aarch64.template import Template, IndirectCall

        class T(Template):
            def build(self, b):
                b.call(IndirectCall(targets=1))

        tc = self._run(T())
        self.assertEqual([i.name for i in self._body(tc)], ["adr", "blr"])
        self.assertEqual(len(tc.functions), 2)              # entry + one callee
        self.assertTrue(tc.functions[1].is_leaf)
        self.assertIsNone(tc.functions[1].dispatch_table)

    def test_multi_target_call_builds_dispatch_table(self):
        from src.aarch64.template import Template, IndirectCall

        class T(Template):
            def build(self, b):
                b.call(IndirectCall(targets=3))

        tc = self._run(T())
        self.assertEqual([i.name for i in self._body(tc)], ["and", "adr", "ldrsw", "add", "blr"])
        self.assertEqual(len(tc.functions), 4)              # entry + three callees
        tables = [f.dispatch_table for f in tc.functions if f.dispatch_table]
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0].size, 4)                 # three callees -> next power of two

    def test_direct_call_is_bl(self):
        from src.aarch64.template import Template, DirectCall

        class T(Template):
            def build(self, b):
                b.call(DirectCall())

        tc = self._run(T())
        self.assertEqual([i.name for i in self._body(tc)], ["bl"])
        self.assertEqual(len(tc.functions), 2)              # entry + one callee

    def test_single_entry_dispatch_table(self):
        # a jump table can hold a single target: IndirectCall(targets=1, dispatch=True)
        from src.aarch64.template import Template, IndirectCall

        class T(Template):
            def build(self, b):
                b.call(IndirectCall(targets=1, dispatch=True))

        tc = self._run(T())
        self.assertEqual([i.name for i in self._body(tc)], ["and", "adr", "ldrsw", "add", "blr"])
        tables = [f.dispatch_table for f in tc.functions if f.dispatch_table]
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0].size, 1)

    def test_dispatch_false_rejects_multiple_targets(self):
        from src.aarch64.template import Template, IndirectCall

        class T(Template):
            def build(self, b):
                b.call(IndirectCall(targets=3, dispatch=False))

        with self.assertRaises(ValueError):
            self._run(T())

    def test_generic_call_is_a_call_and_assembles(self):
        from src.aarch64.template import Template, Call
        from src.aarch64.template.runner import generate_test_case

        class T(Template):
            def build(self, b):
                b.call(Call())

        with tempfile.TemporaryDirectory() as d:
            tc = generate_test_case(T(), self.isa, 3, asm_file=os.path.join(d, "t.asm"), assemble=True)
            self.assertGreaterEqual(len(tc.functions), 2)                  # entry + at least one callee
            self.assertTrue(self._body(tc)[-1].name in ("bl", "blr"))      # ends in a call
            self.assertTrue(os.path.exists(tc.bin_path))

    def test_calls_are_independent_and_assemble(self):
        # two calls with different target counts coexist, each with its own table, and assemble e2e
        from src.aarch64.template import Template, IndirectCall
        from src.aarch64.template.runner import generate_test_case

        class T(Template):
            def build(self, b):
                b.call(IndirectCall(targets=2))
                b.call(IndirectCall(targets=3))

        with tempfile.TemporaryDirectory() as d:
            tc = generate_test_case(T(), self.isa, 1, asm_file=os.path.join(d, "t.asm"), assemble=True)
            tables = [f.dispatch_table for f in tc.functions if f.dispatch_table]
            self.assertEqual(sorted(t.size for t in tables), [2, 4])
            self.assertEqual(len({t.label for t in tables}), 2)   # distinct per-call tables
            self.assertTrue(os.path.exists(tc.bin_path))

    # ---- declared callees, sharing, population ---------------------------------------------------
    @staticmethod
    def _fn_body(tc, i):
        return [inst for inst in tc.functions[i].get_first_bb() if inst.name != "macro"]

    def test_declared_function_has_body(self):
        from src.aarch64.template import Template, Hole, Kind, IndirectCall

        class T(Template):
            def build(self, b):
                leaf = b.function(lambda f: f.hole(Hole(3, kind=Kind.LOAD)))
                b.call(IndirectCall(targets=[leaf]))

        tc = self._run(T())
        body = self._fn_body(tc, 1)
        self.assertEqual(len(body), 3)
        self.assertTrue(all(i.has_memory_access for i in body))

    def test_calls_can_share_a_target(self):
        from src.aarch64.template import Template, IndirectCall, DirectCall

        class T(Template):
            def build(self, b):
                leaf = b.function(populate=False)
                b.call(IndirectCall(targets=[leaf]))
                b.call(DirectCall(target=leaf))

        tc = self._run(T())
        self.assertEqual(len(tc.functions), 2)                 # entry + one shared callee, none created
        body = self._body(tc)
        self.assertEqual([i.name for i in body], ["adr", "blr", "bl"])
        self.assertEqual(body[0].operands[-1].value, ".function_1")   # ADR target
        self.assertEqual(body[2].operands[0].value, ".function_1")    # BL target

    def test_targets_count_reuses_declared_then_creates(self):
        from src.aarch64.template import Template, IndirectCall

        class T(Template):
            def build(self, b):
                b.function(populate=False)          # one declared callee
                b.call(IndirectCall(targets=3))     # reuse it, then create two more

        tc = self._run(T())
        self.assertEqual(len(tc.functions), 4)                 # entry + declared + two created
        table = next(f.dispatch_table for f in tc.functions if f.dispatch_table)
        self.assertIs(table.entries[0], tc.functions[1])       # declared callee used first

    def test_created_callee_populate_hole_constrains_body(self):
        from src.aarch64.template import Template, Hole, Kind, IndirectCall

        class T(Template):
            def build(self, b):
                b.call(IndirectCall(targets=1, populate=Hole(2, kind=Kind.ALU)))

        body = self._fn_body(self._run(T()), 1)
        self.assertEqual(len(body), 2)
        self.assertTrue(all(not i.has_memory_access and not i.control_flow for i in body))

    def test_populate_false_is_empty_leaf(self):
        from src.aarch64.template import Template, DirectCall

        class T(Template):
            def build(self, b):
                b.call(DirectCall(populate=False))

        self.assertEqual(self._fn_body(self._run(T()), 1), [])

    def test_backward_call_is_rejected(self):
        from src.aarch64.template import Template, DirectCall

        class T(Template):
            def build(self, b):
                first = b.function(populate=False)
                b.function(lambda f: f.call(DirectCall(target=first)))   # later fn calls earlier: loop

        with self.assertRaises(ValueError):
            self._run(T())

    def test_bad_target_type_raises(self):
        from src.aarch64.template import Template, IndirectCall

        class T(Template):
            def build(self, b):
                b.call(IndirectCall(targets=[123]))              # not a declared callee handle

        with self.assertRaises(TypeError):
            self._run(T())

    # ---- control flow ----------------------------------------------------------------------------
    def test_if_emits_conditional_then_fallthrough(self):
        from src.aarch64.template import Template, Hole, Kind, IndirectCall

        class T(Template):
            def build(self, b):
                b.hole(Hole(1, kind=Kind.ALU))
                with b.if_():
                    b.call(IndirectCall(targets=1))

        entry = self._run(T()).functions[0]
        pair = [bb for bb in entry if len(bb.terminators) == 2]      # a conditional + fallthrough close a bb
        self.assertTrue(pair, "if_ must close a block with a conditional then an unconditional branch")
        self.assertTrue(all(t.control_flow for t in pair[0].terminators))
        taken = pair[0].successors[0]
        self.assertTrue(any(i.name == "blr" for i in taken), "the call belongs to the taken block")

    def test_multi_bb_hole_spans_blocks_and_assembles(self):
        from src.aarch64.template import Template, Hole, Kind
        from src.aarch64.template.runner import generate_test_case

        class T(Template):
            def build(self, b):
                b.hole(Hole(n=6, n_bb=3, kind=Kind.ALU))

        with tempfile.TemporaryDirectory() as d:
            tc = generate_test_case(T(), self.isa, 1, asm_file=os.path.join(d, "t.asm"), assemble=True)
            self.assertGreater(len(tc.functions[0]), 3)              # >= 3 nodes + a continuation block
            total = sum(1 for bb in tc.functions[0] for i in bb if i.name != "macro")
            self.assertEqual(total, 6)                               # all requested instructions placed
            self.assertTrue(os.path.exists(tc.bin_path))

    def test_max_functions_enforced(self):
        from src.aarch64.template import Template
        self.CONF.max_functions_per_test_case = 2                    # entry + one callee only

        class T(Template):
            def build(self, b):
                b.function(populate=False)
                b.function(populate=False)                           # the second exceeds the limit

        with self.assertRaises(ValueError):
            self._run(T())

    def test_max_calls_enforced(self):
        from src.aarch64.template import Template, DirectCall
        self.CONF.max_calls_per_function = 1

        class T(Template):
            def build(self, b):
                b.call(DirectCall(populate=False))
                b.call(DirectCall(populate=False))                   # the second exceeds the limit

        with self.assertRaises(ValueError):
            self._run(T())

    # ---- loader ----------------------------------------------------------------------------------
    def test_load_template(self):
        from src.aarch64.template import Template
        from src.aarch64.template.runner import load_template
        good = ("from src.aarch64.template import Template, Hole\n"
                "class Foo(Template):\n"
                "    def build(self, b): b.hole(Hole(1))\n")
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "tpl.py")
            with open(p, "w") as f:
                f.write(good)
            self.assertIsInstance(load_template(p), Template)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "none.py")
            with open(p, "w") as f:
                f.write("x = 1\n")
            with self.assertRaises(ValueError):
                load_template(p)


if __name__ == "__main__":
    unittest.main()
