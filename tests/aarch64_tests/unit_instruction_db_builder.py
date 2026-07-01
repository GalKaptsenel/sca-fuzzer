"""Tests for the AArch64 base.json generator (template grammar expansion + tag/format invariants).

Run from any cwd: paths are resolved from __file__.
"""
import json
import os
import string
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.aarch64 import instruction_db_builder as dl   # noqa: E402
from src.aarch64.arm_isa_extractor.models import MemAccess   # noqa: E402
from src.interfaces import MemorySpec, OperandSpec, OT   # noqa: E402
from src.aarch64.aarch64_target_desc import AArch64MemRole   # noqa: E402


def _ospec(name, ot, width, values, role=None):
    return OperandSpec(ot, width, False, True, False, list(values), name, role)


def _memspec(*components):
    """components: (role|None, OT, width, [values], name) -> a MemorySpec wrapping them."""
    inner = [_ospec(name, ot, w, vals, AArch64MemRole(r) if r else None)
             for r, ot, w, vals, name in components]
    return MemorySpec(64, False, True, False, inner, inner[0].name)

_BASE_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "base.json")


def _op(name, kind="reg", **kw):
    return {"name": name, "kind": kind, "read": kw.get("read", False), "write": kw.get("write", False),
            "width": kw.get("width", 64), "signed": False, "values": kw.get("values", []),
            "imm_ranges": kw.get("imm_ranges", []), "reg_range": kw.get("reg_range", [0, 31, 1]),
            "mem_role": kw.get("mem_role", "none")}


def _operand_names(operands):
    """Placeholder names a template needs: top-level operands, with a memory access flattened to its
    address components (mirrors Instruction.to_asm_string)."""
    names = []
    for o in operands:
        if isinstance(o, MemorySpec):
            names.extend(c.name for c in o.inner)
        else:
            names.append(o.name)
    return names


def _forms(template, ops, mem_access=MemAccess.LOAD):
    by = {o["name"]: o for o in ops}
    return {t.rstrip(): _operand_names(ol)
            for t, ol in dl._expand_format(template, by, 0, mem_access)}


class ExpandTemplateTest(unittest.TestCase):
    def test_optional_group_present_and_absent(self):
        forms = _forms("ADDS  <Xd>, <Xn>, #<imm>{, <shift>}",
                       [_op("Xd", write=True), _op("Xn", read=True),
                        _op("imm", kind="imm", imm_ranges=[[0, 0, 3, 1]]),
                        _op("shift", kind="imm", values=["lsl"])])
        self.assertIn("ADDS  {Xd}, {Xn}, #{imm}", forms)
        self.assertIn("ADDS  {Xd}, {Xn}, #{imm}, {shift}", forms)

    def test_alternative_expands_to_each_choice(self):
        forms = _forms("LDR  <Xt>, [<Xn|SP>, (<Wm>|<Xm>)]",
                       [_op("Xt", write=True), _op("Xn|SP", read=True, mem_role="base"),
                        _op("Wm", read=True, width=32, mem_role="index"),
                        _op("Xm", read=True, mem_role="index")])
        self.assertIn("LDR  {Xt}, [{Xn|SP}, {Wm}]", forms)
        self.assertIn("LDR  {Xt}, [{Xn|SP}, {Xm}]", forms)

    def test_width_selector_splits_into_w_and_x(self):
        forms = _forms("TBZ  {R}<t>, #<imm>, <label>",
                       [_op("t", read=True),
                        _op("imm", kind="imm", imm_ranges=[[0, 0, 63, 1]]), _op("label", kind="label")])
        self.assertTrue(any("{Wt}" in f for f in forms))
        self.assertTrue(any("{Xt}" in f for f in forms))

    def test_register_list_braces_are_escaped(self):
        # a literal {…} list must survive str.format as a single { } pair
        forms = _forms("LD1B  {<Zt>.B}, <Pg>/Z, [<Xn|SP>]",
                       [_op("Zt", write=True), _op("Pg", read=True), _op("Xn|SP", read=True, mem_role="base")])
        tmpl = next(iter(forms))
        rendered = tmpl.format(Zt="z0", Pg="p0", **{"Xn|SP": "x1"})
        self.assertIn("{z0.B}", rendered)

    def test_no_stray_brackets_and_format_trick(self):
        ops = [_op("Xt", write=True), _op("Xn|SP", read=True, mem_role="base"),
               _op("Wm", read=True, width=32, mem_role="index"), _op("Xm", read=True, mem_role="index")]
        by = {o["name"]: o for o in ops}
        for tmpl, operands in dl._expand_format("LDR  <Xt>, [<Xn|SP>, (<Wm>|<Xm>)]", by, 0, MemAccess.LOAD):
            self.assertNotIn("<", tmpl)
            self.assertNotIn(">", tmpl)
            names = set(_operand_names(operands))
            tmpl.format(**{n: "x0" for n in names})   # must not raise


def _operands(template, ops, mem_access):
    """The single expanded form's operand objects (templates here have no optional/alternative)."""
    by = {o["name"]: o for o in ops}
    forms = dl._expand_format(template, by, 0, mem_access)
    assert len(forms) == 1, f"expected one form, got {len(forms)}"
    return forms[0][1]


def _mem_ops(operands):
    return [o for o in operands if isinstance(o, MemorySpec)]


class MemoryGroupingTest(unittest.TestCase):
    """A `[...]` holding a base register becomes one MemorySpec wrapping its address components; the
    wrapper carries the access direction, the components keep their own roles and read/write."""

    def test_load_wraps_components_and_reads(self):
        ops = _operands("LDR  <Xt>, [<Xn|SP>, #<simm>]",
                        [_op("Xt", write=True), _op("Xn|SP", read=True, mem_role="base"),
                         _op("simm", kind="imm", read=True, imm_ranges=[[0, -256, 255, 1]], mem_role="offset")],
                        MemAccess.LOAD)
        mems = _mem_ops(ops)
        self.assertEqual(len(mems), 1)
        m = mems[0]
        self.assertEqual(m.type, OT.MEM)
        self.assertEqual((m.src, m.dest), (True, False))          # a load reads the location
        self.assertEqual([c.mem_role for c in m.inner],
                         [AArch64MemRole.BASE, AArch64MemRole.OFFSET])
        self.assertEqual([c.type for c in m.inner], [OT.REG, OT.IMM])

    def test_store_writes_the_location(self):
        ops = _operands("STR  <Xt>, [<Xn|SP>]",
                        [_op("Xt", read=True), _op("Xn|SP", read=True, mem_role="base")], MemAccess.STORE)
        m = _mem_ops(ops)[0]
        self.assertEqual((m.src, m.dest), (False, True))          # a store writes the location

    def test_rmw_reads_and_writes(self):
        ops = _operands("STADD  <Xs>, [<Xn|SP>]",
                        [_op("Xs", read=True), _op("Xn|SP", read=True, mem_role="base")], MemAccess.RMW)
        m = _mem_ops(ops)[0]
        self.assertEqual((m.src, m.dest), (True, True))           # an atomic read-modify-write does both

    def test_writeback_is_on_the_base_component(self):
        # pre-index `!` writes the base register; that lives on the component, not the access direction
        ops = _operands("LDR  <Bt>, [<Xn|SP>, #<simm>]!",
                        [_op("Bt", write=True), _op("Xn|SP", read=True, write=True, mem_role="base"),
                         _op("simm", kind="imm", read=True, imm_ranges=[[0, -256, 255, 1]], mem_role="offset")],
                        MemAccess.LOAD)
        base = _mem_ops(ops)[0].inner[0]
        self.assertEqual(base.mem_role, AArch64MemRole.BASE)
        self.assertTrue(base.dest)                                # writeback survives on the base

    def test_index_extend_offset_grouped_into_one_access(self):
        ops = _operands("LDR  <Bt>, [<Xn|SP>, <Wm>, <extend> <amount>]",
                        [_op("Bt", write=True), _op("Xn|SP", read=True, mem_role="base"),
                         _op("Wm", read=True, width=32, mem_role="index"),
                         _op("extend", kind="extend", values=["uxtw", "sxtw"], mem_role="extend"),
                         _op("amount", kind="imm", imm_ranges=[[0, 0, 0, 1]], mem_role="offset")],
                        MemAccess.LOAD)
        mems = _mem_ops(ops)
        self.assertEqual(len(mems), 1)
        self.assertEqual([c.mem_role for c in mems[0].inner],
                         [AArch64MemRole.BASE, AArch64MemRole.INDEX,
                          AArch64MemRole.EXTEND, AArch64MemRole.OFFSET])

    def test_lane_index_is_not_a_memory_access(self):
        # `Vt.D[index]` is a SIMD lane index (no base) -> stays literal + plain operand; only `[Xn|SP]`
        # (the standalone memory bracket) becomes a MemorySpec.
        ops = _operands("LD1  {<Vt>.D}[<index>], [<Xn|SP>]",
                        [_op("Vt", write=True),
                         _op("index", kind="imm", read=True, imm_ranges=[[0, 0, 1, 1]]),  # role none
                         _op("Xn|SP", read=True, mem_role="base")], MemAccess.LOAD)
        self.assertEqual(len(_mem_ops(ops)), 1)                   # exactly one memory access
        index = next(o for o in ops if o.name == "index")
        self.assertNotIsInstance(index, MemorySpec)
        self.assertIsNone(index.mem_role)                         # the lane index carries no addressing role

    def test_multiple_memory_operands(self):
        # MOPS copy has two memory accesses (`[Xd]!, [Xs]!`); each bracket is its own MemorySpec
        ops = _operands("CPYFP  [<Xd>]!, [<Xs>]!, <Xn>!",
                        [_op("Xd", read=True, write=True, mem_role="base"),
                         _op("Xs", read=True, write=True, mem_role="base"),
                         _op("Xn", read=True, write=True)], MemAccess.RMW)
        mems = _mem_ops(ops)
        self.assertEqual(len(mems), 2)                            # two independent memory accesses
        self.assertEqual([m.inner[0].name for m in mems], ["Xd", "Xs"])
        self.assertEqual([m.inner[0].mem_role for m in mems],
                         [AArch64MemRole.BASE, AArch64MemRole.BASE])
        for m in mems:
            self.assertEqual((m.src, m.dest), (True, True))       # rmw on both regions

    def test_operand_name_reused_with_conflicting_roles_is_loud(self):
        # SME `LDR ZA[<Wv>, <offs>], [<Xn|SP>, #<offs>]` reuses <offs> as a tile index AND a memory
        # offset; the name-keyed template can't represent that -> the downloader must fail loud, never
        # leak the memory role onto the tile-index occurrence.
        inst = {"name": "ldr", "encoding_name": "ldr_za_ri_", "category": "mortlach",
                "control_flow": False, "mem_access": "load", "flags_written": [], "flags_read": [],
                "constraints": [], "asm_template": "LDR  ZA[<Wv>, <offs>], [<Xn|SP>, #<offs>]",
                "operands": [_op("Wv", read=True), _op("offs", kind="imm", imm_ranges=[[0, 0, 15, 1]]),
                             _op("Xn|SP", read=True, mem_role="base"),
                             _op("offs", kind="imm", imm_ranges=[[0, 0, 15, 1]], mem_role="offset")]}
        with self.assertRaises(ValueError):
            dl._expand_instruction(inst)


class RegisterOffsetWidthTest(unittest.TestCase):
    """`_resolve_register_offset`: a memory register-offset's extend is tied to the index width."""

    def _mem(self, index_width, extend_values, with_offset):
        comps = [("base", OT.REG, 64, ["x0"], "Xn|SP"),
                 ("index", OT.REG, index_width, ["x1" if index_width == 64 else "w1"], "m")]
        if extend_values is not None:
            comps.append(("extend", OT.IMM, 0, extend_values, "extend"))
        if with_offset:
            comps.append(("offset", OT.IMM, 0, ["2"], "amount"))
        return _memspec(*comps)

    def test_w_index_keeps_only_w_extends(self):
        m = self._mem(32, ["uxtw", "sxtw", "sxtx"], with_offset=True)
        self.assertTrue(dl._resolve_register_offset(m))
        self.assertEqual(m.inner[2].values, ["uxtw", "sxtw"])      # sxtx (X-only) dropped

    def test_x_index_keeps_only_x_extends(self):
        m = self._mem(64, ["uxtw", "sxtw", "sxtx"], with_offset=True)
        self.assertTrue(dl._resolve_register_offset(m))
        self.assertEqual(m.inner[2].values, ["sxtx"])              # uxtw/sxtw (W-only) dropped

    def test_plain_offset_needs_x_index(self):
        self.assertFalse(dl._resolve_register_offset(self._mem(32, None, with_offset=False)))
        self.assertTrue(dl._resolve_register_offset(self._mem(64, None, with_offset=False)))

    def test_lsl_needs_an_amount(self):
        without = self._mem(64, ["lsl", "sxtx"], with_offset=False)
        self.assertTrue(dl._resolve_register_offset(without))
        self.assertEqual(without.inner[2].values, ["sxtx"])        # lsl dropped (no amount)
        with_amt = self._mem(64, ["lsl", "sxtx"], with_offset=True)
        self.assertTrue(dl._resolve_register_offset(with_amt))
        self.assertEqual(with_amt.inner[2].values, ["lsl", "sxtx"])


class ExtendWidthTest(unittest.TestCase):
    """`_resolve_extend`: a data-processing extend is tied to its source register width."""

    def test_extend_matches_source_width(self):
        ops = [_ospec("Xd", OT.REG, 64, ["x0"]), _ospec("Xn", OT.REG, 64, ["x1"]),
               _ospec("Wm", OT.REG, 32, ["w2"]),
               _ospec("extend", OT.IMM, 0, ["uxtb", "uxtw", "uxtx", "sxtx"]),
               _ospec("amount", OT.IMM, 0, ["2"])]
        self.assertTrue(dl._resolve_extend(ops, {"operands": []}))
        self.assertEqual(ops[3].values, ["uxtb", "uxtw"])          # X-extends dropped for a W source

    def test_lsl_uses_destination_width_and_needs_amount(self):
        base = lambda amt: [_ospec("Xd", OT.REG, 64, ["x0"]), _ospec("Xn", OT.REG, 64, ["x1"]),
                            _ospec("Xm", OT.REG, 64, ["x2"]), _ospec("ext", OT.IMM, 0, ["lsl", "sxtx"])] \
                           + ([_ospec("amount", OT.IMM, 0, ["3"])] if amt else [])
        with_amt = base(True)
        self.assertTrue(dl._resolve_extend(with_amt, {"operands": []}))
        self.assertEqual(with_amt[3].values, ["lsl", "sxtx"])
        no_amt = base(False)
        self.assertTrue(dl._resolve_extend(no_amt, {"operands": []}))
        self.assertEqual(no_amt[3].values, ["sxtx"])               # lsl dropped (no amount)

    def test_missing_extend_for_subwidth_source_is_dropped(self):
        inst = {"operands": [_op("extend", kind="extend", values=["uxtw", "sxtw"])]}
        subwidth = [_ospec("Xd", OT.REG, 64, ["x0"]), _ospec("Xn", OT.REG, 64, ["x1"]),
                    _ospec("Wm", OT.REG, 32, ["w2"])]
        self.assertFalse(dl._resolve_extend(subwidth, inst))       # W source, mandatory extend omitted
        same = [_ospec("Wd", OT.REG, 32, ["w0"]), _ospec("Wn", OT.REG, 32, ["w1"]),
                _ospec("Wm", OT.REG, 32, ["w2"])]
        self.assertTrue(dl._resolve_extend(same, inst))            # same-width source: plain form is fine


class BitTestWidthTest(unittest.TestCase):
    """`_resolve_bit_test`: TBZ/TBNZ bit position must be below the tested register width."""

    def test_w_register_clamps_bit_to_31(self):
        ops = [_ospec("Wt", OT.REG, 32, ["w0"]), _ospec("imm", OT.IMM, 0, [str(i) for i in range(64)]),
               _ospec("label", OT.LABEL, 0, [])]
        self.assertTrue(dl._resolve_bit_test(ops, control_flow=True))
        self.assertEqual(ops[1].values, [str(i) for i in range(32)])

    def test_x_register_allows_63(self):
        ops = [_ospec("Xt", OT.REG, 64, ["x0"]), _ospec("imm", OT.IMM, 0, [str(i) for i in range(64)]),
               _ospec("label", OT.LABEL, 0, [])]
        self.assertTrue(dl._resolve_bit_test(ops, control_flow=True))
        self.assertEqual(len(ops[1].values), 64)

    def test_non_branch_is_untouched(self):
        ops = [_ospec("Wt", OT.REG, 32, ["w0"]), _ospec("imm", OT.IMM, 0, ["40"])]
        self.assertTrue(dl._resolve_bit_test(ops, control_flow=False))
        self.assertEqual(ops[1].values, ["40"])


class GeneratableTest(unittest.TestCase):
    """`_generatable`: a non-branch label operand (adr/adrp, literal load) is not generatable."""

    def test_non_branch_label_is_excluded(self):
        inst = {"control_flow": False, "operands": [_op("Wt", write=True), _op("label", kind="label")]}
        self.assertFalse(dl._generatable(inst))

    def test_branch_label_is_kept(self):
        inst = {"control_flow": True, "operands": [_op("label", kind="label")]}
        self.assertTrue(dl._generatable(inst))

    def test_plain_instruction_is_kept(self):
        inst = {"control_flow": False, "operands": [_op("Xd", write=True), _op("Xn", read=True)]}
        self.assertTrue(dl._generatable(inst))


class BaseJsonRegressionTest(unittest.TestCase):
    """Invariants on the generated base.json (skipped if it has not been generated)."""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(_BASE_JSON):
            raise unittest.SkipTest("base.json not generated")
        cls.specs = json.load(open(_BASE_JSON))

    def test_all_specs_obey_format_trick(self):
        # EVERY spec must be a valid str.format template whose placeholders are exactly its operand
        # names, so the generator can do template.format(**{operand: chosen_value}).
        def names_of(spec):
            # template placeholders = top-level operand names, with memory accesses flattened to their
            # address components (an inner base/index/offset is what the `[...]` placeholders reference)
            names = set()
            for o in spec["operands"] + spec["implicit_operands"]:
                if "inner" in o:
                    names.update(c["name"] for c in o["inner"])
                else:
                    names.add(o["name"])
            return names

        fmt = string.Formatter()
        bad = []
        for s in self.specs:
            names = names_of(s)
            try:
                for _, field, _, _ in fmt.parse(s["template"]):
                    if field is not None and field not in names:
                        bad.append((s["name"], s["category"], f"{{{field}}}", s["template"]))
                s["template"].format(**{n: "x0" for n in names})
            except (ValueError, KeyError, IndexError) as e:
                bad.append((s["name"], s["category"], repr(e), s["template"]))
        self.assertEqual(bad, [], f"{len(bad)} specs fail the format-trick, e.g. {bad[:5]}")

    def test_every_memory_operand_wraps_exactly_one_base(self):
        # structural invariant: a memory access is a MEM operand with `inner` address components, exactly
        # one of which is the base; the base is a register, and the access carries no top-level role.
        bad = []
        for s in self.specs:
            for o in s["operands"] + s["implicit_operands"]:
                if o["type_"] != "MEM":
                    continue
                inner = o.get("inner", [])
                bases = [c for c in inner if c.get("mem_role") == "base"]
                if len(bases) != 1 or bases[0]["type_"] != "REG":
                    bad.append((s["name"], s["template"], [c.get("mem_role") for c in inner]))
        self.assertEqual(bad, [], f"{len(bad)} memory operands not wrapping exactly one base: {bad[:5]}")

    def test_no_role_leaks_onto_top_level_operands(self):
        # an addressing role must only appear on a memory access's inner components, never on a
        # top-level operand.
        bad = [(s["name"], o["name"]) for s in self.specs
               for o in s["operands"] + s["implicit_operands"]
               if o["type_"] != "MEM" and o.get("mem_role") is not None]
        self.assertEqual(bad, [], f"{len(bad)} top-level operands carry an addressing role: {bad[:5]}")


if __name__ == "__main__":
    unittest.main()
