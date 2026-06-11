"""Tests for the AArch64 base.json generator (template grammar expansion + tag/format invariants).

Run from any cwd: paths are resolved from __file__.
"""
import json
import os
import string
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.aarch64 import isa_downloader as dl   # noqa: E402

_BASE_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "base.json")


def _op(name, kind="reg", **kw):
    return {"name": name, "kind": kind, "read": kw.get("read", False), "write": kw.get("write", False),
            "width": kw.get("width", 64), "signed": False, "values": kw.get("values", []),
            "imm_ranges": kw.get("imm_ranges", []), "reg_range": kw.get("reg_range", [0, 31, 1]),
            "mem_role": kw.get("mem_role", "none")}


def _forms(template, ops):
    by = {o["name"]: o for o in ops}
    return {t.rstrip(): [o["name"] for o in ol] for t, ol in dl._expand_format(template, by, 0)}


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
        for tmpl, operands in dl._expand_format("LDR  <Xt>, [<Xn|SP>, (<Wm>|<Xm>)]", by, 0):
            self.assertNotIn("<", tmpl)
            self.assertNotIn(">", tmpl)
            names = {o["name"] for o in operands}
            tmpl.format(**{n: "x0" for n in names})   # must not raise


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
        fmt = string.Formatter()
        bad = []
        for s in self.specs:
            names = {o["name"] for o in s["operands"]} | {o["name"] for o in s["implicit_operands"]}
            try:
                for _, field, _, _ in fmt.parse(s["template"]):
                    if field is not None and field not in names:
                        bad.append((s["name"], s["category"], f"{{{field}}}", s["template"]))
                s["template"].format(**{n: "x0" for n in names})
            except (ValueError, KeyError, IndexError) as e:
                bad.append((s["name"], s["category"], repr(e), s["template"]))
        self.assertEqual(bad, [], f"{len(bad)} specs fail the format-trick, e.g. {bad[:5]}")


if __name__ == "__main__":
    unittest.main()
