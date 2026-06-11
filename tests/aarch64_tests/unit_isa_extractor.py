"""Unit tests for arm_isa_extractor — pure functions on synthetic ASL/encoding inputs
(no XML download, no kernel, no Revizor). Runnable from any cwd."""
import os
import sys
import unittest
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # run from any cwd

from src.aarch64.arm_isa_extractor.models import MemAccess, OperandKind, EncodingCtx, ExtractionError
from src.aarch64.arm_isa_extractor.asl import extract_asl_semantics
from src.aarch64.arm_isa_extractor.immediate import (
    immediate_operand, generate_logical_immediates, constant_fields, _is_label,
)
from src.aarch64.arm_isa_extractor.models import (
    MemRole, RegFile, MemAccess as MA, Instruction, Operand, FlagEffects,
)
from src.aarch64.arm_isa_extractor.asl import AslSemantics
from src.aarch64.arm_isa_extractor.operands import (
    _register_range, _enumerated, _enum_value, build_operands, _writeback_targets, _composite_pair,
)
from src.aarch64.arm_isa_extractor.constraints import unpredictable_constraints
from src.aarch64.arm_isa_extractor.extract import _COMMENT
from src.aarch64.arm_isa_extractor.validate import validate_instruction

_LDADD = """
let value : bits(datasize) = X{datasize}(s);
let data : bits(datasize) = MemAtomic{}(address, comparevalue, value, accdesc);
if n == 31 then address = SP{64}(); else address = X{64}(n); end;
X{datasize}(t) = ZeroExtend{datasize}(data);
"""
_ADDS = """
let operand1 : bits(datasize) = X{}(n);
let operand2 : bits(datasize) = X{}(m);
(result, nzcv) = AddWithCarry{datasize}(operand1, operand2, '0');
X{datasize}(d) = result;
PSTATE.[N,Z,C,V] = nzcv;
"""
_CCMP = """
if ConditionHolds(condition) then (-, flags) = AddWithCarry{datasize}(X{}(n), imm, '0'); end;
PSTATE.[N,Z,C,V] = flags;
"""


def _ctx(decode="", execute="", boxes=None, const_fields=None, arch_constants=None):
    return EncodingCtx(boxes or {}, const_fields or {}, arch_constants or {}, decode, execute)


class TestAslSemantics(unittest.TestCase):
    def test_atomic_rmw_reads_writes_sp(self):
        s = extract_asl_semantics(_LDADD)
        self.assertEqual(s.mem_access, MemAccess.RMW)
        self.assertEqual(s.read_regvars, frozenset({"n", "s"}))
        self.assertEqual(s.written_regvars, frozenset({"t"}))
        self.assertEqual(s.sp_regvars, frozenset({"n"}))

    def test_flag_writer(self):
        s = extract_asl_semantics(_ADDS)
        self.assertEqual(s.mem_access, MemAccess.NONE)
        self.assertEqual(s.written_regvars, frozenset({"d"}))
        self.assertEqual(s.read_regvars, frozenset({"n", "m"}))
        self.assertEqual(s.flags_written, frozenset("NZCV"))
        self.assertEqual(s.flags_read, frozenset())

    def test_conditional_reads_and_writes_flags(self):
        s = extract_asl_semantics(_CCMP)
        self.assertEqual(s.flags_written, frozenset("NZCV"))
        self.assertEqual(s.flags_read, frozenset("NZCV"))  # ConditionHolds

    def test_comments_stripped_no_false_flag_read(self):
        # SETF8 regression: `//PSTATE.C unchanged` must not be read as a flag use
        setf = "PSTATE.N = reg[7];\nPSTATE.Z = z;\n//PSTATE.C unchanged;\nPSTATE.V = v;"
        s = extract_asl_semantics(_COMMENT.sub("", setf))
        self.assertEqual(s.flags_written, frozenset("NZV"))
        self.assertEqual(s.flags_read, frozenset())

    def test_extendreg_with_braces_is_a_read(self):
        # `ExtendReg{}(m, ...)` (braces) must count m as read (the index reg of [Xn, Wm, SXTW])
        s = extract_asl_semantics("let offset = ExtendReg{}(m, extend_type, shift);\nX{}(t) = X{}(n);")
        self.assertIn("m", s.read_regvars)

    def test_sme_simd_accessors(self):
        # ZAtile/ZAslice (SME) and Vpart (SIMD part) accessors must resolve the reg-var role
        s = extract_asl_semantics("ZAtile{}(da, esize) = ZAtile{}(da, esize) + Vpart{}(n, part);")
        self.assertIn("da", s.written_regvars)
        self.assertIn("da", s.read_regvars)
        self.assertIn("n", s.read_regvars)

    def test_pure_rename_alias_resolves_to_var(self):
        # SVE list load: `let transfer = t; Z[transfer] = Mem...` => t (not just 'transfer') is written
        s = extract_asl_semantics("let transfer : integer = t;\nZ[transfer] = data;")
        self.assertIn("t", s.written_regvars)


class TestImmediate(unittest.TestCase):
    def test_plain_unsigned(self):
        op = immediate_operand("imm", "imm12", "unsigned immediate in the range 0 to 4095",
                               _ctx(decode="imm = ZeroExtend{64}(imm12);", boxes={"imm12": 12}), None, "T")
        self.assertEqual(op.kind, OperandKind.IMM)
        self.assertFalse(op.signed)
        self.assertEqual(op.imm_ranges, ((0, 0, 4095, 1),))   # (esize=0 unconditional, lo, hi, stride)

    def test_signed(self):
        op = immediate_operand("simm", "imm9", "signed immediate in the range -256 to 255",
                               _ctx(decode="offset = SignExtend{64}(imm9);", boxes={"imm9": 9}), None, "T")
        self.assertTrue(op.signed)
        self.assertEqual(op.imm_ranges, ((0, -256, 255, 1),))

    def test_scaled_via_const_field(self):
        ctx = _ctx(decode="let scale : integer = UInt(size); offset = LSL(ZeroExtend{64}(imm12), scale);",
                   boxes={"imm12": 12}, const_fields={"size": 3})
        op = immediate_operand("pimm", "imm12", "multiple of 8 in the range 0 to 32760", ctx, None, "T")
        self.assertEqual(op.imm_ranges, ((0, 0, 32760, 8),))   # stride 8 from prose, 4096 values == 2^12

    def test_prose_cardinality_mismatch_loud_fails(self):
        # prose says 4096 values but imm11 only encodes 2048 => loud-fail (value set != bit budget)
        ctx = _ctx(decode="imm = ZeroExtend{64}(imm11);", boxes={"imm11": 11})
        with self.assertRaises(ExtractionError):
            immediate_operand("imm", "imm11", "unsigned immediate in the range 0 to 4095", ctx, None, "T")

    def test_multibox_immediate_from_prose(self):
        # index encoded across two 1-bit boxes H:L => width 2, prose "0 to 3" => 4 values == 2^2
        ctx = _ctx(boxes={"H": 1, "L": 1})
        op = immediate_operand("index", "(H::L)", "index in the range 0 to 3", ctx, None, "T")
        self.assertEqual(op.imm_ranges, ((0, 0, 3, 1),))

    def test_subtractive_shift_amount_from_prose(self):
        # SQRSHR shift = 16 - UInt(imm4): the value SET is [1,16], stride 1, 16 values == 2^4
        ctx = _ctx(decode="let shift : integer = 16 - UInt(imm4);", boxes={"imm4": 4})
        op = immediate_operand("const", "imm4", "shift amount in the range 1 to 16", ctx, None, "T")
        self.assertEqual(op.imm_ranges, ((0, 1, 16, 1),))

    def test_bits_per_element_shift_couples_to_esize(self):
        # SVE tsz shift: tsize 4 bits => esizes 8/16/32/64; "1 to bits per element" => [1,esize] each
        ctx = _ctx(decode="let tsize : bits(4) = tszh::tszl; let shift = (2*esize)-UInt(tsize::imm3);",
                   boxes={"tszh": 2, "tszl": 2, "imm3": 3})
        op = immediate_operand("const", "(tszh::tszl::imm3)",
                               "shift amount in the range 1 to number of bits per element", ctx, None, "T")
        self.assertEqual(op.imm_ranges, ((8, 1, 8, 1), (16, 1, 16, 1), (32, 1, 32, 1), (64, 1, 64, 1)))

    def test_bitmask(self):
        op = immediate_operand("imm", "immr", "bitmask immediate",
                               _ctx(decode="(imm, -) = DecodeBitMasks{32}(N, imms, immr, TRUE);"), 32, "T")
        self.assertEqual(op.kind, OperandKind.IMM)
        self.assertEqual(op.values, generate_logical_immediates(32))

    def test_undef_constraint_narrows_width(self):
        # 32-bit shifted-register: `if sf=='0' && imm6[5]=='1' then UNDEF` forces imm6 MSB to 0 => [0,31]
        ctx = _ctx(decode="if sf == '0' && imm6[5] == '1' then EndOfDecode(Decode_UNDEF); end;",
                   boxes={"imm6": 6}, const_fields={"sf": 0})
        op = immediate_operand("amount", "imm6", "shift amount in the range 0 to 31", ctx, None, "T")
        self.assertEqual(op.imm_ranges, ((0, 0, 31, 1),))

    def test_value_restricted_immediate_high_encodings_reserved(self):
        # extended-register amount: imm3 has 8 encodings but only [0,4] are valid (5<=8 ok, 5-7 reserved)
        op = immediate_operand("amount", "imm3", "shift amount in the range 0 to 4",
                               _ctx(boxes={"imm3": 3}), None, "T")
        self.assertEqual(op.imm_ranges, ((0, 0, 4, 1),))

    def test_prose_more_values_than_field_loud_fails(self):
        with self.assertRaises(ExtractionError):
            immediate_operand("imm", "imm3", "in the range 0 to 15", _ctx(boxes={"imm3": 3}), None, "T")

    def test_enumerated_shift_extend_is_extend_kind(self):
        d = ET.fromstring('<definition><table>'
                          '<row><entry class="symbol">LSL</entry></row>'
                          '<row><entry class="symbol">SXTW</entry></row></table></definition>')
        self.assertEqual(_enumerated("<extend>", d).kind, OperandKind.EXTEND)


class TestMemoryOperands(unittest.TestCase):
    """The registers inside `[...]` are address registers (base/index), not 'memory'; their read/write
    comes from the ASL and the access itself is the instruction's mem_access."""

    @staticmethod
    def _expl(symbol, enc, encodedin):
        e = ET.Element("explanation"); e.set("enclist", enc)
        ET.SubElement(e, "symbol").text = symbol
        ET.SubElement(e, "account").set("encodedin", encodedin)
        return e

    def _sem(self, mem_access, reads, writes, sp):
        return AslSemantics(mem_access=mem_access, read_regvars=frozenset(reads),
                            written_regvars=frozenset(writes), sp_regvars=frozenset(sp),
                            flags_written=frozenset(), flags_read=frozenset())

    def test_store_base_is_read_not_written(self):
        # STR <Wt>, [<Xn|SP>] : base Xn is READ (address); memory is what's written (mem_access=store)
        expls = [self._expl("<Wt>", "E", "Rt"), self._expl("<Xn|SP>", "E", "Rn")]
        sem = self._sem(MA.STORE, reads={"t", "n"}, writes=set(), sp={"n"})
        ctx = _ctx(decode="let t=UInt(Rt); let n=UInt(Rn);", boxes={"Rt": 5, "Rn": 5})
        ops = build_operands("E", "STR <Wt>, [<Xn|SP>]", expls, sem, ctx)
        base = next(o for o in ops if o.name == "Xn|SP")
        self.assertEqual(base.kind, OperandKind.REG)
        self.assertEqual((base.read, base.write), (True, False))
        self.assertEqual(base.mem_role, MemRole.BASE)
        self.assertEqual(base.reg_range, (0, 31, 1))
        self.assertTrue(base.sp_capable)

    def test_two_memory_regions_each_have_a_base(self):
        # cpy-style: two separate `[...]` regions => each bracket's first register is a base (not index)
        expls = [self._expl("<Xd>", "E", "Rd"), self._expl("<Xs>", "E", "Rs")]
        sem = self._sem(MA.RMW, reads={"d", "s"}, writes=set(), sp=set())
        ctx = _ctx(decode="let d=UInt(Rd); let s=UInt(Rs);", boxes={"Rd": 5, "Rs": 5})
        ops = build_operands("E", "OP [<Xd>], [<Xs>]", expls, sem, ctx)
        roles = {o.name: o.mem_role for o in ops}
        self.assertEqual(roles["Xd"], MemRole.BASE)
        self.assertEqual(roles["Xs"], MemRole.BASE)   # second region is a base, not an index

    def test_load_index_register_is_index_role(self):
        # LDR <Wt>, [<Xn|SP>, <Xm>] : Xm is the index (read), tagged INDEX
        expls = [self._expl("<Wt>", "E", "Rt"), self._expl("<Xn|SP>", "E", "Rn"),
                 self._expl("<Xm>", "E", "Rm")]
        sem = self._sem(MA.LOAD, reads={"n", "m"}, writes={"t"}, sp={"n"})
        ctx = _ctx(decode="let t=UInt(Rt); let n=UInt(Rn); let m=UInt(Rm);",
                   boxes={"Rt": 5, "Rn": 5, "Rm": 5})
        ops = build_operands("E", "LDR <Wt>, [<Xn|SP>, <Xm>]", expls, sem, ctx)
        roles = {o.name: o.mem_role for o in ops}
        self.assertEqual(roles["Xn|SP"], MemRole.BASE)
        self.assertEqual(roles["Xm"], MemRole.INDEX)
        self.assertEqual(next(o for o in ops if o.name == "Xm").read, True)


class TestConstraints(unittest.TestCase):
    def test_pair_overlap(self):
        asl = "if t == t2 then let c : Constraint = ConstrainUnpredictable(Unpredictable_LDPOVERLAP);"
        self.assertEqual(unpredictable_constraints(asl, frozenset({"t", "t2", "n"})), (("t", "t2"),))

    def test_writeback_gated_by_false_flag(self):
        guard = "if wback && n == t && n != 31 then c = ConstrainUnpredictable(Unpredictable_WBOVERLAPLD);"
        rv = frozenset({"t", "n"})
        self.assertEqual(unpredictable_constraints("var wback : boolean = TRUE;\n" + guard, rv), (("n", "t"),))
        self.assertEqual(unpredictable_constraints("var wback : boolean = FALSE;\n" + guard, rv), ())

    def test_exclusive_status_aliasing(self):
        asl = ("if s == t then c = ConstrainUnpredictable(Unpredictable_DATAOVERLAP);\n"
               "if s == n && n != 31 then c = ConstrainUnpredictable(Unpredictable_BASEOVERLAP);")
        self.assertEqual(unpredictable_constraints(asl, frozenset({"s", "t", "n"})), (("n", "s"), ("s", "t")))

    def test_or_condition_yields_both_pairs(self):
        asl = "if s == t || (s == t2) then c = ConstrainUnpredictable(Unpredictable_DATAOVERLAP);"
        self.assertEqual(unpredictable_constraints(asl, frozenset({"s", "t", "t2"})), (("s", "t"), ("s", "t2")))

    def test_preceding_feature_check_not_miscaptured(self):
        asl = ("if !IsFeatureImplemented(FEAT_X) then EndOfDecode(Decode_UNDEF); end;\n"
               "if t == t2 then c = ConstrainUnpredictable(Unpredictable_LDPOVERLAP);")
        self.assertEqual(unpredictable_constraints(asl, frozenset({"t", "t2"})), (("t", "t2"),))


class TestRegisterRange(unittest.TestCase):
    def test_full_5bit_field(self):
        ctx = _ctx(decode="let m : integer = UInt(Rm);", boxes={"Rm": 5})
        self.assertEqual(_register_range("m", "Rm", ctx, "T"), (0, 31, 1))

    def test_narrow_governing_predicate(self):
        ctx = _ctx(decode="let g : integer = UInt(Pg);", boxes={"Pg": 3})
        self.assertEqual(_register_range("g", "Pg", ctx, "T"), (0, 7, 1))   # 3-bit => P0..P7

    def test_no_binding_uses_field_width(self):
        ctx = _ctx(boxes={"Rm": 4})                                          # no UInt() binding
        self.assertEqual(_register_range("m", "Rm", ctx, "T"), (0, 15, 1))

    def test_suffix_bits_give_even_registers(self):
        ctx = _ctx(decode="let t : integer = UInt(Rt::'0');", boxes={"Rt": 4})
        self.assertEqual(_register_range("t", "(Rt)", ctx, "T"), (0, 30, 2))  # even regs only

    def test_fixed_prefix_block(self):
        ctx = _ctx(decode="let m : integer = UInt('011'::Rm);", boxes={"Rm": 2})
        self.assertEqual(_register_range("m", "Rm", ctx, "T"), (12, 15, 1))   # base 0b011_00

    def test_unparseable_binding_loud_fails(self):
        ctx = _ctx(decode="let m : integer = UInt(Rm[2:0]);", boxes={"Rm": 5})
        with self.assertRaises(ExtractionError):
            _register_range("m", "Rm", ctx, "T")


class TestCompositeRegister(unittest.TestCase):
    @staticmethod
    def _expl(symbol, enc, encodedin):
        e = ET.Element("explanation"); e.set("enclist", enc)
        ET.SubElement(e, "symbol").text = symbol
        ET.SubElement(e, "account").set("encodedin", encodedin)
        return e

    def test_composite_index_via_extendreg_is_gp(self):
        # extended-register `<R><m>` (add x,x,w,uxtw): index read via ExtendReg{}(m) => GP, not loud-fail
        sem = AslSemantics(MA.NONE, frozenset({"m", "n"}), frozenset({"d"}), frozenset(), frozenset(), frozenset())
        ctx = _ctx(decode="let m = UInt(Rm);", execute="result = X{d} + ExtendReg{}(m, extend_type, sh);",
                   boxes={"Rd": 5, "Rn": 5, "Rm": 5})
        ops = build_operands("E", "OP <Xd>, <Xn>, <R><m>", [self._expl("<Xd>", "E", "Rd"),
              self._expl("<Xn>", "E", "Rn"), self._expl("<m>", "E", "Rm")], sem, ctx)
        m = next(o for o in ops if o.name == "m")
        self.assertEqual((m.kind, m.reg_file), (OperandKind.REG, RegFile.GP))

    def test_R_t_composite_is_one_gp_register(self):
        # TBZ <R><t>: <R> is the width selector (folded in), <t> is a GP register read as X{datasize}(t)
        sem = AslSemantics(MA.NONE, frozenset({"t"}), frozenset(), frozenset(), frozenset(), frozenset())
        ctx = _ctx(decode="let t = UInt(Rt);", execute="when X{datasize}(t) == '0' BranchTo();",
                   boxes={"Rt": 5})
        ops = build_operands("E", "OP <R><t>", [self._expl("<t>", "E", "Rt")], sem, ctx)
        self.assertEqual([o.name for o in ops], ["t"])     # <R> selector is not a separate operand
        self.assertEqual(ops[0].kind, OperandKind.REG)
        self.assertEqual(ops[0].reg_file, RegFile.GP)
        self.assertEqual(ops[0].reg_range, (0, 31, 1))
        self.assertTrue(ops[0].read)


# The ARM ISA XML is needed for these; skip cleanly if the local cache is absent.
_XML_DIR = os.path.expanduser("~/.cache/arm_isa_parser/ISA_A64_xml_A_profile-2025-09_ASL1")


@unittest.skipUnless(os.path.isdir(_XML_DIR), "ARM ISA XML cache not present")
class TestImportantInstructions(unittest.TestCase):
    """Pin the parse of control-flow, conditional, and memory-access instructions against the real XML."""
    byname: dict = {}

    @classmethod
    def setUpClass(cls):
        import glob
        from src.aarch64.arm_isa_extractor.extract import iter_instructions
        from src.aarch64.arm_isa_extractor.immediate import architectural_constants
        ac = architectural_constants(os.path.join(_XML_DIR, "shared_pseudocode.xml"))
        for p in sorted(glob.glob(os.path.join(_XML_DIR, "*.xml"))):
            if p.endswith("shared_pseudocode.xml"):
                continue
            for i in iter_instructions(p, ac, {}):
                cls.byname.setdefault(i.name, i)

    def _i(self, name):
        self.assertIn(name, self.byname, f"{name!r} was not extracted")
        return self.byname[name]

    def _kinds(self, name):
        return {o.name: o for o in self._i(name).operands}

    def test_control_flow_flagged(self):
        for n in ("b", "bl", "br", "blr", "ret", "cbz", "cbnz", "tbz", "tbnz", "b."):
            self.assertTrue(self._i(n).control_flow, f"{n} should be control flow")

    def test_conditional_branch_reads_nzcv(self):
        self.assertEqual(set(self._i("b.").flags.read), set("NZCV"))

    def test_csel_reads_nzcv_ccmp_reads_and_writes(self):
        self.assertEqual(set(self._i("csel").flags.read), set("NZCV"))
        self.assertEqual(set(self._i("ccmp").flags.read), set("NZCV"))
        self.assertEqual(set(self._i("ccmp").flags.written), set("NZCV"))

    def test_cbz_tbz_register_is_read_register(self):
        for n in ("cbz", "tbz"):
            reg = next(o for o in self._i(n).operands if o.kind is OperandKind.REG)
            self.assertTrue(reg.read and not reg.write, f"{n} test register must be read-only")

    def test_memory_access_kinds(self):
        for n, kind in (("ldr", "load"), ("str", "store"), ("ldp", "load"), ("stp", "store"),
                        ("ldxr", "ex-load"), ("stxr", "ex-store"), ("cas", "rmw"),
                        ("swp", "rmw"), ("ldadd", "rmw"), ("prfm", "prefetch")):
            self.assertEqual(self._i(n).mem_access.value, kind, f"{n} mem_access")

    def test_store_base_register_is_read_not_written(self):
        base = next(o for o in self._i("str").operands if o.mem_role is MemRole.BASE)
        self.assertTrue(base.read, "store base register must be read (address)")

    def test_adr_target_is_label(self):
        self.assertTrue(any(o.kind is OperandKind.LABEL for o in self._i("adr").operands))

    def test_ldp_two_destinations_must_differ(self):
        self.assertIn(("t", "t2"), self._i("ldp").constraints)

    def test_mnemonic_strips_optional_suffix(self):
        # `SQDMULL{2}` -> base mnemonic sqdmull (the {2} stays in asm_template, not the name)
        self.assertIn("sqdmull", self.byname)
        self.assertTrue(all("{" not in n and "<" not in n for n in self.byname))

    def test_sve_structured_load_store_recovered(self):
        ld = self._i("ld2b"); st = self._i("st2b")
        self.assertTrue(all(o.write for o in ld.operands if o.reg_file and o.reg_file.value == "sve_z"))
        self.assertTrue(all(o.read and not o.write for o in st.operands if o.reg_file and o.reg_file.value == "sve_z"))

    def test_pac_autia_extracted(self):
        # PAC was dropped because instr-class sits at the iclass (not section) level; must be present
        i = self._i("autia")
        xd = next(o for o in i.operands if o.kind is OperandKind.REG)
        self.assertTrue(xd.read and xd.write)   # pointer authenticated in place


class TestEnumValueFiltering(unittest.TestCase):
    def test_enum_value_keeps_numbers_drops_templates_and_placeholders(self):
        self.assertEqual(_enum_value("8B"), "8b")
        self.assertEqual(_enum_value("#90"), "90")        # enumerated integer immediate
        self.assertEqual(_enum_value("#0.5"), "0.5")      # enumerated FP immediate
        self.assertEqual(_enum_value("#-1"), "-1")
        self.assertIsNone(_enum_value("#uimm5"))          # immediate-syntax template
        self.assertIsNone(_enum_value("RESERVED"))        # unallocated-row marker
        self.assertIsNone(_enum_value("<T>"))             # sub-token

    def test_enumerated_table_filters_correctly(self):
        d = ET.fromstring('<definition><table>'
                          '<row><entry class="symbol">&lt;const&gt;</entry></row>'
                          '<row><entry class="symbol">#90</entry></row>'
                          '<row><entry class="symbol">#270</entry></row>'
                          '<row><entry class="symbol">RESERVED</entry></row></table></definition>')
        self.assertEqual(_enumerated("<const>", d).values, ("90", "270"))


class TestValidation(unittest.TestCase):
    def _reg(self, **kw):
        d = dict(name="Xn", kind=OperandKind.REG, read=True, write=False, width=64, signed=False,
                 reg_file=RegFile.GP, asl_index="n", reg_range=(0, 31, 1))
        d.update(kw)
        return Operand(**d)

    def _inst(self, operands, mem_access=MA.NONE, constraints=()):
        return Instruction(name="x", iclass_id="x", category="general", encoding_name="E",
                           asm_template="", control_flow=False, mem_access=mem_access,
                           flags=FlagEffects(frozenset(), frozenset()), operands=tuple(operands),
                           constraints=constraints)

    def test_valid_instruction_passes(self):
        validate_instruction(self._inst([self._reg()]))

    def test_reg_range_out_of_31_loud_fails(self):
        with self.assertRaises(ExtractionError):
            validate_instruction(self._inst([self._reg(reg_range=(0, 63, 1))]))

    def test_reg_width_must_match_file(self):
        with self.assertRaises(ExtractionError):
            validate_instruction(self._inst([self._reg(width=16)]))   # 16 invalid for GP

    def test_memory_access_needs_base_or_label(self):
        with self.assertRaises(ExtractionError):   # load but the only reg has mem_role NONE
            validate_instruction(self._inst([self._reg()], mem_access=MA.LOAD))

    def test_non_memory_has_no_addressing_role(self):
        with self.assertRaises(ExtractionError):
            validate_instruction(self._inst([self._reg(mem_role=MemRole.BASE)], mem_access=MA.NONE))

    def test_immediate_not_both_range_and_values(self):
        bad = Operand(name="i", kind=OperandKind.IMM, read=True, write=False, width=4, signed=False,
                      values=("x",), imm_ranges=((0, 0, 15, 1),))
        with self.assertRaises(ExtractionError):
            validate_instruction(self._inst([bad]))

    def test_imm_ranges_no_mixed_esize_zero(self):
        bad = Operand(name="i", kind=OperandKind.IMM, read=True, write=False, width=4, signed=False,
                      imm_ranges=((0, 0, 1, 1), (8, 1, 8, 1)))
        with self.assertRaises(ExtractionError):
            validate_instruction(self._inst([bad]))


class TestWriteback(unittest.TestCase):
    def test_writeback_targets_pre_post_standalone(self):
        self.assertEqual(_writeback_targets("LDR <Xt>, [<Xn|SP>, #<simm>]!"), {"Xn|SP"})   # pre-index
        self.assertEqual(_writeback_targets("LDR <Xt>, [<Xn|SP>], #<simm>"), {"Xn|SP"})    # post-index
        self.assertEqual(_writeback_targets("CPYFP [<Xd>]!, [<Xs>]!, <Xn>!"), {"Xd", "Xs", "Xn"})

    def test_no_writeback_for_offset_or_regoffset(self):
        self.assertEqual(_writeback_targets("LDR <Xt>, [<Xn|SP>{, #<pimm>}]"), set())
        self.assertEqual(_writeback_targets("LDR <Xt>, [<Xn|SP>, <Xm>]"), set())


class TestCompositePairDetection(unittest.TestCase):
    def test_adjacent_tokens_are_composite(self):
        self.assertEqual(_composite_pair("TBZ <R><t>, #<imm>, <label>"), ({"<R>"}, {"<t>"}))

    def test_separated_tokens_are_not_composite(self):
        self.assertEqual(_composite_pair("ADD <Wd>, <Wn>, <Wm>"), (set(), set()))


class TestAslRegexes(unittest.TestCase):
    def test_mem_access_classification(self):
        f = lambda a: extract_asl_semantics(a).mem_access
        self.assertEqual(f("x = MemAtomic{}(a, b, c, acc);"), MA.RMW)
        self.assertEqual(f("d = CreateAccDescGPR(MemOp_LOAD, x); v = Mem{8}(addr, d);"), MA.LOAD)
        self.assertEqual(f("d = CreateAccDescGPR(MemOp_STORE, x); Mem{8}(addr, d) = v;"), MA.STORE)
        self.assertEqual(f("d = CreateAccDescExLDA(MemOp_LOAD, x);"), MA.EX_LOAD)
        self.assertEqual(f("y = Prefetch(addr, op);"), MA.PREFETCH)
        self.assertEqual(f("X{}(d) = X{}(n) + X{}(m);"), MA.NONE)

    def test_reg_read_write_expression_struct_helper(self):
        s = extract_asl_semantics(
            "X{datasize}(t) = X{}(n) + ExtendReg{}(m, e, sh);\noffset = X{64}(memcpy.d);")
        self.assertEqual(s.written_regvars, frozenset({"t"}))
        self.assertTrue({"n", "m", "d"} <= s.read_regvars)   # plain / ExtendReg helper / struct index

    def test_shiftreg_and_extendreg_with_and_without_braces(self):
        s = extract_asl_semantics("a = ShiftReg{}(p, ty, am); b = ExtendReg(q, ty, am); X{}(d) = a + b;")
        self.assertTrue({"p", "q"} <= s.read_regvars)

    def test_sp_capable_only_when_31_means_sp(self):
        self.assertIn("n", extract_asl_semantics(
            "if n == 31 then addr = SP{64}(); else addr = X{64}(n); end;").sp_regvars)
        self.assertNotIn("m", extract_asl_semantics("addr = X{64}(m);").sp_regvars)

    def test_nested_paren_index_read_vs_write(self):
        # SVE structured load `Z{VL}((t+r) MOD 32) = ...` is a WRITE of t; the store form is a READ
        load = extract_asl_semantics("Z{VL}((t + r) MOD 32) = data;")
        self.assertIn("t", load.written_regvars)
        self.assertNotIn("t", load.read_regvars)
        store = extract_asl_semantics("values[r] = Z{VL}((t + r) MOD 32);")
        self.assertIn("t", store.read_regvars)
        self.assertNotIn("t", store.written_regvars)

    def test_flag_group_single_read_and_condition(self):
        self.assertEqual(extract_asl_semantics("PSTATE.[N,Z,C,V] = nzcv;").flags_written, frozenset("NZCV"))
        cfinv = extract_asl_semantics("PSTATE.C = NOT(PSTATE.C);")
        self.assertEqual((cfinv.flags_written, cfinv.flags_read), (frozenset("C"), frozenset("C")))
        self.assertEqual(extract_asl_semantics("if ConditionHolds(c) then x; end;").flags_read, frozenset("NZCV"))


class TestLabelAndConstFields(unittest.TestCase):
    def test_label_is_pc_relative(self):
        branch = _ctx(decode="let offset : bits(64) = SignExtend{}(imm19::'00');",
                      execute="BranchTo{64}(PC64() + offset, BranchType_DIR, c);")
        self.assertTrue(_is_label("imm19", branch))

    def test_memory_offset_is_not_label(self):
        mem = _ctx(decode="let offset : bits(64) = SignExtend{}(imm9);",
                   execute="address = AddressAdd(address, offset, accdesc);")
        self.assertFalse(_is_label("imm9", mem))

    def test_constant_fields_merge(self):
        # iclass pins size[1]=1, leaves size[0] variable; encoding pins size[0]=1 => 0b11 = 3
        iclass = ET.fromstring('<regdiagram><box name="size" width="2"><c>1</c><c>x</c></box></regdiagram>')
        enc = ET.fromstring('<encoding><box name="size" width="2"><c></c><c>1</c></box></encoding>')
        self.assertEqual(constant_fields(iclass, enc), {"size": 3})


if __name__ == "__main__":
    unittest.main()
