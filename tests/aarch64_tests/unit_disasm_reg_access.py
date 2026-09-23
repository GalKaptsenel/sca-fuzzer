"""decode_reg_accesses must report every register/flag a generated instruction reads/writes, and must
NOT invent a write it does not perform. Both directions corrupt taint: a missed read (or a missed write
mislabeled as a read) is the taint-SAFE direction (it over-preserves), but a spurious WRITE marks that
operand's input dead so boosting mutates it -> false violation (the SETF16-writes-C bug: SETF8/SETF16
preserve C, RMIF writes only its mask-selected bits, yet Capstone/older code claimed all of NZCV).

Register-operand roles are NO LONGER guessed here: they are taken from the ISA objects (base.json ->
InstructionSpec/Instruction, each operand carrying src/dest) via build_register_role_map, and
decode_reg_accesses applies them verbatim. So an RMW register (CAS's Rs, read AND written), a pure
destination (an LSE LD<op>'s Rt), and a plain source are each categorised from the spec. Capstone still
supplies the register names/order, the memory base, and the flag reads/writes (their per-bit set depends
on the condition code / mask immediate, which no static role captures).

Regression coverage so the categorisation cannot silently break again:
  * CASES pin src/dest (incl. RMW/write/read and flags) for one representative per family.
  * test_flag_writes_are_exact keeps NZCV writes EXACT (the false-positive guard).
  * test_decode_agrees_with_capstone proves, on real encodings, that the map + operand-order zip agree
    with Capstone wherever Capstone reports op.access (an independent oracle for the non-atomic ISA).
  * test_role_map_matches_base_json checks build_register_role_map reproduces base.json's authoritative
    per-operand roles for every loaded spec (and fails loud on a role conflict).
  * test_atomic_and_cas_roles_match_base_json cross-checks the LSE atomic/CAS/CASP roles (incl. the
    blocklisted CAS/CASP, read from raw base.json) -- the empty-op.access families Capstone cannot help.
  * test_every_supported_instruction_covered ensures every generatable mnemonic has a role source.
"""
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # run from any cwd
_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
import capstone
from capstone import CS_AC_READ, CS_AC_WRITE
from capstone.arm64 import ARM64_OP_REG
from src.aarch64.aarch64_disasm import (decode_reg_accesses, is_conditional_branch,
                                        build_register_role_map)
from src.aarch64.aarch64_config import supported_instructions
from src.config import CONF
from src.isa_loader import InstructionSet
from src.interfaces import OT

_MD = capstone.Cs(capstone.CS_ARCH_ARM64, capstone.CS_MODE_LITTLE_ENDIAN)
_MD.detail = True

# Categories broad enough to load every family the tests exercise (base + atomics + acqrel + MTE + PAC).
_TEST_CATEGORIES = ["BASE-ARITH", "BASE-LOGICAL", "BASE-SHIFT", "BASE-BITFIELD", "BASE-CONDSEL",
                    "BASE-BRANCH", "BASE-MEM-LOAD", "BASE-MEM-STORE", "BASE-FLAGOP", "BASE-BITCOUNT",
                    "BASE-CRC", "BASE-BITBYTE", "BASE-MEM-ATOMIC", "BASE-MEM-ACQREL",
                    "MTE-TAGMEM", "MTE-ARITH", "MTE-BASE", "PAC-SIGN", "PAC-STRIP", "PAC-AUTH"]

# Built once (setUpModule) from the loaded ISA: the authoritative register-role map the executor also
# builds, plus the spec list and a raw-base.json fallback for blocklisted specs (CAS/CASP).
_ROLES = None
_ISA_BY_NAME = None
_RAW_BY_NAME = None
_SAVED_CONF = None


def setUpModule():
    global _ROLES, _ISA_BY_NAME, _RAW_BY_NAME, _SAVED_CONF
    _SAVED_CONF = copy.deepcopy(CONF._borg_shared_state)
    CONF.load(os.path.join(_ROOT, "config.yml"))
    isa = InstructionSet(os.path.join(_ROOT, "base.json"), _TEST_CATEGORIES)
    _ROLES = build_register_role_map(isa.instructions)
    _ISA_BY_NAME = {}
    for spec in isa.instructions:
        _ISA_BY_NAME.setdefault(spec.name.lower(), []).append(spec)
    with open(os.path.join(_ROOT, "base.json")) as f:
        raw = json.load(f)
    raw = raw["instructions"] if isinstance(raw, dict) and "instructions" in raw else raw
    raw = list(raw.values()) if isinstance(raw, dict) else raw
    _RAW_BY_NAME = {}
    for i in raw:
        _RAW_BY_NAME.setdefault((i.get("name") or "").lower(), []).append(i)


def tearDownModule():
    CONF._borg_shared_state.clear()
    CONF._borg_shared_state.update(_SAVED_CONF)


def _asm(line, march="armv9-a+memtag"):
    """Assemble one instruction and return its 32-bit little-endian encoding."""
    with tempfile.NamedTemporaryFile("w", suffix=".s", delete=False) as f:
        f.write(".text\n" + line + "\n")
        path = f.name
    obj = path + ".o"
    subprocess.run(["aarch64-linux-gnu-as", "-march=" + march, path, "-o", obj], check=True)
    raw = subprocess.run(["aarch64-linux-gnu-objcopy", "-O", "binary", "-j", ".text", obj,
                          "/dev/stdout"], capture_output=True, check=True).stdout
    os.unlink(path)
    os.unlink(obj)
    return int.from_bytes(raw[:4], "little")


# mnemonic, encoding, required src (reads), required dest (writes). Flag writes ("N"/"Z"/"C"/"V") are
# checked EXACTLY by test_flag_writes_are_exact; register dests and all reads use a subset check.
CASES = [
    # FEAT_FlagM/FlagM2 flag ops Capstone under-reports -- PARTIAL flag writers (kept precise here).
    ("setf8",  0x3a00080d, {"w0"},          {"N", "Z", "V"}),        # C preserved
    ("setf16", 0x3a00480d, {"w0"},          {"N", "Z", "V"}),        # C preserved
    ("rmif",   0xba000421, {"x1"},          {"V"}),                  # mask 0b0001 -> V only
    ("rmif",   0xba000424, {"x1"},          {"Z"}),                  # mask 0b0100 -> Z only
    ("rmif",   0xba018446, {"x2"},          {"Z", "C"}),             # mask 0b0110 -> Z,C
    ("rmif",   0xba00042f, {"x1"},          {"N", "Z", "C", "V"}),   # mask 0b1111 -> all
    ("cfinv",  0xd500401f, {"C"},           {"C"}),                  # C := NOT C
    ("axflag", 0xd500405f, {"Z", "C", "V"}, {"N", "Z", "C", "V"}),   # ARM -> alt FP flag format
    ("xaflag", 0xd500403f, {"C", "Z"},      {"N", "Z", "C", "V"}),   # alt -> ARM FP flag format
    ("pacga",  0x9ac23020, {"x1", "x2"},    {"x0"}),                 # Xm 2nd source (map supplies it)
    # flag readers/writers handled via cc / update_flags ----------------------------
    ("ccmp",   0xfa420020, {"x1", "x2", "Z"}, {"N", "Z", "C", "V"}),
    ("ccmn",   0xba420020, {"x1", "x2", "Z"}, {"N", "Z", "C", "V"}),
    ("adds",   0xab020020, {"x1", "x2"},    {"x0", "N", "Z", "C", "V"}),
    ("subs",   0xeb020020, {"x1", "x2"},    {"x0", "N", "Z", "C", "V"}),
    ("ands",   0xea020020, {"x1", "x2"},    {"x0", "N", "Z", "C", "V"}),
    ("bics",   0xea220020, {"x1", "x2"},    {"x0", "N", "Z", "C", "V"}),
    ("csel",   0x9a820020, {"x1", "x2", "Z"}, {"x0"}),
    ("csinc",  0x9a820420, {"x1", "x2", "Z"}, {"x0"}),
    ("csinv",  0x5a820020, {"w1", "w2", "Z"}, {"w0"}),
    ("csneg",  0x5a820420, {"w1", "w2", "Z"}, {"w0"}),
    # explicit-operand-only controls (one per family) -------------------------------
    ("pacia",  0xdac10020, {"x0", "x1"},    {"x0"}),
    ("pacda",  0xdac10820, {"x0", "x1"},    {"x0"}),
    ("xpacd",  0xdac147e0, {"x0"},          {"x0"}),
    ("and",    0x8a020020, {"x1", "x2"},    {"x0"}),
    ("ldr",    0xf9400020, {"x1"},          {"x0"}),
    ("str",    0xf9000020, {"x0", "x1"},    set()),
    ("ldp",    0xa9400440, {"x2"},          {"x0", "x1"}),
    ("cbz",    0x340000a0, {"w0"},          set()),
    ("b.eq",   0x54000000, {"Z"},           set()),
    # MTE tag stores: base register read; no register dest.
    ("stg",    0xd9200820, {"x1"},          set()),
    ("st2g",   0xd9a00862, {"x3"},          set()),
    ("stzg",   0xd96008a4, {"x5"},          set()),
    ("stz2g",  0xd9e008e6, {"x7"},          set()),
    # MTE tag arithmetic / load (empty op.access -> roles from the ISA map). SUBPS also writes NZCV;
    # LDG is RMW (Xt read+written).
    ("addg",   0x91800420, {"x1"},          {"x0"}),
    ("subg",   0xd1800420, {"x1"},          {"x0"}),
    ("gmi",    0x9ac21420, {"x1", "x2"},    {"x0"}),
    ("subp",   0x9ac20020, {"x1", "x2"},    {"x0"}),
    ("subps",  0xbac20020, {"x1", "x2"},    {"x0", "N", "Z", "C", "V"}),
    ("ldg",    0xd9600020, {"x0", "x1"},    {"x0"}),
    # LSE atomic RMW load forms LD<op> Rs, Rt, [Xn] / SWP (empty op.access -> roles from the ISA map):
    # Rs read, Rt written (old value), Xn read; no NZCV write.
    ("ldadd",  0xf8210062, {"x1", "x3"},    {"x2"}),
    ("ldaddb", 0x38210062, {"w1", "x3"},    {"w2"}),
    ("ldaddal", 0xf8e10062, {"x1", "x3"},   {"x2"}),
    ("ldclr",  0xf8211062, {"x1", "x3"},    {"x2"}),
    ("ldeor",  0xf8212062, {"x1", "x3"},    {"x2"}),
    ("ldset",  0xf8213062, {"x1", "x3"},    {"x2"}),
    ("ldsmax", 0xf8214062, {"x1", "x3"},    {"x2"}),
    ("ldsmin", 0xf8215062, {"x1", "x3"},    {"x2"}),
    ("ldumax", 0xf8216062, {"x1", "x3"},    {"x2"}),
    ("ldumin", 0xf8217062, {"x1", "x3"},    {"x2"}),
    ("swp",    0xf8218062, {"x1", "x3"},    {"x2"}),
    ("swpalh", 0x78e480c5, {"w4", "x6"},    {"w5"}),
    # Acquire/release ordered accesses: Capstone reports these correctly. Pin one load, one store.
    ("ldar",   0xc8dffc20, {"x1"},          {"x0"}),
    ("ldapr",  0xf8bfc020, {"x1"},          {"x0"}),
    ("stlr",   0xc89ffc20, {"x0", "x1"},    set()),
    # MADD/MSUB: Rd write, Rn/Rm/Ra read.
    ("madd",   0x9b020c20, {"x1", "x2", "x3"}, {"x0"}),
    ("msub",   0x9b028c20, {"x1", "x2", "x3"}, {"x0"}),
]

# Register operands whose src/dest exactly match these -- pinned so an RMW/write/read cannot silently
# flip. (mnemonic, encoding) -> (exact register src set, exact register dest set), flags excluded.
EXACT_REG_CASES = [
    ("ldadd",  0xf8210062, {"x1", "x3"}, {"x2"}),        # Rs read, Rt write, base read
    ("swp",    0xf8218062, {"x1", "x3"}, {"x2"}),
    ("ldg",    0xd9600020, {"x0", "x1"}, {"x0"}),        # Xt read+written (RMW)
    ("addg",   0x91800420, {"x1"},       {"x0"}),        # Xd write, Xn read
    ("pacga",  0x9ac23020, {"x1", "x2"}, {"x0"}),        # both sources reported
]

# Acquire/release forms Capstone reports correctly; no register-role fixup needed.
EXPLICIT_OPERAND_ONLY = {
    "autia", "autib", "autiza", "autizb", "autda", "autdb", "autdza", "autdzb",
    "b", "cbnz", "cls", "clz",
    "crc32b", "crc32cb", "crc32ch", "crc32cw", "crc32cx", "crc32h", "crc32w", "crc32x",
    "eor", "orr", "pacib", "paciza", "pacizb", "pacdb", "pacdza", "pacdzb",
    "rbit", "rev", "rev16", "rev32",
    "sdiv", "stp", "tbnz", "tbz", "udiv", "xpaci",
    "ldar", "ldarb", "ldarh", "ldapr", "ldaprb", "ldaprh", "ldlar", "ldlarb", "ldlarh",
    "stlr", "stlrb", "stlrh", "stllr", "stllrb", "stllrh",
}

NO_OPERAND_BARRIERS = {"ssbb", "pssbb"}

# LSE atomic RMW family mnemonics (op x order x size), and CAS/CASP, enumerated exactly so prefix
# matching cannot leak in the FEAT_LSE128 *p pair forms (ldclrp/ldsetp/swpp) or unrelated names.
_LSE_OPS = ("add", "clr", "eor", "set", "smax", "smin", "umax", "umin")
_LSE_ORDER = ("", "a", "l", "al")
_LSE_SIZE = ("", "b", "h")
_LDOP_SWP_NAMES = frozenset(
    [f"ld{op}{o}{s}" for op in _LSE_OPS for o in _LSE_ORDER for s in _LSE_SIZE]
    + [f"swp{o}{s}" for o in _LSE_ORDER for s in _LSE_SIZE])
_CAS_NAMES = frozenset(f"cas{o}{s}" for o in _LSE_ORDER for s in _LSE_SIZE)
_CASP_NAMES = frozenset(f"casp{o}" for o in _LSE_ORDER)

# Capstone 5.0.x reports WRONG (not merely empty) op.access for these, so it is not a trustworthy oracle
# for them -- they are validated against base.json instead. pacga: reports Xd as a read (it is written).
# The MTE arith/tag and LSE atomic families report empty access (auto-skipped by the oracle test), but
# are listed here too for clarity.
_CAPSTONE_UNRELIABLE = ({"pacga", "addg", "subg", "irg", "gmi", "subp", "subps", "ldg",
                         "setf8", "setf16", "rmif", "cfinv", "axflag", "xaflag"}
                        | _LDOP_SWP_NAMES | _CAS_NAMES | _CASP_NAMES)


class DisasmRegAccessTest(unittest.TestCase):
    def test_disassembly_matches_expected_mnemonic(self):
        for mnemonic, encoding, _s, _d in CASES:
            insn = next(_MD.disasm(encoding.to_bytes(4, "little"), 0), None)
            self.assertIsNotNone(insn, f"{mnemonic}: 0x{encoding:08x} did not decode")
            self.assertEqual(insn.mnemonic, mnemonic,
                             f"0x{encoding:08x} decoded as {insn.mnemonic}, expected {mnemonic}")

    def test_no_under_reporting(self):
        for mnemonic, encoding, req_src, req_dest in CASES:
            src, dest = decode_reg_accesses(encoding, 0, _ROLES)
            self.assertLessEqual(req_src, set(src), f"{mnemonic}: missing source(s) {req_src - set(src)}")
            self.assertLessEqual(req_dest, set(dest), f"{mnemonic}: missing dest(s) {req_dest - set(dest)}")

    def test_register_roles_are_exact(self):
        # RMW / write / read register operands pinned exactly (flags excluded): a flipped role here is
        # the class of bug this whole path exists to prevent.
        FLAGS = {"N", "Z", "C", "V"}
        for mnemonic, encoding, exp_src, exp_dest in EXACT_REG_CASES:
            src, dest = decode_reg_accesses(encoding, 0, _ROLES)
            self.assertEqual(set(s for s in src if s not in FLAGS), exp_src, f"{mnemonic}: src regs")
            self.assertEqual(set(d for d in dest if d not in FLAGS), exp_dest, f"{mnemonic}: dest regs")

    def test_flag_writes_are_exact(self):
        # A flag write claimed but not performed drops that flag's live taint -> false violation.
        FLAGS = {"N", "Z", "C", "V"}
        for mnemonic, encoding, _req_src, req_dest in CASES:
            _, dest = decode_reg_accesses(encoding, 0, _ROLES)
            self.assertEqual(req_dest & FLAGS, set(dest) & FLAGS,
                             f"{mnemonic} 0x{encoding:08x}: flag writes {sorted(set(dest) & FLAGS)} "
                             f"!= expected {sorted(req_dest & FLAGS)}")

    def test_decode_agrees_with_capstone(self):
        # Independent oracle: wherever Capstone reports op.access for a register operand, the
        # map-driven decode must agree on that operand's read/write role. This proves the ISA role map
        # and the operand-order zip are correct for every instruction Capstone can vouch for (the
        # atomic/MTE families, which Capstone leaves empty, are checked against base.json separately).
        for mnemonic, encoding, _s, _d in CASES:
            if mnemonic in _CAPSTONE_UNRELIABLE:
                continue
            insn = next(_MD.disasm(encoding.to_bytes(4, "little"), 0))
            src, dest = decode_reg_accesses(encoding, 0, _ROLES)
            for op in insn.operands:
                if op.type != ARM64_OP_REG:
                    continue
                reg = insn.reg_name(op.reg)
                if op.access & CS_AC_READ:
                    self.assertIn(reg, src, f"{mnemonic}: Capstone reads {reg}, decode did not")
                if op.access & CS_AC_WRITE:
                    self.assertIn(reg, dest, f"{mnemonic}: Capstone writes {reg}, decode did not")
                # Never invent a write Capstone (which is authoritative when populated) does not report.
                if op.access & (CS_AC_READ | CS_AC_WRITE) and not (op.access & CS_AC_WRITE):
                    self.assertNotIn(reg, dest,
                                     f"{mnemonic}: decode invented a write to {reg} (Capstone: read-only)")

    def test_every_supported_instruction_covered(self):
        # Every generatable mnemonic must have an authoritative role source: a register-operand entry
        # in the ISA map (so decode categorises it from the spec), or it is a control/branch/barrier
        # with no register operands (Capstone/flag logic suffices).
        mapped = {m for (m, _n) in _ROLES}
        for mnemonic in supported_instructions:
            has_reg_form = any(any(o.type == OT.REG for o in s.operands)
                               for s in _ISA_BY_NAME.get(mnemonic, []))
            covered = (mnemonic in mapped
                       or mnemonic in EXPLICIT_OPERAND_ONLY
                       or mnemonic in NO_OPERAND_BARRIERS
                       or not has_reg_form)
            self.assertTrue(covered, f"{mnemonic!r} is generated but has no register-role source")


class RoleMapCrossCheckTest(unittest.TestCase):
    """build_register_role_map must reproduce base.json's authoritative per-operand roles, so the map
    the executor feeds to taint cannot drift from the ISA."""

    def test_role_map_matches_base_json(self):
        for name, specs in _ISA_BY_NAME.items():
            for spec in specs:
                regs = [op for op in spec.operands if op.type == OT.REG]
                if not regs:
                    continue
                key = (name, len(regs))
                self.assertIn(key, _ROLES, f"{key}: missing from the role map")
                expected = tuple((bool(op.src), bool(op.dest)) for op in regs)
                self.assertEqual(_ROLES[key], expected, f"{key}: role map disagrees with spec")

    def test_atomic_and_cas_roles_match_base_json(self):
        # The LSE atomic/CAS/CASP families are the ones Capstone leaves op.access empty for, so their
        # roles come entirely from the ISA. Verify against raw base.json (CAS/CASP are blocklisted, so
        # absent from the loaded set; read them raw). Rs read, Rt written for LD<op>/SWP; Rs read+written
        # for CAS; Rs pair read+written for CASP.
        def raw_reg_roles(entry):
            regs = [o for o in entry.get("operands", [])
                    if (o.get("values") or [""])[0][:1] in ("x", "w")]
            return [(bool(o.get("src")), bool(o.get("dest"))) for o in regs]

        expected = {}
        for name in _LDOP_SWP_NAMES:
            expected[name] = [(True, False), (False, True)]              # Rs read, Rt written
        for name in _CAS_NAMES:
            expected[name] = [(True, True), (True, False)]               # Rs read+written, Rt read
        for name in _CASP_NAMES:
            expected[name] = [(True, True), (True, True), (True, False), (True, False)]  # pair

        checked = 0
        for name, exp in expected.items():
            for entry in _RAW_BY_NAME.get(name, []):
                self.assertEqual(raw_reg_roles(entry), exp, f"{name}: base.json roles")
                checked += 1
        self.assertGreater(checked, 100, f"too few atomic specs cross-checked ({checked})")


class EmptyAccessFallbackTest(unittest.TestCase):
    def test_no_map_over_approximates_as_read(self):
        # Without a role map (e.g. the display logger), an operand Capstone leaves access-empty is
        # over-approximated as a read -- the taint-SAFE direction (a missed write over-preserves; it
        # never invents a write). GMI's registers are all empty-access in Capstone 5.0.x.
        src, dest = decode_reg_accesses(0x9ac21420, 0, None)   # gmi x0, x1, x2
        self.assertEqual(set(src), {"x0", "x1", "x2"})
        self.assertEqual(set(dest), set())

    def test_no_map_never_invents_a_write(self):
        # The safety invariant across the whole ISA: with no map, decode must never report a register
        # WRITE that the spec does not have (that is the false-violation direction).
        for mnemonic, encoding, _s, _d in CASES:
            _, dest_nomap = decode_reg_accesses(encoding, 0, None)
            _, dest_map = decode_reg_accesses(encoding, 0, _ROLES)
            reg_writes_nomap = {d for d in dest_nomap if d not in ("N", "Z", "C", "V")}
            reg_writes_map = {d for d in dest_map if d not in ("N", "Z", "C", "V")}
            self.assertLessEqual(reg_writes_nomap, reg_writes_map,
                                 f"{mnemonic}: no-map decode invented register write(s) "
                                 f"{reg_writes_nomap - reg_writes_map}")


class FlagWriterCoverageTest(unittest.TestCase):
    def test_every_flag_writer_has_an_exact_case(self):
        # Every generated NZCV writer must have a CASE so test_flag_writes_are_exact pins its per-flag
        # write set (a partial writer like SETF/RMIF must not ship with an over-claimed flag write).
        case_names = {c[0] for c in CASES}
        for name in supported_instructions:
            writes_flags = any(op.type == OT.FLAGS and op.dest
                               for spec in _ISA_BY_NAME.get(name, [])
                               for op in spec.implicit_operands)
            if writes_flags:
                self.assertIn(name, case_names,
                              f"{name} writes NZCV (base.json) but has no exact-flag case in CASES")


class IsConditionalBranchTest(unittest.TestCase):
    def test_bcond_is_conditional(self):
        self.assertTrue(is_conditional_branch(0x54000040))   # B.eq +8
        self.assertTrue(is_conditional_branch(0x5400004B))   # B.lt +8
        self.assertTrue(is_conditional_branch(0xB4000040))   # CBZ  x0, +8
        self.assertTrue(is_conditional_branch(0x36000040))   # TBZ  w0, #0, +8

    def test_bal_bnv_are_not_conditional(self):
        self.assertFalse(is_conditional_branch(0x5400004E))  # B.al +8
        self.assertFalse(is_conditional_branch(0x5400004F))  # B.nv +8


if __name__ == "__main__":
    unittest.main()
