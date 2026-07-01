"""decode_reg_accesses must report every register/flag a generated instruction reads/writes -- under-
reporting hides a data dependency and can turn a real leak into a false negative. Capstone (5.0.x)
under-reports some implicit accesses (rmif/setf8/setf16 expose nothing; pacga drops its 2nd source),
which decode_reg_accesses compensates for.

Each case lists the ARM-defined accesses that MUST appear (subset check -- over-reporting a source is
the safe direction and is allowed). Encodings are confirmed to disassemble to the expected mnemonic so
a wrong encoding fails loudly instead of testing nothing. test_every_supported_instruction_is_classified
forces a case (or an explicit op.access-only classification) for every instruction the fuzzer emits, so
a newly-added instruction with hidden accesses cannot slip through unchecked."""
import copy
import unittest
from unittest.mock import patch

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # run from any cwd
_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
import capstone
import src.aarch64.aarch64_disasm as disasm
from src.aarch64.aarch64_disasm import decode_reg_accesses, is_conditional_branch, _MTE_FIRST_REG_DEST
from src.aarch64.aarch64_config import supported_instructions
from src.config import CONF
from src.isa_loader import InstructionSet
from src.interfaces import OT

_MD = capstone.Cs(capstone.CS_ARCH_ARM64, capstone.CS_MODE_LITTLE_ENDIAN)

# mnemonic, encoding, required src (reads), required dest (writes)
CASES = [
    # implicit accesses Capstone gets wrong (the fix) --------------------------------
    ("rmif",   0xba000421, {"x1"},          {"N", "Z", "C", "V"}),
    ("setf8",  0x3a00080d, {"w0"},          {"N", "Z", "V"}),
    ("setf16", 0x3a00480d, {"w0"},          {"N", "Z", "V"}),
    ("pacga",  0x9ac23020, {"x1", "x2"},    {"x0"}),
    # flag readers/writers handled via cc / update_flags ----------------------------
    ("ccmp",   0xfa420020, {"x1", "x2", "Z"}, {"N", "Z", "C", "V"}),
    ("ccmn",   0xba420020, {"x1", "x2", "Z"}, {"N", "Z", "C", "V"}),
    ("adds",   0xab020020, {"x1", "x2"},    {"x0", "N", "Z", "C", "V"}),
    ("subs",   0xeb020020, {"x1", "x2"},    {"x0", "N", "Z", "C", "V"}),
    ("ands",   0xea020020, {"x1", "x2"},    {"x0", "N", "Z"}),
    ("bics",   0xea220020, {"x1", "x2"},    {"x0", "N", "Z"}),
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
    # MTE tag stores: Capstone reports the base register (read); no register dest in these forms.
    ("stg",    0xd9200820, {"x1"},          set()),
    ("st2g",   0xd9a00862, {"x3"},          set()),
    ("stzg",   0xd96008a4, {"x5"},          set()),
    ("stz2g",  0xd9e008e6, {"x7"},          set()),
    # MTE tag arithmetic / load: Capstone 5.0.x leaves op.access empty, so decode_reg_accesses fills
    # roles by position (first reg = dest, rest = sources); SUBPS also writes NZCV.
    ("addg",   0x91800420, {"x1"},          {"x0"}),
    ("subg",   0xd1800420, {"x1"},          {"x0"}),
    ("irg",    0x9adf1020, {"x1"},          {"x0"}),
    ("gmi",    0x9ac21420, {"x1", "x2"},    {"x0"}),
    ("subp",   0x9ac20020, {"x1", "x2"},    {"x0"}),
    ("subps",  0xbac20020, {"x1", "x2"},    {"x0", "N", "Z", "C", "V"}),
    ("ldg",    0xd9600020, {"x0", "x1"},    {"x0"}),   # RMW: Xt(x0) read+written, Xn(x1) base read
]

# Remaining supported instructions whose explicit operands are reported correctly by Capstone's
# op.access (verified per family: PAC 1-source, ALU, mem, branches) -- no implicit reg/flag access.
EXPLICIT_OPERAND_ONLY = {
    "autia", "autib", "autiza", "autizb", "autda", "autdb", "autdza", "autdzb",
    "b", "cbnz", "cls", "clz",
    "crc32b", "crc32cb", "crc32ch", "crc32cw", "crc32cx", "crc32h", "crc32w", "crc32x",
    "eor", "orr", "pacib", "paciza", "pacizb", "pacdb", "pacdza", "pacdzb",
    "rbit", "rev", "rev16", "rev32",
    "sdiv", "stp", "tbnz", "tbz", "udiv", "xpaci",
}


class DisasmRegAccessTest(unittest.TestCase):
    def test_no_under_reporting(self):
        for mnemonic, encoding, req_src, req_dest in CASES:
            insn = next(_MD.disasm(encoding.to_bytes(4, "little"), 0), None)
            self.assertIsNotNone(insn, f"{mnemonic}: 0x{encoding:08x} did not decode")
            self.assertEqual(insn.mnemonic, mnemonic,
                             f"0x{encoding:08x} decoded as {insn.mnemonic}, expected {mnemonic}")
            src, dest = decode_reg_accesses(encoding, 0)
            self.assertLessEqual(req_src, set(src), f"{mnemonic}: missing source(s) {req_src - set(src)}")
            self.assertLessEqual(req_dest, set(dest), f"{mnemonic}: missing dest(s) {req_dest - set(dest)}")

    def test_every_supported_instruction_is_classified(self):
        tested = {c[0] for c in CASES}
        for mnemonic in supported_instructions:
            covered = (mnemonic in tested
                       or mnemonic in EXPLICIT_OPERAND_ONLY
                       or any(t.startswith(mnemonic) for t in tested))   # "b." <- "b.eq"
            self.assertTrue(covered, f"{mnemonic!r} is generated but has no reg-access test/classification")


class EmptyAccessFallbackTest(unittest.TestCase):
    def test_unknown_role_is_over_approximated_as_source(self):
        # With the positional MTE fixup disabled, GMI's registers (which Capstone leaves access-empty)
        # must fall through to the src-only over-approximation: all read, none written.
        with patch.object(disasm, "_MTE_FIRST_REG_DEST", frozenset()):
            src, dest = decode_reg_accesses(0x9ac21420, 0)   # gmi x0, x1, x2
        self.assertEqual(set(src), {"x0", "x1", "x2"})
        self.assertEqual(set(dest), set())


class BaseJsonRoleCrossCheckTest(unittest.TestCase):
    """base.json carries authoritative per-operand src/dest roles (from ARM's spec). Cross-check the
    hand-maintained positional fixup in decode_reg_accesses against it, so the fixup cannot silently
    drift from the architecture as instructions are added."""

    @classmethod
    def setUpClass(cls):
        cls._saved_conf = copy.deepcopy(CONF._borg_shared_state)
        CONF.load(os.path.join(_ROOT, "config.yml"))
        isa = InstructionSet(os.path.join(_ROOT, "base.json"), CONF.instruction_categories)
        cls.by_name = {}
        for spec in isa.instructions:
            cls.by_name.setdefault(spec.name, []).append(spec)

    @classmethod
    def tearDownClass(cls):
        CONF._borg_shared_state.clear()
        CONF._borg_shared_state.update(cls._saved_conf)

    def test_mte_positional_fixup_agrees_with_base_json(self):
        for mnemonic in _MTE_FIRST_REG_DEST:
            for spec in self.by_name.get(mnemonic, []):   # absent (e.g. irg) => not generated, skip
                regs = [op for op in spec.operands if op.type == OT.REG]
                self.assertTrue(regs, f"{mnemonic}: base.json has no register operands")
                with self.subTest(mnemonic=mnemonic):
                    self.assertTrue(regs[0].dest, "first register must be a destination")
                    for op in regs[1:]:
                        self.assertTrue(op.src and not op.dest, "non-first registers must be src-only")
                    # LDG is the sole RMW: base.json marks Xt both src and dest.
                    self.assertEqual(bool(regs[0].src), mnemonic == "ldg")


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
