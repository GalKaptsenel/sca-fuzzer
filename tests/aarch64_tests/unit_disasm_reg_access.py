"""decode_reg_accesses must report every register/flag a generated instruction reads/writes -- under-
reporting hides a data dependency and can turn a real leak into a false negative. Capstone (5.0.x)
under-reports some implicit accesses (rmif/setf8/setf16 expose nothing; pacga drops its 2nd source),
which decode_reg_accesses compensates for.

Each case lists the ARM-defined accesses that MUST appear (subset check -- over-reporting a source is
the safe direction and is allowed). Encodings are confirmed to disassemble to the expected mnemonic so
a wrong encoding fails loudly instead of testing nothing. test_every_supported_instruction_is_classified
forces a case (or an explicit op.access-only classification) for every instruction the fuzzer emits, so
a newly-added instruction with hidden accesses cannot slip through unchecked."""
import unittest

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # run from any cwd
import capstone
from src.aarch64.aarch64_disasm import decode_reg_accesses
from src.aarch64.aarch64_config import supported_instructions

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
    ("pacda",  0xdac10820, {"x0", "x1"},    {"x0"}),
    ("xpacd",  0xdac147e0, {"x0"},          {"x0"}),
    ("and",    0x8a020020, {"x1", "x2"},    {"x0"}),
    ("ldr",    0xf9400020, {"x1"},          {"x0"}),
    ("str",    0xf9000020, {"x0", "x1"},    set()),
    ("ldp",    0xa9400440, {"x2"},          {"x0", "x1"}),
    ("cbz",    0x340000a0, {"w0"},          set()),
    ("b.eq",   0x54000000, {"Z"},           set()),
]

# Remaining supported instructions whose explicit operands are reported correctly by Capstone's
# op.access (verified per family: PAC 1-source, ALU, mem, branches) -- no implicit reg/flag access.
EXPLICIT_OPERAND_ONLY = {
    "autda", "autdb", "autdza", "autdzb", "b", "cbnz", "cls", "clz",
    "crc32b", "crc32cb", "crc32ch", "crc32cw", "crc32cx", "crc32h", "crc32w", "crc32x",
    "eor", "orr", "pacdb", "pacdza", "pacdzb", "rbit", "rev", "rev16", "rev32",
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


if __name__ == "__main__":
    unittest.main()
