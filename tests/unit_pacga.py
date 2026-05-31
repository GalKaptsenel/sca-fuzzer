"""
PACGA support tests — hardware-required.

Verifies:
  1. The "pacga" ioctl path in handle_pac_sign works and returns a value.
  2. PACGA is deterministic: same (Xn, Xm) always gives the same output.
  3. PACGA is sensitive to both inputs: flipping Xn or Xm changes the output.
  4. PACGA encoding constants (PACGA_MASK / PACGA_BASE) match known instructions.
  5. PACGA output changes when the pinned APGA key changes (key isolation).
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import CONF

_executor = None


def setUpModule():
    global _executor
    if not os.path.exists('/dev/executor'):
        raise unittest.SkipTest(
            "kernel module not loaded — run "
            "'sudo insmod revizor-executor.ko && sudo chmod 777 /dev/executor' "
            "to run these tests")
    from src.aarch64.aarch64_connection import LocalExecutorImp, PacKeys
    CONF.load("config.yml")
    _executor = LocalExecutorImp('/dev/executor', '/sys/executor', '')
    keys = _executor.get_pac_keys()
    _executor.set_pac_keys(keys)


class TestPacgaIoctl(unittest.TestCase):
    """pac_sign("pacga") ioctl correctness."""

    def _pacga(self, xn: int, xm: int) -> int:
        return _executor.pac_sign(xn, xm, "pacga")

    def test_pacga_returns_nonzero_for_nonzero_inputs(self):
        result = self._pacga(0xDEADBEEFCAFE1234, 0x0123456789ABCDEF)
        self.assertNotEqual(result, 0)

    def test_pacga_is_deterministic(self):
        xn, xm = 0xAAAABBBBCCCCDDDD, 0x1111222233334444
        r1 = self._pacga(xn, xm)
        r2 = self._pacga(xn, xm)
        self.assertEqual(r1, r2)

    def test_pacga_sensitive_to_xn(self):
        xm = 0xFEDCBA9876543210
        r1 = self._pacga(0x0000000000000001, xm)
        r2 = self._pacga(0x0000000000000002, xm)
        self.assertNotEqual(r1, r2)

    def test_pacga_sensitive_to_xm(self):
        xn = 0x0123456789ABCDEF
        r1 = self._pacga(xn, 0x0000000000000001)
        r2 = self._pacga(xn, 0x0000000000000002)
        self.assertNotEqual(r1, r2)

    def test_pacga_zero_inputs_returns_value(self):
        result = self._pacga(0, 0)
        # PACGA(0, 0) is still a defined hash — just verify it doesn't crash
        self.assertIsInstance(result, int)

    def test_pacga_result_is_64bit(self):
        result = self._pacga(0xDEADBEEFCAFE0000, 0x0000CAFEBABE0000)
        self.assertGreaterEqual(result, 0)
        self.assertLessEqual(result, 0xFFFFFFFFFFFFFFFF)


class TestPacgaEncoding(unittest.TestCase):
    """Verify PACGA_MASK / PACGA_BASE constants against known encodings."""

    PACGA_MASK = 0xFFE0FC00
    PACGA_BASE = 0xDAC03000

    def _encode_pacga(self, rd: int, rn: int, rm: int) -> int:
        """Encode PACGA Xd, Xn, Xm from register numbers."""
        return self.PACGA_BASE | (rm << 16) | (rn << 5) | rd

    def _matches_pacga(self, inst: int) -> bool:
        return (inst & self.PACGA_MASK) == self.PACGA_BASE

    def test_pacga_x0_x0_x0_encodes_to_base(self):
        self.assertEqual(self._encode_pacga(0, 0, 0), self.PACGA_BASE)

    def test_pacga_x3_x3_x2_matches_mask(self):
        # pacga x3, x3, x2
        inst = self._encode_pacga(rd=3, rn=3, rm=2)
        self.assertTrue(self._matches_pacga(inst))

    def test_pacga_x1_x4_x5_matches_mask(self):
        # pacga x1, x4, x5
        inst = self._encode_pacga(rd=1, rn=4, rm=5)
        self.assertTrue(self._matches_pacga(inst))

    def test_pacda_does_not_match_pacga(self):
        # pacda x0, x1 = 0xDAC10820 (rd=0, rn=1)
        pacda = 0xDAC10820
        self.assertFalse(self._matches_pacga(pacda))

    def test_pacia_does_not_match_pacga(self):
        # pacia x0, x1 = 0xDAC10020
        pacia = 0xDAC10020
        self.assertFalse(self._matches_pacga(pacia))

    def test_rd_rn_rm_extraction(self):
        inst = self._encode_pacga(rd=5, rn=7, rm=11)
        self.assertEqual(inst & 0x1F, 5)            # Rd
        self.assertEqual((inst >> 5) & 0x1F, 7)     # Rn
        self.assertEqual((inst >> 16) & 0x1F, 11)   # Rm


class TestPacgaKeyIsolation(unittest.TestCase):
    """PACGA output must change when the APGA key changes."""

    def test_pacga_changes_with_different_apga_key(self):
        from src.aarch64.aarch64_connection import PacKeys
        xn, xm = 0xDEADBEEF00000000, 0x00000000CAFEBABE

        # Result with current (default) keys
        r1 = _executor.pac_sign(xn, xm, "pacga")

        # Set a different APGA key
        keys_modified = _executor.get_pac_keys()
        keys_modified.apga_lo ^= 0xFFFFFFFFFFFFFFFF
        keys_modified.apga_hi ^= 0xFFFFFFFFFFFFFFFF
        _executor.set_pac_keys(keys_modified)
        r2 = _executor.pac_sign(xn, xm, "pacga")

        # Restore original keys
        keys_orig = _executor.get_pac_keys()
        keys_orig.apga_lo ^= 0xFFFFFFFFFFFFFFFF
        keys_orig.apga_hi ^= 0xFFFFFFFFFFFFFFFF
        _executor.set_pac_keys(keys_orig)

        self.assertNotEqual(r1, r2, "PACGA output must differ under different APGA keys")


if __name__ == '__main__':
    unittest.main()
