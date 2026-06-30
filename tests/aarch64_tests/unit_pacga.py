"""
PACGA support tests — hardware-required.

Verifies:
  1. The "pacga" ioctl path in handle_pac_sign works and returns a value.
  2. PACGA is deterministic: same (Xn, Xm) always gives the same output.
  3. PACGA is sensitive to both inputs: flipping Xn or Xm changes the output.
  4. PACGA encoding constants (PACGA_MASK / PACGA_BASE) match known instructions.
  5. PACGA output changes when the pinned APGA key changes (key isolation).
"""
import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")  # repo root: config.yml / base.json

from src.config import CONF

_executor = None
_SAVED_CONF = None


def setUpModule():
    global _executor, _SAVED_CONF
    if not os.path.exists('/dev/executor'):
        raise unittest.SkipTest(
            "kernel module not loaded — run "
            "'sudo insmod revizor-executor.ko && sudo chmod 777 /dev/executor' "
            "to run these tests")
    from src.aarch64.aarch64_kernel import LocalHWExecutor, PacKeys
    # Snapshot the CONF Borg so CONF.load() does not leak into other modules.
    _SAVED_CONF = copy.deepcopy(CONF._borg_shared_state)
    CONF.load(os.path.join(_ROOT, "config.yml"))
    _executor = LocalHWExecutor('/dev/executor', '/sys/executor')
    _executor.set_pac_keys(None)   # drop any deterministic keys leaked from an earlier module
    keys = _executor.get_pac_keys()
    _executor.set_pac_keys(keys)


def tearDownModule():
    if _SAVED_CONF is not None:
        CONF._borg_shared_state.clear()
        CONF._borg_shared_state.update(_SAVED_CONF)


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
        from src.aarch64.aarch64_kernel import PacKeys
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


class TestPacAuth(unittest.TestCase):
    """PAC sign/auth round-trips, checked two ways.

    A canonical low VA is used so XPAC recovers the original pointer exactly.
    """

    PTR = 0x0000556677889990   # canonical (bits[63:48]=0), 16-byte aligned
    CTX = 0x1234

    def test_real_auth_roundtrip_instruction_key(self):
        # Faithful check: sign, then execute the real AUT* instruction.
        # A correctly-signed pointer authenticates without faulting; the
        # --test dmesg guard catches it if the kernel ever Oopses here.
        signed = _executor.pac_sign(self.PTR, self.CTX, "pacia")
        self.assertNotEqual(signed, self.PTR, "PACIA should add a PAC field")
        authed = _executor.pac_auth(signed, self.CTX, "autia")
        self.assertEqual(authed, self.PTR, "AUTIA of a correctly-signed pointer recovers it")

    def test_real_auth_roundtrip_data_key(self):
        signed = _executor.pac_sign(self.PTR, self.CTX, "pacda")
        authed = _executor.pac_auth(signed, self.CTX, "autda")
        self.assertEqual(authed, self.PTR)

    def test_verify_without_auth_matches_success(self):
        # Non-faulting equivalent of AUTIA: strip (XPAC) then re-sign (PAC) and
        # compare. Equal => auth would succeed; result == the stripped pointer.
        # Uses only instructions that never fault, so it is safe even on a bad
        # pointer.
        signed = _executor.pac_sign(self.PTR, self.CTX, "pacia")
        stripped = _executor.pac_xpac(signed, "xpaci")
        self.assertEqual(stripped, self.PTR, "XPACI recovers the original pointer")
        resigned = _executor.pac_sign(stripped, self.CTX, "pacia")
        self.assertEqual(resigned, signed, "re-signing matches => AUTIA would succeed")

    def test_verify_detects_would_fail_without_faulting(self):
        # Wrong context must NOT re-sign to the same value: the verify path
        # detects an auth failure WITHOUT ever executing a faulting AUT*.
        signed = _executor.pac_sign(self.PTR, self.CTX, "pacia")
        stripped = _executor.pac_xpac(signed, "xpaci")
        resigned_wrong = _executor.pac_sign(stripped, self.CTX ^ 0xABCD, "pacia")
        self.assertNotEqual(resigned_wrong, signed,
                            "wrong context must not authenticate (detected, not faulted)")


class TestInstructionKeyModes(unittest.TestCase):
    """Instruction-key (APIA) sign/auth in both key modes. A swapped APIA that differs
    from the live key used to FPAC-fault the kernel's own pac-ret return and reset the VM."""

    PTR = 0x0000556677889990
    CTX = 0x1234

    def tearDown(self):
        _executor.set_pac_keys(None)
        _executor.set_pac_keys(_executor.get_pac_keys())

    def test_live_keys_roundtrip(self):
        _executor.set_pac_keys(None)
        signed = _executor.pac_sign(self.PTR, self.CTX, "pacia")
        self.assertNotEqual(signed, self.PTR)
        self.assertEqual(_executor.pac_auth(signed, self.CTX, "autia"), self.PTR)

    def test_swapped_keys_equal_live_match(self):
        _executor.set_pac_keys(None)
        live = _executor.pac_sign(self.PTR, self.CTX, "pacia")
        _executor.set_pac_keys(_executor.get_pac_keys())
        self.assertEqual(_executor.pac_sign(self.PTR, self.CTX, "pacia"), live)

    def test_swapped_different_instruction_key(self):
        # regression: a different APIA used to fault the kernel pac-ret RETAA and reset the VM
        _executor.set_pac_keys(None)
        live = _executor.pac_sign(self.PTR, self.CTX, "pacia")
        keys = _executor.get_pac_keys()
        keys.apia_lo ^= 0xA5A5A5A5A5A5A5A5
        keys.apia_hi ^= 0x5A5A5A5A5A5A5A5A
        _executor.set_pac_keys(keys)
        signed = _executor.pac_sign(self.PTR, self.CTX, "pacia")
        self.assertNotEqual(signed, live, "a different APIA must change the signature")
        self.assertEqual(_executor.pac_auth(signed, self.CTX, "autia"), self.PTR,
                         "AUTIA under the same swapped key recovers the pointer")


if __name__ == '__main__':
    unittest.main()
