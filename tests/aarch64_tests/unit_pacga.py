"""
PACGA + PAC sign/auth tests — hardware-required.

Keys travel with every sign/auth request (the kernel keeps no key state of its own), so each test
passes its own key set. Verifies:
  1. The "pacga" ioctl path in handle_pac_sign works and returns a value.
  2. PACGA is deterministic under fixed keys, and sensitive to both inputs.
  3. PACGA encoding constants (PACGA_MASK / PACGA_BASE) match known instructions.
  4. PACGA output changes when the APGA key changes (key isolation).
  5. PAC sign/auth round-trips; a sign/auth request without keys is rejected.
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


def _keys(tag: int):
    """A fixed, arbitrary PAC key set, perturbed by `tag` so callers can build distinct sets."""
    from src.aarch64.aarch64_kernel import PacKeys
    k = PacKeys()
    for name in ("apia", "apib", "apda", "apdb", "apga"):
        setattr(k, name + "_lo", 0x1122334455667788 ^ tag)
        setattr(k, name + "_hi", 0x8877665544332211 ^ (tag << 1))
    return k


def setUpModule():
    global _executor, _SAVED_CONF
    if not os.path.exists('/dev/executor'):
        raise unittest.SkipTest(
            "kernel module not loaded — run "
            "'sudo insmod revizor-executor.ko && sudo chmod 777 /dev/executor' "
            "to run these tests")
    from src.aarch64.aarch64_kernel import LocalHWExecutor
    # Snapshot the CONF Borg so CONF.load() does not leak into other modules.
    _SAVED_CONF = copy.deepcopy(CONF._borg_shared_state)
    CONF.load(os.path.join(_ROOT, "config.yml"))
    _executor = LocalHWExecutor('/dev/executor', '/sys/executor')


def tearDownModule():
    if _SAVED_CONF is not None:
        CONF._borg_shared_state.clear()
        CONF._borg_shared_state.update(_SAVED_CONF)


class TestPacgaIoctl(unittest.TestCase):
    """pac_sign("pacga") ioctl correctness under fixed keys."""

    KEYS = None

    @classmethod
    def setUpClass(cls):
        cls.KEYS = _keys(0)

    def _pacga(self, xn: int, xm: int) -> int:
        return _executor.pac_sign(xn, xm, "pacga", self.KEYS)

    def test_pacga_returns_nonzero_for_nonzero_inputs(self):
        result = self._pacga(0xDEADBEEFCAFE1234, 0x0123456789ABCDEF)
        self.assertNotEqual(result, 0)

    def test_pacga_is_deterministic(self):
        xn, xm = 0xAAAABBBBCCCCDDDD, 0x1111222233334444
        self.assertEqual(self._pacga(xn, xm), self._pacga(xn, xm))

    def test_pacga_sensitive_to_xn(self):
        xm = 0xFEDCBA9876543210
        self.assertNotEqual(self._pacga(0x1, xm), self._pacga(0x2, xm))

    def test_pacga_sensitive_to_xm(self):
        xn = 0x0123456789ABCDEF
        self.assertNotEqual(self._pacga(xn, 0x1), self._pacga(xn, 0x2))

    def test_pacga_zero_inputs_returns_value(self):
        self.assertIsInstance(self._pacga(0, 0), int)

    def test_pacga_result_is_64bit(self):
        result = self._pacga(0xDEADBEEFCAFE0000, 0x0000CAFEBABE0000)
        self.assertGreaterEqual(result, 0)
        self.assertLessEqual(result, 0xFFFFFFFFFFFFFFFF)


class TestPacgaEncoding(unittest.TestCase):
    """Verify PACGA_MASK / PACGA_BASE constants against known encodings."""

    PACGA_MASK = 0xFFE0FC00
    PACGA_BASE = 0xDAC03000

    def _encode_pacga(self, rd: int, rn: int, rm: int) -> int:
        return self.PACGA_BASE | (rm << 16) | (rn << 5) | rd

    def _matches_pacga(self, inst: int) -> bool:
        return (inst & self.PACGA_MASK) == self.PACGA_BASE

    def test_pacga_x0_x0_x0_encodes_to_base(self):
        self.assertEqual(self._encode_pacga(0, 0, 0), self.PACGA_BASE)

    def test_pacga_x3_x3_x2_matches_mask(self):
        self.assertTrue(self._matches_pacga(self._encode_pacga(rd=3, rn=3, rm=2)))

    def test_pacga_x1_x4_x5_matches_mask(self):
        self.assertTrue(self._matches_pacga(self._encode_pacga(rd=1, rn=4, rm=5)))

    def test_pacda_does_not_match_pacga(self):
        self.assertFalse(self._matches_pacga(0xDAC10820))  # pacda x0, x1

    def test_pacia_does_not_match_pacga(self):
        self.assertFalse(self._matches_pacga(0xDAC10020))  # pacia x0, x1

    def test_rd_rn_rm_extraction(self):
        inst = self._encode_pacga(rd=5, rn=7, rm=11)
        self.assertEqual(inst & 0x1F, 5)            # Rd
        self.assertEqual((inst >> 5) & 0x1F, 7)     # Rn
        self.assertEqual((inst >> 16) & 0x1F, 11)   # Rm


class TestPacgaKeyIsolation(unittest.TestCase):
    """PACGA output must change when the APGA key changes."""

    def test_pacga_changes_with_different_apga_key(self):
        xn, xm = 0xDEADBEEF00000000, 0x00000000CAFEBABE
        keys_a = _keys(0)
        keys_b = _keys(0)
        keys_b.apga_lo ^= 0xFFFFFFFFFFFFFFFF
        keys_b.apga_hi ^= 0xFFFFFFFFFFFFFFFF
        r1 = _executor.pac_sign(xn, xm, "pacga", keys_a)
        r2 = _executor.pac_sign(xn, xm, "pacga", keys_b)
        self.assertNotEqual(r1, r2, "PACGA output must differ under different APGA keys")


class TestPacAuth(unittest.TestCase):
    """PAC sign/auth round-trips, checked two ways, under one fixed key set.

    A canonical low VA is used so XPAC recovers the original pointer exactly.
    """

    PTR = 0x0000556677889990   # canonical (bits[63:48]=0), 16-byte aligned
    CTX = 0x1234
    KEYS = None

    @classmethod
    def setUpClass(cls):
        cls.KEYS = _keys(0x99)

    def test_real_auth_roundtrip_instruction_key(self):
        # Sign, then execute the real AUT*: a correctly-signed pointer authenticates without faulting.
        signed = _executor.pac_sign(self.PTR, self.CTX, "pacia", self.KEYS)
        self.assertNotEqual(signed, self.PTR, "PACIA should add a PAC field")
        self.assertEqual(_executor.pac_auth(signed, self.CTX, "autia", self.KEYS), self.PTR,
                         "AUTIA of a correctly-signed pointer recovers it")

    def test_real_auth_roundtrip_data_key(self):
        signed = _executor.pac_sign(self.PTR, self.CTX, "pacda", self.KEYS)
        self.assertEqual(_executor.pac_auth(signed, self.CTX, "autda", self.KEYS), self.PTR)

    def test_verify_without_auth_matches_success(self):
        # Non-faulting equivalent of AUTIA: strip (XPAC) then re-sign and compare.
        signed = _executor.pac_sign(self.PTR, self.CTX, "pacia", self.KEYS)
        stripped = _executor.pac_xpac(signed, "xpaci")
        self.assertEqual(stripped, self.PTR, "XPACI recovers the original pointer")
        resigned = _executor.pac_sign(stripped, self.CTX, "pacia", self.KEYS)
        self.assertEqual(resigned, signed, "re-signing matches => AUTIA would succeed")

    def test_verify_detects_would_fail_without_faulting(self):
        # Wrong context must NOT re-sign to the same value — an auth failure detected without faulting.
        signed = _executor.pac_sign(self.PTR, self.CTX, "pacia", self.KEYS)
        stripped = _executor.pac_xpac(signed, "xpaci")
        resigned_wrong = _executor.pac_sign(stripped, self.CTX ^ 0xABCD, "pacia", self.KEYS)
        self.assertNotEqual(resigned_wrong, signed,
                            "wrong context must not authenticate (detected, not faulted)")


class TestPerRequestKeys(unittest.TestCase):
    """Keys are per-request: distinct key sets give distinct signatures, sign/auth are self-contained,
    and a request without keys is rejected rather than signed under a default."""

    PTR = 0x0000556677889990
    CTX = 0x1234

    def test_roundtrip_under_supplied_keys(self):
        keys = _keys(0x11)
        signed = _executor.pac_sign(self.PTR, self.CTX, "pacia", keys)
        self.assertNotEqual(signed, self.PTR)
        self.assertEqual(_executor.pac_auth(signed, self.CTX, "autia", keys), self.PTR)

    def test_different_instruction_key_changes_signature(self):
        # regression: a different APIA used to fault the kernel pac-ret RETAA and reset the VM.
        keys_a = _keys(0)
        keys_b = _keys(0)
        keys_b.apia_lo ^= 0xA5A5A5A5A5A5A5A5
        keys_b.apia_hi ^= 0x5A5A5A5A5A5A5A5A
        sig_a = _executor.pac_sign(self.PTR, self.CTX, "pacia", keys_a)
        sig_b = _executor.pac_sign(self.PTR, self.CTX, "pacia", keys_b)
        self.assertNotEqual(sig_a, sig_b, "a different APIA must change the signature")
        self.assertEqual(_executor.pac_auth(sig_b, self.CTX, "autia", keys_b), self.PTR,
                         "AUTIA under the same key recovers the pointer")

    def test_sign_without_keys_is_rejected(self):
        # keys=None omits the key set; the kernel must reject the request, not sign under a default.
        from src.aarch64.aarch64_kernel import REVISOR_PAC_SIGN
        with self.assertRaises(OSError):
            _executor._pac_op(REVISOR_PAC_SIGN, self.PTR, self.CTX, "pacia", None)


if __name__ == '__main__':
    unittest.main()
