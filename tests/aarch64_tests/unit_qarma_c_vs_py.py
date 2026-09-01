"""The Python QARMA (aarch64_qarma.py, used to bake signatures) and the CE's C QARMA (qarma.c, used to
model AUT*) MUST be bit-identical — if they diverge, the CE flags correct signatures as "forged" and the
device FPAC-faults on genuine auths (which reset the box until the QARMA3 S-box / TBID bugs were fixed).
This compiles qarma.c into a shared library and cross-checks it against the Python implementation over a
sweep of pointers, modifiers, keys, QARMA versions, TBI/TBID combinations, and instruction/data keys.

Self-contained: resolves qarma.c from __file__ and builds it in a temp dir (needs a C compiler).
"""
import ctypes
import os
import shutil
import subprocess
import tempfile
import unittest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import src.aarch64.aarch64_qarma as q

_CE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "src", "aarch64", "contract_executor")


class _CPacProfile(ctypes.Structure):
    _fields_ = [("iterations", ctypes.c_int), ("tsz", ctypes.c_int),
                ("tbi0", ctypes.c_int), ("tbi1", ctypes.c_int),
                ("pauth2", ctypes.c_bool),
                ("tbid0", ctypes.c_int), ("tbid1", ctypes.c_int)]


@unittest.skipUnless(shutil.which("gcc") or shutil.which("cc"), "no C compiler")
class QarmaCVsPyTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cc = shutil.which("gcc") or shutil.which("cc")
        cls._tmp = tempfile.mkdtemp()
        so = os.path.join(cls._tmp, "libqarma.so")
        src = os.path.join(_CE_DIR, "qarma.c")
        subprocess.run([cc, "-shared", "-fPIC", "-O2", "-I", _CE_DIR, src, "-o", so],
                       check=True, capture_output=True)
        lib = ctypes.CDLL(so)
        lib.qarma_addpac.restype = ctypes.c_uint64
        lib.qarma_addpac.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
                                     ctypes.c_uint64, _CPacProfile, ctypes.c_int]
        lib.qarma_strip.restype = ctypes.c_uint64
        lib.qarma_strip.argtypes = [ctypes.c_uint64, _CPacProfile, ctypes.c_int]
        cls._lib = lib

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmp, ignore_errors=True)

    def _cprof(self, p):
        return _CPacProfile(p.iterations, p.tsz, int(p.tbi0), int(p.tbi1), bool(p.pauth2),
                            int(p.tbid0), int(p.tbid1))

    def test_addpac_and_strip_match(self):
        ptrs = [0x0000000000abc000, 0x0000400020461960, 0xffff400020461960,
                0xffffff8000abc123, 0xffff4000abcde7f0, 0x00000000deadbe00]
        mods = [0, 0x1234, 0xffff400020461ed4]
        keys = [(0x0123456789abcdef, 0xfedcba9876543210),
                (0x1111111111111111, 0x2222222222222222),
                (0xdeadbeefcafef00d, 0x0)]
        checked = 0
        for ver, va in ((3, 48), (5, 48), (3, 39), (5, 52)):
            for tbi0 in (0, 1):
                for tbi1 in (0, 1):
                    for tbid0 in (0, 1):
                        for tbid1 in (0, 1):
                            for pauth2 in (False, True):
                                p = q.profile(ver, va, tbi0, tbi1, pauth2, tbid0, tbid1)
                                cp = self._cprof(p)
                                for (lo, hi) in keys:
                                    for ptr in ptrs:
                                        for mod in mods:
                                            for is_instr in (0, 1):
                                                py = q.addpac(ptr, mod, lo, hi, p, bool(is_instr))
                                                c = self._lib.qarma_addpac(ptr, mod, lo, hi, cp, is_instr)
                                                self.assertEqual(py, c,
                                                    f"addpac ver={ver} va={va} tbi={tbi0}{tbi1} "
                                                    f"tbid={tbid0}{tbid1} p2={pauth2} ptr={ptr:#x} "
                                                    f"mod={mod:#x} instr={is_instr}: py={py:#x} c={c:#x}")
                                                pys = q.strip(py, p, bool(is_instr))
                                                cs = self._lib.qarma_strip(c, cp, is_instr)
                                                self.assertEqual(pys, cs, "strip mismatch")
                                                checked += 1
        self.assertGreater(checked, 1000)


if __name__ == "__main__":
    unittest.main()
