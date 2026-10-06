"""The Python QARMA (aarch64_qarma.py, used to bake signatures) and the CE's C QARMA (qarma.c, used to
model AUT*) MUST be bit-identical — if they diverge, the CE flags correct signatures as "forged" and the
device FPAC-faults on genuine auths (which reset the box until the QARMA3 S-box / TBID bugs were fixed).
This compiles qarma.c into a shared library and cross-checks it against the Python implementation over a
sweep of pointers, modifiers, keys, QARMA versions, TBI/TBID combinations, and instruction/data keys.

Self-contained: resolves qarma.c from __file__ and builds it in a temp dir (needs a C compiler).
"""
import ctypes
import itertools
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
    _fields_ = [("iterations", ctypes.c_int), ("generic_iterations", ctypes.c_int), ("level", ctypes.c_int),
                ("t0sz", ctypes.c_int), ("t1sz", ctypes.c_int),
                ("tbi0", ctypes.c_int), ("tbi1", ctypes.c_int),
                ("tbid0", ctypes.c_int), ("tbid1", ctypes.c_int), ("constpacfield", ctypes.c_bool)]


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
        return _CPacProfile(p.iterations, p.generic_iterations, p.level, p.t0sz, p.t1sz, int(p.tbi0),
                            int(p.tbi1), int(p.tbid0), int(p.tbid1), p.constpacfield)

    def test_addpac_and_strip_match(self):
        ptrs = [0x0000000000abc000, 0x0000400020461960, 0xffff400020461960,
                0xffffff8000abc123, 0xffff4000abcde7f0, 0x00000000deadbe00,
                0x0ae500000ae50000, 0xf000000012345000]   # non-canonical: bit63 != bit55
        mods = [0, 0x1234, 0xffff400020461ed4]
        keys = [(0x0123456789abcdef, 0xfedcba9876543210),
                (0x1111111111111111, 0x2222222222222222),
                (0xdeadbeefcafef00d, 0x0)]
        checked = 0

        for ver, va0, va1 in ((3, 48, 48), (5, 48, 48), (3, 39, 39), (5, 39, 48), (3, 48, 39)):
            for tbi0, tbi1, tbid0, tbid1 in itertools.product((0, 1), repeat=4):
                for level, cpf in itertools.product(range(1, 6), (False, True)):
                    p = q.profile(ver, ver, level, va0, va1, tbi0, tbi1, tbid0, tbid1, cpf, False, False)
                    cp = self._cprof(p)
                    for (lo, hi), ptr, mod, is_instr in itertools.product(keys, ptrs, mods, (0, 1)):
                        py = q.addpac(ptr, mod, lo, hi, p, bool(is_instr))
                        c = self._lib.qarma_addpac(ptr, mod, lo, hi, cp, is_instr)
                        self.assertEqual(py, c, f"addpac {p} ptr={ptr:#x} mod={mod:#x} instr={is_instr}: "
                                                f"py={py:#x} c={c:#x}")
                        self.assertEqual(q.strip(py, p, bool(is_instr)),
                                         self._lib.qarma_strip(c, cp, is_instr), "strip mismatch")
                        checked += 1
        self.assertGreater(checked, 1000)


if __name__ == "__main__":
    unittest.main()
