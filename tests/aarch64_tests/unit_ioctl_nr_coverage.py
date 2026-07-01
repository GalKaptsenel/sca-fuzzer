"""ABI drift guard: the ioctl NR constants mirrored in aarch64_kernel.py must match the canonical
header executor_ioctl_nr.h, every NR must be present on both sides, and the NRs must form a
contiguous 1..N range (so a new ioctl cannot be added to the header without a Python constant, nor
skip a number)."""
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import src.aarch64.aarch64_kernel as kernel

_HDR = os.path.join(os.path.dirname(__file__), "..", "..", "src", "aarch64", "executor",
                    "userapi", "executor_ioctl_nr.h")


def _header_nrs():
    nrs = {}
    with open(_HDR) as f:
        for line in f:
            m = re.match(r"\s*#define\s+(REVISOR_\w+_CONSTANT)\s+(\d+)", line)
            if m:
                nrs[m.group(1)] = int(m.group(2))
    return nrs


class IoctlNrCoverageTest(unittest.TestCase):
    def test_python_constants_match_header(self):
        hdr = _header_nrs()
        self.assertGreater(len(hdr), 0, "no REVISOR_*_CONSTANT defines found in the header")
        for name, val in hdr.items():
            self.assertTrue(hasattr(kernel, name), f"{name} is in the header but missing from aarch64_kernel.py")
            self.assertEqual(getattr(kernel, name), val,
                             f"{name}: python={getattr(kernel, name)} header={val}")

    def test_nrs_are_contiguous_from_1(self):
        vals = sorted(_header_nrs().values())
        self.assertEqual(vals, list(range(1, len(vals) + 1)),
                         "ioctl NRs must be a contiguous 1..N range")


if __name__ == "__main__":
    unittest.main()
