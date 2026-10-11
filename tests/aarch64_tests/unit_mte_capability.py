"""The MTE tag-checking capability gate: ID_AA64PFR1_EL1.MTE (bits [11:8]) must be >= 2 (FEAT_MTE2)
for MTE non-interference. Below that the CPU does not check tags, so MTE sealing is meaningless and,
composed with PAC, a baked signature over a tagged pointer FPAC-faults the kernel -- the executor must
refuse. Device-free: the method runs on a stub whose _read_sysfs returns a chosen ID_AA64PFR1_EL1.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.aarch64.aarch64_kernel import LocalHWExecutor


class _Stub:
    def __init__(self, pfr1: int) -> None:
        self._pfr1 = pfr1

    def _read_sysfs(self, name: str) -> str:
        assert name == "system/id_aa64pfr1_el1", name
        return f"0x{self._pfr1:016x}"


def _supported(pfr1: int) -> bool:
    return LocalHWExecutor.mte_tag_checking_supported(_Stub(pfr1))


class MteCapabilityTest(unittest.TestCase):
    def test_none_and_instructions_only_are_unsupported(self):
        self.assertFalse(_supported(0x0000000000000021))   # MTE field 0 (no MTE)
        self.assertFalse(_supported(0x0000000010010121))   # MTE field 1 -- this N3 (FEAT_MTE, no checking)

    def test_mte2_and_mte3_are_supported(self):
        self.assertTrue(_supported(0x0000000000000221))    # MTE field 2 (FEAT_MTE2, tag checking)
        self.assertTrue(_supported(0x0000000000000321))    # MTE field 3 (FEAT_MTE3)

    def test_only_the_mte_field_decides(self):
        self.assertTrue(_supported(0xFFFFFFFFFFFFF2FF))     # bits outside [11:8] are irrelevant
        self.assertFalse(_supported(0xFFFFFFFFFFFFF1FF))


if __name__ == "__main__":
    unittest.main()
