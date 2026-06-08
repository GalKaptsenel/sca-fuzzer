"""
Tests for the to_executor_input debug utility: it must rewrite ONLY the flags
slot (per-flag -> PSTATE) and leave every other byte of the saved input intact.
"""
import struct
import unittest

from src.aarch64.debugging.to_executor_input import convert
from src.aarch64.aarch64_input_layout import REGISTER_REGION_OFFSET

_FLAGS_BYTE = REGISTER_REGION_OFFSET + 48   # register slot 6


class ToExecutorInputTest(unittest.TestCase):

    def test_flags_slot_converted_to_pstate(self):
        buf = bytearray(12288)
        buf[_FLAGS_BYTE] = 1   # per-flag N = 1
        out = convert(bytes(buf))
        self.assertEqual(struct.unpack_from('<Q', out, _FLAGS_BYTE)[0], 0x80000000)

    def test_only_flags_slot_changes(self):
        buf = bytearray(range(256)) * 48          # 12288 bytes of varied data
        buf[_FLAGS_BYTE] = 1
        out = bytearray(convert(bytes(buf)))
        # zero out the flags slot in both and require everything else identical
        for i in range(8):
            buf[REGISTER_REGION_OFFSET + 48 + i] = 0
            out[REGISTER_REGION_OFFSET + 48 + i] = 0
        self.assertEqual(out, buf, "bytes outside the flags slot were modified")

    def test_too_short_input_rejected(self):
        with self.assertRaises(ValueError):
            convert(b'\x00' * (REGISTER_REGION_OFFSET + 8))


if __name__ == '__main__':
    unittest.main()
