"""Parsing of the contract_executor response blob (contract_trace_t). The wire layout is shared with
the C side via _Static_asserts in simulation_output.h; this locks the Python reader to the same
16-byte header (entry_count + truncated) and fixed 424-byte entry stride."""
import struct
import unittest

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # run from any cwd
from src.aarch64.aarch64_trace import ContractExecutionResult

_NUM_GPRS = 31


def _entry(pc: int = 0, encoding: int = 0, extra_data_size: int = 0) -> bytes:
    cpu = struct.pack(f"<{_NUM_GPRS}Q", *([0] * _NUM_GPRS))       # gpr[31]
    cpu += struct.pack("<QQQQ", 0, pc, 0, encoding)               # sp, pc, nzcv, encoding
    cpu += struct.pack("<Q", extra_data_size)                     # extra_data_size (size_t)
    meta = struct.pack("<QQQQQ", 0, 0, 0, 0, 0)                   # instr_index, has_mem, spec, is_pair, window_id
    meta += struct.pack("<12Q", *([0] * 12))                      # memory_access + memory_access2
    return cpu + meta


def _blob(entries, truncated: int = 0) -> bytes:
    return struct.pack("<QQ", len(entries), truncated) + b"".join(entries)


class CeTraceParseTest(unittest.TestCase):
    def test_entry_stride_is_424(self):
        self.assertEqual(len(_entry()), 424)

    def test_header_and_entries(self):
        cer = ContractExecutionResult(_blob([_entry(pc=0x10, encoding=0xd503201f), _entry(pc=0x14)]),
                                      _NUM_GPRS)
        self.assertEqual(cer.entry_count, 2)
        self.assertEqual(cer.truncated, 0)
        self.assertEqual(len(cer.entries), 2)
        self.assertEqual(cer.entries[0].cpu.pc, 0x10)
        self.assertEqual(cer.entries[1].cpu.pc, 0x14)

    def test_truncated_flag_parsed(self):
        cer = ContractExecutionResult(_blob([_entry()], truncated=1), _NUM_GPRS)
        self.assertEqual(cer.truncated, 1)

    def test_nonzero_extra_data_size_rejected(self):
        with self.assertRaises(AssertionError):
            ContractExecutionResult(_blob([_entry(extra_data_size=8)]), _NUM_GPRS)


if __name__ == "__main__":
    unittest.main()
