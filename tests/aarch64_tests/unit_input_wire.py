"""Round-trip checks for the device input wire format (src/aarch64/aarch64_input_wire.py).

Validates the input_init the Python writer produces against the structure the kernel module documents
in src/aarch64/executor/userapi/executor_input_format.h: a 6*u64 preamble, a 4*u64 section table,
8-aligned payloads located by type, with main/faulty/gpr always present and mte_tags optional.
"""
import struct
import unittest

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # run from any cwd
from src.interfaces import (Input, InputFragment, MAIN_AREA_SIZE, FAULTY_AREA_SIZE,
                            GPR_SUBREGION_SIZE, SIMD_SUBREGION_SIZE)
from src.aarch64.aarch64_input_layout import NZCVScheme
from src.aarch64 import aarch64_input_wire as wire

_PREAMBLE = struct.Struct("<6Q")
_DESC = struct.Struct("<4Q")


def _parse(input_init: bytes):
    """A faithful mirror of the kernel reader: validate the preamble + table, return {type: payload}."""
    magic, version, header_len, n_sections, flags, total_len = _PREAMBLE.unpack_from(input_init, 0)
    assert magic == wire.INPUT_MAGIC, "bad magic"
    assert version == wire.INPUT_VERSION, "bad version"
    assert total_len == len(input_init), "total_len mismatch"
    assert header_len == _PREAMBLE.size + n_sections * _DESC.size, "header_len mismatch"
    sections = {}
    for i in range(n_sections):
        type_, sec_flags, offset, length = _DESC.unpack_from(input_init, _PREAMBLE.size + i * _DESC.size)
        assert sec_flags == 0
        assert header_len <= offset and offset + length <= total_len, "section out of bounds"
        assert offset % 8 == 0, "payload not 8-aligned"
        sections[type_] = input_init[offset:offset + length]
    return flags, sections


class InputWireRoundTrip(unittest.TestCase):
    def _input(self) -> Input:
        inp = Input()
        # deterministic, distinguishable bytes per region
        raw = inp.view("uint64")
        for j in range(MAIN_AREA_SIZE // 8):
            raw[j] = 0x1100 + j
        for j in range(FAULTY_AREA_SIZE // 8):
            raw[MAIN_AREA_SIZE // 8 + j] = 0x2200 + j
        gpr0 = (MAIN_AREA_SIZE + FAULTY_AREA_SIZE) // 8
        for j in range(GPR_SUBREGION_SIZE // 8):
            raw[gpr0 + j] = 0x3300 + j
        return inp

    def test_required_sections_round_trip(self):
        inp = self._input()
        flags, sections = _parse(wire.serialize_input(inp))
        self.assertEqual(flags, 0)
        self.assertEqual(set(sections),
                         {wire.SEC_MEMORY_MAIN, wire.SEC_MEMORY_FAULTY, wire.SEC_GPR, wire.SEC_SIMD})

        frag = inp.tobytes()[:InputFragment.itemsize]
        m0 = InputFragment.fields["main"][1]
        f0 = InputFragment.fields["faulty"][1]
        self.assertEqual(sections[wire.SEC_MEMORY_MAIN], frag[m0:m0 + MAIN_AREA_SIZE])
        self.assertEqual(sections[wire.SEC_MEMORY_FAULTY], frag[f0:f0 + FAULTY_AREA_SIZE])
        self.assertEqual(len(sections[wire.SEC_GPR]), GPR_SUBREGION_SIZE)
        self.assertEqual(len(sections[wire.SEC_SIMD]), SIMD_SUBREGION_SIZE)

    def test_gpr_flags_converted_to_pstate(self):
        inp = self._input()
        # set every per-flag NZCV bit so the PSTATE conversion is observable
        raw = inp.view("uint64")
        gpr0 = (MAIN_AREA_SIZE + FAULTY_AREA_SIZE) // 8
        known = 0
        for byte_off, _ in NZCVScheme._LAYOUT.values():
            known |= 1 << (byte_off * 8)
        raw[gpr0 + NZCVScheme.SLOT_IDX] = known
        _, sections = _parse(wire.serialize_input(inp))
        flags_slot = struct.unpack_from("<Q", sections[wire.SEC_GPR], NZCVScheme.SLOT_IDX * 8)[0]
        self.assertEqual(flags_slot, NZCVScheme.to_pstate(known))

    def test_mte_tags_section_packs_two_per_byte(self):
        inp = self._input()
        tags = [(i % 16) for i in range(wire.MTE_TAG_COUNT)]
        _, sections = _parse(wire.serialize_input(inp, mte_tags=tags))
        self.assertIn(wire.SEC_MTE_TAGS, sections)
        packed = sections[wire.SEC_MTE_TAGS]
        self.assertEqual(len(packed), (wire.MTE_TAG_COUNT + 1) // 2)
        # unpack and compare (low nibble = even granule, high nibble = odd)
        unpacked = []
        for byte in packed:
            unpacked.append(byte & 0xF)
            unpacked.append(byte >> 4)
        self.assertEqual(unpacked[:wire.MTE_TAG_COUNT], tags)

    def test_mte_tags_wrong_count_rejected(self):
        inp = self._input()
        with self.assertRaises(ValueError):
            wire.serialize_input(inp, mte_tags=[0, 1, 2])

    def test_build_input_init_all_sections(self):
        main = b"\xAA" * MAIN_AREA_SIZE
        faulty = b"\xBB" * FAULTY_AREA_SIZE
        gpr = b"\xCC" * GPR_SUBREGION_SIZE
        simd = b"\xDD" * SIMD_SUBREGION_SIZE
        tags = [i % 16 for i in range(wire.MTE_TAG_COUNT)]
        keys = list(range(100, 110))
        _, sections = _parse(wire.build_input_init(main, faulty, gpr, simd, tags, keys))

        self.assertIn(wire.SEC_MTE_TAGS, sections)
        self.assertIn(wire.SEC_PAC_KEYS, sections)
        self.assertEqual(sections[wire.SEC_MEMORY_MAIN], main)
        self.assertEqual(sections[wire.SEC_MEMORY_FAULTY], faulty)
        self.assertEqual(sections[wire.SEC_GPR], gpr)
        self.assertEqual(sections[wire.SEC_SIMD], simd)
        self.assertEqual(len(sections[wire.SEC_MTE_TAGS]), (wire.MTE_TAG_COUNT + 1) // 2)
        self.assertEqual(struct.unpack("<10Q", sections[wire.SEC_PAC_KEYS]), tuple(keys))

    def test_contract_execution_encode_envelope(self):
        """ContractExecution.encode emits a 16*u64 envelope + code + an input initialization whose
        memory == main‖faulty and gpr round-trip; this is what the CE binary parses."""
        from src.aarch64.aarch64_contract_executor import ContractExecution, SimArch, RVZRCE_MAGIC
        memory = bytes((i % 256 for i in range(MAIN_AREA_SIZE + FAULTY_AREA_SIZE)))  # 8192
        registers = bytes((i % 251 for i in range(GPR_SUBREGION_SIZE)))              # 64
        code = b"\x1f\x20\x03\xd5"                                                   # NOP
        ce = ContractExecution(code, memory, registers, SimArch.RVZR_ARCH_AARCH64, 0, 0,
                               req_mem_base_virt=0x1000)
        msg = ce.encode()

        env = struct.unpack_from("<16Q", msg, 0)
        self.assertEqual(env[0], RVZRCE_MAGIC)        # magic
        code_size, init_size = env[13], env[14]
        self.assertEqual(code_size, len(code))
        self.assertEqual(len(msg), 16 * 8 + code_size + init_size)
        self.assertEqual(msg[16 * 8:16 * 8 + code_size], code)

        init = msg[16 * 8 + code_size:]
        _, sections = _parse(init)
        self.assertEqual(sections[wire.SEC_MEMORY_MAIN] + sections[wire.SEC_MEMORY_FAULTY], memory)
        self.assertEqual(sections[wire.SEC_GPR], registers[:GPR_SUBREGION_SIZE])


if __name__ == "__main__":
    unittest.main()
