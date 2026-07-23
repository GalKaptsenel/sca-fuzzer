"""Round-trip checks for the device input wire format (src/aarch64/aarch64_executor_input_encoder.py).

Validates the input_init the Python writer produces against the structure the kernel module documents
in src/aarch64/executor/userapi/executor_input_format.h: a 6*u64 preamble, a 4*u64 section table,
8-aligned payloads located by type, with main/faulty/gpr always present and mte_tags optional.
"""
import struct
import unittest

import numpy as np
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # run from any cwd
from src.interfaces import (Input, InputFragment, MAIN_AREA_SIZE, FAULTY_AREA_SIZE,
                            GPR_SUBREGION_SIZE, SIMD_SUBREGION_SIZE)
from src.aarch64.aarch64_input_layout import NZCVScheme
from src.aarch64.aarch64_relocations import Relocation
from src.aarch64.aarch64_input_generator import AArch64InputGenerator
from src.aarch64 import aarch64_executor_input_encoder as wire


def _rand_relocs(rng, n):
    offs = rng.choice(np.arange(0, 4 * (n + 8), 4), size=n, replace=False)
    return tuple(Relocation(int(o), int(rng.integers(0, 1 << 32))) for o in offs)


def _rand_bpu(rng, n):
    offs = rng.choice(np.arange(0, 4 * (n + 8), 4), size=n, replace=False)
    return tuple((int(o), bool(rng.integers(0, 2))) for o in offs)


def _rand_mte(rng):
    return [int(rng.integers(0, 16)) for _ in range(wire.MTE_TAG_COUNT)]


def _rand_pac(rng):
    return [int(rng.integers(0, 1 << 32)) | (int(rng.integers(0, 1 << 32)) << 32)
            for _ in range(wire.PAC_KEYS_WORDS)]

_PREAMBLE = struct.Struct("<6Q")
_DESC = struct.Struct("<4Q")


def _unpack_bpu(payload: bytes):
    """Mirror the kernel BPU reader: (offset, taken) pairs up to the terminator."""
    out = []
    for i in range(0, len(payload), 8):
        offset, taken = struct.unpack_from("<II", payload, i)
        if offset == wire.BPU_TRAIN_TERMINATOR:
            break
        out.append((offset, taken))
    return out


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
        flags, sections = _parse(wire.ExecutorInput(inp).serialize())
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
        _, sections = _parse(wire.ExecutorInput(inp).serialize())
        flags_slot = struct.unpack_from("<Q", sections[wire.SEC_GPR], NZCVScheme.SLOT_IDX * 8)[0]
        self.assertEqual(flags_slot, NZCVScheme.to_pstate(known))

    def test_mte_tags_section_packs_two_per_byte(self):
        inp = self._input()
        tags = [(i % 16) for i in range(wire.MTE_TAG_COUNT)]
        _, sections = _parse(wire.ExecutorInput(inp, mte_tags=tags).serialize())
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
            wire.ExecutorInput(inp, mte_tags=[0, 1, 2]).serialize()

    def test_build_input_init_all_sections(self):
        main = b"\xAA" * MAIN_AREA_SIZE
        faulty = b"\xBB" * FAULTY_AREA_SIZE
        gpr = b"\xCC" * GPR_SUBREGION_SIZE
        simd = b"\xDD" * SIMD_SUBREGION_SIZE
        tags = [i % 16 for i in range(wire.MTE_TAG_COUNT)]
        keys = list(range(100, 110))
        _, sections = _parse(wire.build_input_init(main, faulty, gpr, simd, mte_tags=tags, pac_keys=keys))

        self.assertIn(wire.SEC_MTE_TAGS, sections)
        self.assertIn(wire.SEC_PAC_KEYS, sections)
        self.assertEqual(sections[wire.SEC_MEMORY_MAIN], main)
        self.assertEqual(sections[wire.SEC_MEMORY_FAULTY], faulty)
        self.assertEqual(sections[wire.SEC_GPR], gpr)
        self.assertEqual(sections[wire.SEC_SIMD], simd)
        self.assertEqual(len(sections[wire.SEC_MTE_TAGS]), (wire.MTE_TAG_COUNT + 1) // 2)
        self.assertEqual(struct.unpack("<10Q", sections[wire.SEC_PAC_KEYS]), tuple(keys))

    def test_bpu_training_section_round_trips(self):
        inp = self._input()
        entries = [(0, True), (4, False), (128, True)]
        _, sections = _parse(wire.ExecutorInput(inp, bpu_training=entries).serialize())
        self.assertIn(wire.SEC_BPU_TRAINING, sections)
        self.assertEqual(_unpack_bpu(sections[wire.SEC_BPU_TRAINING]),
                         [(0, 1), (4, 0), (128, 1)])

    def test_bpu_training_absent_when_empty(self):
        inp = self._input()
        _, sections = _parse(wire.ExecutorInput(inp).serialize())
        self.assertNotIn(wire.SEC_BPU_TRAINING, sections)

    def test_bpu_training_over_cap_rejected(self):
        over = [(4 * i, True) for i in range(wire.MAX_BPU_TRAIN + 1)]
        with self.assertRaises(ValueError):
            wire._pack_bpu_training(over)

    def test_bpu_training_unaligned_offset_rejected(self):
        with self.assertRaises(ValueError):
            wire._pack_bpu_training([(3, True)])

    def _input_with_valid_nzcv(self, seed: int) -> Input:
        inp = self._input()
        gpr0 = (MAIN_AREA_SIZE + FAULTY_AREA_SIZE) // 8
        inp.view("uint64")[gpr0 + NZCVScheme.SLOT_IDX] = NZCVScheme.make_random(np.random.default_rng(seed))
        return inp

    def test_wire_round_trip_all_sections(self):
        ei = wire.ExecutorInput(self._input_with_valid_nzcv(3),
                                code_reloc=(Relocation(0, 0xdeadbeef), Relocation(8, 0x1234)),
                                mte_tags=[i % 16 for i in range(wire.MTE_TAG_COUNT)],
                                pac_keys=list(range(100, 110)),
                                bpu_training=((0, True), (4, False)))
        back = wire.deserialize(ei.serialize())
        self.assertEqual(back.input_.tobytes(), ei.input_.tobytes())
        self.assertEqual(back.code_reloc, ei.code_reloc)
        self.assertEqual(back.mte_tags, ei.mte_tags)
        self.assertEqual(back.pac_keys, ei.pac_keys)
        self.assertEqual(back.bpu_training, ei.bpu_training)
        self.assertEqual(back.serialize(), ei.serialize())

    def test_wire_round_trip_no_optional_sections(self):
        ei = wire.ExecutorInput(self._input_with_valid_nzcv(4))
        back = wire.deserialize(ei.serialize())
        self.assertEqual(back.input_.tobytes(), ei.input_.tobytes())
        self.assertEqual(back.code_reloc, ())
        self.assertIsNone(back.mte_tags)
        self.assertIsNone(back.pac_keys)
        self.assertEqual(back.bpu_training, ())

    def test_deserialize_rejects_bad_magic(self):
        blob = bytearray(wire.ExecutorInput(self._input_with_valid_nzcv(5)).serialize())
        blob[0:8] = (0xDEADBEEF).to_bytes(8, "little")
        with self.assertRaises(ValueError):
            wire.deserialize(bytes(blob))

    def test_round_trip_real_random_inputs(self):
        for inp in AArch64InputGenerator(20250712).generate(8):
            ei = wire.ExecutorInput(inp)
            back = wire.deserialize(ei.serialize())
            self.assertEqual(back.input_.tobytes(), inp.tobytes())
            self.assertEqual(back.serialize(), ei.serialize())

    def test_round_trip_random_relocations(self):
        rng = np.random.default_rng(1)
        inp = AArch64InputGenerator(1).generate(1)[0]
        for n in (0, 1, 5, wire.MAX_CODE_RELOCS):
            relocs = _rand_relocs(rng, n)
            back = wire.deserialize(wire.ExecutorInput(inp, code_reloc=relocs).serialize())
            self.assertEqual(back.code_reloc, relocs)

    def test_round_trip_random_mte_tags(self):
        rng = np.random.default_rng(2)
        inp = AArch64InputGenerator(2).generate(1)[0]
        for _ in range(5):
            tags = _rand_mte(rng)
            back = wire.deserialize(wire.ExecutorInput(inp, mte_tags=tags).serialize())
            self.assertEqual(back.mte_tags, tags)

    def test_round_trip_random_mistraining(self):
        rng = np.random.default_rng(3)
        inp = AArch64InputGenerator(3).generate(1)[0]
        for n in (0, 1, 10, wire.MAX_BPU_TRAIN):
            bpu = _rand_bpu(rng, n)
            back = wire.deserialize(wire.ExecutorInput(inp, bpu_training=bpu).serialize())
            self.assertEqual(back.bpu_training, bpu)

    def test_round_trip_random_all_sections_fuzz(self):
        for seed in range(25):
            rng = np.random.default_rng(1000 + seed)
            inp = AArch64InputGenerator(seed).generate(1)[0]
            ei = wire.ExecutorInput(inp,
                                    code_reloc=_rand_relocs(rng, int(rng.integers(0, 12))),
                                    mte_tags=_rand_mte(rng),
                                    pac_keys=_rand_pac(rng),
                                    bpu_training=_rand_bpu(rng, int(rng.integers(0, 12))))
            back = wire.deserialize(ei.serialize())
            self.assertEqual(back.input_.tobytes(), ei.input_.tobytes())
            self.assertEqual(back.code_reloc, ei.code_reloc)
            self.assertEqual(back.mte_tags, ei.mte_tags)
            self.assertEqual(back.pac_keys, ei.pac_keys)
            self.assertEqual(back.bpu_training, ei.bpu_training)
            self.assertEqual(back.serialize(), ei.serialize())

    def test_contract_execution_encode_envelope(self):
        """ContractExecution.encode emits a 17*u64 envelope + code + an input initialization whose
        memory == main‖faulty and gpr round-trip; this is what the CE binary parses."""
        from src.aarch64.aarch64_contract_executor import ContractExecution, SimArch, RVZRCE_MAGIC
        memory = bytes((i % 256 for i in range(MAIN_AREA_SIZE + FAULTY_AREA_SIZE)))  # 8192
        registers = bytes((i % 251 for i in range(GPR_SUBREGION_SIZE)))              # 64
        code = b"\x1f\x20\x03\xd5"                                                   # NOP
        ce = ContractExecution(code, memory, registers, SimArch.RVZR_ARCH_AARCH64, 0, 0,
                               req_mem_base_virt=0x1000)
        msg = ce.encode()

        env = struct.unpack_from("<17Q", msg, 0)
        self.assertEqual(env[0], RVZRCE_MAGIC)        # magic
        code_size, init_size = env[14], env[15]
        self.assertEqual(code_size, len(code))
        self.assertEqual(len(msg), 17 * 8 + code_size + init_size)
        self.assertEqual(msg[17 * 8:17 * 8 + code_size], code)

        init = msg[17 * 8 + code_size:]
        _, sections = _parse(init)
        self.assertEqual(sections[wire.SEC_MEMORY_MAIN] + sections[wire.SEC_MEMORY_FAULTY], memory)
        self.assertEqual(sections[wire.SEC_GPR], registers[:GPR_SUBREGION_SIZE])


if __name__ == "__main__":
    unittest.main()
