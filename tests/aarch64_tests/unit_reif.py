"""
Unit tests for the REIF object model (src/aarch64/reif.py): builder -> bytes -> parse round-trips,
typed-section codecs, duplicate/bad-magic rejection, and byte-for-byte wire compatibility with the
existing encoder (build_input_init), which the kernel parses. No hardware. Run from root:
    python -m unittest tests.aarch64_tests.unit_reif
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.aarch64.reif import (                                                # noqa: E402
    REIF, REIFBuilder, REIFSectionType, MainMemorySection, FaultyMemorySection, GprSection,
    SimdSection, PacKeysSection, MteTagsSection, PteSettingsSection)
from src.aarch64.seal.environment import EnvironmentPlan, PteOverride         # noqa: E402
from src.aarch64 import aarch64_executor_input_encoder as enc                 # noqa: E402
from src.interfaces import MAIN_AREA_SIZE, FAULTY_AREA_SIZE, GPR_SUBREGION_SIZE  # noqa: E402


class RoundTripTest(unittest.TestCase):
    def _sample(self):
        return (REIFBuilder()
                .add_section(MainMemorySection(b"\x11" * MAIN_AREA_SIZE))
                .add_section(FaultyMemorySection(b"\x22" * FAULTY_AREA_SIZE))
                .add_section(GprSection(bytes(range(GPR_SUBREGION_SIZE))))
                .add_section(SimdSection(b"\xAB" * 256))
                .add_section(PacKeysSection(list(range(10))))
                .build())

    def test_build_parse_round_trips(self):
        s = REIFSectionType
        reif = REIF(self._sample().to_bytes())
        self.assertEqual(reif.section(s.MAIN_MEMORY).content, b"\x11" * MAIN_AREA_SIZE)
        self.assertEqual(reif.section(s.SIMD).content, b"\xAB" * 256)
        keys = PacKeysSection.from_bytes(reif.section(s.PAC_KEYS).content).keys
        self.assertEqual(keys, tuple(range(10)))
        self.assertIsNone(reif.section(s.MTE_TAGS))

    def test_from_file_object(self):
        import io
        reif = REIF(io.BytesIO(self._sample().to_bytes()))
        exp = bytes(range(GPR_SUBREGION_SIZE))
        self.assertEqual(reif.section(REIFSectionType.GPR).content, exp)

    def test_section_payloads_8_byte_aligned(self):
        import struct
        blob = self._sample().to_bytes()
        REIF(blob)                                       # parses
        _, _version, _header, n, _flags, total = struct.unpack_from("<6Q", blob, 0)
        for i in range(n):
            _t, _sf, off, _ln = struct.unpack_from("<4Q", blob, 48 + i * 32)
            self.assertEqual(off % 8, 0)
        self.assertEqual(total, len(blob))


class TypedSectionCodecTest(unittest.TestCase):
    def test_gpr_nzcv_fragment_round_trip(self):
        # Canonical per-flag NZCV slot (bytes 48..55): N=1,Z=0,C=1,V=0, mirrored into bytes 52..55.
        frag = bytearray(range(GPR_SUBREGION_SIZE))
        frag[48:56] = bytes([1, 0, 1, 0, 1, 0, 1, 0])
        frag = bytes(frag)
        sec = GprSection.from_fragment(frag)
        self.assertEqual(sec.to_fragment(), frag)        # convert there and back is identity
        self.assertNotEqual(sec.content, frag)           # on-wire form differs (flags -> PSTATE)

    def test_mte_tags_pack_two_per_byte(self):
        tags = [1, 2, 3, 4, 15, 0]
        sec = MteTagsSection(tags)
        self.assertEqual(sec.content[0], (2 << 4) | 1)
        self.assertEqual(MteTagsSection.from_bytes(sec.content, len(tags)).tags, tuple(tags))

    def test_pte_settings_round_trip(self):
        plan = EnvironmentPlan(pte_overrides=(PteOverride(1, 3, 0b11, 0b10),))
        sec = PteSettingsSection.from_plan(plan)
        self.assertEqual(PteSettingsSection.from_bytes(sec.content).overrides, plan.pte_overrides)


class RejectionTest(unittest.TestCase):
    def test_duplicate_section_rejected(self):
        b = REIFBuilder().add_section(SimdSection(b"\x00" * 16))
        with self.assertRaises(ValueError):
            b.add_section(SimdSection(b"\x01" * 16))

    def test_bad_magic_rejected(self):
        with self.assertRaises(ValueError):
            REIF(b"\x00" * 64)


class WireCompatTest(unittest.TestCase):
    """REIF.to_bytes() must equal the existing encoder byte-for-byte (same sections, same order), so
    the unchanged kernel/CE parser accepts it."""

    def test_matches_build_input_init(self):
        main = b"\x11" * MAIN_AREA_SIZE
        faulty = b"\x22" * FAULTY_AREA_SIZE
        gpr_final = bytes(range(GPR_SUBREGION_SIZE))     # treated as already-final by both paths
        simd = b"\xAB" * 256
        tags = list(range(enc.MTE_TAG_COUNT))
        keys = list(range(10))
        plan = EnvironmentPlan(pte_overrides=(PteOverride(1, 3, 0b11, 0b10),))

        legacy = enc.build_input_init(main, faulty, gpr_final, simd,
                                      mte_tags=tags, pac_keys=keys, env_plan=plan)
        built = (REIFBuilder()
                 .add_section(MainMemorySection(main))
                 .add_section(FaultyMemorySection(faulty))
                 .add_section(GprSection(gpr_final))
                 .add_section(SimdSection(simd))
                 .add_section(MteTagsSection([t & 0xF for t in tags]))
                 .add_section(PacKeysSection(keys))
                 .add_section(PteSettingsSection.from_plan(plan))
                 .build()).to_bytes()
        self.assertEqual(built, legacy)


if __name__ == "__main__":
    unittest.main()
