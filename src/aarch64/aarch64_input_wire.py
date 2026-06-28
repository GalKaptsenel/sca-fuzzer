"""Serialize an input into the /dev/executor input wire format.

This is the Python writer for the format the kernel module defines and documents in
src/aarch64/executor/userapi/executor_input_format.h. The kernel structure is the
source of truth; this module's only job is to convert "the input as the fuzzer sees
it" (an `Input` ndarray) into exactly that wire layout. The constants below mirror
the header and MUST be kept in sync with it.

The input initialization = a 6*u64 little-endian preamble, a section table (4*u64 per entry), then
the section payloads (each 8-byte aligned). Sections are located by type, not offset.
"""
import struct
from typing import List, Optional, Sequence, Tuple

from ..interfaces import (InputFragment, MAIN_AREA_SIZE, FAULTY_AREA_SIZE, GPR_SUBREGION_SIZE,
                          SIMD_SUBREGION_SIZE)
from .aarch64_input_layout import NZCVScheme

# --- mirror of executor_input_format.h --------------------------------------------------------
INPUT_MAGIC = 0x49525A5652      # "RVZRI" magic (matches REVISOR_INPUT_MAGIC)
INPUT_VERSION = 1

SEC_MEMORY_MAIN = 0x01
SEC_MEMORY_FAULTY = 0x02
SEC_GPR = 0x03
SEC_SIMD = 0x04
SEC_PAC_KEYS = 0x05
SEC_MTE_TAGS = 0x06

_PREAMBLE_LEN = 6 * 8           # magic, version, header_len, n_sections, flags, total_len
_SECTION_DESC_LEN = 4 * 8       # type, flags, offset, length
_ALIGN = 8

MTE_GRANULE = 16                # bytes per allocation tag (matches kernel MTE_GRANULE_SIZE)
MTE_TAG_COUNT = (MAIN_AREA_SIZE + FAULTY_AREA_SIZE) // MTE_GRANULE   # tags over the main|faulty span
PAC_KEYS_WORDS = 9              # struct ce_pac_keys: 9 * u64
# ----------------------------------------------------------------------------------------------


def _pack_sections(sections: List[Tuple[int, bytes]]) -> bytes:
    """Assemble (type, payload) pairs into one input initialization: preamble, table, 8-aligned payloads."""
    n = len(sections)
    header_len = _PREAMBLE_LEN + n * _SECTION_DESC_LEN

    descriptors: List[Tuple[int, int, int, int]] = []   # (type, sec_flags, offset, length)
    payload = bytearray()
    offset = header_len
    for type_, data in sections:
        pad = (-offset) % _ALIGN
        payload += b"\x00" * pad
        offset += pad
        sec_flags = 0   # reserved per-section flags
        descriptors.append((type_, sec_flags, offset, len(data)))
        payload += data
        offset += len(data)
    total_len = offset

    out = bytearray()
    out += struct.pack("<6Q", INPUT_MAGIC, INPUT_VERSION, header_len, n, 0, total_len)  # flags reserved
    for type_, sec_flags, off, length in descriptors:
        out += struct.pack("<4Q", type_, sec_flags, off, length)
    assert len(out) == header_len
    out += payload
    assert len(out) == total_len
    return bytes(out)


def _first_fragment(inp) -> bytes:
    """The first actor's InputFragment bytes (the device models a single-actor input)."""
    return inp.tobytes()[:InputFragment.itemsize]


def _gpr_section(frag: bytes) -> bytes:
    """The 64-byte GPR section, with the flags slot converted from per-flag NZCV to ARM PSTATE
    (the kernel loads `flags` verbatim into PSTATE — the conversion is the writer's job)."""
    gpr_off = InputFragment.fields["gpr"][1]
    gpr = bytearray(frag[gpr_off:gpr_off + GPR_SUBREGION_SIZE])
    slot = NZCVScheme.SLOT_IDX
    raw = int.from_bytes(gpr[slot * 8:slot * 8 + 8], "little")
    gpr[slot * 8:slot * 8 + 8] = NZCVScheme.to_pstate(raw).to_bytes(8, "little")
    return bytes(gpr)


def _pack_mte_tags(tags: Sequence[int]) -> bytes:
    """Pack one 4-bit tag per granule, two per byte, low nibble first (granule 2*i then 2*i+1)."""
    if len(tags) != MTE_TAG_COUNT:
        raise ValueError(f"expected {MTE_TAG_COUNT} MTE tags, got {len(tags)}")
    packed = bytearray((MTE_TAG_COUNT + 1) // 2)
    for i, tag in enumerate(tags):
        nibble = tag & 0xF
        if 0 == i % 2:
            packed[i // 2] |= nibble
        else:
            packed[i // 2] |= nibble << 4
    return bytes(packed)


def _pack_pac_keys(keys: Sequence[int]) -> bytes:
    """Pack struct ce_pac_keys (9 * u64, little-endian)."""
    if len(keys) != PAC_KEYS_WORDS:
        raise ValueError(f"expected {PAC_KEYS_WORDS} PAC key words, got {len(keys)}")
    return struct.pack(f"<{PAC_KEYS_WORDS}Q", *keys)


def build_input_init(main: bytes, faulty: bytes, gpr: bytes, simd: Optional[bytes] = None,
                     mte_tags: Optional[Sequence[int]] = None,
                     pac_keys: Optional[Sequence[int]] = None) -> bytes:
    """Assemble a input_init from the raw section payloads. `gpr` is the final 64-byte GPR section (flags
    already in PSTATE form); `simd` is the optional 256-byte vector section. Shared by both consumers:
    the device write (serialize_input) and the contract-executor message (ContractExecution.encode)."""
    sections: List[Tuple[int, bytes]] = [
        (SEC_MEMORY_MAIN, main),
        (SEC_MEMORY_FAULTY, faulty),
        (SEC_GPR, gpr),
    ]
    if simd is not None:
        sections.append((SEC_SIMD, simd))
    if mte_tags is not None:
        sections.append((SEC_MTE_TAGS, _pack_mte_tags(mte_tags)))
    if pac_keys is not None:
        sections.append((SEC_PAC_KEYS, _pack_pac_keys(pac_keys)))
    return _pack_sections(sections)


def serialize_input(inp, mte_tags: Optional[Sequence[int]] = None,
                    pac_keys: Optional[Sequence[int]] = None) -> bytes:
    """Serialize `inp` to a device input initialization. main/faulty/gpr/simd are always present;
    mte_tags and pac_keys are emitted only when supplied (per-input initial state)."""
    frag = _first_fragment(inp)
    main_off = InputFragment.fields["main"][1]
    faulty_off = InputFragment.fields["faulty"][1]
    simd_off = InputFragment.fields["simd"][1]
    return build_input_init(frag[main_off:main_off + MAIN_AREA_SIZE],
                            frag[faulty_off:faulty_off + FAULTY_AREA_SIZE],
                            _gpr_section(frag),
                            frag[simd_off:simd_off + SIMD_SUBREGION_SIZE],
                            mte_tags, pac_keys)
