"""Serialize an input into REIF, the Revizor Extensible Input File format.

This is the Python writer for the format the kernel module defines and documents in
src/aarch64/executor/userapi/executor_input_format.h (see also docs/reif_input_format.md).
The kernel structure is the source of truth; this module's only job is to convert "the input
as the fuzzer sees it" (an `Input` ndarray) into exactly that layout. The constants below
mirror the header and MUST be kept in sync with it.

A REIF file = a 6*u64 little-endian preamble, a section table (4*u64 per entry), then the section
payloads (each 8-byte aligned). Sections are located by type, not offset.
"""
import struct
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..interfaces import (Input, InputFragment, MAIN_AREA_SIZE, FAULTY_AREA_SIZE, GPR_SUBREGION_SIZE,
                          SIMD_SUBREGION_SIZE)
from .aarch64_input_layout import NZCVScheme
from .aarch64_relocations import Relocation, PacSignReloc

# --- mirror of executor_input_format.h --------------------------------------------------------
INPUT_MAGIC = 0x49525A5652      # "RVZRI" magic (matches REVISOR_INPUT_MAGIC)
INPUT_VERSION = 1

SEC_MEMORY_MAIN = 0x01
SEC_MEMORY_FAULTY = 0x02
SEC_GPR = 0x03
SEC_SIMD = 0x04
SEC_PAC_KEYS = 0x05
SEC_MTE_TAGS = 0x06
SEC_CODE_RELOC = 0x07
SEC_BPU_TRAINING = 0x08
SEC_PAC_SIGN_RELOC = 0x09

_PREAMBLE_LEN = 6 * 8           # magic, version, header_len, n_sections, flags, total_len
_SECTION_DESC_LEN = 4 * 8       # type, flags, offset, length
_ALIGN = 8

MTE_GRANULE = 16                # bytes per allocation tag (matches kernel MTE_GRANULE_SIZE)
MTE_TAG_COUNT = (MAIN_AREA_SIZE + FAULTY_AREA_SIZE) // MTE_GRANULE   # tags over the main|faulty span
PAC_KEYS_WORDS = 10             # = struct pac_keys / ce_pac_keys: 5 keys * {lo,hi}
CODE_RELOC_TERMINATOR = 0xFFFFFFFF   # matches REVISOR_CODE_RELOC_TERMINATOR
MAX_CODE_RELOCS = 256                # matches REVISOR_INPUT_MAX_CODE_RELOCS
BPU_TRAIN_TERMINATOR = 0xFFFFFFFF    # matches REVISOR_BPU_TRAIN_TERMINATOR
MAX_BPU_TRAIN = 64                   # matches REVISOR_INPUT_MAX_BPU_TRAIN
PAC_SIGN_RELOC_TERMINATOR = 0xFFFFFFFF   # matches REVISOR_PAC_SIGN_RELOC_TERMINATOR
MAX_PAC_SIGN_RELOCS = 256                # matches REVISOR_INPUT_MAX_PAC_SIGN_RELOCS
TARGET_VA_SIZE = 39                                       # matches REVISOR_TARGET_VA_SIZE
PAC_SIGN_MOVK_WINDOWS = 4 - (TARGET_VA_SIZE // 16)        # matches REVISOR_PAC_SIGN_MOVK_WINDOWS
# revisor_pac_sign_op: mnemonic -> wire opcode (matches enum pac_op sign ops)
PAC_SIGN_OPS = {"pacia": 0, "pacib": 1, "pacda": 2, "pacdb": 3, "pacga": 4,
                "paciza": 5, "pacizb": 6, "pacdza": 7, "pacdzb": 8}
# --------------------------------------------------------------------------------------------------


def _pack_sections(sections: List[Tuple[int, bytes]]) -> bytes:
    """Assemble (type, payload) pairs into one REIF file: preamble, table, 8-aligned payloads."""
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
    """Pack the PAC keys section (little-endian)."""
    if len(keys) != PAC_KEYS_WORDS:
        raise ValueError(f"expected {PAC_KEYS_WORDS} PAC key words, got {len(keys)}")
    return struct.pack(f"<{PAC_KEYS_WORDS}Q", *keys)


def _pack_code_reloc(relocs: Sequence) -> bytes:
    """Pack (offset, value) WORD32 relocations followed by an all-ones terminator (matches struct
    revisor_code_reloc_entry / REVISOR_CODE_RELOC_TERMINATOR)."""
    if len(relocs) > MAX_CODE_RELOCS:
        raise ValueError(f"{len(relocs)} code relocations exceed the {MAX_CODE_RELOCS} cap")
    out = bytearray()
    for r in relocs:
        if 0 != r.offset % 4 or not 0 <= r.offset < CODE_RELOC_TERMINATOR:
            raise ValueError(f"bad code relocation offset {r.offset}")
        if not 0 <= r.value <= 0xFFFFFFFF:
            raise ValueError(f"code relocation value 0x{r.value:x} does not fit 32 bits")
        out += struct.pack("<II", r.offset, r.value)
    out += struct.pack("<II", CODE_RELOC_TERMINATOR, CODE_RELOC_TERMINATOR)
    return bytes(out)


def _pack_bpu_training(entries: Sequence) -> bytes:
    """Pack (byte_offset, train_taken) branch-training entries followed by an all-ones terminator
    (matches struct revisor_bpu_train_entry / REVISOR_BPU_TRAIN_TERMINATOR)."""
    if len(entries) > MAX_BPU_TRAIN:
        raise ValueError(f"{len(entries)} branch-training entries exceed the {MAX_BPU_TRAIN} cap")
    out = bytearray()
    for offset, taken in entries:
        if 0 != offset % 4 or not 0 <= offset < BPU_TRAIN_TERMINATOR:
            raise ValueError(f"bad branch-training offset {offset}")
        out += struct.pack("<II", offset, 1 if taken else 0)
    out += struct.pack("<II", BPU_TRAIN_TERMINATOR, BPU_TRAIN_TERMINATOR)
    return bytes(out)


# struct revisor_pac_sign_reloc_entry: movk_offset[N] u32, (pad to 8), value u64, context u64,
# op u32, rd u32. `_PAC_SIGN_PAD` aligns value to 8; the tail is naturally 8-sized so there is no
# trailing padding.
_PAC_SIGN_PAD = (-4 * PAC_SIGN_MOVK_WINDOWS) % 8
_PAC_SIGN_ENTRY_SIZE = 4 * PAC_SIGN_MOVK_WINDOWS + _PAC_SIGN_PAD + struct.calcsize("<QQII")


def _pack_pac_sign_reloc(relocs: Sequence[PacSignReloc]) -> bytes:
    """Pack PacSignReloc entries followed by an op-terminated sentinel (matches
    struct revisor_pac_sign_reloc_entry / REVISOR_PAC_SIGN_RELOC_TERMINATOR)."""
    if len(relocs) > MAX_PAC_SIGN_RELOCS:
        raise ValueError(f"{len(relocs)} PAC-sign relocations exceed the {MAX_PAC_SIGN_RELOCS} cap")
    out = bytearray()
    for r in list(relocs) + [PacSignReloc(PAC_SIGN_RELOC_TERMINATOR, 0, 0, 0,
                                          (0,) * PAC_SIGN_MOVK_WINDOWS)]:
        if len(r.movk_offsets) != PAC_SIGN_MOVK_WINDOWS:
            raise ValueError(f"expected {PAC_SIGN_MOVK_WINDOWS} MOVK offsets, got {len(r.movk_offsets)}")
        out += struct.pack(f"<{PAC_SIGN_MOVK_WINDOWS}I", *r.movk_offsets)
        out += b"\x00" * _PAC_SIGN_PAD
        out += struct.pack("<QQII", r.value, r.context, r.op, r.rd)
    return bytes(out)


def _unpack_pac_sign_reloc(payload: bytes) -> List[PacSignReloc]:
    out: List[PacSignReloc] = []
    for base in range(0, len(payload), _PAC_SIGN_ENTRY_SIZE):
        offs = struct.unpack_from(f"<{PAC_SIGN_MOVK_WINDOWS}I", payload, base)
        value, context, op, rd = struct.unpack_from("<QQII", payload,
                                                     base + 4 * PAC_SIGN_MOVK_WINDOWS + _PAC_SIGN_PAD)
        if PAC_SIGN_RELOC_TERMINATOR == op:
            break
        out.append(PacSignReloc(op=op, value=value, context=context, rd=rd, movk_offsets=offs))
    return out


def build_input_init(main: bytes, faulty: bytes, gpr: bytes, simd: Optional[bytes] = None, *,
                     mte_tags: Optional[Sequence[int]] = None,
                     pac_keys: Optional[Sequence[int]] = None,
                     code_reloc: Optional[Sequence] = None,
                     bpu_training: Optional[Sequence] = None,
                     pac_sign_reloc: Optional[Sequence] = None) -> bytes:
    """Assemble a REIF file from the raw section payloads. `gpr` is the final 64-byte GPR section (flags
    already in PSTATE form); `simd` is the optional 256-byte vector section. Shared by both consumers:
    the device write (ExecutorInput.serialize) and the contract-executor message (ContractExecution.encode)."""
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
    if code_reloc is not None:
        sections.append((SEC_CODE_RELOC, _pack_code_reloc(code_reloc)))
    if bpu_training is not None:
        sections.append((SEC_BPU_TRAINING, _pack_bpu_training(bpu_training)))
    if pac_sign_reloc is not None:
        if pac_keys is None:
            raise ValueError("PAC-sign relocations require a PAC_KEYS section (on-device signing "
                             "needs this input's keys)")
        sections.append((SEC_PAC_SIGN_RELOC, _pack_pac_sign_reloc(pac_sign_reloc)))
    return _pack_sections(sections)


@dataclass(frozen=True)
class ExecutorInput:
    """One REIF input: an architectural `Input` plus the executor-only sections (relocations, MTE
    tags, PAC keys, branch training)."""
    input_: Input
    code_reloc: Tuple[Relocation, ...] = ()
    mte_tags: Optional[Sequence[int]] = None
    pac_keys: Optional[Sequence[int]] = None
    bpu_training: Tuple[Tuple[int, bool], ...] = ()
    pac_sign_reloc: Tuple[PacSignReloc, ...] = ()

    def serialize(self) -> bytes:
        frag = _first_fragment(self.input_)
        main_off = InputFragment.fields["main"][1]
        faulty_off = InputFragment.fields["faulty"][1]
        simd_off = InputFragment.fields["simd"][1]
        return build_input_init(frag[main_off:main_off + MAIN_AREA_SIZE],
                                frag[faulty_off:faulty_off + FAULTY_AREA_SIZE],
                                _gpr_section(frag),
                                frag[simd_off:simd_off + SIMD_SUBREGION_SIZE],
                                mte_tags=self.mte_tags, pac_keys=self.pac_keys,
                                code_reloc=self.code_reloc if self.code_reloc else None,
                                bpu_training=self.bpu_training if self.bpu_training else None,
                                pac_sign_reloc=self.pac_sign_reloc if self.pac_sign_reloc else None)

    # Delegate the arch-input surface the fuzzer/artifact-store touches, so an ExecutorInput is
    # interchangeable with an arch Input in generic code (no isinstance seams).
    def save(self, path: str) -> None:
        """The complete kernel input: loads verbatim into /dev/executor and round-trips via deserialize()."""
        with open(path, "wb") as f:
            f.write(self.serialize())

    def tobytes(self) -> bytes:
        return self.input_.tobytes()

    def set_arch_trace(self, trace) -> None:
        self.input_.set_arch_trace(trace)

    @property
    def _arch_trace(self):
        return getattr(self.input_, "_arch_trace", None)

    @property
    def seed(self) -> int:
        return self.input_.seed


def _unpack_terminated_pairs(payload: bytes, terminator: int):
    for i in range(0, len(payload), 8):
        a, b = struct.unpack_from("<II", payload, i)
        if terminator == a:
            return
        yield a, b


def _unpack_mte_tags(payload: bytes) -> List[int]:
    tags = []
    for byte in payload:
        tags.append(byte & 0xF)
        tags.append(byte >> 4)
    return tags[:MTE_TAG_COUNT]


def _arch_input_from_sections(sections: dict) -> Input:
    frag = bytearray(InputFragment.itemsize)   # padding stays zero
    main_off = InputFragment.fields["main"][1]
    faulty_off = InputFragment.fields["faulty"][1]
    gpr_off = InputFragment.fields["gpr"][1]
    simd_off = InputFragment.fields["simd"][1]
    frag[main_off:main_off + MAIN_AREA_SIZE] = sections[SEC_MEMORY_MAIN]
    frag[faulty_off:faulty_off + FAULTY_AREA_SIZE] = sections[SEC_MEMORY_FAULTY]

    gpr = bytearray(sections[SEC_GPR])
    slot = NZCVScheme.SLOT_IDX
    pstate = int.from_bytes(gpr[slot * 8:slot * 8 + 8], "little")
    gpr[slot * 8:slot * 8 + 8] = NZCVScheme.from_pstate(pstate).to_bytes(8, "little")
    frag[gpr_off:gpr_off + GPR_SUBREGION_SIZE] = gpr

    if SEC_SIMD in sections:
        frag[simd_off:simd_off + SIMD_SUBREGION_SIZE] = sections[SEC_SIMD]

    inp = Input(1)
    inp.linear_view(0)[:] = np.frombuffer(bytes(frag), dtype=np.uint64)
    return inp


def deserialize(blob: bytes) -> ExecutorInput:
    """Inverse of ExecutorInput.serialize: rebuild the ExecutorInput (arch input + every section)."""
    magic, version, _header_len, n, _flags, _total_len = struct.unpack_from("<6Q", blob, 0)
    if INPUT_MAGIC != magic:
        raise ValueError("not a REIF input file (bad magic)")
    if INPUT_VERSION != version:
        raise ValueError(f"unsupported REIF input version {version}")

    sections = {}
    for i in range(n):
        type_, _sf, off, length = struct.unpack_from("<4Q", blob, _PREAMBLE_LEN + i * _SECTION_DESC_LEN)
        sections[type_] = blob[off:off + length]

    code_reloc = tuple(Relocation(off, val)
                       for off, val in _unpack_terminated_pairs(sections.get(SEC_CODE_RELOC, b""),
                                                                CODE_RELOC_TERMINATOR))
    bpu_training = tuple((off, bool(taken))
                         for off, taken in _unpack_terminated_pairs(sections.get(SEC_BPU_TRAINING, b""),
                                                                    BPU_TRAIN_TERMINATOR))
    mte_tags = _unpack_mte_tags(sections[SEC_MTE_TAGS]) if SEC_MTE_TAGS in sections else None
    pac_keys = (list(struct.unpack(f"<{PAC_KEYS_WORDS}Q", sections[SEC_PAC_KEYS]))
                if SEC_PAC_KEYS in sections else None)
    pac_sign_reloc = tuple(_unpack_pac_sign_reloc(sections.get(SEC_PAC_SIGN_RELOC, b"")))

    return ExecutorInput(_arch_input_from_sections(sections), code_reloc=code_reloc,
                         mte_tags=mte_tags, pac_keys=pac_keys, bpu_training=bpu_training,
                         pac_sign_reloc=pac_sign_reloc)
