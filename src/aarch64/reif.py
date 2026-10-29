"""REIF (Revizor Extensible Input File): the typed input format the kernel and CE read.

Wire layout (must match executor/userapi/executor_input_format.h): a 6*u64 preamble, a 4*u64
descriptor per section, then 8-byte-aligned payloads; sections are located by type. Build with
REIFBuilder().add_section(...).build(); parse with REIF(file_or_bytes). Immutable once built.
"""
import struct
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import List, Optional, Sequence, Tuple, Union

from .aarch64_input_layout import NZCVScheme
from .seal.environment import (EnvironmentPlan, PteOverride, serialize_pte_overrides,
                               deserialize_pte_overrides)

INPUT_MAGIC = 0x49525A5652      # "RVZRI" (REVISOR_INPUT_MAGIC)
INPUT_VERSION = 1

_PREAMBLE = struct.Struct("<6Q")        # magic, version, header_len, n_sections, flags, total_len
_DESC = struct.Struct("<4Q")            # type, flags, offset, length
_ALIGN = 8

MTE_GRANULE = 16                        # bytes per allocation tag (kernel MTE_GRANULE_SIZE)
_PAC_KEY_WORDS = 10                     # 5 keys * {lo, hi}


class REIFSectionType(IntEnum):
    MAIN_MEMORY = 0x01
    FAULTY_MEMORY = 0x02
    GPR = 0x03
    SIMD = 0x04
    PAC_KEYS = 0x05
    MTE_TAGS = 0x06
    PTE_SETTINGS = 0x09


class REIFSection(ABC):
    """A section exposes only its type and its raw content bytes. Typed subclasses add structured
    constructors/parsers but serialize to exactly these bytes."""

    @property
    @abstractmethod
    def type(self) -> REIFSectionType:
        ...

    @property
    @abstractmethod
    def content(self) -> bytes:
        ...


class RawSection(REIFSection):
    """A section as read back from a file: a type plus its undecoded bytes."""

    def __init__(self, type_: REIFSectionType, content: bytes) -> None:
        self._type = type_
        self._content = bytes(content)

    @property
    def type(self) -> REIFSectionType:
        return self._type

    @property
    def content(self) -> bytes:
        return self._content


class MainMemorySection(REIFSection):
    """Initial contents of the main sandbox data region."""

    def __init__(self, content: bytes) -> None:
        self._content = bytes(content)

    @property
    def type(self) -> REIFSectionType:
        return REIFSectionType.MAIN_MEMORY

    @property
    def content(self) -> bytes:
        return self._content


class FaultyMemorySection(REIFSection):
    """Initial contents of the faulty (spec-only) sandbox data region."""

    def __init__(self, content: bytes) -> None:
        self._content = bytes(content)

    @property
    def type(self) -> REIFSectionType:
        return REIFSectionType.FAULTY_MEMORY

    @property
    def content(self) -> bytes:
        return self._content


class SimdSection(REIFSection):
    """Initial vector-register image; 16 bytes/register, lane layout up to the caller."""

    def __init__(self, content: bytes) -> None:
        self._content = bytes(content)

    @property
    def type(self) -> REIFSectionType:
        return REIFSectionType.SIMD

    @property
    def content(self) -> bytes:
        return self._content


class GprSection(REIFSection):
    """Initial GPRs. Flags are PSTATE on the wire but per-flag NZCV in a fuzzer fragment;
    from_fragment()/to_fragment() convert between the two."""

    def __init__(self, content: bytes) -> None:
        self._content = bytes(content)

    @classmethod
    def from_fragment(cls, fragment_gpr: bytes) -> "GprSection":
        """Build from the fuzzer's GPR sub-region (per-flag NZCV slot -> PSTATE)."""
        return cls(_convert_nzcv_slot(fragment_gpr, NZCVScheme.to_pstate))

    def to_fragment(self) -> bytes:
        """The fuzzer's GPR sub-region (PSTATE slot -> per-flag NZCV)."""
        return _convert_nzcv_slot(self._content, NZCVScheme.from_pstate)

    @property
    def type(self) -> REIFSectionType:
        return REIFSectionType.GPR

    @property
    def content(self) -> bytes:
        return self._content


class PacKeysSection(REIFSection):
    """The 5 PAC keys as {lo, hi} little-endian u64 pairs."""

    def __init__(self, keys: Sequence[int]) -> None:
        if len(keys) != _PAC_KEY_WORDS:
            raise ValueError(f"expected {_PAC_KEY_WORDS} PAC key words, got {len(keys)}")
        self._keys = tuple(keys)

    @classmethod
    def from_bytes(cls, content: bytes) -> "PacKeysSection":
        return cls(list(struct.unpack(f"<{_PAC_KEY_WORDS}Q", content)))

    @property
    def keys(self) -> Tuple[int, ...]:
        return self._keys

    @property
    def type(self) -> REIFSectionType:
        return REIFSectionType.PAC_KEYS

    @property
    def content(self) -> bytes:
        return struct.pack(f"<{_PAC_KEY_WORDS}Q", *self._keys)


class MteTagsSection(REIFSection):
    """One 4-bit allocation tag per granule, packed two per byte (low nibble = even granule)."""

    def __init__(self, tags: Sequence[int]) -> None:
        self._tags = tuple(t & 0xF for t in tags)

    @classmethod
    def from_bytes(cls, content: bytes, count: int) -> "MteTagsSection":
        tags: List[int] = []
        for byte in content:
            tags.append(byte & 0xF)
            tags.append(byte >> 4)
        return cls(tags[:count])

    @property
    def tags(self) -> Tuple[int, ...]:
        return self._tags

    @property
    def type(self) -> REIFSectionType:
        return REIFSectionType.MTE_TAGS

    @property
    def content(self) -> bytes:
        packed = bytearray((len(self._tags) + 1) // 2)
        for i, tag in enumerate(self._tags):
            packed[i // 2] |= tag if 0 == i % 2 else tag << 4
        return bytes(packed)


class PteSettingsSection(REIFSection):
    """A variant's page-table overrides (see seal/environment.py)."""

    def __init__(self, overrides: Sequence[PteOverride]) -> None:
        self._overrides = tuple(overrides)

    @classmethod
    def from_plan(cls, plan: EnvironmentPlan) -> "PteSettingsSection":
        return cls(plan.pte_overrides)

    @classmethod
    def from_bytes(cls, content: bytes) -> "PteSettingsSection":
        return cls(deserialize_pte_overrides(content))

    @property
    def overrides(self) -> Tuple[PteOverride, ...]:
        return self._overrides

    @property
    def type(self) -> REIFSectionType:
        return REIFSectionType.PTE_SETTINGS

    @property
    def content(self) -> bytes:
        return serialize_pte_overrides(self._overrides)


def _convert_nzcv_slot(gpr: bytes, convert) -> bytes:
    out = bytearray(gpr)
    base = NZCVScheme.SLOT_IDX * 8
    raw = int.from_bytes(out[base:base + 8], "little")
    out[base:base + 8] = convert(raw).to_bytes(8, "little")
    return bytes(out)


class REIFBuilder:
    """Assembles sections into a REIF. One section per type; `build()` freezes it."""

    def __init__(self, version: int = INPUT_VERSION, flags: int = 0) -> None:
        self._version = version
        self._flags = flags
        self._sections: "OrderedSections" = []

    def add_section(self, section: REIFSection) -> "REIFBuilder":
        if any(s.type == section.type for s in self._sections):
            raise ValueError(f"duplicate REIF section {section.type.name}")
        self._sections.append(section)
        return self

    def build(self) -> "REIF":
        return REIF._from_sections(self._sections, self._version, self._flags)


OrderedSections = List[REIFSection]


class REIF:
    """An immutable REIF: built via REIFBuilder or parsed from a file/bytes. Read sections by type;
    serialize with to_bytes()/save()."""

    def __init__(self, source: Union[bytes, bytearray, "SupportsRead"]) -> None:
        blob = source if isinstance(source, (bytes, bytearray)) else source.read()
        self._version, self._flags, self._sections = _parse(bytes(blob))

    @classmethod
    def _from_sections(cls, sections: OrderedSections, version: int, flags: int) -> "REIF":
        obj = cls.__new__(cls)
        obj._version = version
        obj._flags = flags
        obj._sections = [RawSection(s.type, s.content) for s in sections]
        return obj

    @property
    def version(self) -> int:
        return self._version

    def sections(self) -> Tuple[RawSection, ...]:
        return tuple(self._sections)

    def section(self, type_: REIFSectionType) -> Optional[RawSection]:
        for s in self._sections:
            if s.type == type_:
                return s
        return None

    def to_bytes(self) -> bytes:
        n = len(self._sections)
        header_len = _PREAMBLE.size + n * _DESC.size
        descriptors: List[Tuple[int, int, int, int]] = []
        payload = bytearray()
        offset = header_len
        for s in self._sections:
            pad = (-offset) % _ALIGN
            payload += b"\x00" * pad
            offset += pad
            data = s.content
            descriptors.append((int(s.type), 0, offset, len(data)))
            payload += data
            offset += len(data)
        out = bytearray()
        out += _PREAMBLE.pack(INPUT_MAGIC, self._version, header_len, n, self._flags, offset)
        for d in descriptors:
            out += _DESC.pack(*d)
        out += payload
        return bytes(out)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            f.write(self.to_bytes())


def _parse(blob: bytes) -> Tuple[int, int, List[RawSection]]:
    magic, version, _header_len, n, flags, _total = _PREAMBLE.unpack_from(blob, 0)
    if INPUT_MAGIC != magic:
        raise ValueError("not a REIF file (bad magic)")
    if INPUT_VERSION != version:
        raise ValueError(f"unsupported REIF version {version}")
    sections: List[RawSection] = []
    for i in range(n):
        type_, _sf, off, length = _DESC.unpack_from(blob, _PREAMBLE.size + i * _DESC.size)
        sections.append(RawSection(REIFSectionType(type_), blob[off:off + length]))
    return version, flags, sections


class SupportsRead:                     # typing shim for a binary file object
    def read(self) -> bytes:            # pragma: no cover
        ...
