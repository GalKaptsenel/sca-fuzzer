"""Carrier register file for speculative-exception sealing.

A seal's per-input precondition value (PAC signature, MTE tag, canonicality mask, ...) is not
spliced into the code; it is read at run time from a reserved vector *carrier* register whose bytes
are part of the REIF input. This module owns the carrier byte-math: where each value sits
(`CarrierField`), the top-down allocator over the high vector registers (`CarrierAllocator`), and
the per-input value image shipped in the SIMD input section (`CarrierContribution`). The code a seal
emits is fixed across inputs; only the carrier bytes change.

Registers are reserved from the top down (v31, v30, ...); how many is a campaign parameter. A field
is byte-aligned and never straddles a register (one `UMOV` reads it). The SIMD input section is a
flat 16 x 128-bit image: slot j maps to v(CARRIER_SIMD_BASE_REG + j), so the top slot is v31.
"""
import enum
from dataclasses import dataclass
from typing import List, Tuple

from ...interfaces import SIMD_SUBREGION_SIZE

CARRIER_REG_BYTES = 16                                      # one 128-bit vector register
MAX_CARRIER_REGS = SIMD_SUBREGION_SIZE // CARRIER_REG_BYTES  # SIMD section caps the reserved window
TOP_VECTOR_REG = 31
CARRIER_SIMD_BASE_REG = TOP_VECTOR_REG + 1 - MAX_CARRIER_REGS   # SIMD slot j -> v(base + j)

# width in bytes -> UMOV lane suffix. A field is widened to one of these so a single UMOV reads it.
_UMOV_LANE = {1: "b", 2: "h", 4: "s", 8: "d"}


def _lane_for(width: int) -> int:
    """The smallest UMOV lane (1/2/4/8 bytes) that holds `width` bytes."""
    for lane in (1, 2, 4, 8):
        if width <= lane:
            return lane
    raise ValueError(f"carrier field of {width} bytes exceeds the widest UMOV lane (8)")


class CarrierOverflow(Exception):
    """The reserved carrier window cannot hold another field. Raised, never worked around."""


@dataclass(frozen=True)
class CarrierField:
    """One precondition value's home: a lane within a vector register. `width` is the allocated lane
    size (1/2/4/8 bytes); the value occupies its low bits and the seal's apply op writes only the
    bits it needs."""
    reg_index: int
    byte_offset: int
    width: int

    def __post_init__(self) -> None:
        if self.width not in _UMOV_LANE:
            raise ValueError(f"width {self.width} is not a UMOV lane size (1/2/4/8)")
        if self.byte_offset % self.width:
            raise ValueError(f"byte_offset {self.byte_offset} not {self.width}-aligned")
        if self.byte_offset + self.width > CARRIER_REG_BYTES:
            raise ValueError("field spills past the 16-byte register")

    @property
    def simd_byte_offset(self) -> int:
        """Byte offset of this field in the flat SIMD input image."""
        return (self.reg_index - CARRIER_SIMD_BASE_REG) * CARRIER_REG_BYTES + self.byte_offset

    @property
    def lane_index(self) -> int:
        return self.byte_offset // self.width

    @property
    def umov_source(self) -> str:
        return f"v{self.reg_index}.{_UMOV_LANE[self.width]}[{self.lane_index}]"


class CarrierAllocator:
    """Hands out byte-aligned `CarrierField`s packed top-down from v31. `reg_count` (a campaign
    parameter) fixes the reserved window; a request beyond it raises `CarrierOverflow`."""

    def __init__(self, reg_count: int) -> None:
        if not 1 <= reg_count <= MAX_CARRIER_REGS:
            raise ValueError(f"carrier reg_count {reg_count} outside [1, {MAX_CARRIER_REGS}]")
        self._reg_count = reg_count
        self._lowest_reg = TOP_VECTOR_REG + 1 - reg_count
        self._reg = TOP_VECTOR_REG
        self._byte = 0

    def allocate(self, width: int) -> CarrierField:
        lane = _lane_for(width)
        start = self._byte + (-self._byte % lane)            # align up to the lane
        if start + lane > CARRIER_REG_BYTES:                 # does not fit -> next register down
            self._reg -= 1
            start = 0
        if self._reg < self._lowest_reg:
            raise CarrierOverflow(
                f"carrier window of {self._reg_count} register(s) (v{self._lowest_reg}..v"
                f"{TOP_VECTOR_REG}) cannot hold another {width}-byte field")
        self._byte = start + lane
        return CarrierField(self._reg, start, lane)

    def reserved_registers(self) -> List[int]:
        return list(range(TOP_VECTOR_REG, self._lowest_reg - 1, -1))

    @staticmethod
    def field_taint_offsets(field: CarrierField) -> range:
        """The SIMD-image byte range this field occupies (for sealer-derived carrier taint)."""
        return range(field.simd_byte_offset, field.simd_byte_offset + field.width)


class CarrierRole(enum.Enum):
    """What a carrier field carries, so a seal names its fields instead of ordering them."""
    PAC_SIGNATURE = enum.auto()
    MTE_TAG = enum.auto()
    MTE_RECONVERGE_TAG = enum.auto()
    CANON_MASK = enum.auto()
    BRANCH_TARGET_MASK = enum.auto()
    SIGNEXT_MASK = enum.auto()


class CarrierContribution:
    """The per-input SIMD image: a seal writes each resolved value into its field. Starts zeroed (a
    genuine no-op field is just zero), so an input that omits a field ships zeros there."""

    def __init__(self) -> None:
        self._image = bytearray(SIMD_SUBREGION_SIZE)
        self._written: List[Tuple[int, int]] = []

    def write_field(self, field: CarrierField, value: int) -> None:
        mask = (1 << (8 * field.width)) - 1
        off = field.simd_byte_offset
        self._image[off:off + field.width] = (value & mask).to_bytes(field.width, "little")
        self._written.append((off, field.width))

    def to_simd_bytes(self) -> bytes:
        return bytes(self._image)

    def written_ranges(self) -> Tuple[Tuple[int, int], ...]:
        return tuple(self._written)
