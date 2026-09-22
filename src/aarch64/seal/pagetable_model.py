"""AArch64 VMSAv8-64 page-table descriptor model (4 KB granule, stage 1).

The single source of truth for descriptor bit layout. Everything else (the PTE sealing primitive, the
input encoder, the kernel-facing environment plan) reasons about page-table state by FIELD NAME --
`valid`, `attr_indx`, `ap`, `uxn`, ... -- and never by raw bit offset or magic mask. Add or change a
descriptor bit here once and the rest of the stack follows.

Pure and dependency-free (stdlib only) so the low-level config, the CE, and the sealer can all import it
without a cycle. Two layouts are provided:

  * `LEAF_LAYOUT`  -- a level-3 page descriptor (the leaf that maps a 4 KB page).
  * `TABLE_LAYOUT` -- a level 0-2 table descriptor (points at the next-level table).

`fuzzable_by_default` marks the attribute fields whose value a decoy may vary out of the box; the output
address / next-table address and the descriptor type are never in that set, so a decoy keeps mapping the
same physical page through the same walk shape -- only attributes change.
"""
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class BitField:
    """One named run of bits within a 64-bit descriptor."""
    name: str
    offset: int
    width: int = 1
    fuzzable_by_default: bool = False
    doc: str = ""

    @property
    def mask(self) -> int:
        return ((1 << self.width) - 1) << self.offset

    @property
    def max_value(self) -> int:
        return (1 << self.width) - 1

    def extract(self, descriptor: int) -> int:
        return (descriptor >> self.offset) & self.max_value

    def insert(self, descriptor: int, value: int) -> int:
        if not 0 <= value <= self.max_value:
            raise ValueError(f"{self.name}: value {value:#x} does not fit {self.width} bit(s)")
        return (descriptor & ~self.mask) | (value << self.offset)

    def values(self) -> range:
        """Every legal value of this field."""
        return range(self.max_value + 1)


class DescriptorLayout:
    """A named, non-overlapping set of `BitField`s describing one page-table descriptor level."""

    def __init__(self, name: str, fields: Sequence[BitField]) -> None:
        self.name = name
        self._fields: Tuple[BitField, ...] = tuple(fields)
        self._by_name: Dict[str, BitField] = {}
        covered = 0
        for f in self._fields:
            if f.name in self._by_name:
                raise ValueError(f"layout {name!r}: duplicate field {f.name!r}")
            if covered & f.mask:
                raise ValueError(f"layout {name!r}: field {f.name!r} overlaps another field")
            self._by_name[f.name] = f
            covered |= f.mask

    @property
    def fields(self) -> Tuple[BitField, ...]:
        return self._fields

    @property
    def field_names(self) -> List[str]:
        return [f.name for f in self._fields]

    @property
    def default_fuzzable_field_names(self) -> List[str]:
        return [f.name for f in self._fields if f.fuzzable_by_default]

    def field(self, name: str) -> BitField:
        try:
            return self._by_name[name]
        except KeyError:
            raise KeyError(f"layout {self.name!r} has no field {name!r}; "
                           f"known fields: {self.field_names}")

    def mask_for(self, names: Sequence[str]) -> int:
        """The combined bit mask of the named fields (validates every name)."""
        mask = 0
        for name in names:
            mask |= self.field(name).mask
        return mask


@dataclass(frozen=True)
class PageTableDescriptor:
    """A 64-bit descriptor value paired with its layout, offering read/write by field name."""
    value: int
    layout: DescriptorLayout

    def get(self, name: str) -> int:
        return self.layout.field(name).extract(self.value)

    def with_field(self, name: str, value: int) -> "PageTableDescriptor":
        return PageTableDescriptor(self.layout.field(name).insert(self.value, value), self.layout)

    def differs_only_in(self, other: "PageTableDescriptor", names: Sequence[str]) -> bool:
        """True iff `self` and `other` agree on every bit outside the named fields -- the invariant a
        decoy descriptor must satisfy (it may change only the allowed attribute fields)."""
        allowed = self.layout.mask_for(names)
        return (self.value & ~allowed) == (other.value & ~allowed)


# ==================================================================================================
# The two stage-1 4 KB-granule layouts. Bit positions per Arm ARM (VMSAv8-64 descriptor formats).
# ==================================================================================================

# Level-3 page descriptor (maps a 4 KB page). `oa` (output address) and `type` are never fuzzed so a
# decoy keeps the same physical page and the same walk shape; all other attributes are fair game.
LEAF_LAYOUT = DescriptorLayout("leaf", [
    # BitField(name, offset, width, fuzzable_by_default, doc)
    BitField("valid", 0, 1, True, "descriptor is valid (0 -> translation fault)"),
    BitField("type", 1, 1, False, "must be 1 for a level-3 page descriptor"),
    BitField("attr_indx", 2, 3, True, "MAIR index (memory type / cacheability)"),
    BitField("ns", 5, 1, True, "non-secure"),
    BitField("ap", 6, 2, True, "data access permissions AP[2:1]"),
    BitField("sh", 8, 2, True, "shareability"),
    BitField("af", 10, 1, True, "access flag (0 -> access fault without HW AF)"),
    BitField("ng", 11, 1, True, "not-global"),
    BitField("oa", 12, 36, False, "output address [47:12] -- never fuzzed"),
    BitField("gp", 50, 1, False, "guarded page (FEAT_BTI)"),
    BitField("dbm", 51, 1, True, "dirty bit modifier"),
    BitField("contiguous", 52, 1, True, "contiguous hint"),
    BitField("pxn", 53, 1, True, "privileged execute-never"),
    BitField("uxn", 54, 1, True, "unprivileged execute-never (XN at EL1)"),
    BitField("sw", 55, 4, False, "reserved for software use"),
    BitField("pbha", 59, 4, False, "page-based hardware attributes (FEAT_HPDS2)"),
])

# Level 0-2 table descriptor (points at the next-level table). The next-table address and type are
# fixed (they define the walk); only the table-global attribute overrides are fuzzable.
TABLE_LAYOUT = DescriptorLayout("table", [
    BitField("valid", 0, 1, True, "descriptor is valid (0 -> translation fault)"),
    BitField("type", 1, 1, False, "must be 1 for a table descriptor (0 = block)"),
    BitField("next_addr", 12, 36, False, "next-level table address [47:12] -- never fuzzed"),
    BitField("pxn_table", 59, 1, True, "PXNTable: privileged execute-never for the subtree"),
    BitField("xn_table", 60, 1, True, "UXNTable/XNTable for the subtree"),
    BitField("ap_table", 61, 2, True, "APTable: access-permission override for the subtree"),
    BitField("ns_table", 63, 1, True, "NSTable for the subtree"),
])


LAYOUTS: Dict[str, DescriptorLayout] = {LEAF_LAYOUT.name: LEAF_LAYOUT, TABLE_LAYOUT.name: TABLE_LAYOUT}
