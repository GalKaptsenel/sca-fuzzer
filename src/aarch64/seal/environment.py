"""The per-variant page-table ENVIRONMENT for PTE sealing.

PTE sealing is the first NI primitive whose genuine/decoy difference is the page-table environment,
not an instruction word. This module is that environment, kept separate from code / relocations:

  * `SandboxPage` / `SandboxPageMap` -- the general partition of the sandbox data pages into those
    reachable on the retiring path (arch + spec; identical PTEs in every variant) and those reached
    only speculatively (`spec_only`, the fuzzed axis). No page count is baked in.
  * `PteOverride` / `EnvironmentPlan` -- what a variant asks the kernel to change. An override is a
    position-INDEPENDENT `(mask, value)` on a page's descriptor: the kernel applies
    `new = (live & ~mask) | value` and reverts, so it needs no descriptor-field knowledge and no
    absolute address crosses the boundary. The genuine variant ships an EMPTY plan (pristine PTEs);
    a decoy ships overrides on spec-only pages only.
  * `PteFuzzPolicy` -- turns "these fields are fuzzable" (by name, via `pagetable_model`) into
    genuine / decoy plans. Descriptor-bit knowledge stays in `pagetable_model`; page knowledge
    stays in `SandboxPageMap`; the sealer above sees only clean plans.
"""
import struct
from dataclasses import dataclass
from random import Random
from typing import Dict, List, Optional, Sequence, Tuple

from ...interfaces import PAGE_SIZE, MAIN_AREA_SIZE
from .pagetable_model import DescriptorLayout, LEAF_LAYOUT, TABLE_LAYOUT

# Transport level ids (shared contract with the kernel): the page-table level a descriptor sits at.
LEVEL_LEAF = 3
LEVEL_LAYOUTS: Dict[int, DescriptorLayout] = {LEVEL_LEAF: LEAF_LAYOUT, 0: TABLE_LAYOUT,
                                              1: TABLE_LAYOUT, 2: TABLE_LAYOUT}


@dataclass(frozen=True)
class SandboxPage:
    """One page of the sandbox data region. `index` is the shared contract with the kernel's
    page-role table; `offset` is the page's byte offset from the sandbox base (= the `main` region,
    matching get_sandbox_addr and the kernel's sysfs base) -- the unit the arch-safety guard and the
    kernel use to locate the page. `spec_only` marks a page reached only speculatively (the fuzzed
    axis); the rest are reachable on the retiring path with identical PTEs across variants."""
    index: int
    name: str
    offset: int
    spec_only: bool = False

    def contains(self, address: int, base: int) -> bool:
        """Whether `address` falls in this page, given the sandbox base virtual address."""
        start = base + self.offset
        return start <= address < start + PAGE_SIZE


class SandboxPageMap:
    """The ordered sandbox data pages. Encapsulates the layout knowledge the rest of the stack would
    otherwise hard-code: which pages are spec-only, and -- since an architectural access may spill
    into the next page -- which retiring-reachable pages sit just before a spec-only page and thus
    need an access-size-bounded clamp (a spill into another arch-reachable page is fine; into a
    spec-only page is not)."""

    def __init__(self, pages: Sequence[SandboxPage]) -> None:
        self._pages: Tuple[SandboxPage, ...] = tuple(pages)
        names = set()
        for i, p in enumerate(self._pages):
            if p.index != i:
                raise ValueError(f"page {p.name!r} index {p.index} != position {i}")
            if p.name in names:
                raise ValueError(f"duplicate page name {p.name!r}")
            names.add(p.name)

    def spec_only_containing(self, address: int, base: int) -> Optional[SandboxPage]:
        """The spec-only page `address` falls in (given the sandbox base VA), or None. The
        arch-safety guard uses it to reject an arch access that lands on a spec-only page."""
        for p in self.spec_only_pages:
            if p.contains(address, base):
                return p
        return None

    @property
    def pages(self) -> Tuple[SandboxPage, ...]:
        return self._pages

    @property
    def arch_pages(self) -> List[SandboxPage]:
        return [p for p in self._pages if not p.spec_only]

    @property
    def spec_only_pages(self) -> List[SandboxPage]:
        return [p for p in self._pages if p.spec_only]

    def page(self, name: str) -> SandboxPage:
        for p in self._pages:
            if p.name == name:
                return p
        raise KeyError(f"no sandbox page named {name!r}")

    def successor(self, page: SandboxPage) -> Optional[SandboxPage]:
        nxt = page.index + 1
        return self._pages[nxt] if nxt < len(self._pages) else None

    def arch_access_needs_size_clamp(self, page: SandboxPage) -> bool:
        """True when a full-width arch access based in `page` could spill into a page that is NOT
        arch-reachable (a spec-only successor, or off the end of the region). A spill into another
        arch-reachable page is allowed (returns False); a spec-only page is never an arch base."""
        if page.spec_only:
            return False
        nxt = self.successor(page)
        return nxt is None or nxt.spec_only

    def require_spec_only(self) -> None:
        if not self.spec_only_pages:
            raise ValueError(
                "PTE sealing needs at least one spec-only page in the sandbox page map")


def default_sandbox_page_map() -> SandboxPageMap:
    """The current fuzzable sandbox data pages, addressed by byte offset from the sandbox base (the
    `main` region; see get_sandbox_addr): `main` (arch-reachable) and `faulty` (spec-only, the one
    page a test case may reach only speculatively). The partition is general -- widening the
    spec-only set later (e.g. extra vmap'd pages) changes this one factory and the kernel page-role
    table, not the policy, the executor, or the transport."""
    return SandboxPageMap([
        SandboxPage(0, "main", offset=0),
        SandboxPage(1, "faulty", offset=MAIN_AREA_SIZE, spec_only=True),
    ])


@dataclass(frozen=True)
class PteOverride:
    """A position-independent change to one page's descriptor at one level: set the `mask` bits to
    the corresponding bits of `value` (`value` carries no bits outside `mask`). The kernel applies
    `new = (live & ~mask) | value` and reverts."""
    page_index: int
    level: int
    mask: int
    value: int

    def __post_init__(self) -> None:
        if self.value & ~self.mask:
            raise ValueError(f"override value {self.value:#x} has bits outside mask {self.mask:#x}")


# SEC_PTE_SETTINGS codec: PTE overrides <-> bytes, kept beside PteOverride (the encoder maps this
# payload to its REIF section; the kernel parses the identical record layout). The payload is the
# packed records back to back -- no count header; the count is the payload length / record size.
_PTE_RECORD = struct.Struct("<HHQQ")   # page_index (u16), level (u16), mask (u64), value (u64)


def serialize_pte_overrides(overrides: Sequence[PteOverride]) -> bytes:
    out = bytearray()
    for o in overrides:
        out += _PTE_RECORD.pack(o.page_index, o.level, o.mask, o.value)
    return bytes(out)


def deserialize_pte_overrides(blob: bytes) -> Tuple[PteOverride, ...]:
    if 0 != len(blob) % _PTE_RECORD.size:
        raise ValueError("PTE override section is not a whole number of entries")
    return tuple(PteOverride(*_PTE_RECORD.unpack_from(blob, pos))
                 for pos in range(0, len(blob), _PTE_RECORD.size))


@dataclass(frozen=True)
class EnvironmentPlan:
    """A variant's run-environment deltas: what the kernel applies before the run and reverts after.
    An umbrella over environment ASPECTS, deliberately NOT tied to page tables -- today the only
    aspect is `pte_overrides`, but another (e.g. system-register or cache state) is added as a new
    field here plus a codec beside its data, with no change to how the executor or kernel thread a
    plan through. The genuine variant's plan is empty."""
    pte_overrides: Tuple[PteOverride, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.pte_overrides

    def pte_for_page(self, page_index: int) -> List[PteOverride]:
        return [o for o in self.pte_overrides if o.page_index == page_index]


class PteFuzzPolicy:
    """Turns "these descriptor fields are fuzzable" into genuine / decoy `EnvironmentPlan`s over a
    `SandboxPageMap`. The genuine plan is empty (pristine PTEs); a decoy perturbs each spec-only
    page's descriptor within the allowed fields."""

    def __init__(self, leaf_fields: Sequence[str], table_fields: Sequence[str] = ()) -> None:
        # validate the names against the layouts up front (loud on a typo)
        LEAF_LAYOUT.mask_for(leaf_fields)
        TABLE_LAYOUT.mask_for(table_fields)
        self._leaf_fields = list(leaf_fields)
        self._table_fields = list(table_fields)

    def genuine_plan(self) -> EnvironmentPlan:
        return EnvironmentPlan()

    def decoy_plan(self, page_map: SandboxPageMap, rng: Random) -> EnvironmentPlan:
        page_map.require_spec_only()
        overrides = [self._decoy_override(p.index, LEVEL_LEAF, self._leaf_fields, rng)
                     for p in page_map.spec_only_pages]
        return EnvironmentPlan(pte_overrides=tuple(overrides))

    # -- internals: all descriptor-bit reasoning is delegated to the layout ------------------------
    def _decoy_override(self, page_index: int, level: int, fields: Sequence[str],
                        rng: Random) -> PteOverride:
        layout = LEVEL_LAYOUTS[level]
        chosen = [f for f in fields if rng.random() < 0.5] or [rng.choice(list(fields))]
        mask = value = 0
        for name in chosen:
            f = layout.field(name)
            mask |= f.mask
            value = f.insert(value, rng.randint(0, f.max_value))
        return PteOverride(page_index, level, mask, value)
