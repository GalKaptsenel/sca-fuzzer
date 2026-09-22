"""The per-variant page-table ENVIRONMENT for PTE sealing.

PTE sealing is the first NI primitive whose genuine/decoy difference is the page-table environment, not
an instruction word. This module is that environment, kept entirely separate from code / relocations:

  * `SandboxPage` / `SandboxPageMap` -- the general partition of the sandbox data pages into those
    reachable on the retiring path (arch + spec; identical PTEs in every variant) and those reached only
    speculatively (`spec_only`, the fuzzed axis). No page count is baked in.
  * `PteOverride` / `EnvironmentPlan` -- what a variant asks the kernel to change. An override is a
    position-INDEPENDENT `(mask, value)` on a page's descriptor: the kernel applies
    `new = (live & ~mask) | value` and reverts afterwards, so it needs no knowledge of descriptor fields
    and no absolute address ever crosses the boundary. The genuine variant ships an EMPTY plan (pristine
    PTEs); a decoy ships overrides on spec-only pages only.
  * `PteFuzzPolicy` -- turns "these fields are fuzzable" (by name, via `pagetable_model`) into concrete
    genuine / decoy / forced plans. All descriptor-bit knowledge stays in `pagetable_model`; all page
    knowledge stays in `SandboxPageMap`; the sealer above sees only clean plans.
"""
import struct
from dataclasses import dataclass
from random import Random
from typing import Dict, List, Optional, Sequence, Tuple

from .pagetable_model import DescriptorLayout, LEAF_LAYOUT, TABLE_LAYOUT

# Transport level ids (shared contract with the kernel): the page-table level a descriptor sits at.
LEVEL_LEAF = 3
LEVEL_LAYOUTS: Dict[int, DescriptorLayout] = {LEVEL_LEAF: LEAF_LAYOUT, 0: TABLE_LAYOUT,
                                              1: TABLE_LAYOUT, 2: TABLE_LAYOUT}


@dataclass(frozen=True)
class SandboxPage:
    """One page of the sandbox data region, identified by its index (the shared contract with the
    kernel's page-role table) and a human name. `spec_only` marks a page reached only speculatively --
    the fuzzed axis; the rest are reachable on the retiring path with identical PTEs across variants."""
    index: int
    name: str
    spec_only: bool = False


class SandboxPageMap:
    """The ordered sandbox data pages. Encapsulates the layout knowledge the rest of the stack would
    otherwise hard-code: which pages are spec-only, and -- because an architectural access may spill into
    the next page -- which retiring-reachable pages sit just before a spec-only page and therefore need
    an access-size-bounded clamp (a spill into another arch-reachable page is fine; into a spec-only page
    is not)."""

    def __init__(self, pages: Sequence[SandboxPage]) -> None:
        self._pages: Tuple[SandboxPage, ...] = tuple(pages)
        names = set()
        for i, p in enumerate(self._pages):
            if p.index != i:
                raise ValueError(f"page {p.name!r} index {p.index} != position {i}")
            if p.name in names:
                raise ValueError(f"duplicate page name {p.name!r}")
            names.add(p.name)

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
        """True when a full-width architectural access based in `page` could spill into a page that is
        NOT arch-reachable (a spec-only successor, or off the end of the region). Spilling into another
        arch-reachable page is allowed, so those return False; a spec-only page is never an arch base."""
        if page.spec_only:
            return False
        nxt = self.successor(page)
        return nxt is None or nxt.spec_only

    def require_spec_only(self) -> None:
        if not self.spec_only_pages:
            raise ValueError("PTE sealing needs at least one spec-only page in the sandbox page map")


def default_sandbox_page_map() -> SandboxPageMap:
    """The current sandbox data-page layout (must stay in step with the kernel's page-role table and
    `sandbox_t` in executor/sandbox.h): lower_overflow, main, faulty, upper_overflow. Only `faulty` is
    spec-only for now; the abstraction supports any partition, so widening the spec-only set later is a
    change to this one factory, not to the sealer or the kernel protocol."""
    return SandboxPageMap([
        SandboxPage(0, "lower_overflow"),
        SandboxPage(1, "main"),
        SandboxPage(2, "faulty", spec_only=True),
        SandboxPage(3, "upper_overflow"),
    ])


@dataclass(frozen=True)
class PteOverride:
    """A position-independent change to one page's descriptor at one level: set the bits in `mask` to the
    corresponding bits of `value` (`value` carries no bits outside `mask`). The kernel applies
    `new = (live & ~mask) | value` and reverts."""
    page_index: int
    level: int
    mask: int
    value: int

    def __post_init__(self) -> None:
        if self.value & ~self.mask:
            raise ValueError(f"override value {self.value:#x} has bits outside mask {self.mask:#x}")


# SEC_PTE_SETTINGS codec: PTE overrides <-> bytes, kept beside PteOverride (the encoder maps this payload
# to its REIF section; the kernel parses the identical record layout). A new environment aspect brings its
# own codec beside its own data -- nothing here needs to know about it.
_PTE_RECORD = struct.Struct("<HHQQ")   # page_index (u16), level (u16), mask (u64), value (u64)
_PTE_HEADER = struct.Struct("<I")      # count (u32)


def serialize_pte_overrides(overrides: Sequence[PteOverride]) -> bytes:
    out = bytearray(_PTE_HEADER.pack(len(overrides)))
    for o in overrides:
        out += _PTE_RECORD.pack(o.page_index, o.level, o.mask, o.value)
    return bytes(out)


def deserialize_pte_overrides(blob: bytes) -> Tuple[PteOverride, ...]:
    (count,) = _PTE_HEADER.unpack_from(blob, 0)
    pos = _PTE_HEADER.size
    result = []
    for _ in range(count):
        pi, lvl, mask, val = _PTE_RECORD.unpack_from(blob, pos)
        result.append(PteOverride(pi, lvl, mask, val))
        pos += _PTE_RECORD.size
    return tuple(result)


@dataclass(frozen=True)
class EnvironmentPlan:
    """A variant's run-environment deltas: what the kernel applies before the run and reverts after. An
    umbrella over environment ASPECTS, deliberately NOT tied to page tables -- today the only aspect is
    `pte_overrides`, but another (e.g. system-register or cache state) is added as a new field here plus a
    codec beside its data, with no change to how the executor or kernel thread a plan through. The genuine
    variant's plan is empty."""
    pte_overrides: Tuple[PteOverride, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.pte_overrides

    def pte_for_page(self, page_index: int) -> List[PteOverride]:
        return [o for o in self.pte_overrides if o.page_index == page_index]


class PteFuzzPolicy:
    """Turns "these descriptor fields are fuzzable" into genuine / decoy / forced `EnvironmentPlan`s over
    a `SandboxPageMap`. The genuine plan is empty (pristine PTEs). A decoy plan perturbs each spec-only
    page's descriptor within the allowed fields; a forced plan applies a guaranteed-strong perturbation
    (used by cross-input priming's always-bad lane)."""

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

    def forced_plan(self, page_map: SandboxPageMap) -> EnvironmentPlan:
        """A deterministic, guaranteed-different override on every spec-only page: clear `valid` (a sure
        speculative translation fault) when it is fuzzable, else flip the first allowed field high."""
        page_map.require_spec_only()
        overrides = [self._forced_override(p.index, LEVEL_LEAF, self._leaf_fields)
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

    def _forced_override(self, page_index: int, level: int, fields: Sequence[str]) -> PteOverride:
        layout = LEVEL_LAYOUTS[level]
        if "valid" in fields:
            f = layout.field("valid")
            return PteOverride(page_index, level, f.mask, f.insert(0, 0))
        f = layout.field(list(fields)[0])
        return PteOverride(page_index, level, f.mask, f.insert(0, f.max_value))
