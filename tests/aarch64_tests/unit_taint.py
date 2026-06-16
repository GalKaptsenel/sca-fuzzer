"""
Unit tests for the taint analysis: _TaintTracker, compute_taint, compute_ctrace,
map_register_to_offsets.
"""
import unittest
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from unittest.mock import patch
import numpy as np

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # run from any cwd
from src.aarch64.aarch64_trace import _TaintTracker, compute_taint, compute_ctrace
from src.aarch64.aarch64_input_layout import map_register_to_offsets

# ===========================================================================
# Mock CE trace infrastructure
# ===========================================================================

SANDBOX_BASE = 0x4000_0000


@dataclass
class _MA:
    effective_address: int
    element_size: int
    is_write: bool
    is_atomic: bool = False


@dataclass
class _Meta:
    speculation_nesting: int
    has_memory_access: bool = False
    memory_access: Optional[_MA] = None
    is_pair: bool = False
    memory_access2: Optional[_MA] = None

    def accesses(self):
        if not self.has_memory_access:
            return []
        return [self.memory_access, self.memory_access2] if self.is_pair else [self.memory_access]


@dataclass
class _CPU:
    gpr: List[int]
    pc: int = 0
    encoding: int = 0


@dataclass
class _ITE:
    metadata: _Meta
    cpu: _CPU


def _make_trace(*steps: dict) -> Tuple[List[_ITE], object]:
    """
    Build a mock CE trace.

    Each step is a dict with optional keys:
      depth   : int              speculation nesting (default 0)
      srcs    : List[str]        register source operands
      dests   : List[str]        register destination operands
      mem     : (offset, size, is_write)  memory access relative to sandbox base
      mem2    : (offset, size, is_write)  second pair element (LDP/STP); marks the entry is_pair

    Returns (trace, fake_get_srcs_dests) ready for use with patch().
    """
    ites: List[_ITE] = []
    patch_map: Dict[int, Tuple[List[str], List[str]]] = {}

    for enc, step in enumerate(steps):
        gprs = [0] * 31
        gprs[29] = SANDBOX_BASE

        mem = step.get('mem')
        if mem:
            rel_off, size, is_write = mem
            ma = _MA(SANDBOX_BASE + rel_off, size, is_write)
        else:
            ma = None

        meta = _Meta(step.get('depth', 0), ma is not None, ma)
        mem2 = step.get('mem2')
        if mem2:
            rel_off2, size2, is_write2 = mem2
            meta.is_pair = True
            meta.memory_access2 = _MA(SANDBOX_BASE + rel_off2, size2, is_write2)
        ites.append(_ITE(meta, _CPU(gprs, encoding=enc)))
        patch_map[enc] = (step.get('srcs', []), step.get('dests', []))

    def _fake(encoding, pc):
        return patch_map.get(encoding, ([], []))

    return ites, _fake


_PATCH = 'src.aarch64.aarch64_trace.decode_reg_accesses'


def _taint(*steps: dict):
    trace, fake = _make_trace(*steps)
    with patch(_PATCH, side_effect=fake):
        return compute_taint(trace)


def _ctrace(*steps: dict):
    trace, _ = _make_trace(*steps)
    return compute_ctrace(trace)


def _gpr_preserved(t, slot: int) -> bool:
    return any(t[0]["gpr"].view(np.uint8)[slot * 8: slot * 8 + 8])


def _mem_preserved(t, offset: int) -> bool:
    return bool(t.view(np.uint8)[offset])


_FLAG_BYTE = {'N': 48, 'Z': 49, 'C': 50, 'V': 51}


def _flag_preserved(t, flag: str) -> bool:
    return bool(t[0]["gpr"].view(np.uint8)[_FLAG_BYTE[flag]])


# Slot indices for readability
X0, X1, X2, X3, X4, X5, FLAGS, SP = range(8)


# ===========================================================================
# TaintTracker unit tests
# ===========================================================================

class TestTaintTracker(unittest.TestCase):

    def test_unread_cell_not_preserved(self):
        tt = _TaintTracker()
        tt.set_depth(0)
        tt.on_write([0, 1, 2], depth=0)
        self.assertFalse(tt.must_preserve)

    def test_read_before_write_preserved(self):
        tt = _TaintTracker()
        tt.set_depth(0)
        tt.on_read([5], depth=0)
        self.assertIn(5, tt.must_preserve)

    def test_two_spec_branches_union_of_reads(self):
        # Two sequential spec branches from the same arch point.
        # First branch reads 5; second branch reads 8. Both must be preserved.
        tt = _TaintTracker()
        tt.set_depth(1)
        tt.on_read([5], depth=1)
        tt.set_depth(0)           # first spec exits
        tt.set_depth(1)
        tt.on_read([8], depth=1)
        tt.set_depth(0)
        self.assertIn(5, tt.must_preserve)
        self.assertIn(8, tt.must_preserve)

    def test_second_spec_branch_sees_only_arch_writes(self):
        # First spec branch writes 5. Second spec branch (new, after exit) reads 5.
        # The first spec's write was squashed — second branch sees only arch writes.
        tt = _TaintTracker()
        tt.set_depth(1)
        tt.on_write([5], depth=1)
        tt.set_depth(0)           # first spec exits, write squashed
        tt.set_depth(1)           # new fresh spec sub-tree
        tt.on_read([5], depth=1)  # no arch write or prior spec write of 5 → preserved
        self.assertIn(5, tt.must_preserve)

    def test_deep_nesting_stack_integrity(self):
        # 0→1→2→3→2→1→0, verify each level's writes are scoped correctly.
        tt = _TaintTracker()
        tt.set_depth(0); tt.on_write([0], depth=0)
        tt.set_depth(1); tt.on_write([1], depth=1)
        tt.set_depth(2); tt.on_write([2], depth=2)
        tt.set_depth(3); tt.on_write([3], depth=3)
        # At depth 3: sees 0,1,2,3
        tt.on_read([0, 1, 2, 3], depth=3)
        self.assertFalse(tt.must_preserve)

        tt.set_depth(2)           # depth-3 squashed
        tt.on_read([3], depth=2)  # 3 was written only at depth-3, now gone → preserved
        self.assertIn(3, tt.must_preserve)

        tt.set_depth(1)           # depth-2 squashed
        tt.on_read([2], depth=1)  # 2 was written only at depth-2, now gone → preserved
        self.assertIn(2, tt.must_preserve)

        tt.set_depth(0)           # depth-1 squashed
        tt.on_read([1], depth=0)  # 1 was written only at depth-1, now gone → preserved
        self.assertIn(1, tt.must_preserve)

        tt.on_read([0], depth=0)  # 0 was arch-written at depth-0 → still written → not preserved
        self.assertNotIn(0, tt.must_preserve)

    def test_empty_offsets_noop(self):
        tt = _TaintTracker()
        tt.set_depth(0)
        tt.on_read([], depth=0)
        tt.on_write([], depth=0)
        self.assertFalse(tt.must_preserve)

    def test_partial_overlap_preserved(self):
        # Write covers bytes 0-3, read covers bytes 2-5 → bytes 4-5 not written → preserved.
        tt = _TaintTracker()
        tt.set_depth(0)
        tt.on_write([0, 1, 2, 3], depth=0)
        tt.on_read([2, 3, 4, 5], depth=0)
        self.assertNotIn(2, tt.must_preserve)
        self.assertNotIn(3, tt.must_preserve)
        self.assertIn(4, tt.must_preserve)
        self.assertIn(5, tt.must_preserve)


# ===========================================================================
# Register offset mapping tests
# ===========================================================================

class TestRegisterOffsets(unittest.TestCase):

    def test_x_register_8_bytes(self):
        for n in range(6):
            self.assertEqual(map_register_to_offsets(f'x{n}'),
                             list(range(n * 8, n * 8 + 8)), f'x{n}')

    def test_w_register_maps_to_same_slot_as_x(self):
        # wN is the low 32 bits of xN — same input slot. A read through wN must preserve
        # xN's input bytes, so it maps to the full 8-byte slot (same as xN).
        for n in range(6):
            self.assertEqual(map_register_to_offsets(f'w{n}'),
                             list(range(n * 8, n * 8 + 8)), f'w{n}')
        # Reserved w-registers (w6..w30) are not usable -> empty, like xN.
        for n in range(6, 31):
            self.assertEqual(map_register_to_offsets(f'w{n}'), [], f'w{n}')

    def test_out_of_range_registers_empty(self):
        for n in range(6, 31):
            self.assertEqual(map_register_to_offsets(f'x{n}'), [], f'x{n}')

    def test_ignored_special_registers(self):
        for reg in ('fp', 'lr', 'xzr', 'wzr', 'FP', 'XZR'):
            self.assertEqual(map_register_to_offsets(reg), [], reg)

    def test_flags_per_flag_granularity(self):
        # Each flag gets its own single byte within the flags slot (slot 6, base = 48).
        expected = {'N': 48, 'Z': 49, 'C': 50, 'V': 51}
        for flag, byte in expected.items():
            self.assertEqual(map_register_to_offsets(flag), [byte], flag)
            self.assertEqual(map_register_to_offsets(flag.lower()), [byte], flag.lower())

    def test_flags_are_independent_bytes(self):
        offsets = [map_register_to_offsets(f)[0] for f in ('N', 'Z', 'C', 'V')]
        self.assertEqual(len(offsets), len(set(offsets)), "flag bytes must be distinct")

    def test_sp_8_bytes(self):
        sp_base = 7 * 8
        self.assertEqual(map_register_to_offsets('sp'), list(range(sp_base, sp_base + 8)))

    def test_unknown_operand_empty(self):
        for reg in ('q0', 'v0', 'xyz', '', 'pstate'):
            self.assertEqual(map_register_to_offsets(reg), [])


# ===========================================================================
# compute_taint integration tests
# ===========================================================================

class TestComputeTaint(unittest.TestCase):

    def test_empty_trace_no_taint(self):
        t = _taint()
        self.assertFalse(any(t.view(np.uint8)))

    def test_arch_reg_read_taints(self):
        t = _taint({'depth': 0, 'srcs': ['x1'], 'dests': []})
        self.assertTrue(_gpr_preserved(t, X1))
        self.assertFalse(_gpr_preserved(t, X0))

    def test_arch_write_then_arch_read_no_taint(self):
        t = _taint(
            {'depth': 0, 'srcs': [],     'dests': ['x2']},  # write x2
            {'depth': 0, 'srcs': ['x2'], 'dests': []},      # read x2
        )
        self.assertFalse(_gpr_preserved(t, X2))

    # --- Key spec/arch ordering scenarios ---

    def test_spec_read_before_arch_write_preserved(self):
        # DFS: spec sub-tree comes before the arch write that follows the branch.
        t = _taint(
            {'depth': 1, 'srcs': ['x0'], 'dests': []},      # spec reads x0 → must preserve
            {'depth': 0, 'srcs': [],     'dests': ['x0']},   # arch writes x0 (too late)
        )
        self.assertTrue(_gpr_preserved(t, X0))

    def test_arch_write_before_spec_read_freed(self):
        # Arch writes x0 BEFORE the branch → spec reads the arch-written value.
        t = _taint(
            {'depth': 0, 'srcs': [],     'dests': ['x0']},   # arch write first
            {'depth': 1, 'srcs': ['x0'], 'dests': []},       # spec reads written value
        )
        self.assertFalse(_gpr_preserved(t, X0))

    def test_spec_write_squashed_arch_read_preserved(self):
        # Spec writes x1, spec exits (squash), arch reads x1 → sees original input.
        t = _taint(
            {'depth': 1, 'srcs': [],     'dests': ['x1']},   # spec write (squashed)
            {'depth': 0, 'srcs': ['x1'], 'dests': []},       # arch reads original value
        )
        self.assertTrue(_gpr_preserved(t, X1))

    def test_spec_write_then_spec_read_same_path_freed(self):
        # Within same speculative path: write before read → freed.
        t = _taint(
            {'depth': 1, 'srcs': [],     'dests': ['x2']},   # spec write
            {'depth': 1, 'srcs': ['x2'], 'dests': []},       # spec read of spec-written value
        )
        self.assertFalse(_gpr_preserved(t, X2))

    def test_nested_spec_sees_outer_spec_write(self):
        # Depth-1 writes x4, depth-2 reads x4 — depth-2 sees depth-1's write.
        t = _taint(
            {'depth': 1, 'srcs': [],     'dests': ['x4']},
            {'depth': 2, 'srcs': ['x4'], 'dests': []},
        )
        self.assertFalse(_gpr_preserved(t, X4))

    def test_nested_spec_write_squashed_on_exit(self):
        # Depth-2 writes x5, exits to depth-1, depth-1 reads x5 → squashed → preserved.
        t = _taint(
            {'depth': 2, 'srcs': [],     'dests': ['x5']},
            {'depth': 1, 'srcs': ['x5'], 'dests': []},
        )
        self.assertTrue(_gpr_preserved(t, X5))

    def test_write_N_does_not_cover_Z_per_flag(self):
        # Per-flag granularity: writing N (byte 48) does NOT cover Z (byte 49).
        # N written but never read → not preserved. Z read before written → preserved.
        t = _taint(
            {'depth': 0, 'srcs': [],    'dests': ['N']},
            {'depth': 0, 'srcs': ['Z'], 'dests': []},
        )
        self.assertFalse(_flag_preserved(t, 'N'))
        self.assertTrue(_flag_preserved(t, 'Z'))
        self.assertTrue(_gpr_preserved(t, FLAGS))  # slot has at least Z preserved

    def test_flags_read_before_write_preserved(self):
        t = _taint(
            {'depth': 0, 'srcs': ['C'], 'dests': []},
        )
        self.assertTrue(_gpr_preserved(t, FLAGS))

    # --- Memory taint ---

    def test_memory_read_taints_sandbox(self):
        t = _taint({'depth': 0, 'srcs': [], 'dests': [], 'mem': (0x10, 4, False)})
        for b in range(4):
            self.assertTrue(_mem_preserved(t, 0x10 + b), f'byte {b}')

    def test_pair_load_taints_all_16_bytes(self):
        # LDP x0,x1,[base+0x10] reads 16 CONTIGUOUS bytes (element0 at 0x10, element1 at 0x18).
        # compute_taint must process both pair accesses → all 16 bytes preserved.
        t = _taint({'depth': 0, 'srcs': [], 'dests': ['x0', 'x1'],
                    'mem': (0x10, 8, False), 'mem2': (0x18, 8, False)})
        for b in range(16):
            self.assertTrue(_mem_preserved(t, 0x10 + b), f'byte {b}')

    def test_memory_arch_write_then_read_no_taint(self):
        t = _taint(
            {'depth': 0, 'srcs': [], 'dests': [], 'mem': (0x20, 4, True)},   # arch write
            {'depth': 0, 'srcs': [], 'dests': [], 'mem': (0x20, 4, False)},  # arch read
        )
        for b in range(4):
            self.assertFalse(_mem_preserved(t, 0x20 + b), f'byte {b}')

    def test_memory_spec_write_then_spec_read_same_path_no_taint(self):
        t = _taint(
            {'depth': 1, 'srcs': [], 'dests': [], 'mem': (0x40, 8, True)},   # spec write
            {'depth': 1, 'srcs': [], 'dests': [], 'mem': (0x40, 8, False)},  # spec read
        )
        for b in range(8):
            self.assertFalse(_mem_preserved(t, 0x40 + b), f'byte {b}')

    def test_memory_spec_write_squashed_arch_read_preserved(self):
        t = _taint(
            {'depth': 1, 'srcs': [], 'dests': [], 'mem': (0x100, 4, True)},  # spec write (squashed)
            {'depth': 0, 'srcs': [], 'dests': [], 'mem': (0x100, 4, False)}, # arch read
        )
        for b in range(4):
            self.assertTrue(_mem_preserved(t, 0x100 + b), f'byte {b}')

    def test_memory_spec_read_before_arch_write_preserved(self):
        # DFS: spec read of memory comes before arch write → preserved.
        t = _taint(
            {'depth': 1, 'srcs': [], 'dests': [], 'mem': (0x200, 4, False)},  # spec read
            {'depth': 0, 'srcs': [], 'dests': [], 'mem': (0x200, 4, True)},   # arch write later
        )
        for b in range(4):
            self.assertTrue(_mem_preserved(t, 0x200 + b), f'byte {b}')

    def test_out_of_bounds_memory_not_tainted(self):
        # Accesses outside [0, 0x2000) must not write to taint array.
        t = _taint(
            {'depth': 0, 'mem': (-1, 1, False)},      # one byte before sandbox
            {'depth': 0, 'mem': (0x2000, 1, False)},  # one byte after sandbox
        )
        self.assertFalse(any(t.view(np.uint8)))

    def test_multiple_regs_independence(self):
        # x0 written before read (freed), x1 read before written (preserved).
        t = _taint(
            {'depth': 0, 'srcs': [],        'dests': ['x0']},
            {'depth': 0, 'srcs': ['x0', 'x1'], 'dests': []},
        )
        self.assertFalse(_gpr_preserved(t, X0))
        self.assertTrue(_gpr_preserved(t, X1))

    def test_full_spec_scenario(self):
        # Realistic scenario: arch writes x0, branches, spec reads x0 (freed) and x1
        # (not yet written → preserved), arch then writes x1.
        t = _taint(
            {'depth': 0, 'srcs': [],          'dests': ['x0']},   # arch write x0
            {'depth': 1, 'srcs': ['x0', 'x1'], 'dests': []},      # spec reads both
            {'depth': 0, 'srcs': [],           'dests': ['x1']},   # arch write x1 (late)
        )
        self.assertFalse(_gpr_preserved(t, X0))  # written before spec read
        self.assertTrue(_gpr_preserved(t, X1))   # spec read before arch write


# ===========================================================================
# compute_ctrace tests
# ===========================================================================

class TestComputeCtrace(unittest.TestCase):

    def test_empty_trace_empty_ctrace(self):
        ct = _ctrace()
        self.assertEqual(ct.raw, [])

    def test_single_read_correct_cache_set(self):
        ct = _ctrace({'depth': 0, 'mem': (0, 1, False)})
        self.assertEqual(ct.raw, [0])

    def test_cache_set_formula_in_execution_order(self):
        # The trace is the sequence of cache sets in access order: set1, set2, set0 (64*64 wraps).
        ct = _ctrace(
            {'depth': 0, 'mem': (64,       1, False)},
            {'depth': 0, 'mem': (128,      1, False)},
            {'depth': 0, 'mem': (64 * 64,  1, False)},
        )
        self.assertEqual(ct.raw, [1, 2, 0])

    def test_repeated_same_set_accesses_each_recorded(self):
        # Separate accesses to the same line are each recorded (order-preserving, not set-folded).
        ct = _ctrace(
            {'depth': 0, 'mem': (0,  1, False)},
            {'depth': 0, 'mem': (16, 1, False)},
            {'depth': 0, 'mem': (32, 1, False)},
        )
        self.assertEqual(ct.raw, [0, 0, 0])

    def test_order_is_preserved_not_sorted(self):
        ct = _ctrace(
            {'depth': 0, 'mem': (128, 1, False)},  # set 2
            {'depth': 0, 'mem': (64,  1, False)},  # set 1
            {'depth': 0, 'mem': (0,   1, False)},  # set 0
        )
        self.assertEqual(ct.raw, [2, 1, 0])        # execution order, NOT sorted

    def test_write_accesses_appear_in_ctrace(self):
        # Stores also leak via cache → must appear in ctrace.
        ct = _ctrace({'depth': 0, 'mem': (64, 1, True)})
        self.assertEqual(ct.raw, [1])

    def test_speculative_access_recorded_at_its_position(self):
        # Speculation depth no longer shifts the value — the access is recorded as its plain set,
        # distinguished from architectural accesses only by its position in the sequence.
        ct = _ctrace(
            {'depth': 0, 'mem': (5 * 64, 8, False)},    # arch set 5
            {'depth': 1, 'mem': (3 * 64, 8, False)},    # spec set 3 (recorded after, in order)
        )
        self.assertEqual(ct.raw, [5, 3])

    def test_opposite_branch_directions_separated_by_order(self):
        # A: arch X then spec Y → [X, Y];  B: arch Y then spec X → [Y, X].  Order keeps them distinct
        # (no +offset needed), exactly like the x86 ordered contract tracer.
        X, Y = 5, 10
        a = _ctrace({'depth': 0, 'mem': (X * 64, 8, False)},
                    {'depth': 1, 'mem': (Y * 64, 8, False)})
        b = _ctrace({'depth': 0, 'mem': (Y * 64, 8, False)},
                    {'depth': 1, 'mem': (X * 64, 8, False)})
        self.assertEqual(a.raw, [X, Y])
        self.assertEqual(b.raw, [Y, X])
        self.assertNotEqual(a.raw, b.raw)

    def test_multi_byte_access_spans_lines_in_order(self):
        # A 128-byte load spanning two lines → both sets, in ascending address order, from ONE access.
        ct = _ctrace({'depth': 0, 'mem': (0, 128, False)})
        self.assertEqual(ct.raw, [0, 1])

    def test_pair_records_both_elements_in_order(self):
        # LDP/STP reads two adjacent 8-byte elements (element1 = element0 + 8). Straddling a line
        # boundary, element0 (0x38 → set 0) and element1 (0x40 → set 1) are BOTH recorded, in order.
        ct = _ctrace({'depth': 0, 'mem': (0x38, 8, False), 'mem2': (0x40, 8, False)})
        self.assertEqual(ct.raw, [0, 1])

    def test_no_memory_access_empty_ctrace(self):
        ct = _ctrace(
            {'depth': 0, 'srcs': ['x1'], 'dests': ['x0']},
            {'depth': 0, 'srcs': ['x0'], 'dests': ['x2']},
        )
        self.assertEqual(ct.raw, [])


if __name__ == '__main__':
    unittest.main()
