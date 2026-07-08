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
from src.aarch64.aarch64_disasm import decode_reg_accesses
from src.aarch64.aarch64_input_layout import map_register_to_offsets
from src.config import CONF, ConfigException

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
    before: int = 0


@dataclass
class _Meta:
    speculation_nesting: int
    has_memory_access: bool = False
    memory_access: Optional[_MA] = None
    is_pair: bool = False
    memory_access2: Optional[_MA] = None
    window_id: int = 0

    def accesses(self):
        if not self.has_memory_access:
            return []
        return [self.memory_access, self.memory_access2] if self.is_pair else [self.memory_access]


@dataclass
class _CPU:
    gpr: List[int]
    pc: int = 0
    encoding: int = 0
    sp: int = 0
    nzcv: int = 0


@dataclass
class _ITE:
    metadata: _Meta
    cpu: _CPU


def _make_trace(*steps: dict) -> Tuple[List[_ITE], object]:
    """
    Build a mock CE trace.

    Each step is a dict with optional keys:
      depth   : int              speculation nesting (default 0)
      window  : int              window id override; by default ids are auto-assigned from the depth
                                 transitions (as the CE does). Set this to model a same-depth re-fork
                                 after a hidden unwind — a new flow that shares a depth but not an id.
      srcs    : List[str]        register source operands
      dests   : List[str]        register destination operands
      mem     : (offset, size, is_write[, before])  memory access relative to sandbox base
      mem2    : (offset, size, is_write[, before])  second pair element (LDP/STP); marks is_pair
      pc      : int              instruction PC (default 0)
      sp,nzcv : int              register state at this instruction (default 0)

    Returns (trace, fake_get_srcs_dests) ready for use with patch().
    """
    def _mk_ma(spec):
        rel_off, size, is_write = spec[0], spec[1], spec[2]
        before = spec[3] if len(spec) > 3 else 0
        return _MA(SANDBOX_BASE + rel_off, size, is_write, before=before)

    ites: List[_ITE] = []
    patch_map: Dict[int, Tuple[List[str], List[str]]] = {}

    # Mirror the CE's window bookkeeping: a monotonic counter that never decrements, plus a per-depth
    # stack. Depth+1 opens a fresh window; the same depth continues the same window; an explicit
    # 'window' forces a distinct id (the hidden-unwind sibling the depth number alone cannot express).
    win_stack: List[int] = [0]
    win_counter = 0
    for step in steps:
        depth = step.get('depth', 0)
        if 'window' in step:
            wid = step['window']
            win_stack[depth:] = [wid]
        elif depth >= len(win_stack):
            while len(win_stack) < depth:      # multi-fork: skipped intermediate windows (no instr)
                win_counter += 1
                win_stack.append(win_counter)
            win_counter += 1
            wid = win_counter
            win_stack.append(wid)
        else:                                  # depth < len: same window continues or unwind
            del win_stack[depth + 1:]
            wid = win_stack[depth]
        step['_wid'] = wid

    for enc, step in enumerate(steps):
        gprs = [0] * 31
        gprs[29] = SANDBOX_BASE

        mem = step.get('mem')
        ma = _mk_ma(mem) if mem else None

        meta = _Meta(step.get('depth', 0), ma is not None, ma, window_id=step['_wid'])
        mem2 = step.get('mem2')
        if mem2:
            meta.is_pair = True
            meta.memory_access2 = _mk_ma(mem2)
        ites.append(_ITE(meta, _CPU(gprs, pc=step.get('pc', 0), encoding=enc,
                                    sp=step.get('sp', 0), nzcv=step.get('nzcv', 0))))
        patch_map[enc] = (step.get('srcs', []), step.get('dests', []))

    def _fake(encoding, pc):
        return patch_map.get(encoding, ([], []))

    return ites, _fake


_PATCH = 'src.aarch64.aarch64_trace.decode_reg_accesses'


def _taint(*steps: dict):
    trace, fake = _make_trace(*steps)
    with patch(_PATCH, side_effect=fake):
        return compute_taint(trace)


import copy as _copy

_SAVED_CONF = None


def setUpModule():
    # CONF is a Borg singleton; snapshot it so per-test contract_observation_clause changes here
    # cannot leak into other modules during a full `unittest discover` run.
    global _SAVED_CONF
    _SAVED_CONF = _copy.deepcopy(CONF._borg_shared_state)


def tearDownModule():
    CONF._borg_shared_state.clear()
    CONF._borg_shared_state.update(_SAVED_CONF)


def _ctrace(clause: str, *steps: dict):
    trace, _ = _make_trace(*steps)
    saved = CONF.contract_observation_clause
    CONF.contract_observation_clause = clause
    try:
        return compute_ctrace(trace)
    finally:
        CONF.contract_observation_clause = saved


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
        tt.enter(0, 0)
        tt.on_write([0, 1, 2])
        self.assertFalse(tt.must_preserve)

    def test_read_before_write_preserved(self):
        tt = _TaintTracker()
        tt.enter(0, 0)
        tt.on_read([5])
        self.assertIn(5, tt.must_preserve)

    def test_two_spec_branches_union_of_reads(self):
        # Two sequential spec branches from the same arch point (distinct window ids).
        # First branch reads 5; second branch reads 8. Both must be preserved.
        tt = _TaintTracker()
        tt.enter(1, 1)
        tt.on_read([5])
        tt.enter(0, 0)            # first spec exits
        tt.enter(1, 2)           # second branch: same depth, fresh window
        tt.on_read([8])
        tt.enter(0, 0)
        self.assertIn(5, tt.must_preserve)
        self.assertIn(8, tt.must_preserve)

    def test_second_spec_branch_sees_only_arch_writes(self):
        # First spec branch writes 5. Second spec branch (new window, after exit) reads 5.
        # The first spec's write was squashed — second branch sees only arch writes.
        tt = _TaintTracker()
        tt.enter(1, 1)
        tt.on_write([5])
        tt.enter(0, 0)           # first spec exits, write squashed
        tt.enter(1, 2)           # new fresh spec sub-tree (distinct window)
        tt.on_read([5])          # no arch write or prior spec write of 5 → preserved
        self.assertIn(5, tt.must_preserve)

    def test_same_depth_sibling_write_does_not_mask_read(self):
        # THE FIX: two flows share depth 1 with NO depth dip between them (hidden unwind + re-fork),
        # but carry distinct window ids. A write in the first flow must NOT mask a read in the second.
        tt = _TaintTracker()
        tt.enter(1, 1)
        tt.on_write([7])         # dead sibling writes 7
        tt.enter(1, 2)           # re-fork at the same depth — new window, no depth change
        tt.on_read([7])          # live flow reads the seed value of 7 → must be preserved
        self.assertIn(7, tt.must_preserve)

    def test_same_depth_same_window_write_masks_read(self):
        # Control for the above: same depth AND same window id = one continuous flow, so an earlier
        # write DOES mask the later read (write-before-read on the same path).
        tt = _TaintTracker()
        tt.enter(1, 1)
        tt.on_write([7])
        tt.enter(1, 1)           # same window continues
        tt.on_read([7])
        self.assertNotIn(7, tt.must_preserve)

    def test_deep_nesting_stack_integrity(self):
        # 0→1→2→3→2→1→0, verify each level's writes are scoped to their window.
        tt = _TaintTracker()
        tt.enter(0, 0); tt.on_write([0])
        tt.enter(1, 1); tt.on_write([1])
        tt.enter(2, 2); tt.on_write([2])
        tt.enter(3, 3); tt.on_write([3])
        # At depth 3: the live path is windows {0,1,2,3} → sees 0,1,2,3
        tt.on_read([0, 1, 2, 3])
        self.assertFalse(tt.must_preserve)

        tt.enter(2, 2)            # depth-3 window squashed
        tt.on_read([3])          # 3 was written only in window 3, now gone → preserved
        self.assertIn(3, tt.must_preserve)

        tt.enter(1, 1)            # depth-2 window squashed
        tt.on_read([2])          # 2 was written only in window 2, now gone → preserved
        self.assertIn(2, tt.must_preserve)

        tt.enter(0, 0)            # depth-1 window squashed
        tt.on_read([1])          # 1 was written only in window 1, now gone → preserved
        self.assertIn(1, tt.must_preserve)

        tt.on_read([0])          # 0 was arch-written in window 0 → still live → not preserved
        self.assertNotIn(0, tt.must_preserve)

    def test_multi_fork_depth_jump_fills_empty_windows(self):
        # One instruction can open several windows at once (bpas phase-B push + cond fork), so nesting
        # can rise by >1. The skipped level carries no logged instruction (no writes); a read at the
        # deeper level is still preserved (nothing wrote it) and later use of the skipped level works.
        tt = _TaintTracker()
        tt.enter(0, 0); tt.on_write([1])        # arch writes byte 1
        tt.enter(3, 7)                          # jump 0 -> 3 (windows 1,2 skipped/empty, 3 is real)
        tt.on_read([1])                         # arch write is on the live path -> not preserved
        tt.on_read([9])                         # never written on any live window -> preserved
        self.assertNotIn(1, tt.must_preserve)
        self.assertIn(9, tt.must_preserve)
        # unwinding into a previously-skipped level still works (no dangling placeholder)
        tt.enter(1, 4); tt.on_read([1])
        self.assertNotIn(1, tt.must_preserve)   # arch (window 0) still on the live path

    def test_empty_offsets_noop(self):
        tt = _TaintTracker()
        tt.enter(0, 0)
        tt.on_read([])
        tt.on_write([])
        self.assertFalse(tt.must_preserve)

    def test_partial_overlap_preserved(self):
        # Write covers bytes 0-3, read covers bytes 2-5 → bytes 4-5 not written → preserved.
        tt = _TaintTracker()
        tt.enter(0, 0)
        tt.on_write([0, 1, 2, 3])
        tt.on_read([2, 3, 4, 5])
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
            {'depth': 1, 'srcs': [],     'dests': []},       # fork to depth 1
            {'depth': 2, 'srcs': [],     'dests': ['x5']},   # nested write
            {'depth': 1, 'srcs': ['x5'], 'dests': []},       # back at depth 1: nested write squashed
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

    def test_pair_store_writes_both_elements(self):
        # STP x0,x1,[base+0x10] writes 16 CONTIGUOUS bytes (element0 at 0x10, element1 at 0x18).
        # A pure store doesn't taint, so make the writes observable: a later read of the same 16
        # bytes is fully satisfied by the store → NONE preserved. If the 2nd element's write were
        # missed, reading 0x18..0x1f would be unsatisfied and those bytes would be preserved.
        t = _taint(
            {'depth': 0, 'srcs': ['x0', 'x1'], 'dests': [], 'mem': (0x10, 8, True), 'mem2': (0x18, 8, True)},
            {'depth': 0, 'srcs': [], 'dests': ['x2'], 'mem': (0x10, 16, False)},
        )
        for b in range(16):
            self.assertFalse(_mem_preserved(t, 0x10 + b), f'byte {b}')

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

    def test_sibling_flow_flag_write_does_not_mask_branch_read(self):
        # Regression for violation-260707-213240: a dead speculative sibling writes the N flag
        # (SUBS/ADDS), then execution re-forks at the SAME depth (hidden unwind) and a conditional
        # branch reads the seed N flag. The flag byte must stay must-preserve — otherwise boosting
        # flips the branch and groups two architecturally-distinct inputs (false positive).
        t = _taint(
            {'depth': 1, 'srcs': ['x1', 'x4'], 'dests': ['N', 'Z', 'C', 'V', 'x2']},  # dead sibling: SUBS writes flags
            {'depth': 1, 'srcs': ['N'], 'dests': [], 'window': 99},                    # re-fork: B.pl reads seed N
        )
        self.assertTrue(_flag_preserved(t, 'N'), "seed N flag must survive the sibling's write")


# ===========================================================================
# compute_ctrace tests
# ===========================================================================

class TestComputeCtraceL1D(unittest.TestCase):
    """l1d: an unordered union bitmap of touched cache sets, arch and speculative merged."""

    def test_single_read_sets_its_bit(self):
        ct = _ctrace("l1d", {'mem': (64, 1, False)})   # set 1
        self.assertEqual(ct.raw, [1 << 1])

    def test_union_of_distinct_sets(self):
        ct = _ctrace("l1d",
                     {'mem': (64, 1, False)},          # set 1
                     {'mem': (128, 1, False)})         # set 2
        self.assertEqual(ct.raw, [(1 << 1) | (1 << 2)])

    def test_order_independent(self):
        a = _ctrace("l1d", {'mem': (64, 1, False)}, {'mem': (128, 1, False)})
        b = _ctrace("l1d", {'mem': (128, 1, False)}, {'mem': (64, 1, False)})
        self.assertEqual(a.raw, b.raw)

    def test_arch_and_speculative_merged(self):
        ct = _ctrace("l1d",
                     {'depth': 0, 'mem': (5 * 64, 8, False)},   # arch set 5
                     {'depth': 1, 'mem': (3 * 64, 8, False)})   # spec set 3
        self.assertEqual(ct.raw, [(1 << 5) | (1 << 3)])

    def test_repeated_set_idempotent(self):
        ct = _ctrace("l1d", {'mem': (0, 1, False)}, {'mem': (16, 1, False)})  # both set 0
        self.assertEqual(ct.raw, [1 << 0])

    def test_multi_byte_access_spans_lines(self):
        ct = _ctrace("l1d", {'mem': (0, 128, False)})   # sets 0 and 1
        self.assertEqual(ct.raw, [(1 << 0) | (1 << 1)])

    def test_hash_equals_bitmap(self):
        ct = _ctrace("l1d", {'mem': (128, 1, False)})
        self.assertEqual(ct.hash_, 1 << 2)

    def test_no_memory_access_zero_bitmap(self):
        ct = _ctrace("l1d", {'srcs': ['x1'], 'dests': ['x0']})
        self.assertEqual(ct.raw, [0])


class TestComputeCtraceClauses(unittest.TestCase):
    """The ordered clauses derived from the CE trace, mirroring the x86 tracers."""

    def test_none_is_null(self):
        ct = _ctrace("none", {'pc': 0x100, 'mem': (64, 8, False)})
        self.assertEqual(ct.raw, [])

    def test_pc_records_normalized_pcs(self):
        ct = _ctrace("pc",
                     {'pc': 0x100}, {'pc': 0x104}, {'pc': 0x108})
        self.assertEqual(ct.raw, [0, 4, 8])           # relative to the first PC

    def test_memory_records_sandbox_relative_addresses(self):
        ct = _ctrace("memory",
                     {'pc': 0x100, 'mem': (0x40, 8, False)},
                     {'pc': 0x104, 'mem': (0x80, 8, True)})
        self.assertEqual(ct.raw, [0x40, 0x80])        # PC absent; stores included

    def test_ct_interleaves_pc_then_addresses(self):
        ct = _ctrace("ct",
                     {'pc': 0x100, 'mem': (0x40, 8, False)},
                     {'pc': 0x104, 'mem': (0x80, 8, True)})
        self.assertEqual(ct.raw, [0, 0x40, 4, 0x80])

    def test_ct_nonspecstore_drops_speculative_stores(self):
        ct = _ctrace("ct-nonspecstore",
                     {'pc': 0x100, 'depth': 1, 'mem': (0x40, 8, True)},    # spec store -> dropped
                     {'pc': 0x104, 'depth': 1, 'mem': (0x80, 8, False)})   # spec load  -> kept
        self.assertEqual(ct.raw, [0, 4, 0x80])

    def test_ctr_prepends_initial_registers(self):
        ct = _ctrace("ctr", {'pc': 0x100, 'sp': 0xabc, 'nzcv': 0xf, 'mem': (0x40, 8, False)})
        self.assertEqual(len(ct.raw), 31 + 2 + 2)     # 31 gpr + sp + nzcv, then [pc, addr]
        self.assertEqual(ct.raw[29], SANDBOX_BASE)    # base register seeded
        self.assertEqual(ct.raw[31], 0xabc)           # sp
        self.assertEqual(ct.raw[32], 0xf)             # nzcv
        self.assertEqual(ct.raw[33:], [0, 0x40])

    def test_arch_records_loaded_value_then_address(self):
        ct = _ctrace("arch", {'pc': 0x100, 'mem': (0x40, 8, False, 0xdeadbeef)})
        self.assertEqual(ct.raw[33:], [0, 0xdeadbeef, 0x40])   # pc, loaded value, address

    def test_arch_store_has_no_loaded_value(self):
        ct = _ctrace("arch", {'pc': 0x100, 'mem': (0x40, 8, True, 0x11)})
        self.assertEqual(ct.raw[33:], [0, 0x40])               # pc, address (no value)

    def test_tct_truncates_to_cache_line(self):
        ct = _ctrace("tct", {'pc': 0x104, 'mem': (0x48, 8, False)})   # pc line 0x100, addr line 0x40
        self.assertEqual(ct.raw, [0, 0x40])                          # (pc 4 -> line 0), (0x48 -> 0x40)

    def test_tcto_adds_line_crossing(self):
        ct = _ctrace("tcto", {'pc': 0x100, 'mem': (0x38, 16, False)})  # 0x38..0x47 crosses 0x40
        self.assertEqual(ct.raw, [0, 0x0, 0x40])                       # pc line 0; addr lines 0x00,0x40

    def test_unimplemented_clause_rejected(self):
        with self.assertRaises(ConfigException):
            _ctrace("wibble", {'pc': 0x100})


class TestMemoryOperandRegisterTaint(unittest.TestCase):
    """The address registers of a memory operand are READ (so tainted), and pre/post-index WRITEBACK
    additionally WRITES the base. Encodings assembled with `as` and pinned here."""

    # ldr x0,[x1] / [x1,#8]! / [x1],#8
    LDR_NOWB, LDR_PRE, LDR_POST = 0xf9400020, 0xf8408c20, 0xf8408420
    # str x0,[x1] / [x1,#8]! / [x1],#8
    STR_NOWB, STR_PRE, STR_POST = 0xf9000020, 0xf8008c20, 0xf8008420
    # ldp x0,x1,[x2] / [x2,#16]! / [x2],#16
    LDP_NOWB, LDP_PRE, LDP_POST = 0xa9400440, 0xa9c10440, 0xa8c10440
    # stp x0,x1,[x2] / [x2,#16]! / [x2],#16
    STP_NOWB, STP_PRE, STP_POST = 0xa9000440, 0xa9810440, 0xa8810440

    _NOWB = {LDR_NOWB: "x1", STR_NOWB: "x1", LDP_NOWB: "x2", STP_NOWB: "x2"}
    _WB = {LDR_PRE: "x1", LDR_POST: "x1", STR_PRE: "x1", STR_POST: "x1",
           LDP_PRE: "x2", LDP_POST: "x2", STP_PRE: "x2", STP_POST: "x2"}

    # ---- decode layer ----
    def test_base_register_always_read(self):
        for enc, base in {**self._NOWB, **self._WB}.items():
            src, _ = decode_reg_accesses(enc, 0)
            self.assertIn(base, src, f"0x{enc:08x}: base must be read")

    def test_no_writeback_base_not_written(self):
        for enc, base in self._NOWB.items():
            _, dest = decode_reg_accesses(enc, 0)
            self.assertNotIn(base, dest, f"0x{enc:08x}: no writeback → base not written")

    def test_writeback_base_read_and_written(self):
        for enc, base in self._WB.items():
            src, dest = decode_reg_accesses(enc, 0)
            self.assertIn(base, src, f"0x{enc:08x}: writeback base is read first")
            self.assertIn(base, dest, f"0x{enc:08x}: writeback also writes the base")

    def test_load_writes_data_store_reads_data(self):
        # LDP loads x0,x1 (written); STP stores x0,x1 (read).
        self.assertEqual(set(decode_reg_accesses(self.LDP_NOWB, 0)[1]), {"x0", "x1"})
        self.assertTrue({"x0", "x1"} <= set(decode_reg_accesses(self.STP_NOWB, 0)[0]))

    # ---- compute_taint integration: read-first base stays tainted under writeback ----
    def _trace_one(self, encoding, mem):
        gprs = [0] * 31
        gprs[29] = SANDBOX_BASE
        off, size, is_write = mem
        ma = _MA(SANDBOX_BASE + off, size, is_write)
        return [_ITE(_Meta(0, True, ma), _CPU(gprs, encoding=encoding))]

    def test_compute_taint_preserves_writeback_base(self):
        # ldr x0,[x1],#8 — x1 is read for the address (read-first), so its GPR input bytes are
        # preserved even though the writeback also writes x1.
        t = compute_taint(self._trace_one(self.LDR_POST, (0x10, 8, False)))
        self.assertTrue(_gpr_preserved(t, 1), "writeback base x1 must stay tainted (read-first)")

    def test_compute_taint_preserves_store_writeback_base(self):
        # str x0, [x1, #8]! (pre-index) — base x1 read-first → preserved.
        t = compute_taint(self._trace_one(self.STR_PRE, (0x10, 8, True)))
        self.assertTrue(_gpr_preserved(t, 1), "store writeback base x1 must stay tainted")


if __name__ == '__main__':
    unittest.main()
