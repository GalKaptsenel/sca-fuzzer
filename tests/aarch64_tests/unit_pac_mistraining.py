"""
BPU mistraining logic tests.

Tests set_branch_mistraining() in isolation: given a mock CE trace, verify that
the returned (byte_offset, train_direction) entries are correct.  No hardware is
required; local_executor is replaced by a fake that captures the calls.

Invariant under test:
  For each arch-level (nesting==0) conditional branch at index i in the trace:
    taken         = (cer[i+1].cpu.pc != cer[i].cpu.pc + 4)
    byte_offset   = cer[i].cpu.pc - cer[0].cpu.pc     (code_base = first entry PC)
    train_dir     = not taken                           (opposite → guaranteed mispredict)

Skipped cases:
  - speculation_nesting != 0      (speculative instructions)
  - not a conditional branch      (is_conditional_branch returns False)
  - i+1 >= len(cer)               (no successor — can't determine taken/not-taken)
  - cer is None or empty          (clear_branch_training called instead)
"""
import os
import types
import unittest

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # run from any cwd
from src.aarch64.aarch64_executor import Aarch64LocalExecutor, is_conditional_branch

_CODE_BASE_SYSFS = "/sys/executor/print_code_base"
_MODULE_LOADED = os.path.exists(_CODE_BASE_SYSFS)


def _query_code_base() -> int:
    """Read the executor's code base from the loaded kernel module."""
    with open(_CODE_BASE_SYSFS) as f:
        return int(f.read().strip(), 16)

# ---------------------------------------------------------------------------
# AArch64 conditional branch encodings (bits [31:24])
# B.cond: 0x54; CBZ/CBNZ w/x: 0x34/0x35/0xB4/0xB5; TBZ/TBNZ: 0x36/0x37/0xB6/0xB7
# ---------------------------------------------------------------------------
_BCOND_ENC  = 0x54000000   # B.cond — always a conditional branch
_NOP_ENC    = 0xD503201F   # NOP   — never a conditional branch


def _ite(pc: int, encoding: int, nesting: int = 0, has_mem: bool = False):
    """Minimal fake CE trace entry."""
    ite          = types.SimpleNamespace()
    ite.cpu      = types.SimpleNamespace(pc=pc, encoding=encoding)
    ite.metadata = types.SimpleNamespace(speculation_nesting=nesting,
                                         has_memory_access=has_mem)
    return ite


class _FakeLocalExecutor:
    def __init__(self):
        self.last_entries = None
        self.cleared      = False

    def write_branch_training_config(self, entries):
        self.last_entries = entries
        self.cleared      = False

    def clear_branch_training(self):
        self.cleared      = True
        self.last_entries = None


class _Trainer:
    """Minimal object that exposes branch mistraining without hardware."""
    def __init__(self):
        self.local_executor = _FakeLocalExecutor()

    branch_mistraining_entries = Aarch64LocalExecutor.branch_mistraining_entries
    apply_branch_mistraining   = Aarch64LocalExecutor.apply_branch_mistraining


BASE = _query_code_base() if _MODULE_LOADED else 0   # real code base from the kernel module


class TestMistrainingEncoding(unittest.TestCase):
    """is_conditional_branch correctly identifies the instruction classes."""

    def test_bcond_is_conditional(self):
        self.assertTrue(is_conditional_branch(_BCOND_ENC))

    def test_nop_is_not_conditional(self):
        self.assertFalse(is_conditional_branch(_NOP_ENC))

    def test_cbz_w_is_conditional(self):
        self.assertTrue(is_conditional_branch(0x34000000))

    def test_cbnz_x_is_conditional(self):
        self.assertTrue(is_conditional_branch(0xB5000000))

    def test_tbz_w_is_conditional(self):
        self.assertTrue(is_conditional_branch(0x36000000))

    def test_tbnz_x_is_conditional(self):
        self.assertTrue(is_conditional_branch(0xB7000000))

    def test_unconditional_branch_not_conditional(self):
        # B <label> = 0x14xxxxxx  (bits[31:24] = 0x14)
        self.assertFalse(is_conditional_branch(0x14000001))

    def test_bl_not_conditional(self):
        # BL = 0x94xxxxxx
        self.assertFalse(is_conditional_branch(0x94000001))


@unittest.skipUnless(_MODULE_LOADED, "executor kernel module not loaded")
class TestMistrainingEntries(unittest.TestCase):

    def setUp(self):
        self.trainer = _Trainer()

    def _run(self, cer):
        entries = self.trainer.branch_mistraining_entries(cer)
        self.trainer.apply_branch_mistraining(entries)
        return entries

    # -----------------------------------------------------------------------
    # Core taken / not-taken polarity
    # -----------------------------------------------------------------------

    def test_not_taken_branch_trains_taken(self):
        """Branch where next PC == PC+4 (not-taken) → train as taken (opposite)."""
        cer = [
            _ite(BASE + 0, _BCOND_ENC),   # branch
            _ite(BASE + 4, _NOP_ENC),     # sequential next → not-taken
        ]
        entries = self._run(cer)
        self.assertEqual(len(entries), 1)
        offset, direction = entries[0]
        self.assertEqual(offset, 0)         # first instruction → offset 0
        self.assertTrue(direction,          "not-taken branch should train as taken")

    def test_taken_branch_trains_not_taken(self):
        """Branch where next PC != PC+4 (taken) → train as not-taken (opposite)."""
        cer = [
            _ite(BASE + 0, _BCOND_ENC),    # branch
            _ite(BASE + 0x100, _NOP_ENC),  # jump → taken
        ]
        entries = self._run(cer)
        self.assertEqual(len(entries), 1)
        _, direction = entries[0]
        self.assertFalse(direction, "taken branch should train as not-taken")

    def test_direction_from_arch_successor_not_speculative(self):
        """Speculative (nest>0) entries right after the branch must not flip the
        trained direction: direction comes from the next nest-0 successor."""
        # arch NOT-taken (successor = PC+4) with a speculative "taken"-looking interleave
        cer = [
            _ite(BASE + 0,     _BCOND_ENC),                 # arch branch
            _ite(BASE + 0x100, _NOP_ENC, nesting=1),        # wrong-path → looks taken
            _ite(BASE + 0x104, _NOP_ENC, nesting=1),
            _ite(BASE + 4,     _NOP_ENC),                   # arch successor = PC+4 → not-taken
        ]
        _, direction = self._run(cer)[0]
        self.assertTrue(direction, "arch not-taken must train taken despite spec interleave")

        # arch taken (successor != PC+4) with a speculative "not-taken"-looking interleave
        cer = [
            _ite(BASE + 0,     _BCOND_ENC),
            _ite(BASE + 4,     _NOP_ENC, nesting=1),        # wrong-path fall-through → looks not-taken
            _ite(BASE + 0x200, _NOP_ENC),                   # arch successor = jump → taken
        ]
        _, direction = self._run(cer)[0]
        self.assertFalse(direction, "arch taken must train not-taken despite spec interleave")

    # -----------------------------------------------------------------------
    # Byte offset computation
    # -----------------------------------------------------------------------

    def test_byte_offset_relative_to_first_pc(self):
        """byte_offset = ite.cpu.pc - cer[0].cpu.pc (code_base = first entry's PC)."""
        cer = [
            _ite(BASE + 0,  _NOP_ENC),      # first entry sets code_base
            _ite(BASE + 4,  _NOP_ENC),
            _ite(BASE + 8,  _BCOND_ENC),    # branch at offset 8
            _ite(BASE + 12, _NOP_ENC),      # sequential → not-taken
        ]
        entries = self._run(cer)
        self.assertEqual(len(entries), 1)
        offset, _ = entries[0]
        self.assertEqual(offset, 8)

    def test_multiple_branches_all_recorded(self):
        """Every arch-level conditional branch in the trace gets an entry."""
        cer = [
            _ite(BASE + 0,  _BCOND_ENC),   # branch 1 (not-taken)
            _ite(BASE + 4,  _NOP_ENC),
            _ite(BASE + 8,  _NOP_ENC),
            _ite(BASE + 12, _BCOND_ENC),   # branch 2 (taken)
            _ite(BASE + 0x80, _NOP_ENC),
        ]
        entries = self._run(cer)
        self.assertEqual(len(entries), 2)
        offsets = [e[0] for e in entries]
        self.assertIn(0,  offsets)
        self.assertIn(12, offsets)

    # -----------------------------------------------------------------------
    # Filtering: speculative, non-branch, last-instruction
    # -----------------------------------------------------------------------

    def test_speculative_branch_excluded(self):
        """Branches with speculation_nesting != 0 must be ignored."""
        cer = [
            _ite(BASE + 0, _BCOND_ENC, nesting=1),   # speculative → skip
            _ite(BASE + 4, _NOP_ENC,   nesting=1),
        ]
        entries = self._run(cer)
        self.assertEqual(len(entries), 0)

    def test_non_branch_instruction_excluded(self):
        """Non-conditional-branch instructions must not produce entries."""
        cer = [
            _ite(BASE + 0, _NOP_ENC),   # not a branch
            _ite(BASE + 4, _NOP_ENC),
        ]
        entries = self._run(cer)
        self.assertEqual(len(entries), 0)

    def test_last_branch_without_successor_excluded(self):
        """Branch at the last position in the trace (no i+1) must be skipped."""
        cer = [
            _ite(BASE + 0, _NOP_ENC),
            _ite(BASE + 4, _BCOND_ENC),  # last entry — no successor
        ]
        entries = self._run(cer)
        self.assertEqual(len(entries), 0)

    def test_mixed_trace_only_arch_branches_recorded(self):
        """Only arch-level conditional branches contribute to the entry list."""
        cer = [
            _ite(BASE + 0,  _NOP_ENC),              # NOP arch       → skip
            _ite(BASE + 4,  _BCOND_ENC, nesting=1),  # branch spec    → skip
            _ite(BASE + 8,  _NOP_ENC,   nesting=1),
            _ite(BASE + 12, _BCOND_ENC, nesting=0),  # branch arch    → include
            _ite(BASE + 16, _NOP_ENC),               # sequential → not-taken
        ]
        entries = self._run(cer)
        self.assertEqual(len(entries), 1)
        offset, direction = entries[0]
        self.assertEqual(offset, 12)
        self.assertTrue(direction)   # not-taken → train taken

    # -----------------------------------------------------------------------
    # Empty / None trace
    # -----------------------------------------------------------------------

    def test_empty_trace_clears_training(self):
        """cer=[] must call clear_branch_training and return empty list."""
        entries = self._run([])
        self.assertEqual(entries, [])
        self.assertTrue(self.trainer.local_executor.cleared)

    def test_none_trace_clears_training(self):
        """cer=None must call clear_branch_training and return empty list."""
        entries = self._run(None)
        self.assertEqual(entries, [])
        self.assertTrue(self.trainer.local_executor.cleared)

    # -----------------------------------------------------------------------
    # Hardware write path
    # -----------------------------------------------------------------------

    def test_non_empty_entries_written_to_executor(self):
        """When entries exist, write_branch_training_config must be called with them."""
        cer = [
            _ite(BASE + 0, _BCOND_ENC),
            _ite(BASE + 4, _NOP_ENC),
        ]
        entries = self._run(cer)
        self.assertEqual(self.trainer.local_executor.last_entries, entries)
        self.assertFalse(self.trainer.local_executor.cleared)

    def test_empty_result_calls_clear_not_write(self):
        """When no entries (all non-branch), clear_branch_training must be called."""
        cer = [
            _ite(BASE + 0, _NOP_ENC),
            _ite(BASE + 4, _NOP_ENC),
        ]
        self._run(cer)
        self.assertTrue(self.trainer.local_executor.cleared)
        self.assertIsNone(self.trainer.local_executor.last_entries)


if __name__ == '__main__':
    unittest.main(verbosity=2)
