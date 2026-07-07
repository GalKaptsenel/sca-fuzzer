"""
Copyright (C) Microsoft Corporation
SPDX-License-Identifier: MIT
"""
import unittest
from unittest import mock
from collections import deque

from src.dashboard import Dashboard


def _make_dashboard(paused: bool):
    """Build a Dashboard without curses.initscr() (no TTY in tests), wired up
    with just the attributes the wait-loops touch and a non-None fake screen."""
    d = Dashboard.__new__(Dashboard)
    d.scr = object()            # non-None so tick()/finish_hold() proceed
    d.paused = paused
    d.should_stop = False
    d._io_fail = 0
    d._last_phase = "running"
    d._last_render = 1e18       # skip the throttled pre-loop render
    d.notes = deque(maxlen=500)
    return d


class TerminalLostCounterTest(unittest.TestCase):
    """ Regression for the PID-5537 wedge: a paused dashboard whose terminal
    dies (tmux/ssh gone) must not spin forever holding the executor. """

    def test_counter_trips_only_after_limit_and_resets_on_success(self):
        d = _make_dashboard(paused=False)
        # A run of failures below the limit does not trip.
        for _ in range(Dashboard.IO_FAIL_LIMIT - 1):
            self.assertFalse(d._terminal_lost(False))
        # One successful frame resets the streak.
        self.assertFalse(d._terminal_lost(True))
        self.assertEqual(d._io_fail, 0)
        # It takes a full fresh streak to trip.
        for _ in range(Dashboard.IO_FAIL_LIMIT - 1):
            self.assertFalse(d._terminal_lost(False))
        self.assertTrue(d._terminal_lost(False))

    def test_pause_loop_stops_when_terminal_lost(self):
        d = _make_dashboard(paused=True)
        d._poll_keys = mock.MagicMock()                 # no keys ever arrive
        d._render = mock.MagicMock(return_value=False)   # every frame fails
        with mock.patch("src.dashboard.time.sleep"):
            d.tick(stat=object(), elapsed=1.0)
        # Loop exited (test returned) and asked the fuzzer to stop.
        self.assertTrue(d.should_stop)
        self.assertEqual(d._render.call_count, Dashboard.IO_FAIL_LIMIT)

    def test_finish_hold_breaks_when_terminal_lost(self):
        d = _make_dashboard(paused=False)
        d._poll_keys = mock.MagicMock()
        d._render = mock.MagicMock(return_value=False)
        with mock.patch("src.dashboard.time.sleep"):
            d.finish_hold(stat=object(), elapsed=1.0)
        self.assertEqual(d._render.call_count, Dashboard.IO_FAIL_LIMIT)


if __name__ == "__main__":
    unittest.main()
