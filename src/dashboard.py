"""
File: Interactive curses dashboard for live fuzzing sessions.

Activated automatically when stdout is a TTY (see util.Logger). Falls back to the
plain single-line output otherwise, so piped/backgrounded runs are unaffected.

Controls (applied cooperatively at round boundaries):
  p  pause / resume the fuzzer
  q  stop the fuzzer gracefully (after the current round)
  s  toggle the detailed discard-counter breakdown
  c  toggle the full command line
  up/down  scroll the violations list
"""
import curses
import atexit
import locale
import time
from collections import deque
from datetime import datetime
from typing import Dict, Optional

# ncurses needs the locale set for correct wide/UTF-8 character handling.
locale.setlocale(locale.LC_ALL, "")


def _fmt_dur(seconds: float) -> str:
    s = int(max(0, seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


class Dashboard:
    def __init__(self, meta: Dict):
        self.meta = meta
        self.start_time: datetime = meta["start_time"]
        self.work_dir: str = meta.get("work_dir", ".")

        self.paused = False
        self.should_stop = False
        self.show_stats = False
        self.minimized = False
        self.scroll = 0
        self.notes: deque = deque(maxlen=500)   # scrolling log lines (seeds, events…)
        self.rerun = meta.get("rerun") or meta.get("cmd", "")   # reproduce command
        self._last_render = 0.0
        self._last_phase = ""
        self.sample_size = 0                     # current measurement sample size

        self._viol_cache: list = []   # [(name, detected_at_seconds)] — this session only

        self.scr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        curses.curs_set(0)
        self.scr.keypad(True)
        self.scr.nodelay(True)        # non-blocking getch
        try:
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_RED, -1)
            curses.init_pair(2, curses.COLOR_GREEN, -1)
            curses.init_pair(3, curses.COLOR_CYAN, -1)
            curses.init_pair(4, curses.COLOR_YELLOW, -1)
        except curses.error:
            pass
        atexit.register(self.stop)

    # -- terminal lifecycle -----------------------------------------------------
    def stop(self):
        if self.scr is None:
            return
        try:
            self.scr.nodelay(False)
            curses.nocbreak()
            self.scr.keypad(False)
            curses.echo()
            curses.curs_set(1)
            curses.endwin()
        except curses.error:
            pass
        finally:
            self.scr = None

    def set_message(self, msg: str):
        if msg:
            self.notes.append(msg)

    # -- input ------------------------------------------------------------------
    def _poll_keys(self):
        while True:
            try:
                ch = self.scr.getch()
            except curses.error:
                break
            if ch == -1:
                break
            if ch in (ord('q'), ord('Q')):
                self.should_stop = True
            elif ch in (ord('p'), ord('P'), ord(' ')):
                self.paused = not self.paused
            elif ch in (ord('s'), ord('S')):
                self.show_stats = not self.show_stats
            elif ch in (ord('m'), ord('M')):
                self.minimized = not self.minimized
            elif ch == curses.KEY_UP:
                self.scroll = max(0, self.scroll - 1)
            elif ch == curses.KEY_DOWN:
                self.scroll += 1

    # -- violations -------------------------------------------------------------
    def add_violation(self, name: str, at: float):
        # Recorded when the violation is actually found, so the time-to-detection
        # is correct and old runs' artifacts are never picked up.
        self._viol_cache.append((name, at))

    # -- rendering --------------------------------------------------------------
    def tick(self, stat, elapsed: float, phase: str = "running"):
        if self.scr is None:
            return
        self._poll_keys()
        # Throttle redraws (~10 fps) so very fast rounds don't pay a redraw each
        # time; always redraw promptly on a phase change.
        now = time.monotonic()
        if phase != self._last_phase or (now - self._last_render) >= 0.1:
            self._render(stat, elapsed, phase)
            self._last_render = now
            self._last_phase = phase

        # Pausing blocks the caller (the fuzzer loop) here until resumed/quit.
        while self.paused and not self.should_stop:
            self._poll_keys()
            self._render(stat, elapsed, "PAUSED")
            curses.napms(100)

    def finish_hold(self, stat, elapsed: float):
        """Keep the dashboard open after fuzzing finishes so the user can scroll
        and inspect the results; exits when 'q' is pressed."""
        if self.scr is None:
            return
        self.paused = False
        self.set_message("fuzzing finished — press q to exit")
        while not self.should_stop:
            self._poll_keys()
            self._render(stat, elapsed, "finished")
            curses.napms(100)

    def _eta(self, stat, elapsed: float) -> Optional[float]:
        timeout = self.meta.get("timeout", 0)
        if timeout:
            return max(0.0, timeout - elapsed)
        n = self.meta.get("num_test_cases", 0)
        if n and stat.test_cases and elapsed > 0:
            rate = stat.test_cases / elapsed
            return max(0.0, (n - stat.test_cases) / rate) if rate else None
        return None

    def _render(self, stat, elapsed: float, phase: str):
        scr = self.scr
        try:
            h, w = scr.getmaxyx()
            scr.erase()
            if w < 40:
                scr.addnstr(0, 0, "term too narrow", max(0, w - 1))
                scr.refresh()
                return

            rate = stat.test_cases / elapsed if elapsed > 0 else 0.0
            filtered = stat.fast_path + stat.spec_filter + stat.observ_filter
            eta = self._eta(stat, elapsed)
            timeout = self.meta.get("timeout", 0)

            C_TITLE = curses.color_pair(3) | curses.A_BOLD
            C_VIOL = curses.color_pair(1) | curses.A_BOLD
            C_OK = curses.color_pair(2)
            C_KEY = curses.color_pair(4) | curses.A_BOLD
            C_DIM = curses.A_DIM

            def line(y, x, s, attr=0):
                if 0 <= y < h:
                    scr.addnstr(y, x, s, max(0, w - 1 - x), attr)

            if phase == "finished":
                badge, battr = "✓ finished — press q to exit", C_OK
            elif self.paused:
                badge, battr = "⏸ PAUSED", C_KEY
            elif self.should_stop:
                badge, battr = "■ stopping…", C_VIOL
            else:
                badge, battr = f"▶ {phase}", C_OK
            eta_s = _fmt_dur(eta) if eta is not None else "—"
            tlimit = f" / {_fmt_dur(timeout)}" if timeout else ""

            # Minimized: compact summary (works even in a tiny terminal)
            if self.minimized:
                head = f" Revizor {self.meta.get('mode','?')} · {self.meta.get('isa','?')}"
                line(0, 0, head.ljust(w - 1), C_TITLE)
                line(0, max(0, w - 2 - len(badge)), badge, battr)
                v = stat.violations
                vstr = f"⚠ {v} violations" if v else "0 violations"
                line(1, 1, f"{stat.test_cases} tc · {_fmt_dur(elapsed)}{tlimit} · "
                           f"{rate:.2f} tc/s · {vstr}", C_VIOL if v else 0)
                line(2, 0, " [m] expand  [p]ause  [q]uit ".ljust(w - 1), curses.A_REVERSE)
                scr.refresh()
                return

            if h < 14:
                line(0, 0, "terminal too short — press m to minimize")
                scr.refresh()
                return

            # Header
            title = f" Revizor Fuzzer — {self.meta.get('mode','?')} · {self.meta.get('isa','?')} "
            line(0, 0, title.ljust(w - 1), C_TITLE)
            line(1, 1, f"config  {self.meta.get('config','?')}    spec  {self.meta.get('spec','?')}", C_DIM)
            line(2, 1, f"cmd  {self.meta.get('cmd','')}", C_DIM)

            y = 3
            line(y, 0, "─" * (w - 1), C_DIM)
            y += 1

            line(y, 1, f"elapsed {_fmt_dur(elapsed)}{tlimit}     ETA {eta_s}")
            line(y, w - 2 - len(badge), badge, battr)
            y += 1
            iptc = stat.num_inputs // stat.test_cases if stat.test_cases else 0
            an = stat.analysed_test_cases
            all_cls = (stat.eff_classes + stat.single_entry_classes) // an if an else 0
            eff_cls = stat.eff_classes // an if an else 0
            line(y, 1, f"test cases {stat.test_cases:<8} rate {rate:.2f} tc/s")
            y += 1
            sample_str = str(self.sample_size) if self.sample_size else "—"
            line(y, 1, f"inputs/tc {iptc:<6} classes {eff_cls}/{all_cls} (eff/total)   sample {sample_str}")
            y += 1
            line(y, 1, f"filtered {filtered}")
            y += 1
            v = stat.violations
            line(y, 1, f"⚠ violations {v}" if v else "violations 0", C_VIOL if v else C_OK)
            y += 1

            if self.show_stats:
                line(y, 0, "─" * (w - 1), C_DIM); y += 1
                line(y, 1, f"spec-filter {stat.spec_filter}  obs-filter {stat.observ_filter}  fast-path {stat.fast_path}", C_DIM); y += 1
                line(y, 1, f"nesting {stat.fp_nesting}  taint {stat.fp_taint_mistakes}  early-prime {stat.fp_early_priming}  large-sample {stat.fp_large_sample}  prime {stat.fp_priming}", C_DIM); y += 1

            # Split the remaining space (above the footer) between the violations
            # list and the notes panel, sized to the actual terminal height.
            footer_row = h - 1
            body_rows = footer_row - y               # rows for both panels incl. headers
            n_viol = len(self._viol_cache)
            if body_rows >= 4:
                inner = body_rows - 2                # 2 single-line panel headers
                viol_rows = min(max(n_viol, 1), max(1, inner // 2))
                notes_rows = inner - viol_rows

                # Violations panel
                line(y, 0, "── Violations (time to detection) ".ljust(w - 1, "─"), C_TITLE)
                y += 1
                list_top = y
                max_scroll = max(0, n_viol - viol_rows)
                if self.scroll > max_scroll:
                    self.scroll = max_scroll
                if n_viol == 0:
                    line(list_top, 2, "none yet", C_DIM)
                else:
                    for row, idx in enumerate(range(self.scroll, min(n_viol, self.scroll + viol_rows))):
                        name, at = self._viol_cache[idx]
                        line(list_top + row, 1, f"{idx+1:>3}  {name}   +{_fmt_dur(at)}", C_VIOL)
                y = list_top + viol_rows

                # Notes panel: re-run command pinned on top, then a scrolling tail
                # of recent notes (newest at the bottom; oldest scroll off the top).
                line(y, 0, "── Notes ".ljust(w - 1, "─"), C_TITLE)
                y += 1
                avail = notes_rows
                if self.rerun and avail > 0:
                    line(y, 1, "re-run: " + self.rerun, C_KEY)
                    y += 1
                    avail -= 1
                if avail > 0:
                    for i, note in enumerate(list(self.notes)[-avail:]):
                        line(y + i, 1, "» " + note, C_DIM)

            footer = " [m]inimize  [p]ause  [q]uit  [↑/↓] scroll viols  [s]tats "
            line(footer_row, 0, footer.ljust(w - 1), curses.A_REVERSE)
            scr.refresh()
        except curses.error:
            # Terminal resize / out-of-bounds; skip this frame.
            pass
