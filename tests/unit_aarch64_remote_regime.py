"""
Tests for remote (SSH/ADB) sysfs regime control, which lets cross-input priming run against a device
reached over a Connection (RemoteHWExecutor in aarch64_kernel.py; regime_controllable /
executor_regime in aarch64_fuzzer.py).

A FakeConnection records the shell commands and serves canned sysfs values, so the regime round-trip
is tested with no hardware. Run from the repo root:
    python -m unittest tests.unit_aarch64_remote_regime
"""
import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.aarch64.aarch64_kernel import RemoteHWExecutor, RemoteExecutorConfig  # noqa: E402
from src.aarch64.aarch64_fuzzer import regime_controllable, executor_regime  # noqa: E402

SYSFS = "/sys/executor"


class FakeConnection:
    """Records every shell command; serves sysfs reads from `values` (updated by writes)."""
    def __init__(self, values):
        self.values = dict(values)
        self.calls = []

    def shell(self, cmd, privileged=False):
        self.calls.append(cmd)
        if cmd.startswith("cat "):
            name = cmd[len("cat "):].strip().rsplit("/", 1)[-1]
            return self.values[name] + "\n"
        if cmd.startswith("echo "):
            value = cmd[len("echo "):].split('"')[1]
            name = cmd.rsplit("/", 1)[-1]
            self.values[name] = value
            return ""
        raise AssertionError(f"unexpected shell command: {cmd!r}")


def make_remote(conn):
    """A RemoteHWExecutor wired to `conn`, bypassing _ensure_ready (no device I/O)."""
    exe = RemoteHWExecutor.__new__(RemoteHWExecutor)
    exe._conn = conn
    exe._cfg = RemoteExecutorConfig(device="/dev/executor", sysfs=SYSFS,
                                    module="/x.ko", userland="/x")
    exe._sysfs_written = {}
    return exe


class RemoteSysfsTest(unittest.TestCase):
    def test_write_sysfs_echoes_to_the_path(self):
        conn = FakeConnection({})
        make_remote(conn)._write_sysfs("pin_to_core", b"8")
        self.assertEqual(conn.calls, ['echo "8" > /sys/executor/pin_to_core'])

    def test_read_sysfs_cats_and_strips(self):
        conn = FakeConnection({"enable_phr_flush": "0"})
        self.assertEqual(make_remote(conn)._read_sysfs("enable_phr_flush"), "0")
        self.assertEqual(conn.calls, ['cat /sys/executor/enable_phr_flush'])

    def test_regime_save_value_prefers_written_over_dump_on_read(self):
        # pin_to_core dumps CPU info on read (not round-trippable); the written value is restored.
        conn = FakeConnection({"pin_to_core": "CPU ID : 8\nMPIDR_EL1 : 0x81000800"})
        exe = make_remote(conn)
        exe._write_sysfs("pin_to_core", b"8")
        self.assertEqual(exe.regime_save_value("pin_to_core"), "8")

    def test_regime_save_value_reads_an_unwritten_knob(self):
        conn = FakeConnection({"enable_view_rotation": "0"})
        self.assertEqual(make_remote(conn).regime_save_value("enable_view_rotation"), "0")


class RegimeControlTest(unittest.TestCase):
    def test_remote_executor_is_regime_controllable(self):
        exe = make_remote(FakeConnection({}))
        self.assertTrue(regime_controllable(SimpleNamespace(device=exe)))

    def test_non_hw_device_is_not_regime_controllable(self):
        self.assertFalse(regime_controllable(SimpleNamespace(device=object())))

    def test_executor_regime_applies_then_restores(self):
        conn = FakeConnection({"enable_phr_flush": "1", "pin_to_core": "8"})
        exe = make_remote(conn)
        regime = (("enable_phr_flush", "0"), ("pin_to_core", "-1"))
        with executor_regime(SimpleNamespace(device=exe), regime):
            self.assertEqual(conn.values["enable_phr_flush"], "0")
            self.assertEqual(conn.values["pin_to_core"], "-1")
        self.assertEqual(conn.values["enable_phr_flush"], "1")
        self.assertEqual(conn.values["pin_to_core"], "8")

    def test_executor_regime_restores_a_dump_on_read_knob_to_its_written_value(self):
        # pin_to_core was set to 8 at init (cached); its read dumps CPU info, so restore must use 8.
        conn = FakeConnection({"pin_to_core": "CPU ID : 8\nMPIDR_EL1 : 0x81000800"})
        exe = make_remote(conn)
        exe._write_sysfs("pin_to_core", b"8")
        with executor_regime(SimpleNamespace(device=exe), (("pin_to_core", "-1"),)):
            self.assertEqual(conn.values["pin_to_core"], "-1")
        self.assertEqual(conn.values["pin_to_core"], "8")


if __name__ == "__main__":
    unittest.main()
