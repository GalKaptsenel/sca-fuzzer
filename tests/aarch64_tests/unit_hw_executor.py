"""make_hw_executor selects the measurement backend from CONF, _make_remote_connection dispatches the
transport, the PAC sign backend stays local even when measurement is remote, and _measure ships one
super-batch unit through the device backend. All device-free (backends/connections are mocked)."""
import unittest
from unittest import mock

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # run from any cwd
from src.config import CONF
import src.aarch64.aarch64_kernel as k
import src.aarch64.aarch64_connection as conn_mod
import src.aarch64.aarch64_executor as emod
from src.aarch64.aarch64_executor import Aarch64LocalExecutor
from src.aarch64.aarch64_kernel import HWMeasurement, TraceUnit, TargetInfo


class _ConfRemoteSnapshot(unittest.TestCase):
    """Snapshot and restore every executor_remote_* CONF field per test, so a test that flips the
    backend to remote cannot leak that into the next test (CONF is a process-wide singleton)."""

    def setUp(self):
        self._saved = {n: getattr(CONF, n) for n in dir(CONF) if n.startswith("executor_remote")}
        self.assertIn("executor_remote", self._saved)   # guard: the fields exist to be restored

    def tearDown(self):
        for name, value in self._saved.items():
            setattr(CONF, name, value)


class BackendSelectionTest(_ConfRemoteSnapshot):
    def test_local_backend_selected(self):
        CONF.executor_remote = False
        with mock.patch.object(k, "LocalHWExecutor") as LHW:
            device = k.make_hw_executor()
        LHW.assert_called_once_with("/dev/executor", "/sys/executor")
        self.assertIs(device, LHW.return_value)

    def test_remote_backend_selected_with_config_from_conf(self):
        CONF.executor_remote = True
        CONF.executor_remote_device = "/dev/foo"
        CONF.executor_remote_sysfs = "/sys/foo"
        CONF.executor_remote_module = "/tmp/foo.ko"
        CONF.executor_remote_userland = "/tmp/foo_userland"
        sentinel_conn = object()
        with mock.patch.object(k, "_make_remote_connection", return_value=sentinel_conn), \
             mock.patch.object(k, "RemoteHWExecutor") as RHW:
            device = k.make_hw_executor()
        self.assertIs(device, RHW.return_value)
        (got_conn, got_cfg), _ = RHW.call_args
        self.assertIs(got_conn, sentinel_conn)
        self.assertEqual((got_cfg.device, got_cfg.sysfs, got_cfg.module, got_cfg.userland),
                         ("/dev/foo", "/sys/foo", "/tmp/foo.ko", "/tmp/foo_userland"))


class TransportDispatchTest(_ConfRemoteSnapshot):
    def test_local_transport(self):
        CONF.executor_remote_transport = "local"
        with mock.patch.object(conn_mod, "LocalConnection") as LC:
            connection = k._make_remote_connection()
        LC.assert_called_once_with()
        self.assertIs(connection, LC.return_value)

    def test_ssh_transport_passes_conf(self):
        CONF.executor_remote_transport = "ssh"
        CONF.executor_remote_host = "device.local"
        CONF.executor_remote_port = 2222
        CONF.executor_remote_user = "revizor"
        CONF.executor_remote_key = "/home/revizor/.ssh/id"
        with mock.patch.object(conn_mod, "SSHConnection") as SSH:
            k._make_remote_connection()
        SSH.assert_called_once_with(host="device.local", port=2222, username="revizor",
                                    password=None, key_filename="/home/revizor/.ssh/id")

    def test_unknown_transport_raises(self):
        CONF.executor_remote_transport = "carrier-pigeon"
        with self.assertRaises(ValueError):
            k._make_remote_connection()


class SignBackendSplitTest(unittest.TestCase):
    def _executor(self):
        ex = Aarch64LocalExecutor.__new__(Aarch64LocalExecutor)
        ex.ignore_list = set()
        return ex

    def test_reuses_local_device_when_measurement_is_local(self):
        ex = self._executor()
        ex._sign_hw = "the-local-device"   # set in __init__ when device is a LocalHWExecutor
        self.assertEqual(ex._local_executor(), "the-local-device")

    def test_opens_local_device_when_measurement_is_remote(self):
        ex = self._executor()
        ex._sign_hw = None                 # remote measurement backend: no local handle yet
        with mock.patch.object(emod, "LocalHWExecutor") as LHW:
            first = ex._local_executor()
            second = ex._local_executor()
        LHW.assert_called_once_with("/dev/executor", "/sys/executor")   # opened once, then cached
        self.assertIs(first, LHW.return_value)
        self.assertIs(second, first)


class MeasureRoutingTest(unittest.TestCase):
    def _executor(self, ignore_list, tc_bytes=b"TESTCASE"):
        ex = Aarch64LocalExecutor.__new__(Aarch64LocalExecutor)
        ex.ignore_list = set(ignore_list)
        ex._log_hw_counters = lambda per_input: None
        ex._current_tc_bytes = lambda: tc_bytes
        ex.device = mock.Mock()
        return ex

    def _input(self, blob):
        inp = mock.Mock()
        inp.serialize.return_value = blob
        return inp

    def test_measure_ships_one_unit_and_aggregates_per_input(self):
        ex = self._executor([])
        ex.device.run_batch.return_value = \
            [[[HWMeasurement(7, (1, 2, 3)), HWMeasurement(9, (1, 2, 3))]]]

        htraces = ex._measure([self._input(b"INPUT")], 2)

        (units, n_reps), _ = ex.device.run_batch.call_args
        self.assertEqual(n_reps, 2)
        self.assertEqual(units, [TraceUnit(test_case=b"TESTCASE", inputs=(b"INPUT",))])
        self.assertEqual(len(htraces), 1)
        self.assertEqual(htraces[0].raw, [7, 9])

    def test_ignored_input_is_zeroed_through_measure(self):
        ex = self._executor([0])
        ex.device.run_batch.return_value = [[[HWMeasurement(5, (0, 0, 0))]]]

        htraces = ex._measure([self._input(b"I")], 1)
        self.assertEqual(htraces[0].raw, [0])


class TraceBatchTest(unittest.TestCase):
    """trace_batch measures many make_trace_unit units in one run_batch and returns per-unit htraces."""

    def _executor(self):
        ex = Aarch64LocalExecutor.__new__(Aarch64LocalExecutor)
        ex.ignore_list = set()
        ex._log_hw_counters = lambda per_input: None
        ex.device = mock.Mock()
        return ex

    def _unit(self, ex, tc_bytes, input_blobs):
        ex._current_tc_bytes = lambda: tc_bytes
        inputs = []
        for blob in input_blobs:
            inp = mock.Mock()
            inp.serialize.return_value = blob
            inputs.append(inp)
        return ex.make_trace_unit(inputs)

    def test_one_run_batch_for_all_units_preserving_order(self):
        ex = self._executor()
        units = [self._unit(ex, b"TC0", [b"a", b"b"]), self._unit(ex, b"TC1", [b"c"])]
        self.assertEqual(units, [TraceUnit(b"TC0", (b"a", b"b")), TraceUnit(b"TC1", (b"c",))])
        ex.device.run_batch.return_value = [
            [[HWMeasurement(1, (0, 0, 0))], [HWMeasurement(2, (0, 0, 0))]],
            [[HWMeasurement(3, (0, 0, 0))]],
        ]

        per_unit = ex.trace_batch(units, 1)

        ex.device.run_batch.assert_called_once_with(units, 1)   # exactly one super-batch
        htraces = [[h.raw for h in job] for job in per_unit]
        self.assertEqual(htraces, [[[1], [2]], [[3]]])

    def test_empty_batch_makes_no_device_call(self):
        ex = self._executor()
        self.assertEqual(ex.trace_batch([], 5), [])
        ex.device.run_batch.assert_not_called()

    def test_rejects_nonempty_ignore_list(self):
        ex = self._executor()
        ex.ignore_list = {0}
        with self.assertRaises(AssertionError):
            ex.trace_batch([TraceUnit(b"TC", (b"x",))], 1)


class BackendTransparencyTest(unittest.TestCase):
    """Local and Remote are interchangeable because the executor measures ONLY through the HWExecutor
    interface (target_info + run_batch) — never a backend-specific method."""

    def test_both_backends_implement_the_interface(self):
        self.assertTrue(issubclass(k.LocalHWExecutor, k.HWExecutor))
        self.assertTrue(issubclass(k.RemoteHWExecutor, k.HWExecutor))
        self.assertEqual(k.HWExecutor.__abstractmethods__, frozenset({"target_info", "run_batch"}))

    def test_measurement_path_uses_interface_only(self):
        # autospec of the ABC exposes ONLY target_info/run_batch; if the executor reached for a
        # Local-only method (checkout/write/...), these calls would AttributeError.
        ex = Aarch64LocalExecutor.__new__(Aarch64LocalExecutor)
        ex.ignore_list = set()
        ex._log_hw_counters = lambda per_input: None
        ex._current_tc_bytes = lambda: b"TC"
        ex.device = mock.create_autospec(k.HWExecutor, instance=True)
        ex.device.target_info.return_value = TargetInfo(sandbox_base=0x1000, code_base=0x2000)
        ex.device.run_batch.return_value = [[[HWMeasurement(3, (0, 0, 0))]]]

        self.assertEqual(ex.read_base_addresses(), (0x1000, 0x2000))
        inp = mock.Mock()
        inp.serialize.return_value = b"X"
        htraces = ex._measure([inp], 1)
        self.assertEqual(htraces[0].raw, [3])


if __name__ == "__main__":
    unittest.main()
