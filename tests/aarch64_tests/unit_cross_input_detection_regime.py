"""Regression: cross-input priming must run the CANDIDATE-DETECTION measurement under the same
executor regime (_CROSS_INPUT_PRIMING_REGIME) it uses for localization — not only the localization
pass. Otherwise detection runs under the default regime (e.g. ssbs off) and a v4/SSB-window leftover
never forms, so no candidate is flagged and localization never runs. Device-free: a fake LocalHWExecutor
records the sysfs regime that is live when the inner (super) fuzzing_round runs.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.config import CONF  # noqa: E402
from src.aarch64 import aarch64_fuzzer as F  # noqa: E402
from src.aarch64.aarch64_kernel import LocalHWExecutor  # noqa: E402
from tests.conf_isolation import setUpModule, tearDownModule  # noqa: F401,E402


class _FakeDevice(LocalHWExecutor):
    """A LocalHWExecutor whose sysfs is an in-memory dict (no /dev/executor)."""
    def __init__(self):
        self._sysfs = {"enable_ssbs": "9", "enable_pre_run_flush": "9", "enable_phr_flush": "9",
                       "enable_view_rotation": "9", "pin_to_core": "9"}

    def _read_sysfs(self, name):
        return self._sysfs[name]

    def _write_sysfs(self, name, value: bytes):
        self._sysfs[name] = value.decode()


class _FakeExecutor:
    def __init__(self, device):
        self.device = device


class _RecordingBase:
    """Stands in for the generic fuzzer: records the device regime live during the detection round."""
    def __init__(self, device):
        self.executor = _FakeExecutor(device)
        self.seen = None

    def fuzzing_round(self, test_case, inputs, ignore_list=None):
        dev = self.executor.device
        self.seen = {k: dev._read_sysfs(k) for k in
                     ("enable_ssbs", "enable_pre_run_flush", "enable_phr_flush",
                      "enable_view_rotation", "pin_to_core")}
        return None


class _Harness(F.CrossInputPrimingMixin, _RecordingBase):
    # the speculative-exception seal-lane regime (what the detection pass must also run under)
    _CROSS_INPUT_PRIMING_REGIME = F.Aarch64SpeculativeExceptionFuzzer._CROSS_INPUT_PRIMING_REGIME


class CrossInputDetectionRegimeTest(unittest.TestCase):
    def _run(self):
        dev = _FakeDevice()
        h = _Harness(dev)
        h.fuzzing_round(None, [], None)
        return h.seen, dev

    def test_detection_runs_under_regime(self):
        CONF.enable_cross_input_priming = True
        seen, dev = self._run()
        # The inner (detection) round saw exactly the seal-lane regime values.
        self.assertEqual(seen["enable_ssbs"], "1", "SSB window must be open during detection")
        self.assertEqual(seen["enable_pre_run_flush"], "0")
        self.assertEqual(seen["enable_phr_flush"], "0")
        self.assertEqual(seen["enable_view_rotation"], "0")
        self.assertEqual(seen["pin_to_core"], "-1")
        # And the regime is restored afterward (the fake's sentinel "9" comes back).
        self.assertEqual(dev._read_sysfs("enable_ssbs"), "9", "regime must be restored after the round")

    def test_off_by_default_leaves_regime_untouched(self):
        CONF.enable_cross_input_priming = False
        seen, _ = self._run()
        self.assertEqual(seen["enable_ssbs"], "9", "no regime applied when cross-input priming is off")


if __name__ == "__main__":
    unittest.main()
