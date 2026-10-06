"""Where the PAC profile comes from: all PAC config options unset -> decoded from the device's raw
registers (and recorded in the config); all set -> the config alone, the device is not consulted;
partially set -> ConfigException naming the missing options."""
import copy
import os
import sys
import unittest
from unittest import mock

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, _ROOT)
from src.config import CONF, ConfigException
from src.aarch64 import aarch64_executor as ax
from src.aarch64 import aarch64_qarma as q
from src.aarch64.aarch64_kernel import TargetInfo

_N3_TCR = (16 << 0) | (16 << 16) | (1 << 37) | (1 << 38) | (1 << 52)
_N3_ISAR2 = (3 << 12) | (1 << 8)
_PAC_OPTIONS = [c for _, c in ax._PAC_CONF]


class PacProfileSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._saved_conf = copy.deepcopy(CONF._borg_shared_state)
        CONF.load(os.path.join(_ROOT, "config_pac.yml"))

    @classmethod
    def tearDownClass(cls):
        CONF._borg_shared_state.clear()
        CONF._borg_shared_state.update(cls._saved_conf)

    def setUp(self):
        self._saved = {c: getattr(CONF, c) for c in _PAC_OPTIONS}
        for c in _PAC_OPTIONS:
            setattr(CONF, c, None)
        self.device = mock.Mock()
        self.device.target_info.return_value = TargetInfo(0x1000, 0x2000, _N3_TCR, 0, _N3_ISAR2)

    def tearDown(self):
        for c, v in self._saved.items():
            setattr(CONF, c, v)

    def test_unset_config_decodes_device_and_records_it(self):
        with mock.patch.object(ax.Logger, "warning") as warn:
            regs = ax._pac_registers(self.device)
        warn.assert_called_once()
        self.assertIn("current machine", warn.call_args.args[1])
        self.assertEqual(regs, q.decode_registers(_N3_TCR, 0, _N3_ISAR2))
        self.assertEqual((CONF.va_size, CONF.pac_tbid1, CONF.pac_auth_level), (48, True, 3))

    def test_full_config_overrides_without_touching_device(self):
        remote = q.PacRegisters(5, 5, 4, 39, 39, True, False, False, False, True, False, False)
        for f, c in ax._PAC_CONF:
            setattr(CONF, c, getattr(remote, f))
        with mock.patch.object(ax.Logger, "warning") as warn:
            self.assertEqual(ax._pac_registers(self.device), remote)
        warn.assert_not_called()
        self.device.target_info.assert_not_called()

    def test_partial_config_is_rejected(self):
        CONF.pac_tbi0 = True
        with self.assertRaisesRegex(ConfigException, "pac_qarma_version"):
            ax._pac_registers(self.device)
        self.device.target_info.assert_not_called()


if __name__ == "__main__":
    unittest.main()
