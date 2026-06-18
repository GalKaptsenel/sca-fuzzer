"""LocalHWExecutor's contents deleter frees the currently checked-out input id, not a hardcoded 0
-- it must pass current_region.iid to REVISOR_FREE_INPUT."""
import unittest
from unittest import mock

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # run from any cwd
from src.aarch64.aarch64_kernel import LocalHWExecutor, InputRegion, REVISOR_FREE_INPUT


class FreeInputTest(unittest.TestCase):
    def test_deleter_frees_current_iid(self):
        ex = LocalHWExecutor.__new__(LocalHWExecutor)
        ex.current_region = InputRegion(5)
        ex._ioctl = mock.Mock()
        del ex.contents
        ex._ioctl.assert_called_once_with(REVISOR_FREE_INPUT, 5)


if __name__ == "__main__":
    unittest.main()
