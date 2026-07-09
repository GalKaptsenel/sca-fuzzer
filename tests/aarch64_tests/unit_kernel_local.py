"""LocalHWExecutor's contents deleter frees the currently checked-out input id, not a hardcoded 0
-- it must pass current_region.iid to REVISOR_FREE_INPUT."""
import unittest
from unittest import mock

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # run from any cwd
from src.aarch64.aarch64_kernel import (
    LocalHWExecutor, InputRegion, REVISOR_FREE_INPUT, _raise_access_error)


class FreeInputTest(unittest.TestCase):
    def test_deleter_frees_current_iid(self):
        ex = LocalHWExecutor.__new__(LocalHWExecutor)
        ex.current_region = InputRegion(5)
        ex._ioctl = mock.Mock()
        del ex.contents
        ex._ioctl.assert_called_once_with(REVISOR_FREE_INPUT, 5)


class AccessErrorHintTest(unittest.TestCase):
    """An EACCES on an executor path must surface the exact chmod that fixes it."""

    def test_device_file_target_suggests_plain_chmod(self):
        err = PermissionError(13, "Permission denied")
        with self.assertRaises(PermissionError) as ctx:
            _raise_access_error("/dev/executor", "/dev/executor", err)
        msg = str(ctx.exception)
        self.assertIn("sudo chmod a+rw /dev/executor", msg)
        self.assertNotIn("-R", msg)  # a file target must not get the recursive flag
        self.assertIs(ctx.exception.__cause__, err)

    def test_sysfs_dir_target_suggests_recursive_chmod(self):
        # Point the fix target at a real directory so the isdir() branch is exercised
        # without depending on a loaded kernel module.
        err = PermissionError(13, "Permission denied")
        with self.assertRaises(PermissionError) as ctx:
            _raise_access_error(os.path.join(os.sep, "tmp", "warmups"), "/tmp", err)
        self.assertIn("sudo chmod -R a+rw /tmp", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
