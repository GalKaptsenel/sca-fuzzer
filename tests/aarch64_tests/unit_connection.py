"""SSHConnection.shell surfaces a non-zero remote exit status; ADBConnection rejects an
absent serial. The transports are mocked."""
import unittest
from unittest import mock

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # run from any cwd
from src.aarch64.aarch64_connection import SSHConnection, ADBConnection


def _fake_exec(out, err, rc):
    stdin = mock.Mock()
    stdout = mock.Mock()
    stdout.read.return_value = out.encode()
    stdout.channel.recv_exit_status.return_value = rc
    stderr = mock.Mock()
    stderr.read.return_value = err.encode()
    return stdin, stdout, stderr


class SSHShellExitStatusTest(unittest.TestCase):
    def _conn(self, out, err, rc):
        conn = SSHConnection.__new__(SSHConnection)   # __init__ would open a real SSH connection
        conn.password = None
        conn.client = mock.Mock()
        conn.client.exec_command.return_value = _fake_exec(out, err, rc)
        return conn

    def test_nonzero_exit_raises(self):
        with self.assertRaises(IOError):
            self._conn(out="", err="boom", rc=1).shell("false")

    def test_zero_exit_returns_stdout(self):
        self.assertEqual(self._conn(out="hello", err="", rc=0).shell("echo hello"), "hello")


class ADBSerialTest(unittest.TestCase):
    @staticmethod
    def _client(serials):
        client = mock.Mock()
        client.devices.return_value = [mock.Mock(serial=s) for s in serials]
        return client

    def test_missing_serial_raises(self):
        with mock.patch("src.aarch64.aarch64_connection.AdbClient") as adb_client:
            adb_client.return_value = self._client(["AAA", "BBB"])
            with self.assertRaises(IOError):
                ADBConnection(serial="ZZZ")

    def test_present_serial_sets_device(self):
        with mock.patch("src.aarch64.aarch64_connection.AdbClient") as adb_client:
            adb_client.return_value = self._client(["AAA", "BBB"])
            self.assertEqual(ADBConnection(serial="BBB").device.serial, "BBB")


if __name__ == "__main__":
    unittest.main()
