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


def _fake_run(out_bytes, rc):
    remote_in = mock.Mock()
    remote_out = mock.Mock()
    remote_out.read.return_value = out_bytes
    remote_out.channel.recv_exit_status.return_value = rc
    remote_err = mock.Mock()
    remote_err.read.return_value = b""
    return remote_in, remote_out, remote_err


class SSHRunStreamingTest(unittest.TestCase):
    """SSHConnection.run streams binary stdin, returns raw stdout bytes, checks the exit status, and
    wraps privileged commands in sudo — the path the streamed super-batch crosses."""

    def _conn(self, out_bytes, rc):
        conn = SSHConnection.__new__(SSHConnection)   # __init__ opens a real connection
        conn.client = mock.Mock()
        conn.client.exec_command.return_value = _fake_run(out_bytes, rc)
        return conn

    def test_streams_stdin_and_returns_stdout_bytes(self):
        conn = self._conn(b"\x00\x01\x02\xff", 0)
        out = conn.run("cmd", b"REQUEST", privileged=False)
        self.assertEqual(out, b"\x00\x01\x02\xff")
        remote_in = conn.client.exec_command.return_value[0]
        remote_in.write.assert_called_once_with(b"REQUEST")
        remote_in.channel.shutdown_write.assert_called_once()

    def test_nonzero_exit_raises(self):
        with self.assertRaises(IOError):
            self._conn(b"", 1).run("cmd", b"", privileged=False)

    def test_privileged_wraps_in_sudo(self):
        conn = self._conn(b"", 0)
        conn.run("userland /dev/executor batch - -", b"", privileged=True)
        sent = conn.client.exec_command.call_args.args[0]
        self.assertIn("sudo bash -c", sent)
        self.assertIn("userland /dev/executor batch - -", sent)


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
