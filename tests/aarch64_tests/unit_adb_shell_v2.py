"""ADBConnection.run over adb shell-protocol v2 (raw), self-contained (no device, no adb server):
a fake socket scripts the adbd side and records what the client sends, proving

  * stdout is returned binary-clean and a diagnostic the command writes to stderr never pollutes it
    (the reason we moved off the stderr-merging `exec:` service);
  * the request is framed as stdin packets (id 0) and terminated by a close-stdin packet (id 4),
    reassembling to the exact bytes, across more than one chunk;
  * a nonzero exit raises IOError carrying the stderr text (the SSH/local transport contract).

    python -m unittest tests.aarch64_tests.unit_adb_shell_v2
"""
import os
import struct
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import src.aarch64.aarch64_connection as C  # noqa: E402
from src.aarch64.aarch64_connection import (                                      # noqa: E402
    ADBConnection, _ADB_SHELL_STDIN, _ADB_SHELL_STDOUT, _ADB_SHELL_STDERR,
    _ADB_SHELL_EXIT, _ADB_SHELL_CLOSE_STDIN)


def _frame(packet_id: int, payload: bytes) -> bytes:
    return bytes([packet_id]) + struct.pack("<I", len(payload)) + payload


class _FakeSocket:
    """Serves `recv_script` on recv() and records sendall() bytes; recv() returns b'' when drained."""
    def __init__(self, recv_script: bytes):
        self._recv = bytearray(recv_script)
        self.sent = bytearray()
        self.closed = False

    def settimeout(self, _timeout):
        pass

    def sendall(self, data):
        self.sent += data

    def recv(self, n):
        take = bytes(self._recv[:n])
        del self._recv[:n]
        return take

    def close(self):
        self.closed = True


def _client_stdin(sent: bytes) -> bytes:
    """Reassemble the stdin payload the client framed, after the two `len+body` service requests."""
    off = 0
    for _ in range(2):                                   # host:transport + shell,v2,raw service lines
        length = int(sent[off:off + 4], 16)
        off += 4 + length
    stdin = bytearray()
    saw_close = False
    while off < len(sent):
        packet_id = sent[off]
        (length,) = struct.unpack("<I", sent[off + 1:off + 5])
        payload = sent[off + 5:off + 5 + length]
        off += 5 + length
        if _ADB_SHELL_STDIN == packet_id:
            stdin += payload
        elif _ADB_SHELL_CLOSE_STDIN == packet_id:
            saw_close = True
    assert saw_close, "client never sent a close-stdin packet"
    return bytes(stdin)


class AdbShellV2Test(unittest.TestCase):
    def setUp(self):
        self._real_create = C.socket.create_connection
        self.conn = ADBConnection.__new__(ADBConnection)   # skip __init__ (which needs a real server)
        self.conn.host, self.conn.port, self.conn.serial = "127.0.0.1", 5037, "TESTSERIAL"
        self.fake = None

    def tearDown(self):
        C.socket.create_connection = self._real_create

    def _install(self, response: bytes):
        self.fake = _FakeSocket(b"OKAY" + b"OKAY" + response)   # OKAY for each of the two services
        C.socket.create_connection = lambda *a, **k: self.fake

    def test_stdout_clean_stderr_separated(self):
        payload = bytes(range(256)) * 4            # arbitrary binary, incl. NUL and 0xFF
        self._install(_frame(_ADB_SHELL_STDOUT, payload)
                      + _frame(_ADB_SHELL_STDERR, b"a warning on stderr")
                      + _frame(_ADB_SHELL_EXIT, b"\x00"))
        out = self.conn.run("some cmd", b"the-request-bytes")
        self.assertEqual(out, payload, "stdout must be exactly the stdout stream, no stderr mixed in")
        self.assertTrue(self.fake.closed)

    def test_stdin_is_framed_and_reassembles(self):
        request = os.urandom(3 * (1 << 16) + 123)   # > 3 chunks, exercises the chunk loop
        self._install(_frame(_ADB_SHELL_STDOUT, b"ok") + _frame(_ADB_SHELL_EXIT, b"\x00"))
        self.conn.run("cmd", request)
        self.assertEqual(_client_stdin(bytes(self.fake.sent)), request)

    def test_nonzero_exit_raises_with_stderr(self):
        self._install(_frame(_ADB_SHELL_STDERR, b"batch: unit 0 measurement failed\n")
                      + _frame(_ADB_SHELL_EXIT, b"\x01"))
        with self.assertRaises(IOError) as cm:
            self.conn.run("cmd", b"req")
        self.assertIn("rc=1", str(cm.exception))
        self.assertIn("batch: unit 0 measurement failed", str(cm.exception))

    def test_closed_connection_midstream_raises(self):
        self._install(_frame(_ADB_SHELL_STDOUT, b"partial"))   # no EXIT packet -> socket drains to b""
        with self.assertRaises(IOError):
            self.conn.run("cmd", b"req")


if __name__ == "__main__":
    unittest.main(verbosity=2)
