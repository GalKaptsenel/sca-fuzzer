"""_safe_read serves buffered bytes via read1 before consulting select(), so a read1 over-read
cannot hide already-received data behind a spurious hang/EOF.

On Linux the over-read scenario is exercised over a real non-blocking pipe; on Windows (no
non-blocking pipe / os.set_blocking) the read1/select logic is checked with mocks instead."""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # run from any cwd
import src.aarch64.contract_executor.stream_ipc as sipc
from src.aarch64.contract_executor.stream_ipc import StreamIPC, HEADER_STRUCT

_NONBLOCKING_PIPE = hasattr(os, "set_blocking")


def _mock_ipc(read1_side_effect, poll_return=None):
    ipc = StreamIPC.__new__(StreamIPC)   # __init__ would set O_NONBLOCK on a real fd
    ipc._write_stream = None
    ipc._read_stream = mock.Mock()
    ipc._read_stream.read1.side_effect = read1_side_effect
    ipc._proc = mock.Mock()
    ipc._proc.poll.return_value = poll_return
    ipc._proc.returncode = 0
    return ipc


class SafeReadLogicTest(unittest.TestCase):
    def test_buffered_data_served_without_select(self):
        ipc = _mock_ipc([b"PAYLOAD!"])
        with mock.patch.object(sipc, "_select") as sel:
            sel.select.return_value = ([], [], [])
            self.assertEqual(ipc._safe_read(8), b"PAYLOAD!")
            sel.select.assert_not_called()

    def test_waits_via_select_when_nothing_buffered(self):
        ipc = _mock_ipc([None, b"12345678"])
        with mock.patch.object(sipc, "_select") as sel:
            sel.select.return_value = ([ipc._read_stream], [], [])
            self.assertEqual(ipc._safe_read(8), b"12345678")
            self.assertTrue(sel.select.called)

    def test_eof_when_ce_exited_and_nothing_available(self):
        # exited, and the drain read1 also yields nothing -> EOF
        ipc = _mock_ipc([b"", b""], poll_return=0)
        with mock.patch.object(sipc, "_select") as sel:
            sel.select.return_value = ([], [], [])
            with self.assertRaises(EOFError):
                ipc._safe_read(8)

    def test_drains_bytes_flushed_just_before_exit(self):
        # regression: bytes flushed right before exit must be returned, not dropped as EOF
        ipc = _mock_ipc([b"", b"LASTDATA"], poll_return=0)
        with mock.patch.object(sipc, "_select") as sel:
            sel.select.return_value = ([], [], [])
            self.assertEqual(ipc._safe_read(8), b"LASTDATA")


@unittest.skipUnless(_NONBLOCKING_PIPE, "real non-blocking pipe path is Unix-only")
class SafeReadRealPipeTest(unittest.TestCase):
    def test_overread_payload_not_lost(self):
        r, w = os.pipe()
        payload = b"Z" * 50
        os.write(w, HEADER_STRUCT.pack(len(payload), 5) + payload)   # header+payload both in pipe
        read_stream = os.fdopen(r, "rb", buffering=1 << 20)          # 1MB BufferedReader, as in _spawn
        proc = mock.Mock()
        proc.poll.return_value = None                               # CE alive, write end kept open
        ipc = StreamIPC(None, read_stream, proc)
        try:
            self.assertEqual(ipc.recv_resp(), (5, payload))
        finally:
            read_stream.close()
            os.close(w)

    def test_empty_pipe_is_not_eof_while_ce_alive(self):
        # regression: read1() returns b'' (not None) on an empty non-blocking pipe; an empty
        # read while the CE is alive is would-block, not EOF, so a late response must be read.
        import threading
        import time
        r, w = os.pipe()
        payload = b"R" * 40
        read_stream = os.fdopen(r, "rb", buffering=1 << 20)
        proc = mock.Mock()
        proc.poll.return_value = None          # CE alive: the empty first read must not be EOF
        ipc = StreamIPC(None, read_stream, proc)

        def respond_later():
            time.sleep(0.2)                    # arrives only after recv_resp's first read1 sees b''
            os.write(w, HEADER_STRUCT.pack(len(payload), 7) + payload)

        writer = threading.Thread(target=respond_later)
        writer.start()
        try:
            self.assertEqual(ipc.recv_resp(), (7, payload))
        finally:
            writer.join()
            read_stream.close()
            os.close(w)


if __name__ == "__main__":
    unittest.main()
