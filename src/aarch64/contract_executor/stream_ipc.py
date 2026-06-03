import struct
import subprocess
import select as _select
from typing import Tuple

HEADER_STRUCT = struct.Struct("<II")  # little-endian: length:uint32_t, type:uint32_t

CE_READ_TIMEOUT = 30.0  # seconds before declaring CE hung


class StreamIPC:
    def __init__(self, write_stream: subprocess.PIPE, read_stream: subprocess.PIPE,
                 proc: subprocess.Popen = None):
        self._write_stream: subprocess.PIPE = write_stream
        self._read_stream: subprocess.PIPE = read_stream
        self._proc: subprocess.Popen = proc

    def _safe_write(self, data: bytes):
        total_sent = 0
        while total_sent < len(data):
            n = self._write_stream.write(data[total_sent:])
            if n is None:
                n = 0
            total_sent += n

        self._write_stream.flush()
        assert total_sent == len(data)

    def send_req(self, msg_type: int, payload: bytes) -> None:
        header = HEADER_STRUCT.pack(len(payload), msg_type)
        self._safe_write(header)
        if payload:
            self._safe_write(payload)

    def _safe_read(self, amount: int) -> bytes:
        buff: bytes = bytes()
        while amount > len(buff):
            # Non-blocking peek: if nothing is buffered and the CE has already exited,
            # fail immediately instead of waiting out the full CE_READ_TIMEOUT.
            ready, _, _ = _select.select([self._read_stream], [], [], 0)
            if not ready and self._proc is not None and self._proc.poll() is not None:
                raise EOFError(
                    f"CE exited (code {self._proc.returncode}) while reading {amount} bytes; "
                    f"got {len(buff)}.")
            if not ready:
                ready, _, _ = _select.select([self._read_stream], [], [], CE_READ_TIMEOUT)
            if not ready:
                pid = self._proc.pid if self._proc is not None else "?"
                alive = (self._proc is not None and self._proc.poll() is None)

                wchan = ""
                try:
                    with open(f"/proc/{pid}/wchan") as f:
                        wchan = f.read().strip()
                except Exception:
                    pass

                status = ""
                try:
                    with open(f"/proc/{pid}/status") as f:
                        status = f.read()
                except Exception:
                    pass

                raise RuntimeError(
                    f"CE hung: no response after {CE_READ_TIMEOUT}s "
                    f"(waiting for {amount} bytes, got {len(buff)})\n"
                    f"  pid={pid} alive={alive}\n"
                    f"  wchan={wchan}\n"
                    f"  /proc/{pid}/status:\n{status}"
                )
            chunk = self._read_stream.read1(amount - len(buff))
            if not chunk:
                raise EOFError(f"Stream closed while reading {amount} bytes. Read {len(buff)} bytes.")
            buff += chunk

        assert len(buff) == amount
        return buff

    def recv_resp(self) -> Tuple[int, bytes]:
        header_bytes: bytes = self._safe_read(HEADER_STRUCT.size)
        length, msg_type = HEADER_STRUCT.unpack(header_bytes)
        payload: bytes = self._safe_read(length)
        return msg_type, payload
