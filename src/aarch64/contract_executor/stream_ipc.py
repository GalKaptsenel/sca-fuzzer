import time
import ctypes
import mmap
import os
import platform
import struct
from typing import Tuple, Optional

SHM_NAME = "/contract_executor_stream_shm"
REQ_RING_SIZE = 1 << 21  # 2 MB
RESP_RING_SIZE = 1 << 21  # 2 MB

arch: str = platform.machine().lower()
if arch == "aarch64":
    SYS_futex: int = 98
elif arch in ("x86_64", "amd64"):
    SYS_futex: int = 202
else:
    raise RuntimeError(f"Unsupported architecture: {arch}")

FUTEX_WAIT: int = 0
FUTEX_WAKE: int = 1

HEADER_STRUCT = struct.Struct("<II")  # little-endian: length:uint32_t, type:uint32_t

libc = ctypes.CDLL("libc.so.6", use_errno=True)

class AtomicUInt32:
    """Wraps a ctypes.c_uint32 at a memory address for atomic access + futex."""

    def __init__(self, base_addr: int):
        self._ptr = ctypes.c_uint32.from_address(base_addr)

    @property
    def value(self) -> int:
        return self._ptr.value

    @value.setter
    def value(self, v: int) -> None:
        self._ptr.value = v

    def wait(self, val: int) -> None:
        res = libc.syscall(SYS_futex, ctypes.byref(self._ptr), FUTEX_WAIT, val, None, None, 0)
        if res != 0:
            errno = ctypes.get_errno()
            if errno != 11:  # EINTR
                raise OSError(errno, "futex_wait failed")

    def wake(self, n: int = 1) -> None:
        res = libc.syscall(SYS_futex, ctypes.byref(self._ptr), FUTEX_WAKE, n, None, None, 0)
        if res < 0:
            errno = ctypes.get_errno()
            raise OSError(errno, "futex_wake failed")


class Ring(ctypes.Structure):
    _fields_ = [
        ("head", ctypes.c_uint32),      # _Atomic uint32_t head
        ("tail", ctypes.c_uint32),      # _Atomic uint32_t tail
        ("size", ctypes.c_uint32),      # ring size
        ("offset_from_shm_base", ctypes.c_uint64),  # offset to buffer from the shared memory base address
    ]

    def atomic_head(self) -> AtomicUInt32:
        return AtomicUInt32(ctypes.addressof(self) + Ring.head.offset)

    def atomic_tail(self) -> AtomicUInt32:
        return AtomicUInt32(ctypes.addressof(self) + Ring.tail.offset)


class ShmRegion(ctypes.Structure):
    _fields_ = [
        ("req", Ring),
        ("resp", Ring),
    ]

TMP_BASE = 0

def attach_shm() -> Tuple[ShmRegion, mmap.mmap, ctypes.Array[ctypes.c_uint8], ctypes.Array[ctypes.c_uint8]]:
    global TMP_BASE
    """Attach to the C-created shared memory"""
    fd = os.open(f"/dev/shm{SHM_NAME}", os.O_RDWR)
    total_size = ctypes.sizeof(ShmRegion) + REQ_RING_SIZE + RESP_RING_SIZE
    mem = mmap.mmap(fd, total_size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
    os.close(fd)

    shm = ShmRegion.from_buffer(mem)
    base = ctypes.addressof(shm)
    TMP_BASE = base
    
    TMP_BASE  = base
    # Use the offset_from_shm_base already set by C
    req_buf = (ctypes.c_uint8 * shm.req.size).from_address(base + shm.req.offset_from_shm_base)
    resp_buf = (ctypes.c_uint8 * shm.resp.size).from_address(base + shm.resp.offset_from_shm_base)

    return shm, mem, req_buf, resp_buf


class RingBuffer:
    def __init__(self, ring: Ring, buf: ctypes.Array[ctypes.c_uint8]):
        self._ring = ring
        self._buf = buf
        self._head = ring.atomic_head()
        self._tail = ring.atomic_tail()

    def _used(self) -> int:
        return self._head.value - self._tail.value

    def _free(self) -> int:
        return self._ring.size - self._used()

    def write(self, data: bytes) -> None:
        length = len(data)
        while self._free() < length:
            self._tail.wait(self._tail.value)

        off = self._head.value & (self._ring.size - 1)
        first = self._ring.size - off
        print(f"[Python] Write at offset {ctypes.addressof(self._buf) + off- TMP_BASE}")

        if first >= length:
            ctypes.memmove(ctypes.addressof(self._buf) + off, data, length)
        else:
            ctypes.memmove(ctypes.addressof(self._buf) + off, data[:first], first)
            ctypes.memmove(ctypes.addressof(self._buf), data[first:], length - first)

        self._head.value += length
        self._head.wake()

    def read(self, length: int) -> bytes:
        while self._used() < length:
            self._head.wait(self._head.value)

        off = self._tail.value & (self._ring.size - 1)
        first = self._ring.size - off

        print(f"[Python] Read at offset {ctypes.addressof(self._buf) + off- TMP_BASE}")
        result = (ctypes.c_uint8 * length)()
        dest_base = ctypes.addressof(result)
        if first >= length:
            print("1")
            ctypes.memmove(dest_base, ctypes.addressof(self._buf) + off, length)
            print("2")
        else:
            print("3")
            ctypes.memmove(dest_base, ctypes.addressof(self._buf) + off, first)
            print("4")
            ctypes.memmove(dest_base + first, ctypes.addressof(self._buf), length - first)
            print("5")

        self._tail.value += length
        self._tail.wake()
        return bytes(result)

    def send(self, payload: bytes, msg_type: int) -> None:
        if HEADER_STRUCT.size + len(payload) > self._ring.size:
            raise RuntimeError("Payload+header too big!")

        header = HEADER_STRUCT.pack(len(payload), msg_type)
        self.write(header)
        if payload:
            self.write(payload)

    def recv(self) -> Tuple[int, bytes]:
        header_bytes = self.read(HEADER_STRUCT.size)
        length, msg_type = HEADER_STRUCT.unpack(header_bytes)
        payload = self.read(length) if length > 0 else b""
        return msg_type, payload

MSG_TYPE_TEST = 1

def main():
    # Attach to the shared memory (C already initialized it)
    shm, mem, req_buf, resp_buf = attach_shm()

    # Wrap the rings
    req_ring = RingBuffer(shm.req, req_buf)
    resp_ring = RingBuffer(shm.resp, resp_buf)

    # Example: send a message to C
    counter = 0
    while True:
        payload = b"H"* (100)
        print(f"[Python] Sending: {payload!r}")
        print(f"[Python] Sending: {counter}")
        counter += 1
        req_ring.send(payload, MSG_TYPE_TEST)
        
        # Optionally wait for a reply
        print("[Python] Waiting for reply...")
        msg_type, reply = resp_ring.recv()
        print(f"[Python] Received reply (type {msg_type}): {reply!r}")


if __name__ == "__main__":
    main()

