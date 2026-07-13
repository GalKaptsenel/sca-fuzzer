"""Measurement super-batch wire format, mirrored by the `executor_userland batch` C reader in
executor/userapi/executor_batch_format.h. One request packs many (test case, inputs) units; one
response returns per-input per-repetition htraces + PFCs. Fixed fields are u64, payloads are
byte-length-prefixed (so the prefix is whole u64s and no padding is needed)."""
import struct
from dataclasses import dataclass
from typing import List, Tuple

REQUEST_MAGIC = 0x42525A5652        # "RVZRB"
RESPONSE_MAGIC = 0x52425A5652       # "RVZBR"
VERSION = 1
NUM_PFC = 3                         # matches aarch64_kernel.NUM_PFC / kernel measurement_t

_HEADER = struct.Struct("<4Q")      # magic, version, n_units, n_reps
_UNIT_DESC = struct.Struct("<2Q")   # tc_len, n_inputs
_U64 = struct.Struct("<Q")
_MEASUREMENT = struct.Struct(f"<{1 + NUM_PFC}Q")   # htrace, pfc[NUM_PFC]


@dataclass(frozen=True)
class TraceUnit:
    """A test case plus the serialized REIF inputs to measure on it."""
    test_case: bytes
    inputs: Tuple[bytes, ...]


@dataclass(frozen=True)
class HWMeasurement:
    """One input's htrace + performance counters for one repetition."""
    htrace: int
    pfcs: Tuple[int, ...]


def encode_request(units: List[TraceUnit], n_reps: int) -> bytes:
    out = bytearray()
    out += _HEADER.pack(REQUEST_MAGIC, VERSION, len(units), n_reps)
    for u in units:
        out += _UNIT_DESC.pack(len(u.test_case), len(u.inputs))
    for u in units:
        for inp in u.inputs:
            out += _U64.pack(len(inp))
    for u in units:
        out += u.test_case
        for inp in u.inputs:
            out += inp
    return bytes(out)


def decode_request(blob: bytes) -> Tuple[List[TraceUnit], int]:
    """Inverse of encode_request (the reference the C reader mirrors)."""
    magic, version, n_units, n_reps = _HEADER.unpack_from(blob, 0)
    if REQUEST_MAGIC != magic:
        raise ValueError("not a batch request (bad magic)")
    if VERSION != version:
        raise ValueError(f"unsupported batch version {version}")
    off = _HEADER.size
    shapes = []
    for _ in range(n_units):
        tc_len, n_inputs = _UNIT_DESC.unpack_from(blob, off)
        off += _UNIT_DESC.size
        shapes.append((tc_len, n_inputs))
    input_lens = []
    for _, n_inputs in shapes:
        row = []
        for _ in range(n_inputs):
            (length,) = _U64.unpack_from(blob, off)
            off += _U64.size
            row.append(length)
        input_lens.append(row)
    units = []
    for (tc_len, _), lens in zip(shapes, input_lens):
        tc = blob[off:off + tc_len]
        off += tc_len
        inputs = []
        for length in lens:
            inputs.append(blob[off:off + length])
            off += length
        units.append(TraceUnit(tc, tuple(inputs)))
    return units, n_reps


def encode_response(results: List[List[List[HWMeasurement]]], n_reps: int) -> bytes:
    """`results` is unit -> input -> repetition (the reference the C writer mirrors)."""
    out = bytearray()
    out += _HEADER.pack(RESPONSE_MAGIC, VERSION, len(results), n_reps)
    for unit in results:
        out += _U64.pack(len(unit))
    for unit in results:
        for reps in unit:
            for m in reps:
                out += _MEASUREMENT.pack(m.htrace, *m.pfcs)
    return bytes(out)


def decode_response(blob: bytes) -> List[List[List[HWMeasurement]]]:
    """Parse a results blob into unit -> input -> repetition measurements."""
    magic, version, n_units, n_reps = _HEADER.unpack_from(blob, 0)
    if RESPONSE_MAGIC != magic:
        raise ValueError("not a batch response (bad magic)")
    if VERSION != version:
        raise ValueError(f"unsupported batch response version {version}")
    off = _HEADER.size
    n_inputs_per_unit = []
    for _ in range(n_units):
        (n_inputs,) = _U64.unpack_from(blob, off)
        off += _U64.size
        n_inputs_per_unit.append(n_inputs)
    results = []
    for n_inputs in n_inputs_per_unit:
        unit = []
        for _ in range(n_inputs):
            reps = []
            for _ in range(n_reps):
                htrace, *pfcs = _MEASUREMENT.unpack_from(blob, off)
                off += _MEASUREMENT.size
                reps.append(HWMeasurement(htrace, tuple(pfcs)))
            unit.append(reps)
        results.append(unit)
    return results
