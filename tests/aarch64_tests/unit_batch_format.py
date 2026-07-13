"""Round-trip checks for the measurement super-batch wire format (src/aarch64/aarch64_batch.py).

Validates the request/response blobs the Python remote executor produces against the structure the
`executor_userland batch` C runner mirrors in executor/userapi/executor_batch_format.h.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.aarch64 import aarch64_batch as batch


class BatchRequestRoundTrip(unittest.TestCase):

    def _sample_units(self):
        return [
            batch.TraceUnit(b"\x01\x02\x03\x04" * 3, (b"input-a", b"input-bb", b"input-ccc")),
            batch.TraceUnit(b"\xaa" * 40, (b"only-one-input",)),
            batch.TraceUnit(b"\xde\xad\xbe\xef", ()),   # a unit with no inputs
        ]

    def test_request_round_trips(self):
        units = self._sample_units()
        units2, n_reps = batch.decode_request(batch.encode_request(units, 7))
        self.assertEqual(n_reps, 7)
        self.assertEqual(units2, units)

    def test_empty_batch_round_trips(self):
        units2, n_reps = batch.decode_request(batch.encode_request([], 3))
        self.assertEqual((units2, n_reps), ([], 3))

    def test_request_rejects_bad_magic(self):
        blob = bytearray(batch.encode_request(self._sample_units(), 1))
        blob[0] ^= 0xFF
        with self.assertRaises(ValueError):
            batch.decode_request(bytes(blob))

    def test_request_offsets_are_u64_aligned(self):
        # The fixed prefix (header + unit descs + input lengths) must be a whole number of u64s so
        # the C reader can read it as uint64_t[].
        units = self._sample_units()
        blob = batch.encode_request(units, 1)
        header = batch._HEADER.size
        descs = len(units) * batch._UNIT_DESC.size
        lens = sum(len(u.inputs) for u in units) * batch._U64.size
        self.assertEqual((header + descs + lens) % 8, 0)


class BatchResponseRoundTrip(unittest.TestCase):

    def _sample_results(self, n_reps):
        def meas(seed):
            return batch.HWMeasurement(seed, tuple(seed + i for i in range(batch.NUM_PFC)))
        return [
            [[meas(u * 100 + i * 10 + r) for r in range(n_reps)] for i in range(n_inputs)]
            for u, n_inputs in enumerate((3, 1, 0))
        ]

    def test_response_round_trips(self):
        results = self._sample_results(4)
        self.assertEqual(batch.decode_response(batch.encode_response(results, 4)), results)

    def test_response_rejects_bad_magic(self):
        blob = bytearray(batch.encode_response(self._sample_results(2), 2))
        blob[0] ^= 0xFF
        with self.assertRaises(ValueError):
            batch.decode_response(bytes(blob))

    def test_response_shape_matches_request(self):
        # The response carries its own per-unit input counts, so A can parse it without the request.
        results = self._sample_results(5)
        decoded = batch.decode_response(batch.encode_response(results, 5))
        self.assertEqual([len(u) for u in decoded], [3, 1, 0])
        for unit in decoded:
            for reps in unit:
                self.assertEqual(len(reps), 5)


if __name__ == "__main__":
    unittest.main()
