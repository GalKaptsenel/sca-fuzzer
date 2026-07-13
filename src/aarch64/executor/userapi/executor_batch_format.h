#ifndef EXECUTOR_BATCH_FORMAT_H
#define EXECUTOR_BATCH_FORMAT_H

/*
 * Measurement super-batch wire format — the request/response blobs exchanged between the remote
 * executor (Python, src/aarch64/aarch64_batch.py) and the `executor_userland batch` runner. One
 * request packs M units, each a test-case binary plus the REIF inputs to measure on it; one response
 * returns every input's per-repetition htrace and performance counters. Batching the whole window
 * into one blob replaces thousands of small device round-trips with a single transfer.
 *
 * All fixed fields are u64, little-endian; the variable payloads are byte-length-prefixed, so the
 * fixed prefix is a whole number of u64s and no alignment padding is needed. Keep in sync with
 * aarch64_batch.py.
 *
 * Request:
 *   header:       magic=REVISOR_BATCH_REQUEST_MAGIC, version, n_units, n_reps
 *   unit descs:   n_units   * { tc_len, n_inputs }
 *   input lens:   sum(n_inputs) * { input_len }
 *   payloads:     for each unit: tc bytes, then each input's REIF bytes (no padding)
 *
 * Response:
 *   header:       magic=REVISOR_BATCH_RESPONSE_MAGIC, version, n_units, n_reps
 *   input counts: n_units * { n_inputs }
 *   results:      for each unit, each input, each rep: { htrace, pfc[NUM_PFC] }
 */

#include <stdint.h>

#define REVISOR_BATCH_REQUEST_MAGIC   ((uint64_t)0x42525A5652ULL)   /* "RVZRB" */
#define REVISOR_BATCH_RESPONSE_MAGIC  ((uint64_t)0x52425A5652ULL)   /* "RVZBR" */
#define REVISOR_BATCH_VERSION         ((uint64_t)1ULL)

#endif /* EXECUTOR_BATCH_FORMAT_H */
