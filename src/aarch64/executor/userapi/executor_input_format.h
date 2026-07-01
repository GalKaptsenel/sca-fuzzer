#ifndef EXECUTOR_INPUT_FORMAT_H
#define EXECUTOR_INPUT_FORMAT_H

/*
 * /dev/executor input wire format — the authoritative, self-describing
 * structure of one input initialization written to the device. Any writer (the fuzzer, the
 * executor_userland tool, a test) MUST produce exactly this; the kernel parses and
 * validates strictly against it and rejects anything that does not conform.
 *
 * Layout of one input_init:
 *
 *     struct revisor_input_header     (48 bytes, fixed preamble)
 *     struct revisor_input_section[n] (32 bytes each, the section table)
 *     section payloads                (each begins at its descriptor's `offset`)
 *
 * The reader locates each section by TYPE via the table, never by a hardcoded
 * offset, so new per-input initial state can be added without breaking the layout.
 * Unknown section types are skipped.
 *
 * Conventions:
 *   - All header and descriptor members are u64, little-endian. (u64 throughout so
 *     there are no struct-packing or width rules to get wrong across writers.)
 *   - The preamble is 48 bytes and each descriptor is 32 bytes, so everything is
 *     naturally 8-byte aligned. Payloads start at `header_len`.
 *   - `total_len` is the exact total byte count of the input_init and equals the number of
 *     bytes written to the device.
 *
 * Section payload contents the kernel expects (sizes from sandbox.h / inputs.h):
 *   REVISOR_SEC_MEMORY_MAIN    MAIN_REGION_SIZE bytes   -> sandbox main_region
 *   REVISOR_SEC_MEMORY_FAULTY  FAULTY_REGION_SIZE bytes -> sandbox faulty_region
 *   REVISOR_SEC_GPR            sizeof(registers_t)      -> x0..x5, flags, sp.
 *                              `flags` is the ARM PSTATE value (NZCV in bits 31:28),
 *                              loaded verbatim; the writer is responsible for any
 *                              encoding conversion before packing this section.
 *   REVISOR_SEC_SIMD           256 bytes (v0..v7)       -> reserved (not yet loaded)
 *   REVISOR_SEC_PAC_KEYS       sizeof(struct pac_keys) -> per-input PAC keys
 *   REVISOR_SEC_MTE_TAGS       one 4-bit allocation tag per MTE_GRANULE_SIZE (16-byte)
 *                              granule of the main|faulty span, packed two tags per
 *                              byte, low nibble first (granule 2*i in bits 3:0,
 *                              granule 2*i+1 in bits 7:4).
 *                              Length = (MEMORY_INPUT_SIZE/MTE_GRANULE_SIZE + 1)/2.
 *
 * Included by the kernel module (chardevice.c) and by executor_userland; the parser
 * bodies live in executor/input_format.c.
 */

#ifdef __KERNEL__
#include <linux/types.h>
#else
#include <stdint.h>
#include <stddef.h>   /* NULL */
#endif

/* "RVZRI" magic. A plain u64 sentinel both sides compare for equality. */
#define REVISOR_INPUT_MAGIC      ((uint64_t)0x49525A5652ULL)
#define REVISOR_INPUT_VERSION    ((uint64_t)1ULL)

/* Upper bound on n_sections so header_len arithmetic can never overflow. */
#define REVISOR_INPUT_MAX_SECTIONS  64

/* Section type ids. */
enum revisor_input_section_type {
    REVISOR_SEC_MEMORY_MAIN   = 0x01,
    REVISOR_SEC_MEMORY_FAULTY = 0x02,
    REVISOR_SEC_GPR           = 0x03,
    REVISOR_SEC_SIMD          = 0x04,
    REVISOR_SEC_PAC_KEYS      = 0x05,
    REVISOR_SEC_MTE_TAGS      = 0x06,
};

struct revisor_input_header {
    uint64_t magic;       /* REVISOR_INPUT_MAGIC */
    uint64_t version;     /* REVISOR_INPUT_VERSION */
    uint64_t header_len;  /* 48 + 32*n_sections; offset of the first payload */
    uint64_t n_sections;
    uint64_t flags;       /* reserved */
    uint64_t total_len;    /* total bytes of this input_init */
};

struct revisor_input_section {
    uint64_t type;        /* enum revisor_input_section_type */
    uint64_t flags;       /* reserved per-section, 0 for now */
    uint64_t offset;      /* payload offset from the start of the input_init */
    uint64_t length;      /* payload length in bytes */
};

#define REVISOR_INPUT_HEADER_LEN(n_sections)                                   \
    (sizeof(struct revisor_input_header) +                                     \
     (uint64_t)(n_sections) * sizeof(struct revisor_input_section))

/*
 * Validate a fully-copied input_init of exactly `total_len` bytes. Returns 1 if the header
 * is well-formed and every section lies within [header_len, total_len], else 0.
 */
int revisor_input_header_valid(const void *input_init, uint64_t total_len);

/*
 * Return the descriptor for `type`, or NULL if absent. The input_init must already have
 * passed revisor_input_header_valid().
 */
const struct revisor_input_section *revisor_input_find_section(const void *input_init,
                                                               uint64_t type);

#endif /* EXECUTOR_INPUT_FORMAT_H */
