/* -----------------------------------------------------------------------------
 * taint_instrumentation.S
 *
 * Single-file, pinned, parameterized macros for register and memory tainting.
 *
 * Layout (debug page, 64-bit offsets):
 *   0   : regs_write_bits
 *   8   : regs_read_bits
 *   16  : regs_input_read_bits
 *   24  : mem_write_bits
 *   32  : mem_read_bits
 *   40  : mem_input_read_bits
 *
 * Notes:
 *  - Single-core, no atomics.
 *  - dst/src operands for registers are immediate indices 0..30.
 *  - addr_reg for memory is a physical register holding the effective address.
 *  - tmp registers are scratch registers (e.g., x9).
 * -----------------------------------------------------------------------------
 */

//struct debug_page_t {
//	uint64_t regs_write_bits;       // 0
//	uint64_t regs_read_bits;        // 8
//	uint64_t regs_input_read_bits;  // 16
//	
//	uint64_t mem_write_bits;        // 24
//	uint64_t mem_read_bits;         // 32
//	uint64_t mem_input_read_bits;   // 40	
//};


#define LINE_SHIFT 6   // 64-byte cache lines (64 buckets)

/* -----------------------------------------------------------------------------
 * Register taint macros
 * -----------------------------------------------------------------------------
 */

/* Mark register as written */
.macro TAINT_REG_WRITE dst, base_reg, tmp1, tmp2
    mov     \tmp1, #1
    lsl     \tmp1, \tmp1, #\dst           // tmp1 = 1 << dst
    ldr     \tmp2, [\base_reg, #0]       // load write_bits
    orr     \tmp2, \tmp2, \tmp1
    str     \tmp2, [\base_reg, #0]       // store back
.endm

/* Mark register as read; mark input_read if not written yet */
.macro TAINT_REG_READ src, base_reg, tmp1, tmp2
    mov     \tmp1, #1
    lsl     \tmp1, \tmp1, #\src           // tmp1 = 1 << src

    // check if already written
    ldr     \tmp2, [\base_reg, #0]        // regs_write_bits
    and     \tmp2, \tmp2, \tmp1
    cbnz    \tmp2, 1f                     // already written, skip input_read

    // mark input_read_bits |= tmp
    ldr     \tmp2, [\base_reg, #16]       // regs_input_read_bits
    orr     \tmp2, \tmp2, \tmp1
    str     \tmp2, [\base_reg, #16]

1:  // mark read_bits |= tmp
    ldr     \tmp2, [\base_reg, #8]        // regs_read_bits
    orr     \tmp2, \tmp2, \tmp1
    str     \tmp2, [\base_reg, #8]
.endm

/* -----------------------------------------------------------------------------
 * Memory taint macros

 * Hashing: idx = (addr >> LINE_SHIFT) & 0x3F  (64 buckets).
 * -----------------------------------------------------------------------------
 */

/* Load from memory: mark read and input_read if not written yet */
.macro TAINT_LOAD addr_reg, base_reg, tmp1, tmp2

    // compute 64-bucket hash
    lsr     \tmp1, x\addr_reg, #6
    and     \tmp1, \tmp1, #0x3f           // idx

    mov     \tmp2, #1
    lsl     \tmp2, \tmp2, \tmp1              // mask = 1 << idx

    // check if already written
    ldr     \tmp1, [\base_reg, #24]      // mem_write_bits
    and     \tmp1, \tmp1, \tmp2
    cbnz    \tmp1, 1f                     // already written

    // mark mem_input_read_bits |= mask
    ldr     \tmp1, [\base_reg, #40]
    orr     \tmp1, \tmp1, \tmp2
    str     \tmp1, [\base_reg, #40]

1:  // mark mem_read_bits |= mask
    ldr     \tmp1, [\base_reg, #32]
    orr     \tmp1, \tmp1, \tmp2
    str     \tmp1, [\base_reg, #32]
.endm

/* Store to memory: mark written */
.macro TAINT_STORE addr_reg, base_reg, tmp1, tmp2
    lsr     \tmp1, x\addr_reg, #6
    and     \tmp1, \tmp1, #0x3f           // idx

    mov     \tmp2, #1
    lsl     \tmp2, \tmp2, \tmp1              // mask = 1 << idx

    ldr     \tmp1, [\base_reg, #24]      // mem_write_bits
    orr     \tmp1, \tmp1, \tmp2
    str     \tmp1, [\base_reg, #24]
.endm

