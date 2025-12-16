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

/* --- Helper: mark bit in a taint field --- */
.macro _TAINT_SET_BIT bit, field_offset, base_reg, tmp1, tmp2
    mov     \tmp1, #1
    lsl     \tmp1, \tmp1, #\bit // tmp1 = 1 << dst
    ldr     \tmp2, [\base_reg, #\field_offset] 
    orr     \tmp2, \tmp2, \tmp1
    str     \tmp2, [\base_reg, #\field_offset]
.endm

/* Mark register as written */
.macro TAINT_REG_WRITE dst, base_reg, tmp1, tmp2
    _TAINT_SET_BIT \dst, 0, \base_reg, \tmp1, \tmp2 // regs_write_bits
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
    _TAINT_SET_BIT \src, 16, \base_reg, \tmp1, \tmp2 // regs_input_read_bits

1:  // mark read_bits |= tmp
    _TAINT_SET_BIT \src, 8, \base_reg, \tmp1, \tmp2 // regs_read_bits
.endm

/* Mark FLAGS (NZCV) as written using bit 36*/
.macro TAINT_FLAGS_WRITE n, z, c, v, base_reg, tmp1, tmp2
    TAINT_REG_WRITE 36, \base_reg, \tmp1, \tmp2

    // Optionally mark individual bits
    .if \n
        TAINT_REG_WRITE 32, \base_reg, \tmp1, \tmp2
    .endif
    .if \z
        TAINT_REG_WRITE 33, \base_reg, \tmp1, \tmp2
    .endif
    .if \c
        TAINT_REG_WRITE 34, \base_reg, \tmp1, \tmp2
    .endif
    .if \v
        TAINT_REG_WRITE 35, \base_reg, \tmp1, \tmp2
    .endif
.endm


/* Mark FLAGS (NZCV) as read using bit 36*/
.macro TAINT_FLAGS_READ n, z, c, v, base_reg, tmp1, tmp2
    TAINT_REG_READ 36, \base_reg, \tmp1, \tmp2

    // Optionally mark individual bits
    .if \n
        TAINT_REG_READ 32, \base_reg, \tmp1, \tmp2
    .endif
    .if \z
        TAINT_REG_READ 33, \base_reg, \tmp1, \tmp2
    .endif
    .if \c
        TAINT_REG_READ 34, \base_reg, \tmp1, \tmp2
    .endif
    .if \v
        TAINT_REG_READ 35, \base_reg, \tmp1, \tmp2
    .endif
.endm

/* SP read/write marks bit 31 (SP origin) */
.macro TAINT_SP_READ base_reg, tmp1, tmp2
    TAINT_REG_READ 31, \base_reg, \tmp1, \tmp2     // mark x9 itself
.endm

.macro TAINT_SP_WRITE base_reg, tmp1, tmp2
    TAINT_REG_WRITE 31, \base_reg, \tmp1, \tmp2     // mark x9 itself
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

