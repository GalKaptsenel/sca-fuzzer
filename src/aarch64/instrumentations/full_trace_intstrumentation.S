/* -----------------------------------------------------------------------------
 * taint_instrument.S
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
    TAINT_REG_READ \addr_reg, \base_reg, \tmp1, \tmp2

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
    TAINT_REG_READ \addr_reg, \base_reg, \tmp1, \tmp2
    lsr     \tmp1, x\addr_reg, #6
    and     \tmp1, \tmp1, #0x3f           // idx

    mov     \tmp2, #1
    lsl     \tmp2, \tmp2, \tmp1              // mask = 1 << idx

    ldr     \tmp1, [\base_reg, #24]      // mem_write_bits
    orr     \tmp1, \tmp1, \tmp2
    str     \tmp1, [\base_reg, #24]
.endm

/* ----------------------------------------------------------------------------- 
 * Full instruction trace macros using auto-generated offsets
 * ----------------------------------------------------------------------------- 
 * Requires:
 *   - base_reg: pointer to debug_page_t
 *   - tmp1, tmp2: temporary scratch registers
 * ----------------------------------------------------------------------------- 
 */

/* Compute pointer to next log entry and increment counter */
.macro TRACE_NEXT_ENTRY tmp1, tmp2, base_reg, output
    add     \tmp2, \base_reg, #INSTR_LOG_COUNT_OFFSET
    ldr     \tmp1, [\tmp2]  
    add     \tmp1, \tmp1, #1
    str     \tmp1, [\tmp2]
    sub     \tmp1, \tmp1, #1
    mov     \tmp2, #INSTR_LOG_ENTRY_SIZE
    mul     \tmp1, \tmp1, \tmp2

    // tmp2 = actual address of next log entry
    add     \tmp2, \base_reg, #INSTR_LOG_ARRAY_OFFSET_OFFSET
    ldr     \tmp2, [\tmp2] 
    add	    \tmp2, \base_reg, \tmp2
    add     \output, \tmp2, \tmp1
.endm

/* Log current PC (with delta correction) */
.macro TRACE_PC inst_log_entry, tmp1, tmp2
    add     \tmp2, \inst_log_entry, #INSTR_LOG_PC_OFFSET
    adr     \tmp1, 1f              // get PC of label 1 (we should garantee the logged instruction comes immidiatly after this macro)
    str     \tmp1, [\tmp2]
1:
.endm

.macro TRACE_FLAGS inst_log_entry, tmp1, tmp2
    add     \tmp2, \inst_log_entry, #INSTR_LOG_FLAGS_OFFSET
    mrs     \tmp1, nzcv
    str     \tmp1, [\tmp2]
.endm

.macro TRACE_REG reg_idx, inst_log_entry, tmp1, tmp2
    add     \tmp2, \inst_log_entry, #INSTR_LOG_REGS_OFFSET
    mov     \tmp1, \reg_idx
    lsl     \tmp1, \tmp1, #3
    add     \tmp2, \tmp1, \tmp2
    str     x\reg_idx, [\inst_log_entry, \tmp2]
.endm

/* Trace all general-purpose registers x0..x30 */
.macro TRACE_REGS inst_log_entry, tmp1, tmp2
    TRACE_REG 0, \inst_log_entry, \tmp1, \tmp2
    TRACE_REG 1, \inst_log_entry, \tmp1, \tmp2
    TRACE_REG 2, \inst_log_entry, \tmp1, \tmp2
    TRACE_REG 3, \inst_log_entry, \tmp1, \tmp2
    TRACE_REG 4, \inst_log_entry, \tmp1, \tmp2
    TRACE_REG 5, \inst_log_entry, \tmp1, \tmp2
    TRACE_REG 6, \inst_log_entry, \tmp1, \tmp2
    TRACE_REG 7, \inst_log_entry, \tmp1, \tmp2
    TRACE_REG 8, \inst_log_entry, \tmp1, \tmp2
    TRACE_REG 9, \inst_log_entry, \tmp1, \tmp2
    TRACE_REG 10, \inst_log_entry, \tmp1, \tmp2
    TRACE_REG 11, \inst_log_entry, \tmp1, \tmp2
    TRACE_REG 12, \inst_log_entry, \tmp1, \tmp2
    TRACE_REG 13, \inst_log_entry, \tmp1, \tmp2
    TRACE_REG 14, \inst_log_entry, \tmp1, \tmp2
    TRACE_REG 15, \inst_log_entry, \tmp1, \tmp2
    TRACE_REG 16, \inst_log_entry, \tmp1, \tmp2
    TRACE_REG 17, \inst_log_entry, \tmp1, \tmp2
    TRACE_REG 18, \inst_log_entry, \tmp1, \tmp2
    TRACE_REG 19, \inst_log_entry, \tmp1, \tmp2
    TRACE_REG 20, \inst_log_entry, \tmp1, \tmp2
    TRACE_REG 21, \inst_log_entry, \tmp1, \tmp2
    TRACE_REG 22, \inst_log_entry, \tmp1, \tmp2
    TRACE_REG 23, \inst_log_entry, \tmp1, \tmp2
    TRACE_REG 24, \inst_log_entry, \tmp1, \tmp2
    TRACE_REG 25, \inst_log_entry, \tmp1, \tmp2
    TRACE_REG 26, \inst_log_entry, \tmp1, \tmp2
    TRACE_REG 27, \inst_log_entry, \tmp1, \tmp2
    TRACE_REG 28, \inst_log_entry, \tmp1, \tmp2
    TRACE_REG 29, \inst_log_entry, \tmp1, \tmp2
    TRACE_REG 30, \inst_log_entry, \tmp1, \tmp2
.endm

/* Log memory read */
.macro TRACE_MEM_READ addr_reg, inst_log_entry, tmp1, tmp2
    add     \tmp2, \inst_log_entry, #INSTR_LOG_MEM_ADDR_OFFSET
    // Log the memory address
    str     \addr_reg, [\tmp2]

    // Log value before memory access
    add     \tmp2, \inst_log_entry, #INSTR_LOG_MEM_BEFORE_OFFSET
    ldr     \tmp1, [\addr_reg]
    str     \tmp1, [\tmp2]

    // Log value after memory access
    add     \tmp2, \inst_log_entry, #INSTR_LOG_MEM_AFTER_OFFSET
    str     \tmp1, [\tmp2]
.endm


/* Log memory write */
.macro TRACE_MEM_WRITE addr_reg, val_reg, inst_log_entry, tmp1, tmp2
    // Log the memory address
    add     \tmp2, \inst_log_entry, #INSTR_LOG_MEM_ADDR_OFFSET
    str     \addr_reg, [\tmp2]

    // Log value before memory access
    add     \tmp2, \inst_log_entry, #INSTR_LOG_MEM_BEFORE_OFFSET
    ldr     \tmp1, [\addr_reg]
    str     \tmp1, [\tmp2]

    // Log value after memory access
    add     \tmp2, \inst_log_entry, #INSTR_LOG_MEM_AFTER_OFFSET
    str     \val_reg, [\tmp2]
.endm

/* ----------------------------------------------------------------------------- 
 * TRACE_INSTRUCTION
 * Logs an instruction in order: next entry FLAGS -> REGS -> MEM -> PC
 * ----------------------------------------------------------------------------- 
 *
 * Parameters:
 *   base_reg      	: pointer to debug_page base
 *   tmp1, tmp2, tmp3	: temporary registers
 *   mem_type		: 0 = no mem, 1 = read, 2 = write
 *   addr_re		: register containing memory address (if mem_type != 0) [x0, x1, ..., x30]
 *   val_reg		: register containing value for store (if mem_type = 2) [x0, x1, ..., x30]
 */
.macro TRACE_INSTRUCTION base_reg, tmp1, tmp2, tmp3, mem_type, addr_reg=, val_reg=
    // 1. Compute pointer to next log entry
    TRACE_NEXT_ENTRY \tmp1, \tmp2, \base_reg, \tmp3

    // 2. Log FLAGS
    TRACE_FLAGS \tmp3, \tmp1, \tmp2

    // 3. Log all registers
    TRACE_REGS \tmp3, \tmp1, \tmp2

    // 5. Log memory if requested
    .if \mem_type == 1
        TRACE_MEM_READ \addr_reg, \tmp3, \tmp1, \tmp2
    .elseif \mem_type == 2
        TRACE_MEM_WRITE \addr_reg, \val_reg, \tmp3, \tmp1, \tmp2
    .endif

    // 5. Log PC
    TRACE_PC \tmp3, \tmp1, \tmp2

.endm
