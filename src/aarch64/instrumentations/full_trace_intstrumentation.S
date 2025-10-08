/* -----------------------------------------------------------------------------
 * full_trace_instrumentation.S
 *
 * Single-file, pinned, parameterized macros for register and memory tainting.
 *
 * Requires:
 *   - base_reg: pointer to aux_buffer_t
 *   - tmp1, tmp2, tmp3, tmp4 : temporary scratch registers
 * ----------------------------------------------------------------------------- 
 */

//struct aux_buffer_t {
//	uint64_t instruction_log_array_offset; 	// 0
//	uint64_t instruction_log_entry_count; 	// 8
//	uint64_t instruction_log_max_count; 	// 16
//};

//struct instruction_log_entry_t {
//	uint64_t pc;			// 0
//	uint64_t flags;			// 8
//	uint64_t regs[31];		// 16
//	uint64_t effective_address;	// 264
//	uint64_t mem_before;		// 272
//	uint64_t mem_after;		// 280
//};


#define INSTR_LOG_ARRAY_OFFSET_OFFSET 0
#define INSTR_LOG_COUNT_OFFSET 8
#define INSTR_LOG_MAX_COUNT_OFFSET 16

#define INSTR_LOG_PC_OFFSET 0
#define INSTR_LOG_FLAGS_OFFSET 8
#define INSTR_LOG_REGS_OFFSET 16
#define INSTR_LOG_MEM_ADDR_OFFSET 264
#define INSTR_LOG_MEM_BEFORE_OFFSET 272
#define INSTR_LOG_MEM_AFTER_OFFSET 280
#define INSTR_LOG_ENTRY_SIZE	288

/* initialize */
.macro TRACE_INIT base_reg, tmp1, tmp2
    mov     \tmp2, \base_reg
    mov     \tmp1, #24		// Skip 3 * sizeof(uint64_t)
    str     \tmp1, [\tmp2], #8
    mov     \tmp1, #0		// Initialize INSTR_LOG_COUNT member to 0
    str     \tmp1, [\tmp2], #8
    mov     \tmp1, #255		// Allow to log up to 255 instructions. Notice, auxiliary buffer needs to be of size >= 255*sizeof(instruction_log)+3*sizeof(uint64_t)=255*288+24=73464 bytes, a.k.a >= 18 pages of 4096 bytes log size per (input, testcae) pair
    str     \tmp1, [\tmp2], #8
.endm

/* Compute pointer to next log entry and increment counter */
.macro TRACE_NEXT_ENTRY tmp1, tmp2, base_reg, output
    add     \tmp2, \base_reg, #8 //#INSTR_LOG_COUNT_OFFSET
    ldr     \tmp1, [\tmp2]
    add     \tmp1, \tmp1, #1
    str     \tmp1, [\tmp2]
    sub     \tmp1, \tmp1, #1
    mov     \tmp2, #288 //#INSTR_LOG_ENTRY_SIZE
    mul     \tmp1, \tmp1, \tmp2

    // tmp2 = actual address of next log entry
    add     \tmp2, \base_reg, #0 //#INSTR_LOG_ARRAY_OFFSET_OFFSET
    ldr     \tmp2, [\tmp2] 
    add	    \tmp2, \base_reg, \tmp2
    add     \output, \tmp2, \tmp1
.endm

/* Log current PC (with delta correction) */
.macro TRACE_PC inst_log_entry, tmp1, tmp2
    add     \tmp2, \inst_log_entry, #0 //#INSTR_LOG_PC_OFFSET
    adr     \tmp1, 1f              // get PC of label 1 (we should garantee the logged instruction comes immidiatly after this macro)
    str     \tmp1, [\tmp2]
1:
.endm

.macro TRACE_FLAGS inst_log_entry, tmp1, tmp2
    add     \tmp2, \inst_log_entry, #8 //#INSTR_LOG_FLAGS_OFFSET
    mrs     \tmp1, nzcv
    str     \tmp1, [\tmp2]
.endm

.macro TRACE_REG reg_idx, inst_log_entry, tmp1, tmp2
    add     \tmp2, \inst_log_entry, #16 //#INSTR_LOG_REGS_OFFSET
    mov     \tmp1, \reg_idx
    lsl     \tmp1, \tmp1, #3
    add     \tmp2, \tmp1, \tmp2
    str     x\reg_idx, [\tmp2]
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
    add     \tmp2, \inst_log_entry, #264 //#INSTR_LOG_MEM_ADDR_OFFSET
    // Log the memory address
    str     \addr_reg, [\tmp2]

    // Log value before memory access
    add     \tmp2, \inst_log_entry, #272 //#INSTR_LOG_MEM_BEFORE_OFFSET
    ldr     \tmp1, [\addr_reg]
    str     \tmp1, [\tmp2]

    // Log value after memory access
    add     \tmp2, \inst_log_entry, #280 //#INSTR_LOG_MEM_AFTER_OFFSET
    str     \tmp1, [\tmp2]
.endm


/* Log memory write */
.macro TRACE_MEM_WRITE addr_reg, val_reg, inst_log_entry, tmp1, tmp2
    // Log the memory address
    add     \tmp2, \inst_log_entry, #264 //#INSTR_LOG_MEM_ADDR_OFFSET
    str     \addr_reg, [\tmp2]

    // Log value before memory access
    add     \tmp2, \inst_log_entry, #272 //#INSTR_LOG_MEM_BEFORE_OFFSET
    ldr     \tmp1, [\addr_reg]
    str     \tmp1, [\tmp2]

    // Log value after memory access
    add     \tmp2, \inst_log_entry, #280 //#INSTR_LOG_MEM_AFTER_OFFSET
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
