/* -----------------------------------------------------------------------------
 * simulation_manager.S
 *
 *
 * Requires:
 *   - tmp1, tmp2, tmp3: temporary scratch registers
 * ----------------------------------------------------------------------------- 
 */

//
// struct simulation_checkpoint {
//    uint64_t FLAGS;
//    uint64_t regs[31];
//    uint8_t  memory[sizeof(sandbox_memory);
// }
 
//    uint8_t  stack[sizeof(page)-offsetof(stack, struct simulation_checkpoint)];

// CHECKPOINT_FLAGS_OFFSET = 0
// CHECKPOINT_REGS_OFFSET = 8
// CHECKPOINT_MEMORY_OFFSET = 256

.macro LOG_FLAGS checkpoint_entry, tmp
    mrs     \tmp, nzcv
    str     \tmp, [\checkpoint_entry, #0] // #INSTR_LOG_FLAGS_OFFSET
.endm

.macro LOG_REG reg_idx, checkpoint_entry, tmp
    mov     \tmp, \reg_idx
    lsl     \tmp, \tmp, #3 // sizeof(uint64_t)
    add     \tmp, \tmp, \checkpoint_entry
    str     x\reg_idx, [\tmp, #8] // #INSTR_LOG_REGS_OFFSET
.endm


.macro LOG_REGS checkpoint_entry, tmp
    .rept 31
        LOG_REG %0, \checkpoint_entry, \tmp
        .set %0, %0+1
    .endr
.endm

.macro RELOAD_FLAGS checkpoint_entry, tmp
    ldr     \tmp, [\checkpoint_entry, #0] // #INSTR_LOG_FLAGS_OFFSET
    msr     nzcv, \tmp
.endm

.macro RELOAD_REG reg_idx, checkpoint_entry, tmp
    mov     \tmp, \reg_idx
    lsl     \tmp, \tmp, #3 // sizeof(uint64_t)
    add     \tmp, \tmp, \checkpoint_entry
    ldr     x\reg_idx, [\tmp, #8] // #INSTR_LOG_REGS_OFFSET
.endm

.macro RELOAD_REGS checkpoint_entry, tmp
    .rept 31
        RELOAD_REG %0, \checkpoint_entry, \tmp
        .set %0, %0+1
    .endr
.endm


// MOV_BUFFER: copy `size` bytes from [src] -> [dst] using caller-provided temps
// params: src, dst, size, tmp_ptr_src, tmp_ptr_dst, tmp_size, tmp_val
.macro MOV_BUFFER src, dst, size, tmp_ptr_src, tmp_ptr_dst, tmp_size, tmp_val
    mov     \tmp_ptr_src, \src      // Xn holding source pointer
    mov     \tmp_ptr_dst, \dst      // Xn holding destination pointer
    mov     \tmp_size, \size        // Xn holding remaining size

1:
    ldrb    \tmp_val, [\tmp_ptr_src], #1
    strb    \tmp_val, [\tmp_ptr_dst], #1
    subs    \tmp_size, \tmp_size, #1
    b.ne    1b
.endm


// TAKE_SNAPSHOT: log regs/flags then copy sandbox -> snapshot
// params: sandbox_addr, snapshot_buffer, size, regs_buffer, tmp_ptr_src, tmp_ptr_dst, tmp_size, tmp_val
.macro TAKE_SNAPSHOT checkpoint_entry, sandbox_addr, size, tmp_ptr_src, tmp_ptr_dst, tmp_size, tmp_val, tmp
    LOG_FLAGS  \checkpoint_entry, \tmp
    LOG_REGS   \checkpoint_entry, \tmp
    add        \tmp, \checkpoint_entry, #256 // #CHECKPOINT_MEMORY_OFFSET
    MOV_BUFFER \sandbox_addr, \tmp, \size, \
               \tmp_ptr_src, \tmp_ptr_dst, \tmp_size, \tmp_val
.endm


// RELOAD_SNAPSHOT: copy snapshot -> sandbox, then reload regs/flags
// params: snapshot_buffer, sandbox_addr, size, regs_buffer, tmp_ptr_src, tmp_ptr_dst, tmp_size, tmp_val
.macro RELOAD_SNAPSHOT checkpoint_entry, sandbox_addr, size, tmp_ptr_src, tmp_ptr_dst, tmp_size, tmp_val, tmp
    add        \tmp, \checkpoint_entry, #256 // #CHECKPOINT_MEMORY_OFFSET
    MOV_BUFFER  \tmp, \sandbox_addr, \size, \
                \tmp_ptr_src, \tmp_ptr_dst, \tmp_size, \tmp_val
    RELOAD_REGS  \checkpoint_entry, \tmp
    RELOAD_FLAGS \checkpoint_entry, \tmp
.endm


// RET_IF_EXHAUSTED_WINDOW: if window_counter >= max_window pop RSB and return
// params: window_counter_ptr, max_window_imm_or_reg, rsb_address, tmp_load, tmp_addr
.macro RET_IF_EXHAUSTED_WINDOW window_counter_ptr, max_window, rsb_address, tmp_load, tmp_addr
    ldr     \tmp_load, [\window_counter_ptr]    // load current counter
    cmp     \tmp_load, \max_window
    b.lt    1f

    // exhausted -> pop (counter, return_addr) from software RSB and branch to return_addr
    ldp     \tmp_load, \tmp_addr, [\rsb_address], #16

    str     \tmp_load, [\window_counter_ptr]
    br      \tmp_addr
1:
.endm


.macro RET_IF_EXHAUSTED_NESTING_WINDOW nesting_window_counter_ptr, max_window, rsb_address, tmp_load, tmp_addr
    RET_IF_EXHAUSTED_WINDOW \nesting_window_counter_ptr, \max_window, \rsb_address, \tmp_load, \tmp_addr
.endm


// COND_SPECULATION_SIMULATION:
// - computes primary target and opposite target using ADR/CSEL
// - pushes (window_counter, opposite_target) onto software RSB
// - increments window_counter and branches to primary_target
//
// params:
//   cond              - condition code token (eq/ne/lt/...)
//   taken_label       - label (symbol) for taken path
//   not_taken_label   - label (symbol) for not-taken path
//   window_counter_ptr
//   nesting_window_counter_ptr
//   nesting_max_window
//   rsb_address
//   tmp_load      - temp reg for loads/counters
//   tmp_at       - temp reg for address taken
//   tmp_nt       - temp reg for address not-taken
//   tmp_sel      - temp reg for selected primary/opposite target
.macro COND_SPECULATION_SIMULATION cond, taken_label, not_taken_label, \
                                      window_counter_ptr, nesting_window_counter_ptr, nesting_max_window, \
                                      rsb_address, tmp_load, tmp_at, tmp_nt, tmp_sel
    // Check nested speculation exhaustion
    RET_IF_EXHAUSTED_NESTING_WINDOW \nesting_window_counter_ptr, \
                                    \nesting_max_window, \
                                    \rsb_address, \tmp_load, \tmp_sel

    ldr     \tmp_load, [\window_counter_ptr]

    adr     \tmp_at, \taken_label
    adr     \tmp_nt, \not_taken_label

    // primary = cond ? taken : not_taken
    csel    \tmp_sel, \tmp_at, \tmp_nt, \cond

    // opposite = cond ? not_taken : taken
    csel    \tmp_at, \tmp_nt, \tmp_at, \cond   // reuse tmp_at as opposite

    // push (window_counter, primary) onto software RSB (pre-decrement)
    stp     \tmp_load, \tmp_sel, [\rsb_address, #-16]!

    // increment and store window_counter
    add     \tmp_load, \tmp_load, #1
    str     \tmp_load, [\window_counter_ptr]

    // jump to opposite target
    br      \tmp_at
.endm


// SIMULATE_INIT: initialize nesting counter and push exit address onto RSB
// params: nesting_window_counter_ptr, rsb_address, exit_address_symbol, tmp_zero, tmp_addr
.macro SIMULATE_INIT nesting_window_counter_ptr, rsb_address, exit_address, tmp_zero, tmp_addr
    mov     \tmp_zero, #0
    adr     \tmp_addr, \exit_address
    stp     \tmp_zero, \tmp_addr, [\rsb_address, #-16]!
    str     \tmp_zero, [\nesting_window_counter_ptr]
.endm

