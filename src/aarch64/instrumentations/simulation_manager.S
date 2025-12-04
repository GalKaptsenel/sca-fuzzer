/* -----------------------------------------------------------------------------
 * simulation_manager.S
 * -----------------------------------------------------------------------------
 */

// struct simulation_checkpoint {
//    uint64_t FLAGS;
//    uint64_t regs[31];
//    uint8_t  memory[...];    // at offset 256
// }

// CHECKPOINT_FLAGS_OFFSET	= 0
// CHECKPOINT_REGS_OFFSET	= 8
// CHECKPOINT_MEMORY_OFFSET	= 256

// struct simulation_mgmt{
//    uint64_t  current_nesting;
//    uint64_t* stack_top;
//    struct simulation_checkpoint snapshots_array[];
//    uint64_t  snapshot_size;
//    uint8_t stack[...];
// }

// SIM_MGMT_CURRENT_NESTING_OFFSET	= 0
// SIM_MGMT_STACK_TOP_OFFSET		= 8
// SIM_MGMT_SNAPSHOTS_ARRAY_OFFSET	= 16
// SIM_MGMT_SNAPSHOT_SIZE_OFFSET	= 24
// SIM_MGMT_STACK_END_OFFSET		= 32
// SIM_MGMT_STACK_BASE_OFFSET		= sizeof(PAGE) - SIM_MGMT_STACK_END_OFFSET = 4096-32 = 4064
// SIM_MGMT_SIM_MGMT_SIZEOF		= sizeof(struct simulation_mgmt)


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
    .irp r,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30
        LOG_REG \r, \checkpoint_entry, \tmp
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
    .irp r,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30
        RELOAD_REG \r, \checkpoint_entry, \tmp
    .endr
.endm


// MOV_BUFFER: copy `size` bytes from [src] -> [dst] using caller-provided temps
// params: src, dst, size, tmp_ptr_src, tmp_ptr_dst, tmp_size, tmp_val
.macro MOV_BUFFER src, dst, size, tmp_size, tmp_val
    mov     \tmp_size, \size

1:
    subs    \tmp_size, \tmp_size, #8

    add     \tmp_size, \tmp_size, \src
    ldr    \tmp_val, [\tmp_size]
    sub     \tmp_size, \tmp_size, \src

    add     \tmp_size, \tmp_size, \dst
    str    \tmp_val, [\tmp_size]
    sub     \tmp_size, \tmp_size, \dst

    b.ne    1b
.endm


// Compute simulation_checkpoint entry address
// checkpoint_id is register or immidiate
.macro GET_SNAPSHOT_ADDRESS sim_mgmt, checkpoint_id, result, tmp
    // tmp_ptr_src = (byte)sim_mgmt->snapshots_array + sim_mgmt->snapshot_size * checkpoint_id
    ldr	       \tmp, [\sim_mgmt, #24]			// #SIM_MGMT_SNAPSHOT_SIZE_OFFSET
    mov        \result, \checkpoint_id
    mul        \result, \tmp, \result
    ldr	       \tmp, [\sim_mgmt, #16]			// #SIM_MGMT_SNAPSHOTS_ARRAY_OFFSET
    add        \result, \tmp, \result
.endm

.macro TAKE_SNAPSHOT sim_mgmt, checkpoint_id, sandbox_addr, size, tmp1, tmp2, tmp3, tmp4
    GET_SNAPSHOT_ADDRESS \sim_mgmt, \checkpoint_id, \tmp1, \tmp2
    LOG_FLAGS  \tmp1, \tmp2
    LOG_REGS   \tmp1, \tmp2

    add        \tmp1, \tmp1, #256 	// #CHECKPOINT_MEMORY_OFFSET

    movz       \tmp4, #(\sandbox_addr & 0xFFFF)
    movk       \tmp4, #(\sandbox_addr & 0xFFFF), lsl #16
    movk       \tmp4, #(\sandbox_addr & 0xFFFF), lsl #32
    movk       \tmp4, #(\sandbox_addr & 0xFFFF), lsl #48

    MOV_BUFFER \tmp4, \tmp1, \size, \tmp2, \tmp3
.endm


.macro RELOAD_SNAPSHOT sim_mgmt, snapshot_address, sandbox_addr, size, tmp1, tmp2, tmp3, tmp4
    add          \tmp1, \snapshot_address, #256 // #CHECKPOINT_MEMORY_OFFSET

    movz         \tmp4, #(\sandbox_addr & 0xFFFF)
    movk         \tmp4, #(\sandbox_addr & 0xFFFF), lsl #16
    movk         \tmp4, #(\sandbox_addr & 0xFFFF), lsl #32
    movk         \tmp4, #(\sandbox_addr & 0xFFFF), lsl #48

    MOV_BUFFER   \tmp1, \tmp4, \size, \tmp2, \tmp3
    RELOAD_REGS  \snapshot_address, \tmp1
    RELOAD_FLAGS \snapshot_address, \tmp1
.endm

.macro SAFE_SIMULATION_SPEC_RET sim_mgmt, sandbox_addr, size, tmp1, tmp2, tmp3, tmp4, tmp5
    // exhausted -> pop (counter, return_addr, snapshot_id, reserved) from software RSB and branch to return_addr

    ldr     \tmp1, [\sim_mgmt, #8] 	// #SIM_MGMT_STACK_TOP_OFFSET
    ldp     \tmp2, xzr, [\tmp1], #16
    str     \tmp1, [\sim_mgmt, #8]     	// #SIM_MGMT_STACK_TOP_OFFSET

    GET_SNAPSHOT_ADDRESS \sim_mgmt, \tmp2, \tmp3, \tmp4
    RELOAD_SNAPSHOT \sim_mgmt, \tmp3, \sandbox_addr, \size, \tmp1, \tmp2, \tmp4, \tmp5

    ldr     \tmp1, [\sim_mgmt, #8] 	// #SIM_MGMT_STACK_TOP_OFFSET
    ldp     \tmp3, \tmp4, [\tmp1], #16
    str     \tmp1, [\sim_mgmt, #8]     	// #SIM_MGMT_STACK_TOP_OFFSET
    str     \tmp3, [\sim_mgmt, #0]    	// #SIM_MGMT_CURRENT_NESTING_OFFSET

    br      \tmp4
.endm

.macro _RET_IF_EXHAUSTED_WINDOW sim_mgmt, max_window, sandbox_addr, size, tmp1, tmp2, tmp3, tmp4, tmp5
    ldr     \tmp1, [\sim_mgmt, #0]    	// #SIM_MGMT_CURRENT_NESTING_OFFSET
    cmp     \tmp1, \max_window
    b.lt    1f

    SAFE_SIMULATION_SPEC_RET \sim_mgmt, \sandbox_addr, \size, \tmp1, \tmp2, \tmp3, \tmp4, \tmp5
1:
.endm

.macro _RET_IF_EXHAUSTED_NESTING_WINDOW sim_mgmt, max_window, sandbox_addr, size, tmp1, tmp2, tmp3, tmp4, tmp5
    _RET_IF_EXHAUSTED_WINDOW \sim_mgmt, \max_window, \sandbox_addr, \size, \tmp1, \tmp2, \tmp3, \tmp4, \tmp5
.endm


.macro SIMULATION_COND_SPECULATION cond, taken_label, not_taken_label, sim_mgmt, snapshot_id, max_nesting,  sandbox_addr, sandbox_size tmp1, tmp2, tmp3, tmp4, tmp5, tmp6

    mrs     \tmp1, nzcv

    // Check for nested speculation exhaustion
    _RET_IF_EXHAUSTED_NESTING_WINDOW \sim_mgmt, \max_nesting, \sandbox_addr, \sandbox_size, \tmp2, \tmp3, \tmp4, \tmp5, \tmp6

    msr     nzcv, \tmp1

    adr     \tmp1, \taken_label
    adr     \tmp2, \not_taken_label

    // primary = cond ? taken : not_taken
    csel    \tmp3, \tmp1, \tmp2, \cond

    // opposite = cond ? not_taken : taken
    csel    \tmp4, \tmp2, \tmp1, \cond


    // If nesting is 0, we are actually not speculating, and therefore push 0 nesting to the "architectualy*
    // Otherwise, increment and store window_counter

    mov     \tmp5, \snapshot_id
    ldr     \tmp2, [\sim_mgmt, #8]	// #SIM_MGMT_STACK_TOP_OFFSET
    ldr     \tmp1, [\sim_mgmt, #0] 	// #SIM_MGMT_CURRENT_NESTING_OFFSET
    cbz     \tmp1, 1f 	

    // Increment and then push pair
			
    add     \tmp1, \tmp1, #1
    str     \tmp1, [\sim_mgmt, #0]	// #SIM_MGMT_CURRENT_NESTING_OFFSET

    // push (nesting_counter, primary, snapshot_id, reserved) onto software RSB (pre-decrement)
    stp     \tmp1, \tmp3, [\tmp2, #-16]!
    stp     \tmp5, xzr, [\tmp2, #-16]!
    str     \tmp2, [\sim_mgmt, #8]	// #SIM_MGMT_STACK_TOP_OFFSET

    b 2f
1:
    // Push pair and then increment

    // push (nesting_counter, primary, snapshot_id, reserved) onto software RSB (pre-decrement)
    stp     \tmp1, \tmp3, [\tmp2, #-16]!
    stp     \tmp5, xzr, [\tmp2, #-16]!
    str     \tmp2, [\sim_mgmt, #8]	// #SIM_MGMT_STACK_TOP_OFFSET

    // increment and store window_counter
    add     \tmp1, \tmp1, #1
    str     \tmp1, [\sim_mgmt, #0]	// #SIM_MGMT_CURRENT_NESTING_OFFSET

2:
    // jump to opposite target
    br      \tmp4
.endm

.macro SIMULATION_INIT sim_mgmt, snapshot_array, snapshot_size, tmp, tmp_addr
    str     xzr, [\sim_mgmt, #0]			// #SIM_MGMT_CURRENT_NESTING_OFFSET
    mov     \tmp, \snapshot_array
    str     \tmp, [\sim_mgmt, #16]			// #SIM_MGMT_SNAPSHOTS_ARRAY_OFFSET
    mov     \tmp, \snapshot_size
    str     \tmp, [\sim_mgmt, #24]			// #SIM_MGMT_SNAPSHOT_SIZE_OFFSET
    add     \tmp, \sim_mgmt, #4064			// #SIM_MGMT_STACK_BASE_OFFSET
    str     \tmp, [\sim_mgmt, #8]			// #SIM_MGMT_STACK_TOP_OFFSET
.endm

.macro SIMULATION_SPEC_RET sim_mgmt, sandbox_addr, size, tmp1, tmp2, tmp3, tmp4, tmp5

    ldr     \tmp1, [\sim_mgmt, #0]    	// #SIM_MGMT_CURRENT_NESTING_OFFSET
    cbz     \tmp1, 1f

    SAFE_SIMULATION_SPEC_RET \sim_mgmt, \sandbox_addr, \size, \tmp1, \tmp2, \tmp3, \tmp4, \tmp5
1:
.endm

