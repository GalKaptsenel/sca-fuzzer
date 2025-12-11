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
//    uint8_t stack[PAGE_SIZE];
//    uint8_t snapshot_array_data[...];
// }

// SIM_MGMT_CURRENT_NESTING_OFFSET	= 0
// SIM_MGMT_STACK_TOP_OFFSET		= 8
// SIM_MGMT_SNAPSHOTS_ARRAY_OFFSET	= 16
// SIM_MGMT_SNAPSHOT_SIZE_OFFSET	= 24
// SIM_MGMT_STACK_END_OFFSET		= 32
// SIM_MGMT_STACK_BASE_OFFSET		= sizeof(PAGE) = 4096 			// Size of stack is PAGE_SIZE - SIM_MGMT_STACK_END_OFFSET
// SIM_MGMT_SNAPSHOT_ARRAY_DATA_OFFSET	= SIM_MGMT_STACK_BASE_OFFSET + 8
// SIM_MGMT_SIM_MGMT_SIZEOF		= sizeof(struct simulation_mgmt)


// Helper: perform action on reg_idx if it is NOT in skip list
.macro REG_ACTION_IF_NOT_SKIP reg_idx, action_macro, checkpoint_entry, tmp, skip0=-1, skip1=-1, skip2=-1, skip3=-1, skip4=-1, skip5=-1, skip6=-1, skip7=-1, skip8=-1, skip9=-1, skip10=-1, skip11=-1, skip12=-1, skip13=-1, skip14=-1, skip15=-1, skip16=-1, skip17=-1, skip18=-1, skip19=-1, skip20=-1, skip21=-1, skip22=-1, skip23=-1, skip24=-1, skip25=-1, skip26=-1, skip27=-1, skip28=-1, skip29=-1, skip30=-1
    .ifc x\reg_idx, \skip0
    .else
    .ifc x\reg_idx, \skip1
    .else
    .ifc x\reg_idx, \skip2
    .else
    .ifc x\reg_idx, \skip3
    .else
    .ifc x\reg_idx, \skip4
    .else
    .ifc x\reg_idx, \skip5
    .else
    .ifc x\reg_idx, \skip6
    .else
    .ifc x\reg_idx, \skip7
    .else
    .ifc x\reg_idx, \skip8
    .else
    .ifc x\reg_idx, \skip9
    .else
    .ifc x\reg_idx, \skip10
    .else
    .ifc x\reg_idx, \skip11
    .else
    .ifc x\reg_idx, \skip12
    .else
    .ifc x\reg_idx, \skip13
    .else
    .ifc x\reg_idx, \skip14
    .else
    .ifc x\reg_idx, \skip15
    .else
    .ifc x\reg_idx, \skip16
    .else
    .ifc x\reg_idx, \skip17
    .else
    .ifc x\reg_idx, \skip18
    .else
    .ifc x\reg_idx, \skip19
    .else
    .ifc x\reg_idx, \skip20
    .else
    .ifc x\reg_idx, \skip21
    .else
    .ifc x\reg_idx, \skip22
    .else
    .ifc x\reg_idx, \skip23
    .else
    .ifc x\reg_idx, \skip24
    .else
    .ifc x\reg_idx, \skip25
    .else
    .ifc x\reg_idx, \skip26
    .else
    .ifc x\reg_idx, \skip27
    .else
    .ifc x\reg_idx, \skip28
    .else
    .ifc x\reg_idx, \skip29
    .else
    .ifc x\reg_idx, \skip30
    .else // Don't skip this register
        \action_macro \reg_idx, \checkpoint_entry, \tmp
    .endif
    .endif
    .endif
    .endif
    .endif
    .endif
    .endif
    .endif
    .endif
    .endif
    .endif
    .endif
    .endif
    .endif
    .endif
    .endif
    .endif
    .endif
    .endif
    .endif
    .endif
    .endif
    .endif
    .endif
    .endif
    .endif
    .endif
    .endif
    .endif
    .endif
    .endif
.endm

.macro LOOP_REGS action_macro, checkpoint_entry, tmp, skip0=-1, skip1=-1, skip2=-1, skip3=-1, skip4=-1, skip5=-1, skip6=-1, skip7=-1, skip8=-1, skip9=-1, skip10=-1, skip11=-1, skip12=-1, skip13=-1, skip14=-1, skip15=-1, skip16=-1, skip17=-1, skip18=-1, skip19=-1, skip20=-1, skip21=-1, skip22=-1, skip23=-1, skip24=-1, skip25=-1, skip26=-1, skip27=-1, skip28=-1, skip29=-1, skip30=-1
    .irp r,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30
        REG_ACTION_IF_NOT_SKIP \r, \action_macro, \checkpoint_entry, \tmp, \skip0, \skip1, \skip2, \skip3, \skip4, \skip5, \skip6, \skip7, \skip8, \skip9, \skip10, \skip11, \skip12, \skip13, \skip14, \skip15, \skip16, \skip17, \skip18, \skip19, \skip20, \skip21, \skip22, \skip23, \skip24, \skip25, \skip26, \skip27, \skip28, \skip29, \skip30
    .endr
.endm


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


.macro LOG_REGS checkpoint_entry, tmp, skip0=-1, skip1=-1, skip2=-1, skip3=-1, skip4=-1, skip5=-1, skip6=-1, skip7=-1, skip8=-1, skip9=-1, skip10=-1, skip11=-1, skip12=-1, skip13=-1, skip14=-1, skip15=-1, skip16=-1, skip17=-1, skip18=-1, skip19=-1, skip20=-1, skip21=-1, skip22=-1, skip23=-1, skip24=-1, skip25=-1, skip26=-1, skip27=-1, skip28=-1, skip29=-1, skip30=-1

    LOOP_REGS LOG_REG, \checkpoint_entry, \tmp, \skip0, \skip1, \skip2, \skip3, \skip4, \skip5, \skip6, \skip7, \skip8, \skip9, \skip10, \skip11, \skip12, \skip13, \skip14, \skip15, \skip16, \skip17, \skip18, \skip19, \skip20, \skip21, \skip22, \skip23, \skip24, \skip25, \skip26, \skip27, \skip28, \skip29, \skip30

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

.macro RELOAD_REGS checkpoint_entry, tmp, skip0=-1, skip1=-1, skip2=-1, skip3=-1, skip4=-1, skip5=-1, skip6=-1, skip7=-1, skip8=-1, skip9=-1, skip10=-1, skip11=-1, skip12=-1, skip13=-1, skip14=-1, skip15=-1, skip16=-1, skip17=-1, skip18=-1, skip19=-1, skip20=-1, skip21=-1, skip22=-1, skip23=-1, skip24=-1, skip25=-1, skip26=-1, skip27=-1, skip28=-1, skip29=-1, skip30=-1

    LOOP_REGS RELOAD_REG, \checkpoint_entry, \tmp, \skip0, \skip1, \skip2, \skip3, \skip4, \skip5, \skip6, \skip7, \skip8, \skip9, \skip10, \skip11, \skip12, \skip13, \skip14, \skip15, \skip16, \skip17, \skip18, \skip19, \skip20, \skip21, \skip22, \skip23, \skip24, \skip25, \skip26, \skip27, \skip28, \skip29, \skip30

.endm

// MOV_BUFFER: copy `size` bytes from [src] -> [dst] using caller-provided temps
// params: src, dst, size, tmp_ptr_src, tmp_ptr_dst, tmp_size, tmp_val
.macro MOV_BUFFER src, dst, size, tmp_size, tmp_val
    mov     \tmp_size, \size

1:
    subs    \tmp_size, \tmp_size, #8

    ldr    \tmp_val, [\src, \tmp_size]
    str    \tmp_val, [\dst, \tmp_size]

    cbnz    \tmp_size, 1b
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
    .set _take_sa0, ((\sandbox_addr)      & 0xFFFF)
    .set _take_sa1, ((\sandbox_addr >> 16) & 0xFFFF)
    .set _take_sa2, ((\sandbox_addr >> 32) & 0xFFFF)
    .set _take_sa3, ((\sandbox_addr >> 48) & 0xFFFF)

//TODO: TRY to understand why It looks like I corrctly compute this address, but the store in the MOV macro fails
    GET_SNAPSHOT_ADDRESS \sim_mgmt, \checkpoint_id, \tmp1, \tmp2
    LOG_FLAGS  \tmp1, \tmp2
    LOG_REGS   \tmp1, \tmp2

    add        \tmp1, \tmp1, #256 	// #CHECKPOINT_MEMORY_OFFSET

    movz       \tmp4, #_take_sa0
    movk       \tmp4, #_take_sa1, lsl #16
    movk       \tmp4, #_take_sa2, lsl #32
    movk       \tmp4, #_take_sa3, lsl #48

    MOV_BUFFER \tmp4, \tmp1, \size, \tmp2, \tmp3
.endm


.macro RELOAD_SNAPSHOT sim_mgmt, snapshot_addr, sandbox_addr, size, tmp1, tmp2, tmp3, tmp4
    .set _reload_sa0, ((\sandbox_addr)      & 0xFFFF)
    .set _reload_sa1, ((\sandbox_addr >> 16) & 0xFFFF)
    .set _reload_sa2, ((\sandbox_addr >> 32) & 0xFFFF)
    .set _reload_sa3, ((\sandbox_addr >> 48) & 0xFFFF)


    add          \tmp1, \snapshot_addr, #256 // #CHECKPOINT_MEMORY_OFFSET

    movz       \tmp4, #_reload_sa0
    movk       \tmp4, #_reload_sa1, lsl #16
    movk       \tmp4, #_reload_sa2, lsl #32
    movk       \tmp4, #_reload_sa3, lsl #48

    MOV_BUFFER   \tmp1, \tmp4, \size, \tmp2, \tmp3
    RELOAD_REGS  \snapshot_addr, \tmp1, skip0=\snapshot_addr, skip1=\tmp1
    RELOAD_FLAGS \snapshot_addr, \tmp1
.endm

.macro _SAFE_SIMULATION_SPEC_RET sim_mgmt, sandbox_addr, size, tmp1, tmp2, tmp3, tmp4, tmp5
    // exhausted -> pop (counter, return_addr, snapshot_id, reserved) from software RSB and branch to return_addr

    ldr     \tmp1, [\sim_mgmt, #8] 	// #SIM_MGMT_STACK_TOP_OFFSET
    ldp     \tmp2, xzr, [\tmp1]

    GET_SNAPSHOT_ADDRESS \sim_mgmt, \tmp2, \tmp3, \tmp4

    RELOAD_SNAPSHOT \sim_mgmt, \tmp3, \sandbox_addr, \size, \tmp1, \tmp2, \tmp4, \tmp5

    ldr     \tmp1, [\sim_mgmt, #8] 	// #SIM_MGMT_STACK_TOP_OFFSET
    ldp     \tmp3, \tmp4, [\tmp1, #16]

    str     \tmp3, [\sim_mgmt, #0]    	// #SIM_MGMT_CURRENT_NESTING_OFFSET

    add     \tmp1, \tmp1, #32
    str     \tmp1, [\sim_mgmt, #8]     	// #SIM_MGMT_STACK_TOP_OFFSET

    br      \tmp4
.endm

.macro _RET_IF_EXHAUSTED_WINDOW sim_mgmt, max_window, sandbox_addr, size, tmp1, tmp2, tmp3, tmp4, tmp5
    ldr     \tmp1, [\sim_mgmt, #0]    	// #SIM_MGMT_CURRENT_NESTING_OFFSET
    cmp     \tmp1, \max_window
    b.lt    2f

    _SAFE_SIMULATION_SPEC_RET \sim_mgmt, \sandbox_addr, \size, \tmp1, \tmp2, \tmp3, \tmp4, \tmp5
2:
.endm

.macro _RET_IF_EXHAUSTED_NESTING_WINDOW sim_mgmt, max_window, sandbox_addr, size, tmp1, tmp2, tmp3, tmp4, tmp5
    _RET_IF_EXHAUSTED_WINDOW \sim_mgmt, \max_window, \sandbox_addr, \size, \tmp1, \tmp2, \tmp3, \tmp4, \tmp5
.endm


.macro SIMULATION_COND_SPECULATION cond, taken_label, not_taken_label, sim_mgmt, snapshot_id, max_nesting,  sandbox_addr, sandbox_size, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, cmpre_reg=-1

    mrs     \tmp1, nzcv

    // Check for nested speculation exhaustion
    _RET_IF_EXHAUSTED_NESTING_WINDOW \sim_mgmt, \max_nesting, \sandbox_addr, \sandbox_size, \tmp2, \tmp3, \tmp4, \tmp5, \tmp6

    msr     nzcv, \tmp1

    adr     \tmp1, \taken_label
    adr     \tmp2, \not_taken_label

    .ifc \cmpre_reg, -1

        // primary = cond ? taken : not_taken
        csel    \tmp3, \tmp1, \tmp2, \cond
    
        // opposite = cond ? not_taken : taken
        csel    \tmp4, \tmp2, \tmp1, \cond

    .else

        .ifc \cond, eq
            cbz \cmpre_reg, 6f
        .else
            cbnz \cmpre_reg, 6f
        .endif

        mov \tmp3, \tmp2
        mov \tmp4, \tmp1
        b 7f
6:
        mov \tmp3, \tmp1
        mov \tmp4, \tmp2
7:

    .endif


    // If nesting is 0, we are actually not speculating, and therefore push 0 nesting to the "architectualy*
    // Otherwise, increment and store window_counter

    mov     \tmp5, \snapshot_id
    ldr     \tmp2, [\sim_mgmt, #8]	// #SIM_MGMT_STACK_TOP_OFFSET
    ldr     \tmp1, [\sim_mgmt, #0] 	// #SIM_MGMT_CURRENT_NESTING_OFFSET
    cbz     \tmp1, 3f 	

    // Increment and then push pair
			
    add     \tmp1, \tmp1, #1
    str     \tmp1, [\sim_mgmt, #0]	// #SIM_MGMT_CURRENT_NESTING_OFFSET

    // push (nesting_counter, primary, snapshot_id, reserved) onto software RSB (pre-decrement)
    stp     \tmp1, \tmp3, [\tmp2, #-16]!
    stp     \tmp5, xzr, [\tmp2, #-16]!
    str     \tmp2, [\sim_mgmt, #8]	// #SIM_MGMT_STACK_TOP_OFFSET

    b 4f
3:
    // Push pair and then increment

    // push (nesting_counter, primary, snapshot_id, reserved) onto software RSB (pre-decrement)
    stp     \tmp1, \tmp3, [\tmp2, #-16]!
    stp     \tmp5, xzr, [\tmp2, #-16]!
    str     \tmp2, [\sim_mgmt, #8]	// #SIM_MGMT_STACK_TOP_OFFSET

    // increment and store window_counter
    add     \tmp1, \tmp1, #1
    str     \tmp1, [\sim_mgmt, #0]	// #SIM_MGMT_CURRENT_NESTING_OFFSET

4:
    mrs     \tmp1, nzcv
    TAKE_SNAPSHOT \sim_mgmt, \snapshot_id, \sandbox_addr, \sandbox_size, \tmp2, \tmp3, \tmp6, \tmp5
    msr     nzcv, \tmp1

    // jump to opposite target
    br      \tmp4
.endm

.macro SIMULATION_INIT sim_mgmt, snapshot_size, tmp
    .set _init_sz0, ((\snapshot_size)      & 0xFFFF)
    .set _init_sz1, ((\snapshot_size >> 16) & 0xFFFF)
    .set _init_sz2, ((\snapshot_size >> 32) & 0xFFFF)
    .set _init_sz3, ((\snapshot_size >> 48) & 0xFFFF)

    str     xzr, [\sim_mgmt, #0]			// #SIM_MGMT_CURRENT_NESTING_OFFSET
    add     \tmp, \sim_mgmt, #4096			// #SIM_MGMT_SNAPSHOT_ARRAY_DATA_OFFSET
    add     \tmp, \tmp, #32				// #SIM_MGMT_SNAPSHOT_ARRAY_DATA_OFFSET
    str     \tmp, [\sim_mgmt, #16]			// #SIM_MGMT_SNAPSHOTS_ARRAY_OFFSET
    movz    \tmp, #_init_sz0
    movk    \tmp, #_init_sz1, lsl #16
    movk    \tmp, #_init_sz2, lsl #32
    movk    \tmp, #_init_sz3, lsl #48
    str     \tmp, [\sim_mgmt, #24]			// #SIM_MGMT_SNAPSHOT_SIZE_OFFSET
    add     \tmp, \sim_mgmt, #4096			// #SIM_MGMT_STACK_BASE_OFFSET
    str     \tmp, [\sim_mgmt, #8]			// #SIM_MGMT_STACK_TOP_OFFSET
.endm

.macro SIMULATION_SPEC_RET sim_mgmt, sandbox_addr, size, tmp1, tmp2, tmp3, tmp4, tmp5

    ldr     \tmp1, [\sim_mgmt, #0]    	// #SIM_MGMT_CURRENT_NESTING_OFFSET
    cbz     \tmp1, 5f

    _SAFE_SIMULATION_SPEC_RET \sim_mgmt, \sandbox_addr, \size, \tmp1, \tmp2, \tmp3, \tmp4, \tmp5
5:
.endm

