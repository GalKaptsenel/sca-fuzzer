#ifndef SIMULATION_CALL_STACK_H
#define SIMULATION_CALL_STACK_H

#include <stddef.h>
#include <stdint.h>

/* The architectural return-address stack. A BL/BLR pushes the address of the instruction after it; a
 * RET pops the address to return to. This models what the hardware achieves with X30 plus the memory
 * spills a callee's prologue/epilogue perform, without the contract executor having to track X30
 * through its single-step machinery (which reuses the real LR). An empty stack means "no active call",
 * so a RET then falls back to the simulation's top-level return address.
 *
 * The stack is snapshot into / restored from speculation checkpoints (see take_checkpoint), so a
 * call or return taken on a mispredicted path is undone when that window is squashed. */

void   call_stack_reset(void);
void   call_stack_push(uintptr_t return_addr);
/* Pop the top return address into *out; returns 1 on success, 0 when the stack is empty. */
int    call_stack_pop(uintptr_t *out);

/* Snapshot support for speculation checkpoints. */
size_t call_stack_snapshot_bytes(void);
void   call_stack_snapshot(void *buf);
void   call_stack_restore(const void *buf);

#endif // SIMULATION_CALL_STACK_H
