#ifndef PAC_SIGN_PLUGIN_H
#define PAC_SIGN_PLUGIN_H

#include "simulation_hook.h"

/*
 * Must be called once at process startup (before any IPC requests are served)
 * to open the persistent /dev/executor file descriptor used for signing ioctls.
 */
void pac_sign_plugin_init(void);

/*
 * Seed the keys every kernel sign/auth runs under from the current input's PAC_KEYS section
 * (`present` false when the input carries none). The kernel keeps no key state of its own.
 */
void pac_keys_init(const uint64_t* keys, bool present);

/*
 * Must be called at process shutdown to close the device fd.
 */
void pac_sign_plugin_cleanup(void);

/*
 * Simulation hook: intercepts PAC signing instructions and replaces the result
 * with the value produced by the kernel executor via a single ioctl().
 * Install this BEFORE log_instr_execution_cluase_hook so the trace records
 * the kernel-signed value.
 */
void *pac_sign_hook(struct simulation_state *sim_state);

/*
 * Simulation hook: intercepts PAC auth instructions (AUTIA/AUTIB/AUTDA/AUTDB
 * and their zero-context variants), proxies authentication through the kernel
 * executor at EL1 (matching the EL1 keys used during signing), writes the clean
 * address into Xd, and patches the instruction to NOP.
 * Install immediately after pac_sign_hook.
 */
void *auth_verify_hook(struct simulation_state *sim_state);

void *xpac_hook(struct simulation_state *sim_state);

#endif /* PAC_SIGN_PLUGIN_H */
