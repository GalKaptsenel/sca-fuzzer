#ifndef SIMULATION_TAGE_PY_H
#define SIMULATION_TAGE_PY_H
#include <stdint.h>
int tagebp_init(const char *module_dir, const char *module_name);

int  tagebp_predict(uintptr_t address);
void tagebp_update(uintptr_t address, int taken, uintptr_t target);  /* retire: update tables (counters) */
void tagebp_advance(uintptr_t address, int taken, uintptr_t target); /* advance speculative history */
void tagebp_checkpoint(void);                                        /* snapshot history (window opens) */
void tagebp_rollback(void);                                          /* restore history (misprediction) */
void tagebp_commit(void);                                            /* drop the latest snapshot */
void tagebp_reset(void);
void tagebp_destroy_instance();


void python_finalize();
int python_init();
#endif // SIMULATION_TAGE_PY_H
