#ifndef SIMULATION_TAGE_PY_H
#define SIMULATION_TAGE_PY_H
int tagebp_init(const char *module_dir, const char *module_name,
		int num_state_bits, int init_state_val, int num_base_entries);

int tagebp_predict(uintptr_t address);
void tagebp_update(uintptr_t address, int taken);
void tagebp_finalize();

#endif // SIMULATION_TAGE_PY_H
