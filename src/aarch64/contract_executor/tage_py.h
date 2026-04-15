#ifndef SIMULATION_TAGE_PY_H
#define SIMULATION_TAGE_PY_H
int tagebp_init(const char *module_dir, const char *module_name);

int tagebp_predict(uintptr_t address);
void tagebp_update(uintptr_t address, int taken, uintptr_t target);
void tagebp_destroy_instance();


void python_finalize();
int python_init();
#endif // SIMULATION_TAGE_PY_H
