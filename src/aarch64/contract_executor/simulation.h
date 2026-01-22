#ifndef SIMULATION_H
#define SIMULATION_H

#include "main.h"

struct simulation {
	uintptr_t return_address;
	struct simulation_input sim_input;
	struct simulation_code sim_code;
	uint8_t* simulation_memory;
	uint8_t* code_tmp;
};

struct simulation simulation = { 0 };
extern struct simulation simulation;

#endif // SIMULATION_H
