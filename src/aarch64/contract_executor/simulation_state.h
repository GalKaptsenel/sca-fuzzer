#ifndef SIMULATION_STATE_H
#define SIMULATION_STATE_H

#include "main.h"

struct gprs {
	uintptr_t x29;
	uintptr_t x28;
	uintptr_t x27;
	uintptr_t x26;
	uintptr_t x25;
	uintptr_t x24;
	uintptr_t x23;
	uintptr_t x22;
	uintptr_t x21;
	uintptr_t x20;
	uintptr_t x19;
	uintptr_t x18;
	uintptr_t x17;
	uintptr_t x16;
	uintptr_t x15;
	uintptr_t x14;
	uintptr_t x13;
	uintptr_t x12;
	uintptr_t x11;
	uintptr_t x10;
	uintptr_t x9;
	uintptr_t x8;
	uintptr_t x7;
	uintptr_t x6;
	uintptr_t x5;
	uintptr_t x4;
	uintptr_t x3;
	uintptr_t x2;
	uintptr_t x1;
	uintptr_t x0;
};

_Static_assert(sizeof(struct gprs) == 30 * sizeof(uintptr_t),
               "gprs size mismatch");

/* Explicit order checks (descending register numbers) */
_Static_assert(offsetof(struct gprs, x29) ==  0 * sizeof(uintptr_t), "x29 offset");
_Static_assert(offsetof(struct gprs, x28) ==  1 * sizeof(uintptr_t), "x28 offset");
_Static_assert(offsetof(struct gprs, x27) ==  2 * sizeof(uintptr_t), "x27 offset");
_Static_assert(offsetof(struct gprs, x26) ==  3 * sizeof(uintptr_t), "x26 offset");
_Static_assert(offsetof(struct gprs, x25) ==  4 * sizeof(uintptr_t), "x25 offset");
_Static_assert(offsetof(struct gprs, x24) ==  5 * sizeof(uintptr_t), "x24 offset");
_Static_assert(offsetof(struct gprs, x23) ==  6 * sizeof(uintptr_t), "x23 offset");
_Static_assert(offsetof(struct gprs, x22) ==  7 * sizeof(uintptr_t), "x22 offset");
_Static_assert(offsetof(struct gprs, x21) ==  8 * sizeof(uintptr_t), "x21 offset");
_Static_assert(offsetof(struct gprs, x20) ==  9 * sizeof(uintptr_t), "x20 offset");
_Static_assert(offsetof(struct gprs, x19) == 10 * sizeof(uintptr_t), "x19 offset");
_Static_assert(offsetof(struct gprs, x18) == 11 * sizeof(uintptr_t), "x18 offset");
_Static_assert(offsetof(struct gprs, x17) == 12 * sizeof(uintptr_t), "x17 offset");
_Static_assert(offsetof(struct gprs, x16) == 13 * sizeof(uintptr_t), "x16 offset");
_Static_assert(offsetof(struct gprs, x15) == 14 * sizeof(uintptr_t), "x15 offset");
_Static_assert(offsetof(struct gprs, x14) == 15 * sizeof(uintptr_t), "x14 offset");
_Static_assert(offsetof(struct gprs, x13) == 16 * sizeof(uintptr_t), "x13 offset");
_Static_assert(offsetof(struct gprs, x12) == 17 * sizeof(uintptr_t), "x12 offset");
_Static_assert(offsetof(struct gprs, x11) == 18 * sizeof(uintptr_t), "x11 offset");
_Static_assert(offsetof(struct gprs, x10) == 19 * sizeof(uintptr_t), "x10 offset");
_Static_assert(offsetof(struct gprs, x9)  == 20 * sizeof(uintptr_t), "x9 offset");
_Static_assert(offsetof(struct gprs, x8)  == 21 * sizeof(uintptr_t), "x8 offset");
_Static_assert(offsetof(struct gprs, x7)  == 22 * sizeof(uintptr_t), "x7 offset");
_Static_assert(offsetof(struct gprs, x6)  == 23 * sizeof(uintptr_t), "x6 offset");
_Static_assert(offsetof(struct gprs, x5)  == 24 * sizeof(uintptr_t), "x5 offset");
_Static_assert(offsetof(struct gprs, x4)  == 25 * sizeof(uintptr_t), "x4 offset");
_Static_assert(offsetof(struct gprs, x3)  == 26 * sizeof(uintptr_t), "x3 offset");
_Static_assert(offsetof(struct gprs, x2)  == 27 * sizeof(uintptr_t), "x2 offset");
_Static_assert(offsetof(struct gprs, x1)  == 28 * sizeof(uintptr_t), "x1 offset");
_Static_assert(offsetof(struct gprs, x0)  == 29 * sizeof(uintptr_t), "x0 offset");

struct cpu_state {
	uintptr_t sp;
	uintptr_t nzcv;
	uintptr_t pc;
	uintptr_t lr;
	struct gprs gprs;
};

_Static_assert(sizeof(struct cpu_state) == 34 * sizeof(uintptr_t),
               "cpu_state ABI mismatch");
_Static_assert(offsetof(struct cpu_state, sp)   == 0,   "sp offset wrong");
_Static_assert(offsetof(struct cpu_state, nzcv) == 8,   "nzcv offset wrong");
_Static_assert(offsetof(struct cpu_state, pc)   == 16,  "pc offset wrong");
_Static_assert(offsetof(struct cpu_state, lr)   == 24,  "lr offset wrong");
_Static_assert(offsetof(struct cpu_state, gprs) == 32,  "gprs offset wrong");


struct simulation_state {
	struct cpu_state cpu_state;
	uint8_t* memory;
};

#endif // SIMULATION_STATE_H
