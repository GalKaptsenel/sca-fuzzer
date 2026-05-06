#ifndef EXECUTOR_PAC_H
#define EXECUTOR_PAC_H
#include <linux/types.h>
#include <linux/percpu.h>

void trigger_pauth_fault(void);
void pauth_restore_cpu(void);
int pauth_init_cpu(
		uint64_t ia_hi, uint64_t ia_lo,
		uint64_t ib_hi, uint64_t ib_lo,
		uint64_t da_hi, uint64_t da_lo,
		uint64_t db_hi, uint64_t db_lo,
		uint64_t ga_hi,  uint64_t ga_lo);

void pauth_restore_all_cpus(void);
int pauth_init_all_cpus(
		uint64_t ia_hi, uint64_t ia_lo,
		uint64_t ib_hi, uint64_t ib_lo,
		uint64_t da_hi, uint64_t da_lo,
		uint64_t db_hi, uint64_t db_lo,
		uint64_t ga_hi,  uint64_t ga_lo);

uint64_t pacia(uint64_t ptr, uint64_t mod);
uint64_t pacib(uint64_t ptr, uint64_t mod);
uint64_t pacda(uint64_t ptr, uint64_t mod);
uint64_t pacdb(uint64_t ptr, uint64_t mod);
uint64_t pacga(uint64_t x, uint64_t y);
uint64_t paciza(uint64_t ptr);
uint64_t pacizb(uint64_t ptr);
uint64_t pacdza(uint64_t ptr);
uint64_t pacdzb(uint64_t ptr);
uint64_t autia(uint64_t ptr, uint64_t mod);
uint64_t autib(uint64_t ptr, uint64_t mod);
uint64_t autda(uint64_t ptr, uint64_t mod);
uint64_t autdb(uint64_t ptr, uint64_t mod);
uint64_t autiza(uint64_t ptr);
uint64_t autizb(uint64_t ptr);
uint64_t autdza(uint64_t ptr);
uint64_t autdzb(uint64_t ptr);
uint64_t xpaci(uint64_t ptr);
uint64_t xpacd(uint64_t ptr);


#endif // EXECUTOR_PAC_H
