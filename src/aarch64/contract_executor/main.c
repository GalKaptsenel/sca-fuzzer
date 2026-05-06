#include "main.h"
#include "simulation.h"
#include "simulation_output.h"
#include "simulation_execution_clause_hook.h"
#include "pac_sign_plugin.h"
#include "tage_py.h"
#include <signal.h>

extern void base_hook();
extern void base_hook_end();
extern uint64_t base_hook_c_target;

simulation_hook_fn hooks_to_install[] = {
	execution_clause_hook,
	pac_sign_hook,              /* must run before logging so trace sees kernel-signed value */
	auth_verify_hook,           /* must run after pac_sign_hook, before logging */
//	stdout_print_hook,
	log_instr_execution_cluase_hook,
//	log_instr_hook,
	handle_ret_hook,
};

/* ---- watchdog ---------------------------------------------------------- */
#define CE_ITERATION_TIMEOUT_SEC 30

static volatile sig_atomic_t g_iter_phase = 0;  /* 0=idle 1=read 2=sim 3=output */
static volatile size_t       g_iter_num   = 0;

static void ce_alarm_handler(int sig) {
	(void)sig;
	const char* phase_names[] = {"idle", "read-input", "simulation", "write-output"};
	const char* phase = (g_iter_phase < 4) ? phase_names[g_iter_phase] : "unknown";
	fprintf(stderr,
		"[CE WATCHDOG] iteration %zu hung in phase '%s' after %d seconds — aborting\n",
		g_iter_num, phase, CE_ITERATION_TIMEOUT_SEC);
	fflush(stderr);
	abort();
}
/* ----------------------------------------------------------------------- */

int main() {
	int ret = 0;

	const size_t base_hook_size = (size_t)(base_hook_end - base_hook);
	base_hook_c_target = (uint64_t)base_hook_c;

	pac_sign_plugin_init();

	/* Install SIGALRM watchdog */
	struct sigaction sa = { 0 };
	sa.sa_handler = ce_alarm_handler;
	sigemptyset(&sa.sa_mask);
	sigaction(SIGALRM, &sa, NULL);

	while(1) {
		++g_iter_num;

		g_iter_phase = 0; /* idle / python_init */
		alarm(CE_ITERATION_TIMEOUT_SEC);

		ret = python_init();
		if(ret < 0) {
			alarm(0);
			fprintf(stderr, "Failed to init python interpreter\n");
			goto main_out;
		}

		g_iter_phase = 1; /* read-input */

		ret = simulation_input_from_file(stdin, &simulation.sim_input);

		if (0 >= ret) {
			alarm(0);
			if(-1 != ret) {
				fprintf(stderr, "Failed to load input from stdin\n");
			} else {
				fprintf(stderr, "EOF reached\n");
			}
			goto main_failure;
		}

		ret = simulation_code_init(&simulation.sim_input, &simulation.sim_code, 4 + base_hook_size);
		if (ret < 0) {
			alarm(0);
			fprintf(stderr, "Failed to allocate simulation code\n");
			goto main_input_free;
		}

		size_t sandbox_size = simulation.sim_input.hdr.mem_size + 0x1000; // Add a single page for overflow
		simulation.simulation_memory = (uint8_t*)malloc(sandbox_size);
		if(NULL == simulation.simulation_memory) {
			alarm(0);
			fprintf(stderr, "was unable to allocate enough memory for sandbox\n");
			ret = -1;
			goto main_code_free;
		}


		if(RVZR_ARCH_AARCH64 == simulation.sim_input.hdr.arch) {
			if(0 != hook_aarch64_instructions(&simulation.sim_input, &simulation.sim_code, base_hook, base_hook_size)) {
				alarm(0);
				fprintf(stderr, "Failed installing hooks\n");
				ret = -1;
				goto main_simulation_memory_free;
			}
		} else {
			alarm(0);
			fprintf(stderr, "ONLY AARCH64 arch currently supported for simulation\n");
			ret = -1;
			goto main_simulation_memory_free;
		}

		memset(simulation.simulation_memory, 0, sandbox_size);
		memcpy(simulation.simulation_memory, simulation.sim_input.memory, simulation.sim_input.hdr.mem_size);

		simulation.n_hooks = 0;
		memset(simulation.hooks, 0, sizeof(simulation.hooks));
		simulation.return_address = 0;

		simulation.n_hooks = install_hooks(&simulation, MAX_HOOKS, hooks_to_install, (sizeof(hooks_to_install)/sizeof(hooks_to_install[0])));

		uint64_t* regs_blob = (uint64_t*)simulation.sim_input.regs;

		void* kernel_sandbox_base = 0;
		if(CONFIG_FLAG_REQ_MEM_BASE_VIRT & simulation.sim_input.hdr.config.flags) {
			kernel_sandbox_base = (void*)simulation.sim_input.hdr.config.requested_mem_base_virt;
		} else {
			alarm(0);
			fprintf(stderr, "[ERR] Expected memory base for the sandbox!\n");
			exit(0);
		}

		g_iter_phase = 2; /* simulation */

		asm volatile (
				"mov x29, %2\n"
				"adr x9, 1f\n"
				"str x9, %0\n"
				"mov x0, %3\n"
				"mov x1, %4\n"
				"mov x2, %5\n"
				"mov x3, %6\n"
				"mov x4, %7\n"
				"mov x5, %8\n"
				"mov x6, %9\n"
				"msr nzcv, x6\n"
				"mov x7, %10\n"
				"mov x8, %11\n"
				"blr %1\n"
				"1:\n"
				: "=m"(simulation.return_address)
				: "r"(simulation.sim_code.code), "r"(kernel_sandbox_base),
				"r"(regs_blob[0]), "r"(regs_blob[1]), "r"(regs_blob[2]), "r"(regs_blob[3]),
				"r"(regs_blob[4]), "r"(regs_blob[5]), "r"(regs_blob[6]), "r"(regs_blob[7]),
				"r"(regs_blob[8])
				: "x9", "x29", "memory", "cc", "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x10", "x30"
			);

		g_iter_phase = 3; /* write-output */

		reset_execution_clause_state(); /* free checkpoint memory + reset TAGE between TCs */
		destroy_trace_log();
		free(simulation.simulation_memory);

		alarm(0); /* cancel watchdog — iteration completed */
	}


main_simulation_memory_free:
	free(simulation.simulation_memory);
main_code_free:
	simulation_code_free(&simulation.sim_code, 4 + base_hook_size);
main_input_free:
	simulation_input_free(&simulation.sim_input);
main_failure:
	python_finalize();
main_out:
	pac_sign_plugin_cleanup();
	return ret;
}
