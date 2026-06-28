/*
 * Exhaustive test of the non-faulting PAC auth-verification logic used by
 * pac_sign_plugin's debug_auth_no_fault: for every AUT* variant, a correctly-signed
 * pointer must verify as WOULD-SUCCEED and a forged one as WOULD-FPAC. The check is
 * "re-sign the XPAC'd canonical pointer and compare to the actual pointer", using only
 * REVISOR_PAC_SIGN + REVISOR_PAC_XPAC (never REVISOR_PAC_AUTH) — so it can never FPAC
 * the machine. Requires /dev/executor (the kernel signs with its EL1 keys).
 *
 * Build:  make test_pac_auth_verify   Run: ./test_pac_auth_verify
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>

#include "userapi/executor_pac_api.h"

static int g_fd = -1;
static int g_run = 0;
static int g_fail = 0;

static uint64_t pac_op(unsigned long req_nr, const char *mn, uint64_t ptr, uint64_t ctx)
{
	struct pac_sign_req req = { .ptr = ptr, .ctx = ctx, .result = 0 };
	strncpy(req.mnemonic, mn, sizeof(req.mnemonic) - 1);
	if (ioctl(g_fd, req_nr, &req) < 0) {
		perror("ioctl");
		return 0;
	}
	return req.result;
}

/* Mirror of debug_auth_no_fault: a real AUT* succeeds iff the pointer re-signs to itself. */
static int would_succeed(const char *pac_mn, const char *xpac_mn, uint64_t signed_ptr, uint64_t ctx)
{
	uint64_t canonical = pac_op(REVISOR_PAC_XPAC, xpac_mn, signed_ptr, 0);
	uint64_t resigned  = pac_op(REVISOR_PAC_SIGN, pac_mn, canonical, ctx);
	return resigned == signed_ptr;
}

#define CHECK(cond, ...) do { \
	g_run++; \
	if (!(cond)) { g_fail++; printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); } \
} while (0)

struct variant {
	const char *auth;
	const char *pac;
	const char *xpac;
	int zero_ctx;   /* the *Z* variants ignore the context */
};

static const struct variant VARIANTS[] = {
	{ "autia",  "pacia",  "xpaci", 0 },
	{ "autib",  "pacib",  "xpaci", 0 },
	{ "autda",  "pacda",  "xpacd", 0 },
	{ "autdb",  "pacdb",  "xpacd", 0 },
	{ "autiza", "paciza", "xpaci", 1 },
	{ "autizb", "pacizb", "xpaci", 1 },
	{ "autdza", "pacdza", "xpacd", 1 },
	{ "autdzb", "pacdzb", "xpacd", 1 },
};

int main(void)
{
	g_fd = open("/dev/executor", O_RDWR);
	if (g_fd < 0) {
		perror("open /dev/executor (load the executor module first)");
		return 2;
	}

	/* Corner-case pointers (all < 2^48 so bits[54:48] are the PAC field) and contexts. */
	const uint64_t addrs[] = { 0x0ULL, 0x4ULL, 0x0000c2191000ULL, 0x0000fffffffffff0ULL };
	const uint64_t ctxs[]  = { 0x0ULL, 0xdeadbeefULL, 0xffffffffffffffffULL };
	const uint64_t PAC_FIELD_FLIP = 0x7FULL << 48;   /* bits the PAC occupies; XPAC strips them */

	for (size_t v = 0; v < sizeof(VARIANTS) / sizeof(VARIANTS[0]); v++) {
		const struct variant *V = &VARIANTS[v];
		for (size_t a = 0; a < sizeof(addrs) / sizeof(addrs[0]); a++) {
			for (size_t c = 0; c < sizeof(ctxs) / sizeof(ctxs[0]); c++) {
				uint64_t ctx = V->zero_ctx ? 0 : ctxs[c];
				uint64_t addr = addrs[a];

				uint64_t s = pac_op(REVISOR_PAC_SIGN, V->pac, addr, ctx);

				/* A correctly-signed pointer must verify as WOULD-SUCCEED. */
				CHECK(would_succeed(V->pac, V->xpac, s, ctx),
				      "%s addr=%#018lx ctx=%#018lx: valid sig misread as WOULD-FPAC (s=%#018lx)",
				      V->auth, (unsigned long)addr, (unsigned long)ctx, (unsigned long)s);

				/* A forged sig (PAC field perturbed) must verify as WOULD-FPAC. */
				uint64_t forged = s ^ PAC_FIELD_FLIP;
				CHECK(forged == s || !would_succeed(V->pac, V->xpac, forged, ctx),
				      "%s addr=%#018lx ctx=%#018lx: forged sig misread as WOULD-SUCCEED (f=%#018lx)",
				      V->auth, (unsigned long)addr, (unsigned long)ctx, (unsigned long)forged);

				/* For non-Z variants, a sig made with the wrong context must verify as WOULD-FPAC. */
				if (!V->zero_ctx) {
					uint64_t wrong = pac_op(REVISOR_PAC_SIGN, V->pac, addr, ctx ^ 0xA5A5A5A5ULL);
					CHECK(wrong == s || !would_succeed(V->pac, V->xpac, wrong, ctx),
					      "%s addr=%#018lx ctx=%#018lx: wrong-context sig misread as WOULD-SUCCEED",
					      V->auth, (unsigned long)addr, (unsigned long)ctx);
				}

				if (V->zero_ctx) {
					break;   /* context is irrelevant for the Z variants */
				}
			}
		}
	}

	printf("%d checks, %d failed\n", g_run, g_fail);
	close(g_fd);
	return g_fail ? 1 : 0;
}
