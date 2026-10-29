// SPDX-License-Identifier: GPL-2.0
/*
 * Cross-privilege (EL1 victim -> EL0 attacker) cache side-channel demo.
 *
 * The victim (this module) runs the vulnerable "Gk back-edge" gadget: it loads an anchor line [x1]
 * unconditionally, and -- guarded by a branch that a training loop biases not-taken -- SPECULATIVELY
 * loads a secret-dependent line [x0]. Both live in a region SHARED (mmap) with userspace, so a
 * userspace attacker can Flush+Reload that region to try to recover the secret. The module also
 * enables EL0 access to the PMU so the attacker can measure with hardware counters, not just a timer.
 *
 * Layout: NSYM entries strided by 4160 = 4096+64 -> each entry on its own page (no next-line prefetch
 * bleed) and in a distinct L1D set. Entry i is at offset i*STRIDE. secret -> [x0], ANCHOR -> [x1].
 */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/fs.h>
#include <linux/miscdevice.h>
#include <linux/mm.h>
#include <linux/vmalloc.h>
#include <linux/uaccess.h>
#include <linux/random.h>
#include <linux/preempt.h>

#define DEVNAME  "flvictim"
#define STRIDE   4160u
#define NSYM     64u
#define REGION   (NSYM * STRIDE)
#define ANCHOR   3u                    /* fixed set for the architectural [x1] load */

#define FLV_MAGIC     'F'
#define FLV_RANDOMIZE _IO(FLV_MAGIC, 1)
#define FLV_TRIGGER   _IO(FLV_MAGIC, 2)
#define FLV_GET       _IOR(FLV_MAGIC, 3, unsigned long)
#define FLV_SET       _IOW(FLV_MAGIC, 4, unsigned long)
#define FLV_REGION    _IOR(FLV_MAGIC, 5, unsigned long)
#define FLV_ANCHOR    _IOR(FLV_MAGIC, 6, unsigned long)
#define FLV_ENABLE_PMU _IO(FLV_MAGIC, 7)     /* enable EL0 PMU access on the caller's core */

static char *shmem;                    /* vmalloc'd shared region (kernel VA) */
static unsigned int secret = 42;

/* The vulnerable gadget (no barriers, so speculation is possible), written as inline asm so it runs
 * from the module's executable .text. x29 = region base; x0 = secret*STRIDE; x1 = ANCHOR*STRIDE.
 * The 20-iteration loop trains `cbnz x3` not-taken; on the final pass (x3=1) it should be taken, but
 * the misprediction speculatively runs `ldr x5,[x0]` -- the secret-dependent access. */
static noinline void run_gadget(unsigned long base, unsigned long x0_off, unsigned long x1_off)
{
	asm volatile(
		"stp x29, x30, [sp, #-16]!\n"
		"mov x29, %[base]\n"
		"mov x0, %[x0o]\n"
		"mov x1, %[x1o]\n"
		"add x1, x1, x29\n"
		"add x0, x0, x29\n"
		"mov x2, #20\n"
		"mov x3, #0\n"
		"1:\n"
		"ldr x4, [x1]\n"
		"cbnz x3, 2f\n"
		"ldr x5, [x0]\n"
		"dc civac, x0\n"
		"dc civac, x1\n"
		"sub x2, x2, #1\n"
		"cbnz x2, 1b\n"
		"mov x3, #1\n"
		"b 1b\n"
		"2:\n"
		"ldp x29, x30, [sp], #16\n"
		:
		: [base]"r"(base), [x0o]"r"(x0_off), [x1o]"r"(x1_off)
		: "x0", "x1", "x2", "x3", "x4", "x5", "memory", "cc");
}

/* Enable EL0 access to the PMU on the current core, with counter0 = L1D_CACHE_REFILL (0x03) and the
 * cycle counter running (no /64 divider), so the attacker can read pmevcntr0_el0 / pmccntr_el0. */
static void pmu_enable_el0(void)
{
	u64 v;
	asm volatile("msr pmuserenr_el0, %0\n isb\n" :: "r"((u64)0xf));   /* EN|SW|CR|ER */
	asm volatile("msr pmcntenclr_el0, %0\n isb\n" :: "r"((u64)((1u << 0) | (1u << 1) | (1u << 31))));
	asm volatile("msr pmevtyper0_el0, %0\n isb\n" :: "r"((u64)0x0017)); /* counter0 = L2D_CACHE_REFILL */
	asm volatile("msr pmevtyper1_el0, %0\n isb\n" :: "r"((u64)0x002a)); /* counter1 = L3D_CACHE_REFILL */
	asm volatile("msr pmccfiltr_el0, %0\n isb\n" :: "r"((u64)0));
	asm volatile("msr pmcntenset_el0, %0\n isb\n" :: "r"((u64)((1u << 0) | (1u << 1) | (1u << 31))));
	asm volatile("mrs %0, pmcr_el0" : "=r"(v));
	asm volatile("msr pmcr_el0, %0\n isb\n" :: "r"((v | 0b111) & ~(1ULL << 3)));
}

static int flv_mmap(struct file *f, struct vm_area_struct *vma)
{
	if (vma->vm_end - vma->vm_start > REGION)
		return -EINVAL;
	return remap_vmalloc_range(vma, shmem, 0);
}

static long flv_ioctl(struct file *f, unsigned int cmd, unsigned long arg)
{
	switch (cmd) {
	case FLV_RANDOMIZE:
		secret = get_random_u32() % NSYM;
		return 0;
	case FLV_SET:
		secret = (unsigned int)(arg % NSYM);
		return 0;
	case FLV_GET:
		return put_user((unsigned long)secret, (unsigned long __user *)arg);
	case FLV_REGION:
		return put_user((unsigned long)REGION, (unsigned long __user *)arg);
	case FLV_ANCHOR:
		return put_user((unsigned long)ANCHOR, (unsigned long __user *)arg);
	case FLV_ENABLE_PMU:
		preempt_disable();
		pmu_enable_el0();          /* runs on the caller's (attacker's) core */
		preempt_enable();
		return 0;
	case FLV_TRIGGER:
		preempt_disable();
		/* x0 = SECRET (speculatively loaded when the trained branch mispredicts);
		 * x1 = fixed ANCHOR (architectural, always resident) -- must differ from the secret. */
		run_gadget((unsigned long)shmem,
		           (unsigned long)(secret * STRIDE),
		           (unsigned long)(ANCHOR * STRIDE));
		preempt_enable();
		return 0;
	}
	return -EINVAL;
}

static const struct file_operations flv_fops = {
	.owner          = THIS_MODULE,
	.mmap           = flv_mmap,
	.unlocked_ioctl = flv_ioctl,
};

static struct miscdevice flv_dev = {
	.minor = MISC_DYNAMIC_MINOR,
	.name  = DEVNAME,
	.fops  = &flv_fops,
	.mode  = 0666,
};

static int __init flv_init(void)
{
	shmem = vmalloc_user(REGION);
	if (!shmem)
		return -ENOMEM;
	return misc_register(&flv_dev);
}

static void __exit flv_exit(void)
{
	misc_deregister(&flv_dev);
	vfree(shmem);
}

module_init(flv_init);
module_exit(flv_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("EL1 victim: Gk back-edge gadget with speculative secret access into a shared page");
