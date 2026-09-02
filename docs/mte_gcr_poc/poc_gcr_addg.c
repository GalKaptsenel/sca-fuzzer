// PoC: on a KASAN_HW_TAGS kernel the executor inherits GCR_EL1.Exclude (reserved tags 14/15), so
// ADDG's tag arithmetic is NOT plain modular — it skips excluded tags (ChooseNonExcludedTag). The
// Revizor seal retags a pointer with `ADDG xN,xN,#0,#delta` where delta = (correct_tag - ptr_tag) % 16
// (plain modular), and the contract executor models ADDG assuming GCR_EL1.Exclude == 0. So for any
// access whose tag walk crosses tag 14 or 15, the genuine sealed retag lands on a DIFFERENT tag on HW
// than the model computed -> a tag-mismatched architectural access -> EL1 MTE tag-check fault (the
// observed ptrX/mem6 KASAN reports; content-dependent, hence intermittent). The fix
// (src/aarch64/executor/mte.c mte_gcr_clear_exclude) clears Exclude for the run.
//
// This module sweeps every (base tag, delta) pair on the real device and reports, for the kernel's
// own GCR_EL1 vs Exclude=0, exactly which pairs diverge. init returns -EINVAL so it never stays loaded.

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <asm/barrier.h>
#include <asm/sysreg.h>

#ifndef SYS_GCR_EL1
#define SYS_GCR_EL1 sys_reg(3, 0, 1, 0, 6)
#endif
#define GCR_EXCL_MASK 0xFFFFUL

// ADDG x0, x1, #0, #delta — encoding 0x91800020 | (delta << 10). .inst needs a compile-time constant,
// so switch over the 16 deltas. Returns the tag nibble [59:56] of the result.
static u64 addg_tag(u8 base_tag, int delta)
{
	register u64 in asm("x1") = ((u64)base_tag << 56) | 0x0000ff8800000000ULL;
	register u64 out asm("x0") = 0;
#define A(d, word) case d: asm volatile(".inst " #word : "=r"(out) : "r"(in)); break
	switch (delta) {
	A(0, 0x91800020); A(1, 0x91800420); A(2, 0x91800820);  A(3, 0x91800c20);
	A(4, 0x91801020); A(5, 0x91801420); A(6, 0x91801820);  A(7, 0x91801c20);
	A(8, 0x91802020); A(9, 0x91802420); A(10, 0x91802820); A(11, 0x91802c20);
	A(12, 0x91803020); A(13, 0x91803420); A(14, 0x91803820); A(15, 0x91803c20);
	}
#undef A
	return (out >> 56) & 0xF;
}

static int __init poc_init(void)
{
	u64 gcr = read_sysreg_s(SYS_GCR_EL1);
	int base, delta, diverge = 0, example_shown = 0;

	pr_err("=== MTE GCR_EL1.Exclude ADDG-fault PoC ===\n");
	pr_err("kernel-inherited GCR_EL1 = 0x%016llx  (Exclude bits[15:0] = 0x%04llx)\n",
	       gcr, gcr & GCR_EXCL_MASK);
	if ((gcr & GCR_EXCL_MASK) == 0)
		pr_err("NOTE: Exclude is already 0 here (no KASAN reservation) — sweep will show no divergence.\n");

	for (base = 0; base < 16; base++) {
		for (delta = 1; delta < 16; delta++) {
			u8 t_kernel, t_zero;

			t_kernel = addg_tag(base, delta);          // under the kernel's GCR (KASAN Exclude)
			write_sysreg_s(gcr & ~GCR_EXCL_MASK, SYS_GCR_EL1);
			isb();
			t_zero = addg_tag(base, delta);            // under Exclude=0 (seal/contract assumption)
			write_sysreg_s(gcr, SYS_GCR_EL1);
			isb();

			if (t_kernel != t_zero) {
				diverge++;
				if (example_shown < 6) {
					pr_err("  DIVERGE: ADDG(tag=%2d, #%2d)  kernel-GCR -> tag %2u   Exclude=0 -> tag %2u  "
					       "(modular=(%d+%d)%%16=%d)\n",
					       base, delta, t_kernel, t_zero, base, delta, (base + delta) % 16);
					example_shown++;
				}
			}
		}
	}

	pr_err("SWEEP: %d of 240 (base tag, delta) pairs give a DIFFERENT ADDG tag under the kernel's "
	       "GCR_EL1.Exclude=0x%04llx than under Exclude=0.\n", diverge, gcr & GCR_EXCL_MASK);
	pr_err("Each such pair is a genuine sealed access whose retag lands on a tag != the modeled/granule "
	       "tag -> an architectural EL1 MTE tag-check fault. The executor fix clears Exclude for the run.\n");
	pr_err("=== end PoC (module intentionally not loaded) ===\n");
	return -EINVAL;
}

module_init(poc_init);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("PoC: GCR_EL1.Exclude makes ADDG non-modular -> Revizor MTE seal architectural tag fault");
