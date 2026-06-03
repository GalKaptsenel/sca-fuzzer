#include "main.h"

/* ============================================================================
 * Module-static state
 * ============================================================================ */

static branch_training_config_t saved_config = { .count = 0 };

/* xorshift64 PRNG — IRQ-safe, no allocation, no kernel RNG dependency. */
static uint64_t bpu_prng_state = 0xdeadbeefcafebabe;

static inline uint64_t bpu_rand(void)
{
    bpu_prng_state ^= bpu_prng_state << 13;
    bpu_prng_state ^= bpu_prng_state >> 7;
    bpu_prng_state ^= bpu_prng_state << 17;
    return bpu_prng_state;
}

/* ============================================================================
 * Core BPU operations
 * ============================================================================ */

void *invalidate_bpu_entries(void) {
    static size_t current_view = 0;
    return executor.measurement_code_views[current_view++ % MAX_MEASUREMENT_VIEWS];
}

void flush_bpu_phr(void) {
    /* Official ARM Spectre-BHB mitigation sequence (see ARM-EPM-048486 v4).
     * PHR on Neoverse N3: 75 records × 4 bits/record = 300 bits (shift register).
     * Each taken branch shifts in a new 4-bit footprint; 75 taken branches fully
     * overwrite all 300 bits of old history.
     *
     * K=38 iterations: "b 2f" always taken (38 updates); "b.ne 1b" taken when
     * x0 ≠ 0 after subs (37 times).  Total = 38+37 = 75 taken-branch updates. */
    asm volatile (
        "mov x0, #38\n"
        "1:\n"
        "b 2f\n"
        "2:\n"
        "subs x0, x0, #1\n"
        "b.ne 1b\n"
        "dsb nsh\n"
        "isb\n"
        :
        :
        : "x0", "cc"
    );
}

void *load_training_entry_at(uint32_t *view, size_t loc) {
    view[loc]   = INSN_BTI_C;
    view[loc+1] = INSN_CBNZ_X0;
    view[loc+2] = INSN_RET;
    flush_icache_range((unsigned long)&view[loc],
                       (unsigned long)&view[loc + 3]);
    return &view[loc];
}

/* ============================================================================
 * PHR perturbation
 * ============================================================================
 *
 * Executes 1–16 unconditional taken branches (count from xorshift64 PRNG).
 * Each taken branch shifts 4 new bits into the 300-bit PHR, so each call
 * advances the PHR state by 4–64 bits.  Called between training iterations
 * so each iteration maps to a different TAGE table entry, isolating the
 * training effect to the base PHT.
 */
int debug_phr_mode = 0;     /* DEBUG: 0=flush, 1=random, 2=none (see bpu.h) */

noinline void __nocfi set_phr_random(void)
{
    uint64_t r     = bpu_rand();
    uint32_t n     = (uint32_t)(r & 0xF) + 1; /* 1–16 branches */
    uint32_t dists = (uint32_t)(r >> 4);       /* 2 bits per branch → distance */

    for (uint32_t i = 0; i < n; i++) {
        /* Distance bits select target offset: +4 / +8 / +12 / +16 bytes.
         * Different t bits → different PHR footprint per iteration. */
        switch ((dists >> (i * 2)) & 0x3) {
        case 0: asm volatile("b 1f\n\t1:\n\t"                      : : : ); break;
        case 1: asm volatile("b 1f\n\tnop\n\t1:\n\t"               : : : ); break;
        case 2: asm volatile("b 1f\n\tnop\n\tnop\n\t1:\n\t"        : : : ); break;
        default:asm volatile("b 1f\n\tnop\n\tnop\n\tnop\n\t1:\n\t" : : : ); break;
        }
    }
}

/* ============================================================================
 * Branch training config — parse / format / apply
 * ============================================================================ */

int parse_branch_training_config(const char *buf, size_t len,
                                 branch_training_config_t *out) {
    out->count = 0;
    const char *p   = buf;
    const char *end = buf + len;

    while (p < end && out->count < MAX_BRANCH_TRAINING_ENTRIES) {
        char *next;

        while (p < end && (*p == ',' || *p == ' ' || *p == '\n' || *p == '\t'))
            ++p;
        if (p >= end || *p == '\0')
            break;

        unsigned long off = simple_strtoul(p, &next, 10);
        if (next == p)
            break;
        p = next;

        while (p < end && *p == ' ') ++p;
        if (p >= end || *p != ':')
            break;
        ++p;

        while (p < end && *p == ' ') ++p;
        unsigned long dir = simple_strtoul(p, &next, 10);
        if (next == p)
            break;
        p = next;

        out->entries[out->count].byte_offset = (uint32_t)off;
        out->entries[out->count].train_taken  = (dir != 0) ? 1 : 0;
        ++out->count;
    }

    return out->count;
}

int format_branch_training_config(char *buf, size_t size) {
    int len = 0;
    for (int i = 0; i < saved_config.count; ++i) {
        len += scnprintf(buf + len, size - len, "%s%u:%u",
                         i ? "," : "",
                         saved_config.entries[i].byte_offset,
                         saved_config.entries[i].train_taken);
    }
    len += scnprintf(buf + len, size - len, "\n");
    return len;
}

/*
 * Local-only ICache flush.  We are pinned to CPU 0 and have IRQs disabled,
 * so we skip flush_icache_range's SMP-wide kick_all_cpus_sync.
 */
static void local_icache_flush(unsigned long start, unsigned long end)
{
    unsigned long addr;
    for (addr = start; addr < end; addr += 64)
        asm volatile("dc cvau, %0" : : "r"(addr) : "memory");
    asm volatile("dsb ish" : : : "memory");
    for (addr = start; addr < end; addr += 64)
        asm volatile("ic ivau, %0" : : "r"(addr) : "memory");
    asm volatile("dsb ish\n isb" : : : "memory");
}

/*
 * Train the branch at byte_offset within active_view to saturate the base PHT.
 *
 * Only active_view is trained — not all views.  Each view starts at a different
 * sub-page offset (VIEW_STRIDE = MAX_MEASUREMENT_CODE_SIZE + 8 = 0x4008, so
 * view j starts at page offset j×8 bytes).  The BTB is indexed by virtual
 * address, so training a different view trains a different BTB entry that the
 * upcoming test-case execution will never look up.
 *
 * Between each of the 16 training repetitions, set_phr_random() advances the
 * PHR by a PRNG-driven number of taken branches (1–16).  Each rep therefore
 * maps to a different TAGE table entry, preventing TAGE from accumulating a
 * consistent prediction that would override the base predictor.
 */
/* Decode the signed word offset of an imm19-form conditional branch
 * (CBZ/CBNZ/B.cond).  Returns true and sets *doff when recognized. */
static bool branch_word_offset(uint32_t insn, int64_t *doff)
{
    bool is_cbz_cbnz = (insn & 0x7E000000u) == 0x34000000u;
    bool is_bcond    = (insn & 0xFF000010u) == 0x54000000u;
    if (!is_cbz_cbnz && !is_bcond)
        return false;

    int32_t imm19 = (int32_t)((insn >> 5) & 0x7FFFFu);
    if (imm19 & 0x40000)            /* sign-extend bit 18 */
        imm19 |= ~0x7FFFF;
    *doff = imm19;
    return true;
}

/* Build a CBNZ/CBZ x0 with the given imm19 word offset, reproducing the real
 * branch's target while x0 (=&sandbox≠0) drives the direction. */
static uint32_t make_x0_branch(bool taken, int64_t doff)
{
    uint32_t imm19 = (uint32_t)(int32_t)doff & 0x7FFFFu;
    uint32_t base  = taken ? 0xB5000000u   /* cbnz x0, <off> */
                           : 0xB4000000u;  /* cbz  x0, <off> */
    return base | (imm19 << 5);            /* Rt = x0 = 0 */
}

void __nocfi apply_branch_training(void *active_view,
                                   const branch_training_config_t *cfg)
{
    if (!cfg || cfg->count == 0 || !active_view)
        return;

    size_t tc_off = get_tc_insert_offset_words();
    const size_t view_words = MAX_MEASUREMENT_CODE_SIZE / sizeof(uint32_t);
    uint32_t *view = (uint32_t *)active_view;

    for (int i = 0; i < cfg->count; ++i) {
        const branch_training_entry_t *e = &cfg->entries[i];
        size_t k = tc_off + e->byte_offset / sizeof(uint32_t);

        if (k < 1) {
            module_err("bpu: entry[%d] offset %u → k=%zu < 1, skipped\n",
                       i, e->byte_offset, k);
            continue;
        }

        uint32_t saved[3] = { view[k-1], view[k], view[k+1] };

        /* Reproduce the real branch's target so the BTB is poisoned with the
         * CORRECT target.  A fixed +4 stub gives the real (differently-targeted)
         * branch a target-misprediction instead of the intended direction
         * mistraining. */
        int64_t doff = 1;                       /* fallback: +4 (next word) */
        bool matched = branch_word_offset(saved[1], &doff);

        view[k-1] = INSN_BTI_C;
        view[k]   = matched ? make_x0_branch(e->train_taken, doff)
                            : (e->train_taken ? INSN_CBNZ_X0 : INSN_CBZ_X0);
        view[k+1] = INSN_RET;                   /* not-taken fall-through */
        local_icache_flush((unsigned long)&view[k-1],
                           (unsigned long)&view[k+2]);

        /* A TAKEN stub jumps to view[k+doff] (the real target).  Drop a
         * temporary RET there so it returns instead of running real TC code.
         * doff==1 already lands on the k+1 RET above. */
        int64_t tgt_s = (int64_t)k + doff;
        bool tgt_placed = false;
        size_t tgt = 0;
        uint32_t saved_tgt = 0;
        if (e->train_taken && matched && doff != 1 &&
            tgt_s >= 0 && (size_t)tgt_s < view_words &&
            (size_t)tgt_s != k - 1 && (size_t)tgt_s != k && (size_t)tgt_s != k + 1) {
            tgt = (size_t)tgt_s;
            saved_tgt = view[tgt];
            view[tgt] = INSN_RET;
            local_icache_flush((unsigned long)&view[tgt],
                               (unsigned long)&view[tgt+1]);
            tgt_placed = true;
        }

        void *entry = (void *)&view[k-1];
        for (int t = 0; t < 16; ++t) {
            if (debug_phr_mode != 2) /* phr_mode==2: train with natural PHR */
                set_phr_random();
            ((void (*)(void *))entry)(&executor.sandbox);
        }

        if (tgt_placed) {
            view[tgt] = saved_tgt;
            local_icache_flush((unsigned long)&view[tgt],
                               (unsigned long)&view[tgt+1]);
        }
        view[k-1] = saved[0];
        view[k]   = saved[1];
        view[k+1] = saved[2];
        local_icache_flush((unsigned long)&view[k-1],
                           (unsigned long)&view[k+2]);
    }
}

void __nocfi set_branch_training_config(const char *buf, size_t len) {
    parse_branch_training_config(buf, len, &saved_config);
}

void __nocfi reapply_branch_training(void *active_view) {
    apply_branch_training(active_view, &saved_config);
}
