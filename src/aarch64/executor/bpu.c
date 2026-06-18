#include "main.h"

static branch_training_config_t saved_config = { .count = 0 };

void *invalidate_bpu_entries(void) {
    static size_t current_view = 0;
    size_t idx = current_view % MAX_MEASUREMENT_VIEWS;
    ++current_view;
    return executor.measurement_code_views[idx];
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

static int parse_branch_training_config(const char *buf, size_t len,
                                 branch_training_config_t *out) {
    out->count = 0;
    const char *p   = buf;
    const char *end = buf + len;

    while (p < end && out->count < MAX_BRANCH_TRAINING_ENTRIES) {
        char *next;

        while (p < end && (',' == *p || ' ' == *p || '\n' == *p || '\t' == *p)) {
            ++p;
        }
        if (p >= end || '\0' == *p) {
            break;
        }

        unsigned long off = simple_strtoul(p, &next, 10);
        if (next == p) {
            break;
        }
        p = next;

        while (p < end && ' ' == *p) {
            ++p;
        }
        if (p >= end || ':' != *p) {
            break;
        }
        ++p;

        while (p < end && ' ' == *p) {
            ++p;
        }
        unsigned long dir = simple_strtoul(p, &next, 10);
        if (next == p) {
            break;
        }
        p = next;

        out->entries[out->count].byte_offset = (uint32_t)off;
        out->entries[out->count].train_taken  = (0 != dir) ? 1 : 0;
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
 * Local-only ICache flush.  During measurement we are pinned to a single CPU
 * with IRQs disabled, so we skip flush_icache_range's SMP-wide kick_all_cpus_sync.
 */
static void local_icache_flush(unsigned long start, unsigned long end)
{
    unsigned long addr;
    for (addr = start; addr < end; addr += 64) {
        asm volatile("dc cvau, %0" : : "r"(addr) : "memory");
    }
    asm volatile("dsb ish" : : : "memory");
    for (addr = start; addr < end; addr += 64) {
        asm volatile("ic ivau, %0" : : "r"(addr) : "memory");
    }
    asm volatile("dsb ish\n isb" : : : "memory");
}

/* Decode the signed word offset of a conditional branch (CBZ/CBNZ/B.cond imm19,
 * TBZ/TBNZ imm14).  Returns true and sets *doff when recognized. */
static bool branch_word_offset(uint32_t insn, int64_t *doff)
{
    bool is_cbz_cbnz = (0x34000000u == (insn & 0x7E000000u));
    bool is_bcond    = (0x54000000u == (insn & 0xFF000010u));
    bool is_tbz_tbnz = (0x36000000u == (insn & 0x7E000000u));
    if (!is_cbz_cbnz && !is_bcond && !is_tbz_tbnz) {
        return false;
    }

    if (is_tbz_tbnz) {                /* TBZ/TBNZ use a 14-bit imm at bits [18:5] */
        int32_t imm14 = (int32_t)((insn >> 5) & 0x3FFFu);
        if (imm14 & 0x2000) {         /* sign-extend bit 13 */
            imm14 |= ~0x3FFF;
        }
        *doff = imm14;
        return true;
    }

    int32_t imm19 = (int32_t)((insn >> 5) & 0x7FFFFu);  /* CBZ/CBNZ/B.cond: 19-bit imm */
    if (imm19 & 0x40000) {            /* sign-extend bit 18 */
        imm19 |= ~0x7FFFF;
    }
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

/* Invalidate icache for [k0,k1) words at EVERY view VA.  The views alias the
 * same physical pages, so a write via one VA must be made visible at all the
 * other view VAs the predictor will fetch from. */
static void flush_all_views_at(size_t k0, size_t k1)
{
    for (int v = 0; v < MAX_MEASUREMENT_VIEWS; ++v) {
        uint32_t *view = (uint32_t *)executor.measurement_code_views[v];
        if (NULL == view) {
            continue;
        }
        local_icache_flush((unsigned long)&view[k0], (unsigned long)&view[k1]);
    }
}

/* Train each configured branch toward its direction.  The views alias the same
 * physical pages, so the direction-forcing stub (cbz/cbnz x0, x0=&sandbox!=0) is
 * written once into the shared page (via the active view) and is then trained by
 * executing it at every view VA, saturating the shared base-PHT entry.  The real
 * branch is restored before the caller measures. */
static void __nocfi apply_branch_training(void *active_view,
                                   const branch_training_config_t *cfg)
{
    if (NULL == cfg || 0 == cfg->count || NULL == active_view) {
        return;
    }

    size_t tc_off = current_tc_insert_offset_bytes() / sizeof(uint32_t);
    const size_t view_words = MAX_MEASUREMENT_CODE_SIZE / sizeof(uint32_t);
    uint32_t *view = (uint32_t *)active_view;   /* the shared physical page (stub write/restore) */

    for (int i = 0; i < cfg->count; ++i) {
        const branch_training_entry_t *e = &cfg->entries[i];
        size_t k = tc_off + e->byte_offset / sizeof(uint32_t);

        /* We write view[k-1], view[k], view[k+1], so k must be in [1, view_words-2]. */
        if (1 > k || k + 1 >= view_words) {
            module_err("bpu: entry[%d] offset %u → k=%zu out of range [1, %zu), skipped\n",
                       i, e->byte_offset, k, view_words - 1);
            continue;
        }

        uint32_t saved[3] = { view[k-1], view[k], view[k+1] };
        int64_t doff = 1;                       /* fallback: +4 (next word) */
        bool matched = branch_word_offset(saved[1], &doff);

        /* An exact-offset TAKEN stub jumps to k+doff and needs a RET planted there; if that
         * target is backward/out-of-range/aliasing, use the +4 direction stub (lands on the
         * in-stub RET at k+1). Base-PHT is PC-indexed, so +4 trains the same entry. NOT-taken
         * stubs fall through to k+1, so the exact form is always safe for them. */
        int64_t tgt_s = (int64_t)k + doff;
        bool target_safe = matched && 1 != doff &&
                           0 <= tgt_s && (size_t)tgt_s < view_words &&
                           (size_t)tgt_s != k - 1 && (size_t)tgt_s != k && (size_t)tgt_s != k + 1;
        bool use_exact = matched && (!e->train_taken || target_safe);

        /* write the stub (aliased -> visible at every view, flush all) */
        view[k-1] = INSN_BTI_C;
        view[k]   = use_exact ? make_x0_branch(e->train_taken, doff)
                              : (e->train_taken ? INSN_CBNZ_X0 : INSN_CBZ_X0);
        view[k+1] = INSN_RET;
        flush_all_views_at(k-1, k+2);

        /* exact TAKEN stub jumps to k+doff; drop a temp RET there so it returns */
        bool tgt_placed = false;
        size_t tgt = 0;
        uint32_t saved_tgt = 0;
        if (use_exact && e->train_taken) {
            tgt = (size_t)tgt_s;
            saved_tgt = view[tgt];
            view[tgt] = INSN_RET;
            flush_all_views_at(tgt, tgt+1);
            tgt_placed = true;
        }

        /* Train the branch across ALL views to saturate the shared base-PHT
         * entry: the views differ only in upper VA bits, so they map to a common
         * base-predictor index.  The stub is in the shared physical page (written
         * via the active view above), so it is present at every view VA. */
        for (int vw = 0; vw < MAX_MEASUREMENT_VIEWS; ++vw) {
            uint32_t *vv = (uint32_t *)executor.measurement_code_views[vw];
            if (NULL == vv) {
                continue;
            }
            flush_bpu_phr();
            ((void (*)(void *))&vv[k-1])(executor.sandbox);
            asm volatile("isb" ::: "memory");
        }
        asm volatile("dsb sy\n isb\n" ::: "memory");

        /* restore the real branch (and target) */
        if (tgt_placed) {
            view[tgt] = saved_tgt;
            flush_all_views_at(tgt, tgt+1);
        }
        view[k-1] = saved[0];
        view[k]   = saved[1];
        view[k+1] = saved[2];
        flush_all_views_at(k-1, k+2);
    }
}

void __nocfi set_branch_training_config(const char *buf, size_t len) {
    parse_branch_training_config(buf, len, &saved_config);
}

void __nocfi reapply_branch_training(void *active_view) {
    apply_branch_training(active_view, &saved_config);
}
