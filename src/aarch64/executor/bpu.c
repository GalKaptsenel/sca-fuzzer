#include "main.h"

/* ============================================================================
 * Module-static state
 * ============================================================================ */

static branch_training_config_t saved_config = { .count = 0 };

/* ============================================================================
 * Core BPU operations (consolidated from measurement.c)
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
     * K=38 iterations: "b 2f" is always taken (38 updates); "b.ne 1b" is taken
     * when x0 ≠ 0 after subs, i.e. 37 times.  Total = 38 + 37 = 75 taken-branch
     * updates = exactly the PHR depth → old history fully flushed. */
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
    /* Write a 3-word stub [ BTI c | CBNZ x0, +4 | RET ] at view[loc].
     * BTI c is required because we call via BLR.
     * The CBNZ at view[loc+1] trains the base-predictor entry for that PC. */
    view[loc]   = INSN_BTI_C;
    view[loc+1] = INSN_CBNZ_X0;
    view[loc+2] = INSN_RET;
    flush_icache_range((unsigned long)&view[loc],
                       (unsigned long)&view[loc + 3]);
    return &view[loc];
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

        /* skip separators and whitespace */
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

void __nocfi apply_branch_training(const branch_training_config_t *cfg) {
    if (!cfg || cfg->count == 0)
        return;

    size_t tc_off = get_tc_insert_offset_words();

    for (int v = 0; v < MAX_MEASUREMENT_VIEWS; ++v) {
        uint32_t *view = (uint32_t *)executor.measurement_code_views[v];

        for (int i = 0; i < cfg->count; ++i) {
            const branch_training_entry_t *e = &cfg->entries[i];
            size_t k = tc_off + e->byte_offset / sizeof(uint32_t);

            if (k < 1) {
                module_err("bpu: entry[%d] offset %u → k=%zu < 1, skipped\n",
                           i, e->byte_offset, k);
                continue;
            }

            /* Save the 3 words we will temporarily overwrite */
            uint32_t saved[3] = { view[k-1], view[k], view[k+1] };

            /* Write the training stub so the branch under training is at
             * exactly the right PC (view[k]) — the same address the test
             * case will use — then execute it 16 times to saturate the
             * 4-bit base-predictor counter in the requested direction. */
            view[k-1] = INSN_BTI_C;
            view[k]   = e->train_taken ? INSN_CBNZ_X0 : INSN_CBZ_X0;
            view[k+1] = INSN_RET;
            flush_icache_range((unsigned long)&view[k-1],
                               (unsigned long)&view[k+2]);

            void *entry = (void *)&view[k-1];
            for (int t = 0; t < 16; ++t)
                ((void (*)(void *))entry)(&executor.sandbox);

            /* Restore original instructions */
            view[k-1] = saved[0];
            view[k]   = saved[1];
            view[k+1] = saved[2];
            flush_icache_range((unsigned long)&view[k-1],
                               (unsigned long)&view[k+2]);
        }
    }
}

void __nocfi set_branch_training_config(const char *buf, size_t len) {
    parse_branch_training_config(buf, len, &saved_config);
    //module_info("bpu: branch training configured with %d entries\n", saved_config.count);
}

void __nocfi reapply_branch_training(void) {
    apply_branch_training(&saved_config);
}
