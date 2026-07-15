/* Unit test for the engine's N-way sibling-window exploration (spec_request_window +
 * resolve_pending_windows + handle_window_end). No production clause requests >1 distinct window yet
 * (that is the indirect-branch-predictor case), so a synthetic clause drives it here: it requests a set
 * of windows sharing one architectural continuation, and we check the engine explores each DISTINCT
 * one off a single checkpoint (in request order, after dedup) and then resumes. Links the real engine
 * with minimal stubs for its externals. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <unistd.h>
#include <sys/wait.h>

#include "simulation_state.h"
#include "simulation_input.h"
#include "simulation.h"
#include "execution_clauses.h"
#include "simulation_execution_clause_hook.h"

static int g_run = 0, g_fail = 0;
#define EXPECT_EQ(a, b) do {                                                        \
    ++g_run; unsigned long _a = (unsigned long)(a), _b = (unsigned long)(b);        \
    if (_a != _b) { ++g_fail;                                                       \
        fprintf(stderr, "FAIL %s:%d: %s (0x%lx) != %s (0x%lx)\n",                   \
                __func__, __LINE__, #a, _a, #b, _b); }                              \
} while (0)

/* The marker where the clause forks, the shared continuation, and the out-of-simulation threshold
 * (a path "ends" once pc reaches it). Sibling entries are 0x2000, 0x2100, ... (all below PC_END). */
enum { PC_MARK = 0x1000, PC_R = 0x4000, PC_END = 0x9000, ENTRY_BASE = 0x2000, ENTRY_STEP = 0x100 };

/* ---- stubs of the engine's externals ---- */
struct simulation simulation;

bool  out_of_simulation(struct cpu_state* s) { return s->pc >= PC_END; }
void* log_instr_with_speculation_nesting(struct simulation_state* s, uint64_t n, uint64_t w) {
    (void)s; (void)n; (void)w; return NULL;
}
size_t mte_tagmem_bytes(void)              { return 0; }
void   mte_tagmem_snapshot(uint8_t* d)     { (void)d; }
void   mte_tagmem_restore(const uint8_t* s){ (void)s; }
int    execution_clauses_supported(uint64_t c) { (void)c; return 1; }

/* Synthetic clause: at the marker, request g_entries[] (each sharing continuation PC_R). */
static uint64_t  tc_index;
static uintptr_t g_entries[MAX_PENDING_WINDOWS + 1];
static int       g_entry_count;
static void tc_on_init(uint64_t i) { tc_index = i; }
static void tc_on_instruction(struct simulation_state* s) {
    if (PC_MARK != s->cpu_state.pc) return;
    for (int i = 0; i < g_entry_count; ++i) spec_request_window(g_entries[i], PC_R, tc_index);
}
static const struct execution_clause_descriptor tc_clause = {
    .name = "testmulti", .clause_bit = (1u << 0), .on_init = tc_on_init,
    .on_reset = NULL, .on_instruction = tc_on_instruction, .on_barrier = NULL, .on_rollback = NULL,
};
int execution_clause_count(void) { return 1; }
const struct execution_clause_descriptor* execution_clause_at(int i) { (void)i; return &tc_clause; }

static struct simulation_state* fresh_state(uint8_t** mem_out) {
    memset(&simulation, 0, sizeof(simulation));
    simulation.sim_input.hdr.config.max_misspred_branch_nesting = 4;
    simulation.sim_input.hdr.config.max_misspred_instructions   = 0;   /* no per-window cap */
    simulation.sim_input.hdr.config.execution_clauses           = (1u << 0);
    simulation.sim_input.mem_size                               = 0x100;
    reset_execution_clause_state();

    uint8_t* mem = calloc(1, simulation.sim_input.mem_size + 0x1000);
    static struct simulation_state st;
    memset(&st, 0, sizeof(st));
    st.memory = mem;
    *mem_out = mem;
    return &st;
}

/* Request `entries` at the marker; assert the engine explores `expected` (the deduped set, in request
 * order) one per path-end, then resumes PC_R — the whole set at one nesting level. */
static void run_scenario(uintptr_t* entries, int n_entries, uintptr_t* expected, int n_expected) {
    memcpy(g_entries, entries, (size_t)n_entries * sizeof(uintptr_t));
    g_entry_count = n_entries;
    uint8_t* mem;
    struct simulation_state* st = fresh_state(&mem);

    st->cpu_state.pc = PC_MARK;
    EXPECT_EQ((uintptr_t)execution_clause_hook(st), expected[0]);   /* dive the first sibling */
    EXPECT_EQ(spec_nesting(), 1);

    for (int i = 1; i < n_expected; ++i) {
        st->cpu_state.pc = PC_END;
        EXPECT_EQ((uintptr_t)execution_clause_hook(st), expected[i]);   /* rewind + dive the next */
        EXPECT_EQ(spec_nesting(), 1);                                   /* still one window */
    }

    st->cpu_state.pc = PC_END;
    EXPECT_EQ((uintptr_t)execution_clause_hook(st), PC_R);   /* exhausted -> resume the continuation */
    EXPECT_EQ(spec_nesting(), 0);

    reset_execution_clause_state();
    free(mem);
}

static void test_coalesce_same_window(void) {
    uintptr_t in[]  = { 0x2000, 0x2000 };   /* same (entry,return) twice */
    uintptr_t exp[] = { 0x2000 };           /* -> one window */
    run_scenario(in, 2, exp, 1);
}
static void test_two_distinct_windows(void) {
    uintptr_t in[]  = { 0x2000, 0x2100 };
    uintptr_t exp[] = { 0x2000, 0x2100 };
    run_scenario(in, 2, exp, 2);
}
static void test_three_distinct_windows(void) {
    uintptr_t in[]  = { 0x2000, 0x2100, 0x2200 };
    uintptr_t exp[] = { 0x2000, 0x2100, 0x2200 };
    run_scenario(in, 3, exp, 3);
}
static void test_dedup_among_distinct(void) {
    uintptr_t in[]  = { 0x2000, 0x2100, 0x2000, 0x2200 };   /* 0x2000 repeats */
    uintptr_t exp[] = { 0x2000, 0x2100, 0x2200 };
    run_scenario(in, 4, exp, 3);
}
static void test_max_windows(void) {
    uintptr_t in[MAX_PENDING_WINDOWS];
    for (int i = 0; i < MAX_PENDING_WINDOWS; ++i) in[i] = ENTRY_BASE + (uintptr_t)i * ENTRY_STEP;
    run_scenario(in, MAX_PENDING_WINDOWS, in, MAX_PENDING_WINDOWS);   /* all distinct -> all explored */
}
static void test_overflow_traps(void) {
    /* One past the capacity must trap. It aborts the process, so drive it in a child. */
    pid_t pid = fork();
    if (0 == pid) {
        for (int i = 0; i <= MAX_PENDING_WINDOWS; ++i)   /* MAX_PENDING_WINDOWS + 1 distinct requests */
            spec_request_window(0x50000 + (uintptr_t)i * 0x100, PC_R, 0);
        _exit(0);   /* reached only if the overflow was NOT caught */
    }
    int status = 0;
    waitpid(pid, &status, 0);
    ++g_run;
    if (!WIFSIGNALED(status)) {
        ++g_fail;
        fprintf(stderr, "FAIL %s: expected a trap signal on overflow, child exited normally\n", __func__);
    }
}

int main(void) {
    printf("Running CE sibling-window tests...\n");
    test_coalesce_same_window();
    test_two_distinct_windows();
    test_three_distinct_windows();
    test_dedup_among_distinct();
    test_max_windows();
    test_overflow_traps();
    printf("\n%d tests, %d failed\n", g_run, g_fail);
    return g_fail ? 1 : 0;
}
