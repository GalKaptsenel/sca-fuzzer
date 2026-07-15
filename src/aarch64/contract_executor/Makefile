CC = gcc

# Base flags
CFLAGS = -Wall -Wextra -O0 -g -march=native -I../executor
LDFLAGS =

# Add Python3 embedding flags
PYTHON_CFLAGS = $(shell python3-config --embed --cflags)
PYTHON_LDFLAGS = $(shell python3-config --embed --ldflags)

# All headers — sources are compiled in a single shot (no .o/.d files to track),
# so any header change must force a full rebuild of every target.
HDRS = $(wildcard *.h)

# Source files
SRCS = \
  base_hook.S \
  ../executor/input_format.c \
  instruction_encodings.c \
  simulation_code.c \
  simulation_hook.c \
  simulation_input.c \
  simulation_output.c \
  simulation.c \
  simulation_execution_clause_hook.c \
  branch_speculation.c \
  execution_clauses.c \
  execution_clause_cond.c \
  execution_clause_bpu.c \
  execution_clause_bpas.c \
  execution_clause_sls.c \
  branch_predictors.c \
  neoverse_n3_bpu.c \
  utils.c \
  pac_sign_plugin.c \
  mte_tag_plugin.c \
  tage_py.c \
  main.c

.PHONY: all clean test test-ci
all: contract_executor

# Build + run every C test suite. The binaries resolve paths from their own location, so this works
# from any cwd. test_ce_integration forks ./contract_executor, so that target is built too; this
# requires an AArch64 host (contract_executor embeds base_hook.S + python-embed).
test: test_ce test_ce_integration test_mte_tagmem test_sibling_windows contract_executor
	./test_ce
	./test_ce_integration
	./test_mte_tagmem
	./test_sibling_windows

# Host-portable subset for CI on a non-AArch64 runner: only the self-contained suites (test_ce,
# test_mte_tagmem and test_sibling_windows build and run anywhere; test_ce_integration needs the
# AArch64 contract_executor).
test-ci: test_ce test_mte_tagmem test_sibling_windows
	./test_ce
	./test_mte_tagmem
	./test_sibling_windows

# Target
contract_executor: $(SRCS) $(HDRS)
	$(CC) $(CFLAGS) $(PYTHON_CFLAGS) $(SRCS) -o $@ $(PYTHON_LDFLAGS)

TEST_SRCS = ../executor/input_format.c instruction_encodings.c simulation_input.c branch_speculation.c utils.c test_ce.c
test_ce: $(TEST_SRCS) $(HDRS)
	$(CC) $(CFLAGS) -I. $(TEST_SRCS) -o $@

INT_TEST_SRCS = ../executor/input_format.c instruction_encodings.c simulation_input.c test_ce_integration.c
test_ce_integration: $(INT_TEST_SRCS) $(HDRS)
	$(CC) $(CFLAGS) -I. $(INT_TEST_SRCS) -o $@

# MTE tag-memory + instruction emulation unit tests (pure software, no device).
MTE_TEST_SRCS = mte_tag_plugin.c test_mte_tagmem.c
test_mte_tagmem: $(MTE_TEST_SRCS) $(HDRS)
	$(CC) $(CFLAGS) -I. $(MTE_TEST_SRCS) -o $@

# Speculation-engine sibling-window unit test: the real engine + stubbed externals + a synthetic clause.
SIBLING_TEST_SRCS = simulation_execution_clause_hook.c test_sibling_windows.c
test_sibling_windows: $(SIBLING_TEST_SRCS) $(HDRS)
	$(CC) $(CFLAGS) -I. $(SIBLING_TEST_SRCS) -o $@

clean:
	rm -f contract_executor test_ce test_ce_integration test_mte_tagmem test_sibling_windows

