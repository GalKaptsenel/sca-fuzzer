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
  branch_predictors.c \
  neoverse_n3_bpu.c \
  utils.c \
  pac_sign_plugin.c \
  mte_tag_plugin.c \
  tage_py.c \
  main.c

.PHONY: all clean test
all: contract_executor

# Build + run both C test suites. The binaries resolve paths from their own location, so this
# works from any cwd (no need to cd anywhere first).
test: test_ce test_ce_integration contract_executor
	./test_ce
	./test_ce_integration

# Target
contract_executor: $(SRCS) $(HDRS)
	$(CC) $(CFLAGS) $(PYTHON_CFLAGS) $(SRCS) -o $@ $(PYTHON_LDFLAGS)

TEST_SRCS = instruction_encodings.c simulation_input.c test_ce.c
test_ce: $(TEST_SRCS) $(HDRS)
	$(CC) $(CFLAGS) -I. $(TEST_SRCS) -o $@

INT_TEST_SRCS = instruction_encodings.c simulation_input.c test_ce_integration.c
test_ce_integration: $(INT_TEST_SRCS) $(HDRS)
	$(CC) $(CFLAGS) -I. $(INT_TEST_SRCS) -o $@

clean:
	rm -f contract_executor test_ce test_ce_integration

