CC=gcc
CFLAGS=-Wall -Wextra -O0 -g

SRCS = \
  base_hook.S \
  instruction_encodings.c \
  simulation_code.c \
  simulation_hook.c \
  simulation_input.c \
  simulation.c \
  simulation_execution_clause_hook.c \
  main.c

contract_executor: $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) -o $@

