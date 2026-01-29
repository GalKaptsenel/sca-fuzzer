CC = gcc

# Base flags
CFLAGS = -Wall -Wextra -O0 -g
LDFLAGS =

# Add Python3 embedding flags
PYTHON_CFLAGS = $(shell python3-config --embed --cflags)
PYTHON_LDFLAGS = $(shell python3-config --embed --ldflags)

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
  stream_ipc.c \
  tage_py.c \
  main.c

# Target
contract_executor: $(SRCS)
	$(CC) $(CFLAGS) $(PYTHON_CFLAGS) $(SRCS) -o $@ $(PYTHON_LDFLAGS)

