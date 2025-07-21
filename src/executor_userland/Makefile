CC = aarch64-linux-gnu-gcc
CFLAGS = -I../utils -static -march=armv9-a+sve+memtag -Wall -Wextra -O2

SRC_DIR = .
UTILS_DIR = ../utils

SRCS = executor_userland.c trace_writer.c $(UTILS_DIR)/cJSON.c
OBJS = $(SRCS:.c=.o)

TARGET = executor_userland

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

$(UTILS_DIR)/%.o: $(UTILS_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

