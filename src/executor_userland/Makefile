CC = aarch64-linux-gnu-gcc
CFLAGS = -I../include -Iinclude -static -march=armv9-a+sve+memtag -Wall -Wextra -O2

SRC_DIR = ./src
THIRD_PARTY_DIR = ../third_party

SRCS = $(SRC_DIR)/executor_userland.c $(SRC_DIR)/trace_writer.c $(SRC_DIR)/buffer.c $(THIRD_PARTY_DIR)/cJSON.c
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

