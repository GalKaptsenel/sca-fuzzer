CC = aarch64-linux-gnu-gcc
CFLAGS = -g -I../include -Iinclude -I../aarch64/executor/userapi -static -march=armv9-a+sve+memtag -Wall -Wextra -O2

SRC_DIR = ./src

SRCS = $(SRC_DIR)/executor_userland.c
OBJS = $(SRCS:.c=.o)

TARGET = executor_userland

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

