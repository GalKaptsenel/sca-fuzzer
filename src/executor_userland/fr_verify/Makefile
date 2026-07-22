# Standalone userspace Flush+Reload verification harness (independent of /dev/executor and the PMU).
CC = aarch64-linux-gnu-gcc
CFLAGS = -static -O2 -Wall -Wextra
all: fr_verify
fr_verify: fr_verify.c
	$(CC) $(CFLAGS) -o $@ $<
clean:
	rm -f fr_verify
