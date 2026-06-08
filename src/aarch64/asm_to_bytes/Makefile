CC = gcc
CFLAGS = -Wall -Wextra -O2 -g

asm_to_bytes: asm_to_bytes.c
	$(CC) $(CFLAGS) $< -o $@

.PHONY: clean
clean:
	rm -f asm_to_bytes
