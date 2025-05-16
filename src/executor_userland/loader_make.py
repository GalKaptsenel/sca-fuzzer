#!/usr/bin/python
import os
import sys
import subprocess

TOOLCHAIN = 'aarch64-linux-gnu'
CC = f'{TOOLCHAIN}-gcc'
LD = f'{TOOLCHAIN}-ld'
OBJCOPY = f'{TOOLCHAIN}-objcopy'
AS = f'{TOOLCHAIN}-as'

def run_command(command):
	print(f"Running: {command}")
	subprocess.run(command, shell=True, check=True)

def convert_binary_data_to_object(binary_file, symbol_base, object_file):
	run_command(f"cp {binary_file} {symbol_base}")
	run_command(f"{OBJCOPY} -I binary -O elf64-littleaarch64 -B aarch64 --rename-section .data=.data,alloc,load,data,contents {symbol_base} {object_file}")
	run_command(f"rm {symbol_base}")

def create_c_source(symbol_base, c_file):
	c_code = f"""#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <assert.h>
#include <arm_acle.h>
#include <sys/prctl.h>
#include <linux/prctl.h>
#include "executor_utils.h"

// Declare the external symbols created by objcopy
extern unsigned char _binary_{symbol_base}_start[];
extern unsigned char _binary_{symbol_base}_end[];
extern unsigned char _binary_{symbol_base}_size;
extern void func(uint64_t x0, uint64_t x1, uint64_t x2, uint64_t x3, uint64_t x4, uint64_t x5);

#define DEFAULT_TEST_TAG 6

static inline void* mte_create_tag(void* ptr, uint8_t tag) {{
    uintptr_t p = (uintptr_t)ptr;
    p &= ~(0xFULL << 56);
    p |= ((uintptr_t)tag << 56);
    return (void*)p;
}} 

static inline void mte_set_tag(void* tagged_ptr) {{
    asm volatile ("stg %0, [%0]" :: "r"(tagged_ptr) : "memory");
}}

static void set_tag_for_range(void* ptr, size_t length, uint8_t tag) {{
    for(size_t i = 0; i < length; ++i) {{
        void* p = (void*)(((uintptr_t)ptr + i) & ~0xFULL);
        void* tagged_ptr = mte_create_tag(p, tag);
        mte_set_tag(tagged_ptr);
    }}
}}

int main() {{
    unsigned char* data = _binary_{symbol_base}_start;
    size_t size = _binary_{symbol_base}_end - _binary_{symbol_base}_start;

    printf("Data size: %zu bytes\\n", size);

    const uint64_t SANDBOX_SIZE = 0x4000;
	input_t* input = (input_t*)data;	
    void* sandbox_base = mmap(NULL, SANDBOX_SIZE, PROT_READ | PROT_WRITE | PROT_MTE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    memcpy(sandbox_base, data, size);
    assert(sandbox_base);

    prctl(PR_SET_TAGGED_ADDR_CTRL, PR_TAGGED_ADDR_ENABLE | PR_MTE_TCF_SYNC | PR_MTE_TAG_MASK, 0, 0, 0);
    set_tag_for_range(sandbox_base, SANDBOX_SIZE, DEFAULT_TEST_TAG);

	uint64_t x0 = input->regs.x0;
	uint64_t x1 = input->regs.x1;
	uint64_t x2 = input->regs.x2;
	uint64_t x3 = input->regs.x3;
	uint64_t x4 = input->regs.x4;
	uint64_t x5 = input->regs.x5;
	uint64_t nzcv = input->regs.flags;
	uint64_t sp = (uint64_t)sandbox_base + SANDBOX_SIZE - 2 * sizeof(uint64_t);
    sp = (uint64_t)mte_create_tag((void*)sp, DEFAULT_TEST_TAG);
	asm volatile(			\\
        " stp x0, x1, [sp, #-16]! \\n"	\\
        " stp x2, x3, [sp, #-16]! \\n"	\\
        " stp x4, x5, [sp, #-16]! \\n"	\\
        " stp x6, x7, [sp, #-16]! \\n"	\\
        " stp x8, x9, [sp, #-16]! \\n"	\\
        " stp x10, x11, [sp, #-16]! \\n" \\
        " stp x12, x13, [sp, #-16]! \\n" \\
        " stp x14, x15, [sp, #-16]! \\n" \\
        " stp x16, x17, [sp, #-16]! \\n" \\
        " stp x18, x19, [sp, #-16]! \\n" \\
        " stp x20, x21, [sp, #-16]! \\n" \\
        " stp x22, x23, [sp, #-16]! \\n" \\
        " stp x24, x25, [sp, #-16]! \\n" \\
        " stp x26, x27, [sp, #-16]! \\n" \\
        " stp x28, x29, [sp, #-16]! \\n" \\
        " mrs x7, nzcv	\\n" 		\\
        " stp x30, x7,  [sp, #-16]! \\n" \\
	" mov x0, %[x0_reg] \\n"		\\
	" mov x1, %[x1_reg] \\n"		\\
	" mov x2, %[x2_reg] \\n"		\\
	" mov x3, %[x3_reg] \\n"		\\
	" mov x4, %[x4_reg] \\n"		\\
	" mov x5, %[x5_reg] \\n"		\\
	" mov x6, %[sandbox_base_reg]  \\n"		\\
    " mov x7, sp    \\n"     \\
    " str x7, [%[sp_reg], #-16]!   \\n"     \\
	" mov sp, %[sp_reg]  \\n"		\\
	" msr nzcv, %[nzcv_reg] \\n"		\\
	"pre_call:	    \\n"		\\
	" bl func	    \\n"		\\
    " ldr %[sp_reg], [sp], #16  \\n"  \\
	" mov sp, %[sp_reg]  \\n"		\\
        " ldp x30, x7,  [sp], #16 \\n"	\\
        " msr nzcv, x7	\\n"		\\
        " ldp x28, x29, [sp], #16 \\n"	\\
        " ldp x26, x27, [sp], #16 \\n"	\\
        " ldp x24, x25, [sp], #16 \\n"	\\
        " ldp x22, x23, [sp], #16 \\n"	\\
        " ldp x20, x21, [sp], #16 \\n"	\\
        " ldp x18, x19, [sp], #16 \\n"	\\
        " ldp x16, x17, [sp], #16 \\n"	\\
        " ldp x14, x15, [sp], #16 \\n"	\\
        " ldp x12, x13, [sp], #16 \\n"	\\
        " ldp x10, x11, [sp], #16 \\n"	\\
        " ldp x8, x9, [sp], #16 \\n"	\\
        " ldp x6, x7, [sp], #16 \\n"	\\
        " ldp x4, x5, [sp], #16 \\n"	\\
        " ldp x2, x3, [sp], #16 \\n"	\\
        " ldp x0, x1, [sp], #16 \\n"	\\
		: 
		: [x0_reg] "r"(x0), [x1_reg] "r"(x1), [x2_reg] "r"(x2), [x3_reg] "r"(x3), [x4_reg] "r"(x4), [x5_reg] "r"(x5), [sandbox_base_reg] "r"(sandbox_base), [nzcv_reg] "r"(nzcv), [sp_reg] "r"(sp)
		: "memory", "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7");

    munmap(sandbox_base, SANDBOX_SIZE);
    return 0;
}}
"""
	with open(c_file, 'w') as f:
		f.write(c_code)


def append_information_to_assembly(input_filename, output_filename):
	with open(input_filename, 'r') as file:
		content = file.readlines()

	pre_lines = """
.globl func
.text
func:
str x30, [sp, #-16]!
mov x30, x6
"""
	post_lines = """
    .text
    ldr x30, [sp], #16
    ret
    """
	filtered_contents =  list(filter(lambda l: '.section' not in l, content))
	new_content = [pre_lines] + filtered_contents + [post_lines]

	with open(output_filename, 'w') as file:
		file.writelines(new_content)

	print(f"File '{output_filename}' created successfully with appended information.")

def assemble_testcase(filename, output_file):
	modified = filename + '_mod'
	append_information_to_assembly(filename, modified);
	run_command(f'{AS} -march=armv9-a+sve+memtag {modified} -o {output_file}');
	run_command(f'rm {modified}')

def compile_and_link(c_file, testcase_file, data_file, output_file):
	run_command(f"{CC} -I/home/gal/revizor_aarch64/src/executor_userland -march=armv9-a+memtag+sve -Wall -g -static -o {output_file} {c_file} {testcase_file} {data_file}")

def clean(files):
	for f in files:
		if os.path.exists(f):
			os.remove(f)

def main():
	if len(sys.argv) != 3:
		print(f"Usage: {sys.argv[0]} <input_binary_file> <test_case_file>")
		sys.exit(1)
	
	test_case_file = sys.argv[2]
	test_case_object = test_case_file + '.o'
	data_file = sys.argv[1]
	data_object_file = "data.o"
	c_file = "loader_aarch64.c"
	output_file = "program"
	symbol_base = 'data_base'

	try:
		print("== Building ==")
		convert_binary_data_to_object(data_file, symbol_base, data_object_file)
		create_c_source(symbol_base, c_file)
		assemble_testcase(test_case_file, test_case_object)
		compile_and_link(c_file, test_case_object, data_object_file, output_file)
		print(f"✅ Build successful: {output_file}")
	except subprocess.CalledProcessError as e:
		print(f"❌ Build failed: {e}")
	finally:
		clean([test_case_object, data_object_file, c_file])

if __name__ == "__main__":
	main()

