# Force AArch64 cross-compiler
ARCH := arm64
CROSS_COMPILE := aarch64-linux-gnu-
CC := aarch64-linux-gnu-gcc

# Module name
obj-m := revizor-executor.o

# Detect hardware MTE support (overrideable from command line)
HW_SUPPORTS_MTE ?= $(shell grep -q 'Features.*mte' /proc/cpuinfo 2>/dev/null && echo y || echo n)

ifeq ($(HW_SUPPORTS_MTE),y)
$(info Hardware supports MTE)
ccflags-y += -DCONFIG_ARM64_MTE_HW=1 -march=armv8.5-a+sve+memtag+pauth
else
$(info Hardware does NOT support MTE)
ccflags-y += -DCONFIG_ARM64_MTE_HW=0 -march=armv8.3-a+sve+pauth
endif

# Common flags
ccflags-y += -g -DL1D_SIZE_K=64 -DL1D_ASSOCIATIVITY=4
ccflags-y += -I$(src)/userapi

# Object list
revizor-executor-y := \
	main.o \
	chardevice.o \
	cpu.o \
	executor.o \
	inputs.o \
	measurement.o \
	mte.o \
	pac.o \
	pmu.o \
	pagetable.o \
	sandbox.o \
	sysfs.o \
	jit.o \
	templates_jit.o \
	globals.o \
	bpu.o \

KDIR := /lib/modules/$(shell uname -r)/build
KERNEL_DIR := $(KDIR)

# Default target: build kernel module
all:
	@echo "Building Revizor executor (MTE HW: $(HW_SUPPORTS_MTE))..."
	$(MAKE) -C $(KERNEL_DIR) M=$(CURDIR) ARCH=$(ARCH) CROSS_COMPILE=$(CROSS_COMPILE) CC=$(CC)

# Host-compiled unit test for the kernel-free PMU logic (no hardware, no module).
HOSTCC ?= cc
test_pmu: test_pmu.c pmu_logic.h
	$(HOSTCC) -Wall -Wextra -O2 -o $@ test_pmu.c

test: test_pmu
	./test_pmu

# Cleanup
clean:
	$(MAKE) -C $(KERNEL_DIR) M=$(CURDIR) clean
	rm -f test_pmu

