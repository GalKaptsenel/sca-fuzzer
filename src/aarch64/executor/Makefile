# Force AArch64 cross-compiler
ARCH := arm64
CROSS_COMPILE := aarch64-linux-gnu-
CC := aarch64-linux-gnu-gcc

# Module name
obj-m := revizor-executor.o

# Detect hardware MTE support (overrideable from command line)
HW_SUPPORTS_MTE ?= $(shell grep -q 'Features.*mte' /proc/cpuinfo 2>/dev/null && echo y || echo n)
# Detect hardware PAC support (overrideable from command line)
HW_SUPPORTS_PAC ?= $(shell grep -q 'Features.*paca' /proc/cpuinfo 2>/dev/null && echo y || echo n)

ifeq ($(HW_SUPPORTS_MTE),y)
MTE_BANNER := Hardware supports MTE
ARM_ARCH := armv8.5-a+sve+memtag
ccflags-y += -DCONFIG_ARM64_MTE_HW=1
else
MTE_BANNER := Hardware does NOT support MTE
ARM_ARCH := armv8.3-a+sve
ccflags-y += -DCONFIG_ARM64_MTE_HW=0
endif

ifeq ($(HW_SUPPORTS_PAC),y)
PAC_BANNER := Hardware supports PAC
ARM_ARCH := $(ARM_ARCH)+pauth
ccflags-y += -DCONFIG_ARM64_PAC_HW=1
else
PAC_BANNER := Hardware does NOT support PAC
ccflags-y += -DCONFIG_ARM64_PAC_HW=0
endif

# $(info) fires on every parse. kbuild parses this file twice (the outer `make` and
# the `make -C $(KDIR) M=...` re-read), so gate the banner to the kbuild pass only
# (KBUILD_EXTMOD is set solely then) to print it exactly once for either invocation.
ifdef KBUILD_EXTMOD
$(info $(MTE_BANNER))
$(info $(PAC_BANNER))
endif

ccflags-y += -march=$(ARM_ARCH)

# Common flags
ccflags-y += -g -DL1D_SIZE_K=64 -DL1D_ASSOCIATIVITY=4
ccflags-y += -I$(src)/userapi

# Object list
revizor-executor-y := \
	main.o \
	chardevice.o \
	input_format.o \
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

# Host-compiled, device-backed test of the kernel PAC sign/xpac ioctls via the non-faulting
# verification (never AUTHs). Needs /dev/executor at runtime.
test_pac_auth_verify: test_pac_auth_verify.c userapi/executor_pac_api.h userapi/executor_ioctl_nr.h
	$(HOSTCC) -Wall -Wextra -O2 -Iuserapi -o $@ test_pac_auth_verify.c

test: test_pmu
	./test_pmu

# Cleanup
clean:
	$(MAKE) -C $(KERNEL_DIR) M=$(CURDIR) clean
	rm -f test_pmu test_pac_auth_verify

