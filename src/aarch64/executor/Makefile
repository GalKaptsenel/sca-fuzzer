# Force AArch64 cross-compiler
ARCH := arm64
CROSS_COMPILE := aarch64-linux-gnu-
CC := aarch64-linux-gnu-gcc-13

# Module name
obj-m := revizor-executor.o

# Detect hardware MTE support (overrideable from command line)
HW_SUPPORTS_MTE ?= $(shell grep -q 'Features.*mte' /proc/cpuinfo 2>/dev/null && echo y || echo n)

ifeq ($(HW_SUPPORTS_MTE),y)
$(info Hardware supports MTE)
ccflags-y += -DCONFIG_ARM64_MTE_HW=1 -march=armv8.5-a+sve+memtag
else
$(info Hardware does NOT support MTE)
ccflags-y += -DCONFIG_ARM64_MTE_HW=0 -march=armv8.2-a+sve
endif

# Common flags
ccflags-y += -g -DL1D_SIZE_K=64 -DL1D_ASSOCIATIVITY=4
ccflags-y += -I$(PWD)/userapi

# Object list
revizor-executor-y := \
	main.o \
	chardevice.o \
	cpu.o \
	executor.o \
	inputs.o \
	measurement.o \
	mte.o \
	pagewalk.o \
	sandbox.o \
	sysfs.o \
	templates.o \
	globals.o \
	aux_buffer.o

KDIR ?= /home/rvzr_prj/linux

ifneq ($(KDIR),)
    KERNEL_DIR := $(KDIR)
else
    KERNEL_DIR := /lib/modules/$(shell uname -r)/build
endif

# Default target: build kernel module
all:
	@echo "Building Revizor executor (MTE HW: $(HW_SUPPORTS_MTE))..."
	$(MAKE) -C $(KERNEL_DIR) M=$(PWD) ARCH=$(ARCH) CROSS_COMPILE=$(CROSS_COMPILE) CC=$(CC)

# Cleanup
clean:
	$(MAKE) -C $(KERNEL_DIR) M=$(PWD) clean

