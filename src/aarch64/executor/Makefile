# Force AArch64 cross-compiler
ARCH := arm64
CROSS_COMPILE := aarch64-linux-gnu-
CC := aarch64-linux-gnu-gcc-13

# Your module object(s)
obj-m := revizor-executor.o 

ccflags-y +=-g -march=armv8.2-a+sve  -DL1D_SIZE_K=64 -DL1D_ASSOCIATIVITY=4

revizor-executor-y := main.o chardevice.o cpu.o executor.o inputs.o measurement.o mte.o pagewalk.o sandbox.o sysfs.o templates.o globals.o aux_buffer.o

# Tell kbuild to build the module with the forced CC
all:
	$(MAKE) -C /lib/modules/$(shell uname -r)/build M=$(PWD) ARCH=$(ARCH) CROSS_COMPILE=$(CROSS_COMPILE) CC=$(CC)

clean:
	$(MAKE) -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean

