# Build the victim kernel module. Pass KDIR explicitly on this box, e.g.:
#   make KDIR=/usr/src/linux-headers-$(uname -r)
obj-m += victim.o

KDIR ?= /lib/modules/$(shell uname -r)/build

all:
	$(MAKE) -C $(KDIR) M=$(CURDIR) modules

clean:
	$(MAKE) -C $(KDIR) M=$(CURDIR) clean
