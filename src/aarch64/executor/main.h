#ifndef ARM64_EXECUTOR_H
#define ARM64_EXECUTOR_H

// Includes
#include <linux/module.h>
#include <linux/init.h>
MODULE_LICENSE("GPL");
MODULE_AUTHOR("ACSL - Gal Kaptsenel");
MODULE_DESCRIPTION("AArch64 implementation of Revisor's executor");
#include <linux/kernel.h>
#include <linux/version.h>
#include <linux/types.h>
#include <linux/mm.h>
#include <linux/io.h>
#include <linux/vmalloc.h>
#include <linux/kprobes.h>
#include <linux/bug.h>
#include <linux/minmax.h>
#include <linux/cdev.h>
#include <linux/uaccess.h>
#include <linux/memory.h>
#include <linux/smp.h>
#include <linux/cpu.h>
#include <linux/ctype.h>
#include <asm/sysreg.h>
#include <asm/cputype.h>
#include <asm/cacheflush.h>
#include <linux/sched.h>
#include <linux/delay.h>
#include <linux/cpumask.h>
#include <linux/perf_event.h>
#include <linux/hw_breakpoint.h>

#if LINUX_VERSION_CODE < KERNEL_VERSION(5, 4, 0)
#include <linux/set_memory.h>
#endif

#include "aux_buffer.h"
#include "cache_configuration.h"
#include "utils.h"
#include "measurement.h"
#include "sandbox.h"
#include "executor.h"
#include "templates.h"
#include "chardevice.h"
#include "sysfs.h"
#include "pfc.h"
#include "inputs.h"
#include "mte.h"
#include "cpu.h"
#include "globals.h"

#define DEBUG 0

#include "pagewalk.h"

#define kernel_module_name	"executor"

#endif // ARM64_EXECUTOR_H
