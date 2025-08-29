#ifndef ARM64_EXECUTOR_H
#define ARM64_EXECUTOR_H

// Includes
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/init.h>
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

#if LINUX_VERSION_CODE < KERNEL_VERSION(5, 4, 0)
#include <linux/set_memory.h>
#endif

#include "utils.h"
#include "sysfs.h"
#include "pfc.h"
#include "cache_configuration.h"
#include "templates.h"
#include "measurement.h"
#include "sandbox.h"
#include "inputs.h"
#include "chardevice.h"
#include "executor.h"
#include "mte.h"
#include "cpu.h"

#define DEBUG 0

#include "pagewalk.h"

#define kernel_module_name	"executor"

#endif // ARM64_EXECUTOR_H
