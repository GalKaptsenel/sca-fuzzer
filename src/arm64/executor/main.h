#ifndef ARM64_EXECUTOR_H
#define ARM64_EXECUTOR_H

// Includes
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/init.h>
#include <linux/types.h>
#include <linux/mm.h>
#include <linux/io.h>
#include <linux/vmalloc.h>
#include <linux/kprobes.h>
#include <linux/bug.h>
#include <linux/minmax.h>

#include "pagewalk.h"
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

#define DEBUG 0

#define kernel_module_name	"executor"

#endif // ARM64_EXECUTOR_H
