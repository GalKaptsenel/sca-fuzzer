#ifndef EXECUTOR_GLOBALS_H
#define EXECUTOR_GLOBALS_H

#include <linux/version.h>
#include <linux/types.h>
#include "main.h"

// Extern Globals
extern kallsyms_lookup_name_t kallsyms_lookup_name_fn;
extern executor_t executor;

#endif // EXECUTOR_GLOBALS_H

