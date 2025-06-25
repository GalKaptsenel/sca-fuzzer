#ifndef ARM64_EXECUTOR_UTILS_H
#define ARM64_EXECUTOR_UTILS_H

#define xxstr(s) 	xstr(s)
#define xstr(s) 	_str(s)
#define _str(s) 	str(s)
#define str(s) 		#s

#define KB		(1024UL)
#define MB		(1024UL * (KB))
#define PAGESIZE	(4UL * (KB))

// MACROS
#define ALIGN_UP(x, align)   (((x) + ((align) - 1)) & ~((typeof(x))(align) - 1))
//#define ALIGN_DOWN(x, align) ((x) & ~((typeof(x))(align) - 1))


#define module_msg(printer_fn, format, ...)             printer_fn(kernel_module_name ": " format, ##__VA_ARGS__)
#define module_emerg(format, ...)                       module_msg(pr_emerg, format, ##__VA_ARGS__)
#define module_alert(format, ...)                       module_msg(pr_alert, format, ##__VA_ARGS__)
#define module_crit(format, ...)                        module_msg(pr_crit, format, ##__VA_ARGS__)
#define module_err(format, ...)                         module_msg(pr_err, format, ##__VA_ARGS__)
#define module_warn(format, ...)                        module_msg(pr_warn, format, ##__VA_ARGS__)
#define module_notice(format, ...)                      module_msg(pr_notice, format,##__VA_ARGS__)
#define module_info(format, ...)                        module_msg(pr_info, format, ##__VA_ARGS__)
#define module_debug(format, ...)                       module_msg(pr_err, format, ##__VA_ARGS__)
#define module_devel(format, ...)                       module_msg(pr_devel, format, ##__VA_ARGS__)
#define load_global_symbol(lookup_fn, type, local_symbol, symbol)                                                   \
       do {                                                                                                         \
                local_symbol = (type)lookup_fn(#symbol);                                                            \
                module_info("%s = (%s)%s(\"%s\") = %px", #local_symbol, #type, #lookup_fn, #symbol, local_symbol);   \
       } while(0)

// TYPEDEFS
typedef unsigned long (*kallsyms_lookup_name_t)(const char *name);

// GLOBALS

typedef int (*set_memory_t)(unsigned long, int);
extern kallsyms_lookup_name_t kallsyms_lookup_name_fn;

#endif // ARM64_EXECUTOR_UTILS_H
