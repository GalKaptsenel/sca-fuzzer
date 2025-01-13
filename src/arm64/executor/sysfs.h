#ifndef ARM64_EXECUTOR_SYSFS_H
#define ARM64_EXECUTOR_SYSFS_H

/* warning! need write-all permission so overriding check */
#undef VERIFY_OCTAL_PERMISSIONS
#define VERIFY_OCTAL_PERMISSIONS(perms) (perms)

#define SYSFS_DIRNAME           kernel_module_name"_configuration"

int initialize_sysfs(void);
void free_sysfs(void);

#endif // ARM64_EXECUTOR_SYSFS_H
