#include <linux/module.h>
#include <linux/export-internal.h>
#include <linux/compiler.h>

MODULE_INFO(name, KBUILD_MODNAME);

__visible struct module __this_module
__section(".gnu.linkonce.this_module") = {
	.name = KBUILD_MODNAME,
	.init = init_module,
#ifdef CONFIG_MODULE_UNLOAD
	.exit = cleanup_module,
#endif
	.arch = MODULE_ARCH_INIT,
};

KSYMTAB_FUNC(initialize_device_interface, "", "");
KSYMTAB_FUNC(free_device_interface, "", "");
KSYMTAB_FUNC(get_cpu_info, "", "");
KSYMTAB_FUNC(execute_on_pinned_cpu, "", "");
KSYMTAB_FUNC(initialize_executor, "", "");
KSYMTAB_FUNC(free_executor, "", "");
KSYMTAB_FUNC(initialize_inputs_db, "", "");
KSYMTAB_FUNC(allocate_input, "", "");
KSYMTAB_FUNC(get_measurement, "", "");
KSYMTAB_FUNC(get_input, "", "");
KSYMTAB_FUNC(remove_input, "", "");
KSYMTAB_FUNC(destroy_inputs_db, "", "");
KSYMTAB_FUNC(initialize_measurement, "", "");
KSYMTAB_FUNC(free_measurement, "", "");
KSYMTAB_FUNC(execute, "", "");
KSYMTAB_FUNC(stg, "", "");
KSYMTAB_FUNC(mte_randomly_tag_region, "", "");
KSYMTAB_FUNC(mte_init_sandbox_tags, "", "");
KSYMTAB_FUNC(enable_TCMA1_bit, "", "");
KSYMTAB_FUNC(disable_TCMA1_bit, "", "");
KSYMTAB_FUNC(enable_TCO_bit, "", "");
KSYMTAB_FUNC(disable_TCO_bit, "", "");
KSYMTAB_FUNC(enable_mte_tag_checking, "", "");
KSYMTAB_FUNC(page_walk_explorer, "", "");
KSYMTAB_FUNC(disable_mte_for_region, "", "");
KSYMTAB_FUNC(initialize_sandbox, "", "");
KSYMTAB_FUNC(initialize_sysfs, "", "");
KSYMTAB_FUNC(free_sysfs, "", "");
KSYMTAB_FUNC(get_tc_insert_offset_words, "", "");
KSYMTAB_FUNC(load_template, "", "");
KSYMTAB_DATA(kallsyms_lookup_name_fn, "", "");
KSYMTAB_DATA(executor, "", "");

SYMBOL_CRC(initialize_device_interface, 0x3e54c60e, "");
SYMBOL_CRC(free_device_interface, 0x0ac16929, "");
SYMBOL_CRC(get_cpu_info, 0xd610ff84, "");
SYMBOL_CRC(execute_on_pinned_cpu, 0xc4088fc0, "");
SYMBOL_CRC(initialize_executor, 0x0e6e487d, "");
SYMBOL_CRC(free_executor, 0xda69da7f, "");
SYMBOL_CRC(initialize_inputs_db, 0x7200b986, "");
SYMBOL_CRC(allocate_input, 0xcbd07701, "");
SYMBOL_CRC(get_measurement, 0x1f221787, "");
SYMBOL_CRC(get_input, 0x8ee65be9, "");
SYMBOL_CRC(remove_input, 0x1d4d7e6e, "");
SYMBOL_CRC(destroy_inputs_db, 0x750ecb8d, "");
SYMBOL_CRC(initialize_measurement, 0xa6f50b99, "");
SYMBOL_CRC(free_measurement, 0x05b27d1f, "");
SYMBOL_CRC(execute, 0xf72d94f4, "");
SYMBOL_CRC(stg, 0x8c02af45, "");
SYMBOL_CRC(mte_randomly_tag_region, 0x77e3b646, "");
SYMBOL_CRC(mte_init_sandbox_tags, 0xd50d7e18, "");
SYMBOL_CRC(enable_TCMA1_bit, 0x9cfbc417, "");
SYMBOL_CRC(disable_TCMA1_bit, 0x0988eabf, "");
SYMBOL_CRC(enable_TCO_bit, 0xbde562b5, "");
SYMBOL_CRC(disable_TCO_bit, 0xd5b83f01, "");
SYMBOL_CRC(enable_mte_tag_checking, 0x2f93900f, "");
SYMBOL_CRC(page_walk_explorer, 0x5728dfd2, "");
SYMBOL_CRC(disable_mte_for_region, 0xe98d75f5, "");
SYMBOL_CRC(initialize_sandbox, 0x62b3e5f4, "");
SYMBOL_CRC(initialize_sysfs, 0x89bee7e3, "");
SYMBOL_CRC(free_sysfs, 0xf044d70e, "");
SYMBOL_CRC(get_tc_insert_offset_words, 0xe529d99d, "");
SYMBOL_CRC(load_template, 0xfa500a34, "");
SYMBOL_CRC(kallsyms_lookup_name_fn, 0x309078fb, "");
SYMBOL_CRC(executor, 0x2c0fd385, "");

static const struct modversion_info ____versions[]
__used __section("__versions") = {
	{ 0x6228c21f, "smp_call_function_single" },
	{ 0xe3ec2f2b, "alloc_chrdev_region" },
	{ 0x88db9f48, "__check_object_size" },
	{ 0xa6e1a69d, "kick_all_cpus_sync" },
	{ 0xa46dc12a, "sysfs_create_file_ns" },
	{ 0x74d858a7, "on_each_cpu_cond_mask" },
	{ 0xca9360b5, "rb_next" },
	{ 0xf0b2d0b7, "class_destroy" },
	{ 0x96848186, "scnprintf" },
	{ 0xaf56600a, "arm64_use_ng_mappings" },
	{ 0x4829a47e, "memcpy" },
	{ 0x94961283, "vunmap" },
	{ 0x37a0cba, "kfree" },
	{ 0x60113a5d, "kernel_kobj" },
	{ 0xa5526619, "rb_insert_color" },
	{ 0x92997ed8, "_printk" },
	{ 0x3d9ee9f0, "clear_page" },
	{ 0xf0fdf6cb, "__stack_chk_fail" },
	{ 0x6cbbfc54, "__arch_copy_to_user" },
	{ 0x46f219c3, "__free_pages" },
	{ 0xae34ebce, "cdev_add" },
	{ 0x8c8569cb, "kstrtoint" },
	{ 0x730d398a, "device_create" },
	{ 0x4502ef44, "class_create" },
	{ 0x1236618a, "set_cpus_allowed_ptr" },
	{ 0x9688de8b, "memstart_addr" },
	{ 0x4d9b652b, "rb_erase" },
	{ 0x60f90a11, "vmap" },
	{ 0x564405cb, "__cpu_online_mask" },
	{ 0xbcab6ee6, "sscanf" },
	{ 0x75ca79b5, "__fortify_panic" },
	{ 0x11089ac7, "_ctype" },
	{ 0xdcb764ad, "memset" },
	{ 0x17de3d5, "nr_cpu_ids" },
	{ 0xece784c2, "rb_first" },
	{ 0x3e4f8768, "vmalloc_array_noprof" },
	{ 0xb63d562f, "kobject_create_and_add" },
	{ 0x3c3ff9fd, "sprintf" },
	{ 0x999e8297, "vfree" },
	{ 0x6091b333, "unregister_chrdev_region" },
	{ 0xc2e168ab, "caches_clean_inval_pou" },
	{ 0x180d6c31, "device_destroy" },
	{ 0x453acf41, "__kmalloc_cache_noprof" },
	{ 0xbee3ddd5, "vzalloc_noprof" },
	{ 0xd36dc10c, "get_random_u32" },
	{ 0x20000329, "simple_strtoul" },
	{ 0x12a4e128, "__arch_copy_from_user" },
	{ 0x472cf3b, "register_kprobe" },
	{ 0xc2ec7438, "alloc_pages_noprof" },
	{ 0xa65c6def, "alt_cb_patch_nops" },
	{ 0xeb78b1ed, "unregister_kprobe" },
	{ 0x8e662847, "cdev_init" },
	{ 0x9bf5c14b, "kmalloc_caches" },
	{ 0xfd1725f0, "cdev_del" },
	{ 0x5c411c74, "kobject_put" },
	{ 0x92393144, "module_layout" },
};

MODULE_INFO(depends, "");

