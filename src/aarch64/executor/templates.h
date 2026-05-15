#ifndef ARM64_EXECUTOR_TEMPLATES_H
#define ARM64_EXECUTOR_TEMPLATES_H

#define TEMPLATE_ENTER          0x00001111
#define TEMPLATE_INSERT_TC      0x00002222
#define TEMPLATE_RETURN         0x00003333

enum Templates {
	UNSET_TEMPLATE,
	PRIME_AND_PROBE_TEMPLATE,
	FLUSH_AND_RELOAD_TEMPLATE,
};

int load_template(size_t tc_size);

/* Returns the word index (in uint32_t units) within any measurement_code_view
 * where the test case starts.  Valid after the first successful load_template(). */
size_t get_tc_insert_offset_words(void);

#endif // ARM64_EXECUTOR_TEMPLATES_H
