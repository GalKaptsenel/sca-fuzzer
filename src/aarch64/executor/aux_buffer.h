#ifndef DEBUG_PAGE_H
#define DEBUG_PAGE_H

struct aux_buffer_t {
	void* addr;
	size_t size; // Actual size of the buffer pointed by 'addr'.
};

struct aux_buffer_t* aux_buffer_alloc(size_t size);
void aux_buffer_free(struct aux_buffer_t* auxb);
void aux_buffer_init(struct aux_buffer_t* auxb);
void aux_buffer_dump_range(const struct aux_buffer_t* auxb, size_t offset, size_t length);
void aux_buffer_dump(const struct aux_buffer_t* auxb);

//struct debug_page_t {
//	uint64_t regs_write_bits;       // 0
//	uint64_t regs_read_bits;        // 8
//	uint64_t regs_input_read_bits;  // 16
//	
//	uint64_t mem_write_bits;        // 24
//	uint64_t mem_read_bits;         // 32
//	uint64_t mem_input_read_bits;   // 40	
//
//	uint64_t instruction_log_array_offset; // 48
//	uint64_t instruction_log_entry_count; // 56
//	uint64_t instruction_log_max_count; // 64
//};
//
//struct instruction_log_entry_t {
//	uint64_t pc;
//	uint64_t flags;
//	uint64_t regs[31];
//	uint64_t effective_address;
//	uint64_t mem_before;
//	uint64_t mem_after;
//};
//
//void debug_page_print(const struct debug_page_t* dp);
//struct debug_page_t* debug_page_alloc(void);
//void debug_page_free(struct debug_page_t* dp);
//void debug_page_init(struct debug_page_t* dp);

#endif // DEBUG_PAGE_H
