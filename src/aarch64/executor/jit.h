#ifndef JIT_H
#define JIT_H

#include <linux/types.h>

/* JIT memory context */
typedef struct {
    uint8_t* base;   /* base of the allocated memory */
    uint8_t* cur;    /* current write position */
    size_t size;     /* total allocated size */
    bool overflow;   /* set if a write was dropped because the buffer was full */
} jit_t;

/* Branch condition codes */
enum {
    COND_EQ = 0b0000,
    COND_NE = 0b0001,
};
typedef enum {
	LSL,
	LSR,
	ASR,
	ROR,
} shift_type_t;
typedef enum {
	DC_IVAC,
	DC_ISW,
	
	DC_IGVAC,   // FEAT_MTE2
	DC_IGSW,    // FEAT_MTE2
	DC_IGDVAC,  // FEAT_MTE2
	DC_IGDSW,   // FEAT_MTE2
	
	DC_CSW,
	
	DC_CGSW,    // FEAT_MTE2
	DC_CGDSW,   // FEAT_MTE2
	
	DC_CISW,
	
	DC_CIGSW,   // FEAT_MTE2
	DC_CIGDSW,  // FEAT_MTE2
	
	DC_ZVA,
	
	DC_GVA,     // FEAT_MTE
	DC_GZVA,    // FEAT_MTE
	
	DC_CVAC,
	
	DC_CGVAC,   // FEAT_MTE
	DC_CGDVAC,  // FEAT_MTE
	
	DC_CVAU,
	
	DC_CVAP,    // FEAT_DPB
	
	DC_CGVAP,   // FEAT_MTE
	DC_CGDVAP,  // FEAT_MTE
	
	DC_CVADP,   // FEAT_DPB2
	
	DC_CGVADP,  // FEAT_MTE
	DC_CGDVADP, // FEAT_MTE
	
	DC_CIVAC,
	
	DC_CIGVAC,  // FEAT_MTE
	DC_CIGDVAC, // FEAT_MTE
	
	DC_CIPAE,   // FEAT_MEC
	DC_CIGDPAE, // FEAT_MEC
	
	DC_CIPAPA,  // FEAT_RME
	DC_CIGDPAPA // FEAT_RME
} dc_op_t;




/* initialization */
jit_t* jit_init(size_t size, uint32_t* buffer);
void jit_free(jit_t* jit);

/* pointer control */
uint8_t* jit_get_cur(jit_t* jit);
void jit_set_cur(jit_t* jit, uint8_t* ptr);

/* memory control */
uint8_t* jit_align(jit_t* jit, size_t alignment, ssize_t offset);
uint8_t* jit_next_with_mask(jit_t* jit, uintptr_t value_mask, uintptr_t bit_mask, bool forward);

int jit_perm_rw(jit_t* jit);
int jit_perm_rx(jit_t* jit);

/* instructions: emit bytes into JIT memory */
void jit_emit(jit_t* jit, uint32_t insn);

void jit_cfp_rctx(jit_t* jit, int rt);
void jit_isb(jit_t* jit);
void jit_dsb_sy(jit_t* jit);
void jit_nop(jit_t* jit);
void jit_svc0(jit_t* jit);
void jit_ret(jit_t* jit, int rn);
void jit_bti(jit_t* jit, bool calls, bool branches);

void jit_sys(jit_t* jit, int op1, int CRn, int CRm, int op2, int rt);
void jit_dc(jit_t* jit, int rt, dc_op_t dc_op);
void jit_ubfx64(jit_t* jit, int rd, int rn, int lsb, int width);
void jit_add64(jit_t* jit, int rd, int rn, int imm);
void jit_add64_shift(jit_t* jit, int rd, int rn, int imm);
void jit_sub64(jit_t* jit, int rd, int rn, int imm);
void jit_addr64(jit_t* jit, int rd, int rn, int rm);
void jit_subr64(jit_t* jit, int rd, int rn, int rm);
void jit_subs64(jit_t* jit, int rd, int rn, int imm);
void jit_cmp32(jit_t* jit, int rn, int imm);
void jit_and32(jit_t* jit, int rd, int rn, int imm);
void jit_and64(jit_t* jit, int rd, int rn, int imm);
void jit_andr64(jit_t* jit, int rd, int rn, int rm);
void jit_orr64(jit_t* jit, int rd, int rn, int rm);
void jit_orr64_shift(jit_t* jit, int rd, int rn, int rm, shift_type_t shift_type, int amount);
void jit_eor64(jit_t* jit, int rd, int rn, int rm);
void jit_cbnz64(jit_t* jit, int rt, uint8_t* target);
void jit_cbz32(jit_t* jit, int rt, uint8_t* target);
void jit_b(jit_t* jit, uint8_t* target);

void jit_mrs_pmevcntr0_el0_64(jit_t* jit, int rd);
void jit_msr(jit_t* jit, uint8_t o0, uint8_t op1, uint8_t CRn, uint8_t CRm, uint8_t op2, uint8_t Rt);
void jit_mrs(jit_t* jit, uint8_t o0, uint8_t op1, uint8_t CRn, uint8_t CRm, uint8_t op2, uint8_t Rt);
void jit_read64_pmu(jit_t* jit, uint8_t pmu, uint8_t Rt);
void jit_msr_nzcv(jit_t* jit, uint8_t Rt);
void jit_mrs_nzcv(jit_t* jit, uint8_t Rt);
void jit_stp64(jit_t* jit, int rt1, int rt2, int rn, int imm);
void jit_str64(jit_t* jit, int rt, int rn);
void jit_stp64_post_index(jit_t* jit, int rt1, int rt2, int rn, int imm);
void jit_stp64_pre_index(jit_t* jit, int rt1, int rt2, int rn, int imm);
void jit_ldp64(jit_t* jit, int rt1, int rt2, int rn, int imm);
void jit_ldp64_post_index(jit_t* jit, int rt1, int rt2, int rn, int imm);
void jit_ldr64(jit_t* jit, int rt, int rn);
void jit_ldr64shift0(jit_t* jit, int rt, int rn, int rm);
void jit_str64shift0(jit_t* jit, int rt, int rn, int rm);
void jit_ldr32shift2(jit_t* jit, int rt, int rn, int rm);
void jit_ldr64shift3(jit_t* jit, int rt, int rn, int rm);
void jit_mov64(jit_t* jit, int rd, int imm);
void jit_movr64(jit_t* jit, int rd, int rm);
void jit_smc(jit_t* jit, int imm);
void jit_movk64(jit_t* jit, int rd, int imm, int shift);
void jit_li64(jit_t* jit, int rd, uint64_t imm);
void jit_csel64(jit_t* jit, int rd, int rn, int rm, int cond);
void jit_br64(jit_t* jit, int rn);
void jit_blr64(jit_t* jit, int rn);
void jit_lsr64(jit_t* jit, int rd, int rn, int shift);
void jit_lsl64(jit_t* jit, int rd, int rn, int shift);
void jit_lslr64(jit_t* jit, int rd, int rn, int rm);
void jit_rbit64(jit_t* jit, int rd, int rn);
void jit_lsrr64(jit_t* jit, int rd, int rn, int rm);
void jit_ror64(jit_t* jit, int rd, int rs, int shift);
void jit_udiv64(jit_t* jit, int rd, int rn, int rm);
void jit_msub64(jit_t* jit, int rd, int rn, int rm, int ra);

void jit_set_phr_neoversen3(jit_t* jit, uint64_t value, int rtmp);
void jit_set_phr_neoversen3_dynamic(jit_t* jit, uint64_t reg_bit, uint64_t rtmp1, uint64_t rtmp2, int pos, bool prev_zero);
#endif
