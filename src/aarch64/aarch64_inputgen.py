# Requires: capstone, z3-solver
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from capstone import *
from capstone.arm64 import *
from z3 import *
import struct

from .aarch64_generator import InstructionLog

# ------------------------------
# Symbolic state representation
# ------------------------------
class SymbolicState:
	def __init__(self, z3_ctx: Context, sandbox_base_addr: int, sandbox_size: int):
		self.ctx: Context  = z3_ctx

		# 31 general purpose registers x0..x30, use 64-bit BitVecs
		self.regs: Dict[int, BitVecRef] = {}  # reg index -> BitVec
		
		self.chunk_size = 64
		# Z3 array based memory: addr(64) -> byte(8)
		Addr = BitVecSort(64, ctx=self.ctx)
		Chunk = BitVecSort(self.chunk_size * 8, ctx=self.ctx)
		self.memory_array = Array('mem', Addr, Chunk)
		# We'll keep mapping of known concrete regs (if provided)
		self.solver: Solver  = Solver(ctx=self.ctx)

		# We'll create symbolic variables for x0..x5 and flags and the sandbox pages
		self.input_symbols: Dict[str, BitVecRef] = {}
		for r in range(0, 6):  # x0..x5 as inputs
			name = f"x{r}_in"
			bv = BitVec(name, 64, ctx=self.ctx)
			self.input_symbols[name] = bv
			self.regs[r] = bv

		# flags register as symbolic
		flags_bv = BitVec("flags_in", 64, ctx=self.ctx)
		self.input_symbols["flags_in"] = flags_bv
		self.flags: BitVec = flags_bv

		# sp and pc
		self.sp = BitVec("sp", 64, ctx=self.ctx)
		self.pc = BitVec("pc", 64, ctx=self.ctx)

		# sandbox pages meta
		self.sandbox_base: int = sandbox_base_addr
		self.sandbox_size: int = sandbox_size

		# create byte symbols (you can change to larger chunks if too many vars)
		for chunk_offset in range(0, self.sandbox_size, self.chunk_size):
			addr = self.sandbox_base + chunk_offset
			chunk_sym = BitVec(f"mem_chunk_{chunk_offset:x}", self.chunk_size * 8, ctx=self.ctx)
			self.memory_array = Store(self.memory_array, BitVecVal(addr, 64, ctx=self.ctx), chunk_sym)

		# For other regs, default to fresh unconstrained symbolic variables (but will often be constrained by concrete snapshot)
		for r in range(6, 31):
			self.regs[r] = BitVec(f"x{r}_sym", 64, ctx=self.ctx)

	def read_reg(self, reg_index: int) -> BitVecRef:
		return self.regs[reg_index]

	def write_reg(self, reg_index: int, value: BitVecRef):
		self.regs[reg_index] = value

	def _read_mem_chunk(self, addr_bv) -> BitVecRef:
		if isinstance(addr_bv, int):
			addr_bv = BitVecVal(addr_bv, 64, ctx=self.ctx)
		
		addr_aligned = addr_bv & ~BitVecVal(self.chunk_size - 1, 64, ctx=self.ctx)
		return Select(self.memory_array, addr_aligned)

	def _write_mem_chunk(self, addr_bv, chunk_bv):
		if isinstance(addr_bv, int):
			addr_bv = BitVecVal(addr_bv, 64, ctx=self.ctx)
		
		addr_aligned = addr_bv & ~BitVecVal(self.chunk_size - 1, 64, ctx=self.ctx)
		self.memory_array = Store(self.memory_array, addr_aligned, chunk_bv)

	def read_mem_byte(self, addr_bv) -> BitVecRef:
		# addr_bv must be a BitVec(64)
		if isinstance(addr_bv, int):
			addr_bv = BitVecVal(addr_bv, 64, ctx=self.ctx)
		chunk = self._read_mem_chunk(addr_bv)
		byte_idx = addr_bv & BitVecVal(self.chunk_size - 1, 64, ctx=self.ctx)
		shift_amount = zero_extend(byte_idx * 8, chunk.size())
		shifted = LShR(chunk, shift_amount)
		return Extract(7, 0, shifted)

	def write_mem_byte(self, addr_bv, byte_bv):
		if isinstance(addr_bv, int):
			addr_bv = BitVecVal(addr_bv, 64, ctx=self.ctx)
		if isinstance(byte_bv, int):
			byte_bv = BitVecVal(byte_bv & 0xFF, 8, ctx=self.ctx)

		chunk_addr = addr_bv & ~BitVecVal(self.chunk_size - 1, 64, ctx=self.ctx)
		chunk = self._read_mem_chunk(addr_bv)
		byte_idx = addr_bv & BitVecVal(self.chunk_size - 1, 64, ctx=self.ctx)
		shift_amount = zero_extend(byte_idx * 8, chunk.size())
		mask = BitVecVal(0xFF, chunk.size(), ctx=self.ctx) << shift_amount
		new_chunk = (chunk & ~mask) | (zero_extend(byte_bv, chunk.size()) << shift_amount)
		self._write_mem_chunk(chunk_addr, new_chunk)

	def read_mem_u64(self, addr_bv) -> BitVecRef:
		if isinstance(addr_bv, int):
			addr_bv = BitVecVal(addr_bv, 64, ctx=self.ctx)
		result = BitVecVal(0, 64, ctx=self.ctx)
		for i in range(8):
			byte = self.read_mem_byte(addr_bv + BitVecVal(i, 64, ctx=self.ctx))
			byte64 = ZeroExt(56, byte)
			shifted = byte64 << (8 * i)
			result = result | shifted
		return result

	def write_mem_u64(self, addr_bv, value64: BitVecRef):
		if isinstance(addr_bv, int):
			addr_bv = BitVecVal(addr_bv, 64, ctx=self.ctx)
		if isinstance(value64, int):
			value64 = BitVecVal(value64 & ((1 << 64) - 1), 64, ctx=self.ctx)

		assert hasattr(value64, 'size') and value64.size() == 64, f"Expected 64-bit value, got {value64.size()} bits"

		# split value64 into bytes (little endian)
		for i in range(8):
			lo = 8 * i
			hi = 8 * (i + 1) - 1
			byte_i = Extract(hi, lo, value64)  # Extract bits [8*i+7 : 8*i]
			self.write_mem_byte(addr_bv + BitVecVal(i, 64, ctx=self.ctx), byte_i)

# ------------------------------
# Utility: compute cache set from address (symbolic or concrete)
# ------------------------------
def compute_cache_set_from_addr(addr_bv: BitVecRef, line_size: int, num_sets: int, ctx: Context) -> BitVecRef:
	off = line_size.bit_length() - 1
	mask = num_sets - 1

	if isinstance(addr_bv, int):
		cset = (addr_bv >> off) & mask
		return BitVecVal(cset, mask.bit_length(), ctx=ctx)

	elif not isinstance(addr_bv, BitVecRef):
		raise TypeError(f"addr_bv must be int or BitVecRef, got {type(addr)}")

	# set = (addr >> log2(line_size)) & (num_sets - 1)
	nbits = addr_bv.size()

	off = line_size.bit_length() - 1
	mask = num_sets - 1

	off_bv = BitVecVal(off, nbits, ctx=ctx)
#	mask_bv = BitVecVal(mask, nbits, ctx=ctx)

	shifted_bv = LShR(addr_bv, off_bv)
	needed = max(1, mask.bit_length())
	return Extract(needed - 1, 0, shifted_bv)

def norm(mn):
	return mn.lower().replace('.', '_')

# Utility: sign-extend or zero-extend bitvectors
def sign_extend(bv: BitVecRef, to_bits: int):
	assert bv.size() <= to_bits
	return SignExt(to_bits - bv.size(), bv)

def zero_extend(bv: BitVecRef, to_bits: int):
	assert bv.size() <= to_bits
	return ZeroExt(to_bits - bv.size(), bv)

# Utility: rotate right
def ror(bv: BitVecRef, sh):
	n = bv.size()
	if isinstance(sh, int):
		s = sh % n
		return LShR(bv, s) | (bv << (n - s))
	if isinstance(sh, BitVecRef):
		s = sh & BitVecVal(n-1, sh.size(), ctx=sh.ctx)
		return LShR(bv, s) | (bv << (n - s))
	raise TypeError(f"Unsupported shift type: {type(sh)}")

def apply_shift(val, shift, bits=64):
	val = val if isinstance(val, BitVecRef) else BitVecVal(val, bits, ctx=self.ctx)

	if shift.type == ARM64_SFT_LSL:
		return val << shift.value
	elif shift.type == ARM64_SFT_LSR:
		return LShR(val, shift.value)
	elif shift.type == ARM64_SFT_ASR:
		if bits == 32:
			val32 = Extract(31, 0, val)
			shifted32 = val32 >> shift.value
			return zero_extend(shifted32, 64)
		return val >> shift.value
	elif shift.type == ARM64_SFT_ROR:
		return ror(val, shift.value)

	raise ValueError(f"Unsupported shift type: {shift.type}")



# ------------------------------
# ARM64 instruction handler: simplified, handles common cases
# ------------------------------
class Aarch64Emulator:
	def __init__(self, ctx: Context, state: SymbolicState):
		self.ctx = ctx
		self.state = state
		self.cs = Cs(CS_ARCH_ARM64, CS_MODE_ARM)
		self.cs.detail = True
	
	# -------------------------
	# Public: emulate a single capstone instruction
	# returns: dict with optional keys:
	#   - eff_addr_bv (BitVecRef) if mem access effective address computed
	#   - written_mem (addr, bv) if we performed a concrete or symbolic store
	#   - pc (new PC value) if instruction changes PC
	# -------------------------
	def emulate(self, insn: CsInsn):
		m = norm(insn.mnemonic)
		# normalize some alias patterns
		if m.startswith('b_'):
			m = "b." + m[2:]  # keep original-ish

		# dispatch by group
		if m in ('add', 'adds', 'sub', 'subs', 'mul', 'smulh', 'udiv', 'sdiv', 'neg', 'negw', 'madd', 'msub', 'smaddl', 'smsubl', 'umaddl', 'umsubl', 'adc', 'sbc', 'ngc'):
			return self._handle_arith(insn)
		if m in ('and', 'orr', 'eor', 'orn', 'eon', 'bic', 'ands', 'orrs', 'eors', 'orns', 'eons', 'bics'):
			return self._handle_bitwise(insn)
		if m in ('lsl', 'lsr', 'asr', 'ror', 'lslv', 'lsrv', 'asrv', 'rorv', 'ubfm', 'ubfx', 'ubfiz', 'uxtb', 'uxth'):
			return self._handle_shift(insn)
		if m in ('mov', 'movk', 'movz', 'movn', 'mvn', 'adr', 'adrp', 'fmov'):
			return self._handle_mov(insn)
		if m in ('ldr', 'ldrb', 'ldrsb', 'ldrsh', 'ldtr', 'ldtrb', 'ldtrh', 'ldxr', 'ldxrb', 'ldxrh', 'ldaxr', 'prfm', 'ldrh', 'ldrsw', 'ldr_w', 'ldur', 'ldnp', 'ldp', 'ldp_x', 'ldrb_x'):
			return self._handle_load(insn)
		if m in ('str', 'strb', 'stxr', 'stxrb', 'stxrh', 'stlxr', 'sttr', 'sttrb', 'sttrh', 'stp', 'stnp', 'str_w', 'str_x'):
			return self._handle_store(insn)
		if m in ('cbz', 'cbnz', 'tbz', 'tbnz'):
			return self._handle_test_branch(insn)
		if m in ('b', 'bl', 'ret', 'br', 'blr', 'b_cond') or m.startswith('b.'):
			return self._handle_branch(insn)
		if m in ('nop', 'yield', 'wfe', 'wfi', 'sev', 'sevl', 'isb', 'dsb', 'dmb'):
			return {}
		# fallback: unhandled or pseudo-instruction (safe no-op)
		return {}


	def _bv64_const(self, v: int) -> BitVecRef:
		return BitVecVal(v & ((1<<64)-1), 64, ctx=self.ctx)

	def _update_flags(self, result: BitVecRef, carry: BitVecRef = None, overflow: BitVecRef = None):
		"""Update NZCV flags in self.state.flags from result, carry, overflow."""
		N = Extract(63, 63, result)
		Z = If(result == 0, BitVecVal(1, 1, ctx=self.ctx), BitVecVal(0, 1, ctx=self.ctx))
		C = carry if carry is not None else Extract(29,29, self.state.flags)
		V = overflow if overflow is not None else Extract(28,28, self.state.flags)

		upper = Extract(63, 32, self.state.flags)
		lower = Extract(27, 0, self.state.flags)
		
		# pack back into flags
		self.state.flags = Concat(upper, N, Z, C, V, lower)

	def _operand_reg_val(self, op):
		reg_name = self.cs.reg_name(op.reg).lower()

		if reg_name == "lr":
			reg_name = "x30"

		if reg_name.startswith('x'):
			idx = int(reg_name[1:])
			return self.state.read_reg(idx)
		elif reg_name.startswith("w"):
			idx = int(reg_name[1:])
			return zero_extend(Extract(31, 0, self.state.read_reg(idx)), 64)
		elif reg_name == "sp":
			return self.state.sp
		elif reg_name in ("zr", "xzr", "wzr"):
			return self._bv64_const(0)
		else:
			# fallback
			raise NotImplementedError(f"unsupported reg name {reg_name}")

	def _operand_write_reg(self, op, value: BitVecRef):
		"""Write 64-bit value into destination register operand (handles w/x/sp/zr)."""
		reg_name = self.cs.reg_name(op.reg).lower()

		if reg_name == "lr":
			reg_name = "x30"

		if reg_name.startswith('x'):
			idx = int(reg_name[1:])
			self.state.write_reg(idx, value)
			return
		if reg_name.startswith('w'):
			idx = int(reg_name[1:])
			# write lower 32 bits into Xn preserving semantics: we can mask
			masked = Concat(BitVecVal(0, 32, ctx=self.ctx), Extract(31, 0, value))  # zero-extend to 64 but in correct layout
			self.state.write_reg(idx, masked)
			return
		if reg_name == 'sp':
			self.state.sp = value
			return
		if reg_name in ('xzr', 'wzr', 'zr'):
			# writes to zero register are ignored
			return
		raise NotImplementedError(f"unsupported reg write {reg_name}")

	def _parse_mem_op(self, mem_op, insn):
		"""
		mem_op is capstone operand.mem object
		returns (base_bv, index_bv_or_none, disp_int, pre_index_flag, post_index_flag)
		Note: Capstone may not expose pre/post flags uniformly, so we infer from insn.mnemonic when needed.
		"""
		base_bv = BitVecVal(0, 64, ctx=self.ctx)
		idx_bv = None
		disp = getattr(mem_op, 'disp', 0) or 0

		if mem_op.base != 0:
			fake_op = type("FakeOp", (), {"reg": mem_op.base})
			base_bv = self._operand_reg_val(fake_op)

		if getattr(mem_op, 'index', 0) not in (0, ARM64_REG_SP, ARM64_REG_XZR, ARM64_REG_WZR):
			fake_op = type("FakeOp", (), {"reg": mem_op.index})
			idx_bv = self._operand_reg_val(fake_op)
			if getattr(mem_op, 'shift', None):
				shift_type = mem_op.shift.type
				shift_val = mem_op.shift.value
				if shift_type == ARM64_SFT_LSL:
					idx_bv = idx_bv << shift_val
				elif shift_type == ARM64_SFT_LSR:
					idx_bv = LShR(idx_bv, shift_val)
				elif shift_type == ARM64_SFT_ASR:
					idx_bv = idx_bv >> shift_val
				# ROR not allowed for memory operand in AArch64

		op_str = insn.op_str.replace(" ", "").lower()
		pre = op_str.endswith("]!")
		post = not pre and "]," in op_str
		return base_bv, idx_bv, disp, pre, post

	def _get_val(self, op, bits=64):
		if op.type == ARM64_OP_REG: # REG
			val = self._operand_reg_val(op)
			if bits == 32:
				val32 = Extract(31, 0, val)
				val = zero_extend(val32, 64)
			return val
		elif op.type == ARM64_OP_IMM: # IMM
			v = op.imm
			if bits == 32:
				val32 = BitVecVal(v & 0xFFFFFFFF, 32, ctx=self.ctx)
				return zero_extend(val32, 64)
			return self._bv64_const(v)
		else:
			raise NotImplementedError(f"Unsupported operand type {op.type}")
	
	def _capstone_reg_to_reg_index(self, reg_op) -> int:
		if reg_op.type != ARM64_OP_REG:
			raise TypeError(f"Operand is not a register: {op}")
		reg_id = reg_op.reg
		reg_name = self.cs.reg_name(reg_id).lower()
		if reg_name.startswith("x") or reg_name.startswith("w"):
			return int(reg_name[1:])
		elif reg_name == "sp":
			return 31
		elif reg_name == "pc":
			return 32
		else:
			raise ValueError(f"Unsupported register: {reg_name}")

	def _handle_arith(self, insn: CsInsn):
		m = insn.mnemonic.lower()
		ops = insn.operands

		if len(ops) < 2:
			raise ValueError(f"{m} missing operands (Got {len(ops)}, expected at least 2)")

		dest_reg = ops[0]
		bits = 64
		if dest_reg.type == ARM64_OP_REG:
			reg_name = self.cs.reg_name(dest_reg.reg).lower()
			if reg_name.startswith("w"):
				bits = 32

		op0_val = self._get_val(ops[0])
		op1_val = self._get_val(ops[1])
		op2_val = self._get_val(ops[2]) if len(ops) > 2 else BitVecVal(0, bits, ctx=self.ctx)

		if hasattr(ops[1], 'shift') and ops[1].shift.type != 0:
			op1_val = apply_shift(op1_val, ops[1].shift, bits)
		if len(ops) > 2 and hasattr(ops[2], 'shift') and ops[2].shift.type != 0:
			op2_val = apply_shift(op2_val, ops[2].shift, bits)

		result = None
		carry = None
		overflow = None

		if m in ("add", "adds"):
			result = op1_val + op2_val
			if m.endswith("s"):
				carry = ULT(result, op1_val)
				overflow = And(op1_val[bits-1] == op2_val[bits-1], result[bits-1] != op1_val[bits-1])
		elif m in ("sub", "subs"):
			result = op1_val - op2_val
			if m.endswith("s"):
				carry = UGE(op1_val, op2_val)
				overflow = And(op1_val[bits-1] != op2_val[bits-1], result[bits-1] != op1_val[bits-1])
		elif m == "mul":
			result = op1_val * op2_val
		elif m == "smulh":
			# high 64 bits of signed multiply
			result = Extract(127, 64, sign_extend(op1_val, 128) * sign_extend(op2_val, 128))
		elif m in ("madd", "msub", "smaddl", "umaddl", "smsubl", "umsubl"):
			mul = op1_val * op2_val
			if "add" in m:
				result = op0_val + mul
			else:
				result = op0_val - mul
		elif m in ("neg", "negw"):
			result = -op0_val
			if bits == 32:
				result = sign_extend(Extract(31,0,result), 64)
		elif m == "adc":
			c = Extract(29,29,self.state.flags)
			op2_c = op2_val + zero_extend(c, bits)
			result = op1_val + op2_c
			carry = ULT(result, op1_val)
			overflow = And(op1_val[bits-1] == op2_c[bits-1], result[bits-1] != op1_val[bits-1])
		elif m in ("sbc", "ngc"):
			c = Extract(29,29,self.state.flags)
			op2_c = op2_val + (1 - c)
			result = op1_val - op2_c
			carry = UGE(op1_val, op2_c)
			overflow = And(op1_val[bits-1] != op2_c[bits-1], result[bits-1] != op1_val[bits-1])
		elif m == "udiv":
			result = UDiv(op1_val, op2_val)
		elif m == "sdiv":
			result = SDiv(op1_val, op2_val)
		else:
			raise ValueError(f"{m} not supported mnemonic")

		if bits == 32:
			result = zero_extend(Extract(31, 0, result), 64)

		if dest_reg.type == ARM64_OP_REG:
			reg_idx = self._capstone_reg_to_reg_index(dest_reg)
			self.state.write_reg(reg_idx, result)

		if m.endswith("s") or m in ("adc","sbc"):
			self._update_flags(result, carry, overflow)

		return {"result": result}


	def _handle_bitwise(self, insn: CsInsn):
		m = insn.mnemonic.lower()
		ops = insn.operands

		dest_reg = ops[0] if len(ops) > 0 else None
		bits = 64
		if dest_reg and dest_reg.type == ARM64_OP_REG:
			reg_name = self.cs.reg_name(dest_reg.reg).lower()
			if reg_name.startswith("w"):
				bits = 32

		if len(ops) < 3:
			raise ValueError(f"{m} missing operands (Got {len(ops)} operands, 3 required)")

		op1_val = self._get_val(ops[1])
		op2_val = self._get_val(ops[2])

		if hasattr(ops[1], 'shift') and ops[1].shift.type != 0:
			op1_val = apply_shift(op1_val, ops[1].shift, bits)
		if len(ops) > 2 and hasattr(ops[2], 'shift') and ops[2].shift.type != 0:
			op2_val = apply_shift(op2_val, ops[2].shift, bits)

		result = None
		if m in ('and', 'ands', 'bic', 'bics'):
			result = op1_val & (~op2_val if "bic" in m else op2_val)
		elif m in ('orr', 'orrs'):
			result = op1_val | op2_val
		elif m in ('eor', 'eors', 'eon', 'orn'):
			if m in ('eor', 'eors'):
				result = op1_val ^ op2_val
			else:
				result = op1_val ^ ~op2_val

		if bits == 32:
			result = zero_extend(Extract(31, 0, result), 64)

		if len(ops) > 0:
			reg_idx = self._capstone_reg_to_reg_index(ops[0])
			self.state.write_reg(reg_idx, result)

		# update flags if instruction has 's' variant
		if m.endswith('s'):
			self._update_flags(result)

		return {"result": result}

	def _handle_shift(self, insn: CsInsn):
		m = insn.mnemonic.lower()
		ops = insn.operands

		def apply_shift(val, mnem, amount, bits=64):
			val = val if isinstance(val, BitVecRef) else BitVecVal(val, bits, ctx=self.ctx)
			amount = amount if isinstance(amount, BitVecRef) else BitVecVal(amount, bits, ctx=self.ctx)
			amount = amount & BitVecVal(bits - 1, amount.size(), ctx=self.ctx)

			if mnem in ('lsl', 'lslv'):
				return val << amount
			elif mnem in ('lsr', 'lsrv'):
				return LShR(val, amount)
			elif mnem in ("asr", "asrv"):
				if bits == 32:
					val32 = Extract(31, 0, val)
					shifted32 = val32 >> amount
					return zero_extend(shifted32, 64)
				return val >> amount
			elif mnem in ("ror", "rorv"):
				return ror(val, amount)
			return val
		
		dest_reg = ops[0] if len(ops) > 0 else None
		bits = 64
		if dest_reg and dest_reg.type == ARM64_OP_REG:
			reg_name = self.cs.reg_name(dest_reg.reg).lower()
			if reg_name.startswith("w"):
				bits = 32

		op_val = self._get_val(ops[1]) if len(ops) > 1 else None
		if len(ops) > 2:
			amount = self._get_val(ops[2], bits)
		else:
			amount = 0

		result = apply_shift(op_val, m, amount, bits)

		if bits == 32:
			result = zero_extend(Extract(31,0,result), 64)

		if len(ops) > 0:
			dest_idx = self._capstone_reg_to_reg_index(dest_reg)
			self.state.write_reg(dest_idx, result)

		return {"result": result}

	def _handle_mov(self, insn: CsInsn):
		m = insn.mnemonic.lower()
		ops = insn.operands

		dest_reg = ops[0] if len(ops) > 0 else None
		bits = 64
		if dest_reg and dest_reg.type == ARM64_OP_REG:
			reg_name = self.cs.reg_name(dest_reg.reg).lower()
			if reg_name.startswith("w"):
				bits = 32

		src_val = self._get_val(ops[1]) if len(ops) > 1 else None

		# Handle MOVK/MOVN/MOVZ variants (immediate with shift)
		if m in ('movk', 'movn', 'movz'):
			imm = ops[1].imm
			shift = getattr(ops[1].shift, 'value', 0)
			imm_val = BitVecVal(imm, bits, ctx=self.ctx) << shift

			if m == 'movk':
				dest_idx = self._capstone_reg_to_reg_index(dest_reg)
				dest_val = self.state.read_reg(dest_idx)
				mask = (~(BitVecVal(0xFFFF, bits, ctx=self.ctx) << shift)) & BitVecVal((1<<bits)-1, bits, ctx=self.ctx)
				src_val = (dest_val & mask) | imm_val
			elif m == 'movz':
				src_val = imm_val 
			elif m == 'movn':
				src_val = ~imm_val & BitVecVal((1 << bits) - 1, bits, ctx=self.ctx)
		elif isinstance(src_val, int):
			src_val = BitVecVal(src_val, bits, ctx=self.ctx)

		# Truncate to 32-bit if needed
		if bits == 32:
			src_val = zero_extend(Extract(31, 0, src_val), 64)

		if dest_reg:
			reg_idx = self._capstone_reg_to_reg_index(dest_reg)
			self.state.write_reg(reg_idx, src_val)

		return {"result": src_val}

	def _handle_load(self, insn: CsInsn):
		m = insn.mnemonic.lower()
		ops = insn.operands

		dest_reg = ops[0] if len(ops) > 0 else None
		bits = 64
		if dest_reg and dest_reg.type == ARM64_OP_REG:
			reg_name = self.cs.reg_name(dest_reg.reg).lower()
			if reg_name.startswith("w"):
				bits = 32

		mem_op = ops[1].mem
		base, index, disp, pre, post = self._parse_mem_op(mem_op, insn)
		if index is not None: # register-indexed [Xn, Xm]
			addr_val = base + index
		elif pre: # pre-indexed [Xn, #imm]!
			addr_val = base + BitVecVal(disp, 64, ctx=self.ctx)
			fake_op = type("FakeOp", (), {"reg": mem_op.base, "type": ARM64_OP_REG})
			base_idx = self._capstone_reg_to_reg_index(fake_op)
			self.state.write_reg(base_idx, addr_val)
		elif post: # post-indexed [Xn], #imm
			addr_val = base
		else: # simple offseet [xn, #inn]
			addr_val = base + BitVecVal(disp, 64, ctx=self.ctx)

		if 'b' in m:
			result = self.state.read_mem_byte(addr_val)
		elif 'h' in m or 'w' in m or bits == 32:
			nbytes = 2 if 'h' in m else 4
			result = BitVecVal(0, nbytes * 8, ctx=self.ctx)
			for i in range(nbytes):
				byte = self.state.read_mem_byte(addr_val + BitVecVal(i, 64, ctx=self.ctx))
				extended_byte = zero_extend(byte, nbytes * 8)
				result |= extended_byte << (8*i)
		else:
			result = self.state.read_mem_u64(addr_val)

		if bits == 32:
			result = zero_extend(Extract(31, 0, result), 64)

		if dest_reg:
			dest_idx = self._capstone_reg_to_reg_index(dest_reg)
			self.state.write_reg(dest_idx, result)

		if post:
			fake_op = type("FakeOp", (), {"reg": mem_op.base, "type": ARM64_OP_REG})
			base_idx = self._capstone_reg_to_reg_index(fake_op)
			self.state.write_reg(base_idx, addr_val + BitVecVal(disp, 64, ctx=self.ctx))

		return {"result": result, "addr": addr_val}

	def _handle_store(self, insn: CsInsn):
		m = insn.mnemonic.lower()
		ops = insn.operands

		src_reg = ops[0] if len(ops) > 0 else None
		bits = 64
		if src_reg and src_reg.type == ARM64_OP_REG:
			reg_name = self.cs.reg_name(src_reg.reg).lower()
			if reg_name.startswith("w"):
				bits = 32

		src_val = self._get_val(src_reg) if src_reg else None
		if src_val is None:
			raise RuntimeError("Store with no source value")

		if hasattr(src_val, "size"):
			src_val = Extract(bits - 1, 0, src_val)
		else:
			src_val = BitVecVal(src_val & ((1 << bits) - 1), bits, ctx=self.ctx)

		mem_op = ops[1].mem
		base, index, disp, pre, post = self._parse_mem_op(mem_op, insn)

		if index is not None: # register-indexed [Xn, Xm]
			addr_val = base + index
		elif pre: # pre-indexed [Xn, #imm]!
			addr_val = base + BitVecVal(disp, 64, ctx=self.ctx)
			fake_op = type("FakeOp", (), {"reg": mem_op.base, "type": ARM64_OP_REG})
			base_idx = self._capstone_reg_to_reg_index(fake_op)
			self.state.write_reg(base_idx, addr_val)
		elif post: # post-indexed [Xn], #imm
			addr_val = base
		else: # simple offset [Xn, #imm]
			addr_val = base + BitVecVal(disp, 64, ctx=self.ctx)

		if 'b' in m:
			byte_val = Extract(7, 0, src_val)
			self.state.write_mem_byte(addr_val, byte_val)
		elif 'h' in m or 'w' in m or bits == 32:
			nbytes = 2 if 'h' in m else 4
			for i in range(nbytes):
				byte_val = Extract(8*i + 7, 8*i, src_val)
				self.state.write_mem_byte(addr_val + BitVecVal(i, 64, ctx=self.ctx), byte_val)
		else:
			self.state.write_mem_u64(addr_val, src_val)

		if post:
			fake_op = type("FakeOp", (), {"reg": mem_op.base, "type": ARM64_OP_REG})
			base_idx = self._capstone_reg_to_reg_index(fake_op)
			self.state.write_reg(base_idx, addr_val + BitVecVal(disp, 64, ctx=self.ctx))

		return {"addr": addr_val, "value": src_val}

	def _branch_to(self, target):
		self.state.pc = target
		return {'branch_target': target}

	def _handle_test_branch(self, insn: CsInsn):
		m = insn.mnemonic.lower()
		ops = insn.operands
		reg_val = self._get_val(ops[0])

		next_pc = self.state.pc + BitVecVal(4, 64, ctx=self.ctx)

		if m in ('cbz', 'cbnz'):
			target = self._get_val(ops[1])
			cond = (reg_val == BitVecVal(0, reg_val.size(), ctx=self.ctx))
			taken = cond if m == 'cbz' else Not(cond)
		elif m in ('tbz', 'tbnz'):
			bitpos = self._get_val(ops[1])
			target = self._get_val(ops[2])
			if not isinstance(bitpos, int):
				raise ValueError("Symbolic bit positions not supported in tbz/tbnz")

			bit_test = Extract(bitpos, bitpos, reg_val)
			cond = (bit_test == BitVecVal(0, 1, ctx=self.ctx))
			taken = cond if m == 'tbz' else Not(cond)
		else:
			raise ValueError(f"Unsupported test branch {m}")

		self.state.pc = If(taken, target, next_pc)
		return {'branch_target': target, 'cond': taken}

	def _handle_branch(self, insn: CsInsn):
		m = insn.mnemonic.lower()
		ops = insn.operands
		taken = False

		target = self._get_val(ops[0]) if len(ops) > 0 else None

		next_pc = self.state.pc + BitVecVal(4, 64, ctx=self.ctx)

		if m in ('b.eq', 'b.ne', 'b.cs', 'b.hs', 'b.cc', 'b.lo', 'b.mi', 'b.pl',
				'b.vs', 'b.vc', 'b.hi', 'b.ls', 'b.ge', 'b.lt', 'b.gt', 'b.le'):
			flags = self.state.flags
			N = Extract(31, 31, flags)
			Z = Extract(30, 30, flags)
			C = Extract(29, 29, flags)
			V = Extract(28, 28, flags)

			T = BitVecVal(1, 1, ctx=self.ctx)
			F = BitVecVal(0, 1, ctx=self.ctx)

			cond_map = {
					'b.eq': Z == T,
					'b.ne': Z == F,
					'b.cs': C == T,
					'b.hs': C == T,
					'b.cc': C == F,
					'b.lo': C == F,
					'b.mi': N == T,
					'b.pl': N == F,
					'b.vs': V == T,
					'b.vc': V == F,
					'b.hi': And(C == T, Z == F),
					'b.ls': Or(C == F, Z == T),
					'b.ge': N == V,
					'b.lt': N != V,
					'b.gt': And(Z == F, N == V),
					'b.le': Or(Z == T, N != V),
				}
			cond = cond_map[m]
			self.state.pc = If(cond, target, next_pc)
			return {"branch_target": target, "cond": cond}

		elif m in ('b', 'br', 'bl', 'blr'):
			if m in ('bl', 'blr'):
				self.state.write_reg(30, next_pc)
			return self._branch_to(target)
		
		elif m == 'ret':
			LR = self.state.read_reg(30) # X30 = LR (Link Register)
			return self._branch_to(LR)

		else:
			raise ValueError(f"Unexpected branch: {m}")


## ------------------------------
## Decode helpers
## ------------------------------

def decode_encoding(cs: Cs, enc32: int) -> Optional[CsInsn]:
    # Capstone expects bytes
    try:
        b = struct.pack("<I", enc32 & 0xffffffff)
        insns = list(cs.disasm(b, 0x0))
        if len(insns) == 0:
            return None
        return insns[0]
    except Exception as e:
        return None

def debug_solver(solver: Solver):
    constraints = solver.assertions()
    print(f"Solver has {len(constraints)} constraints.\n")

    # Simplify each constraint
    print("Simplified constraints:")
    for i, c in enumerate(constraints):
        simple_c = simplify(c)
        print(f"  [{i}] {simple_c}")
    print("\n")

    # Individual check: only add if it's Boolean
    print("Checking individual constraints for satisfiability:")
    for i, c in enumerate(constraints):
        tmp_solver = Solver()
        print(f"Adding constraint: {c}-------> It is of type {type(c)}")
        # Wrap into a boolean if it's a BitVec
        if isinstance(c, BitVecRef):
            tmp_solver.add(c != 0)
        else:
            tmp_solver.add(c)
        result = tmp_solver.check()
        print(f"  [{i}] {'SAT' if result == sat else 'UNSAT' if result == unsat else 'UNKNOWN'}")

    # Incremental cumulative check
    print("Incremental satisfiability check:")
    tmp_solver = Solver()
    for i, c in enumerate(constraints):
        if isinstance(c, BitVecRef):
            tmp_solver.add(c != 0)
        else:
            tmp_solver.add(c)
        result = tmp_solver.check()
        print(f"  After adding [{i}] constraint: {'SAT' if result == sat else 'UNSAT' if result == unsat else 'UNKNOWN'}")
        if result == unsat:
            print(f"    Conflict likely introduced by constraint [{i}].")
            break

def simplify_expr(expr):
	t = Tactic('ctx-solver-simplify', expr.ctx)
	goal = Goal(ctx=expr.ctx)
	goal.add(expr)
	res = t(goal)
	return simplify(res.as_expr())


# ------------------------------
# Main solver function
# ------------------------------
def solve_for_inputs(instruction_logs: List[InstructionLog],
                     num_sets: int,
                     line_size: int,
                     sandbox_base: int,
                     sandbox_size: int) -> Optional[Dict[str, int]]:
	"""
	Attempt to produce inputs (x0..x5 and sandbox pages) that produce the same cache set trace
	as seen in instruction_logs. Returns model mapping symbol names to concrete values if successful.
	"""
	ctx = Context()
	state = SymbolicState(ctx, sandbox_base, sandbox_size)
	emulator = Aarch64Emulator(ctx, state)

	observed_sets = []
	for log in instruction_logs:
		if log.effective_address != 0:
			set_idx_bv = compute_cache_set_from_addr(log.effective_address, line_size, num_sets, ctx)
			observed_sets.append(set_idx_bv)

	obs_idx = 0
	mem_access_constraints = []

	for i, log in enumerate(instruction_logs):
		enc = log.encoding & 0xffffffff
		insn = decode_encoding(emulator.cs, enc)
		if insn is None:
			print(f"[ERROR] Unable to simulate instruction {insn}")

		emulation_log = emulator.emulate(insn)
		if log.effective_address != 0:
			if "addr" not in emulation_log:
				print(f"[ERROR] Incorrect decode of instruction: {insn}")
			eff_bv = emulation_log["addr"]
			set_bv = compute_cache_set_from_addr(eff_bv, line_size, num_sets, ctx)
			#state.solver.add(simplify_expr(set_bv == observed_sets[obs_idx]))
			c = simplify_expr(set_bv == observed_sets[obs_idx])
			state.solver.add(c)
			if state.solver.check() == sat:
				print(f"[SAT] after adding constraint {len(state.solver.assertions())}")
			else:
				print(f"[UNSAT] after adding constraint {len(state.solver.assertions())}")
			obs_idx += 1

	s = state.solver

	if s.check() == sat:
		m = s.model()
		result = {}
		for k, v in state.input_symbols.items():
			if m.eval(v, model_completion=True) is not None:
				val = m.eval(v, model_completion=True).as_long()
				result[k] = val

		mem_result = {}
		for chunk_offset in range(0, state.sandbox_size, state.chunk_size):
			addr = state.sandbox_base + chunk_offset
			assert sandbox_base <= addr <= state.sandbox_base + state.sandbox_size
			chunk_bv = Select(state.memory_array, BitVecVal(addr, 64, ctx=state.ctx))
			chunk_val = m.eval(chunk_bv, model_completion=True).as_long()
			for i in range(state.chunk_size):
				byte_val = (chunk_val >> (8 * i)) & 0xFF
				mem_result[addr + i] = val
		for c in state.solver.assertions():
			lhs = c.arg(0)
			rhs = c.arg(1)
			lhs_val = m.eval(lhs, model_completion=True)
			rhs_val = m.eval(rhs, model_completion=True)
			print(f"Constraint: {c}")
			print(f"		{lhs} ---> {lhs_val}")
			print(f"		{rhs} ---> {rhs_val}")
		result["memory"] = mem_result
		import pdb; pdb.set_trace()
		return result
	else:
		return None

