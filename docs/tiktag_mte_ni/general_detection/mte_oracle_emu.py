#!/usr/bin/env python3
"""Independent MTE architectural oracle (Unicorn).

Single-steps the EXACT enacted test-case binary with correct MTE semantics: ADDG/SUBG tag arithmetic
and the LDG/STG* tag-memory family are hand-emulated, and every load/store runs a synchronous tag check
(SCTLR_EL1.TCF=SYNC, TCMA1: pointer tags 0/15 are Unchecked) against a modeled per-granule tag memory.
TBI is modeled by aliasing one host sandbox buffer at every tagged page. It reports the architectural
instruction path, the direction taken at each conditional branch, and the first tag-check fault (the
point and the pointer-vs-granule tags).

Usage: mte_oracle_emu.py <violation-dir> <enacted.bin> [--tag29 N] [--base 0xADDR]
"""
import ctypes
import glob
import os
import sys

from unicorn import (Uc, UC_ARCH_ARM64, UC_MODE_ARM, UC_HOOK_CODE, UC_HOOK_MEM_READ,
                     UC_HOOK_MEM_WRITE, UC_HOOK_MEM_UNMAPPED, UC_PROT_ALL, UcError)
from unicorn import arm64_const as A
from capstone import Cs, CS_ARCH_ARM64, CS_MODE_ARM

REPO = "/home/gal_k_1_1998/revizor/sca-fuzzer"
sys.path.insert(0, REPO)

MTE_GRANULE = 16
DEFAULT_TAG = 6           # MTE_INITIAL_DEFAULT_TAG
SANDBOX_UNTAGGED = 0x00ffff8851c21000   # bits[55:0] of the kernel sandbox base (TBI strips [63:56])
SANDBOX_SIZE = 0x40000    # main+faulty+overflow, generous
CODE_BASE = 0x400000
STACK_OFF = 0x20000       # SP within the sandbox

# x-reg -> unicorn const
XREG = {i: getattr(A, f"UC_ARM64_REG_X{i}") for i in range(31)}
XREG[29] = A.UC_ARM64_REG_X29
SP = A.UC_ARM64_REG_SP
PC = A.UC_ARM64_REG_PC


def addg_subg(enc):
    """(is_mte_reg, kind, rd, rn, uimm6, uimm4) for ADDG/SUBG; else (False,...)."""
    if (enc & 0xFFC0C000) == 0x91800000:
        return True, "ADDG", enc & 0x1F, (enc >> 5) & 0x1F, (enc >> 16) & 0x3F, (enc >> 10) & 0xF
    if (enc & 0xFFC0C000) == 0xD1800000:
        return True, "SUBG", enc & 0x1F, (enc >> 5) & 0x1F, (enc >> 16) & 0x3F, (enc >> 10) & 0xF
    return False, None, 0, 0, 0, 0


def is_stg_family(enc):
    """STG/STZG/ST2G/STZ2G (sets a granule tag from Xt[59:56]); returns (hit, two, zero, rt, rn, mode)."""
    if (enc >> 24) == 0xD9:
        top3 = (enc >> 21) & 0x7
        mode = (enc >> 10) & 0x3
        kinds = {0x1: ("STZGM", "STG"), 0x3: ("LDG", "STZG"), 0x5: ("STGM", "ST2G"), 0x7: ("LDGM", "STZ2G")}
        if top3 in kinds:
            name = kinds[top3][0] if mode == 0 else kinds[top3][1]
            if name in ("STG", "STZG", "ST2G", "STZ2G"):
                two = name in ("ST2G", "STZ2G")
                zero = name in ("STZG", "STZ2G")
                return True, two, zero, enc & 0x1F, (enc >> 5) & 0x1F, mode, name
    return False, 0, 0, 0, 0, 0, None


class Oracle:
    def __init__(self, tag29):
        self.uc = Uc(UC_ARCH_ARM64, UC_MODE_ARM)
        self.cs = Cs(CS_ARCH_ARM64, CS_MODE_ARM)
        self.tag29 = tag29
        self.sbuf = (ctypes.c_char * SANDBOX_SIZE)()
        self.tagmem = bytearray([DEFAULT_TAG]) * (SANDBOX_SIZE // MTE_GRANULE)
        self.fault = None
        self.taken = []          # (pc_off, 'taken'/'not', target_off)
        self.dbg = []
        self._mapped_pages = set()
        self._map_alias(SANDBOX_UNTAGGED)   # untagged alias
        self.uc.hook_add(UC_HOOK_MEM_READ, self._tagcheck)
        self.uc.hook_add(UC_HOOK_MEM_WRITE, self._tagcheck)
        self.uc.hook_add(UC_HOOK_MEM_UNMAPPED, self._on_unmapped)
        self.uc.hook_add(UC_HOOK_CODE, self._on_code)

    def _map_alias(self, addr):
        page = addr & ~0xFFF
        base = SANDBOX_UNTAGGED & ~0xFFF
        # map the whole sandbox buffer at the tagged base aligned to the untagged base
        tbase = (addr & 0xFF00000000000000) | base
        if tbase in self._mapped_pages:
            return
        self.uc.mem_map_ptr(tbase, SANDBOX_SIZE, UC_PROT_ALL, ctypes.cast(self.sbuf, ctypes.c_void_p))
        self._mapped_pages.add(tbase)

    def _granule(self, addr):
        off = (addr & ((1 << 56) - 1)) - SANDBOX_UNTAGGED
        return off // MTE_GRANULE if 0 <= off < SANDBOX_SIZE else None

    def _on_unmapped(self, uc, access, address, size, value, user):
        if self._granule(address) is not None or (address & ((1 << 56) - 1)) - (SANDBOX_UNTAGGED & ~0xFFF) < SANDBOX_SIZE:
            self._map_alias(address)
            return True
        self.fault = ("unmapped", address)
        return False

    def _tagcheck(self, uc, access, address, size, value, user):
        g = self._granule(address)
        if g is None:
            return True
        ptag = (address >> 56) & 0xF
        if ptag in (0, 15):
            return True                       # TCMA1 unchecked
        if self.tagmem[g] != ptag:
            if self.fault is None:
                self.fault = ("tagfault", uc.reg_read(PC) - CODE_BASE, address, ptag, self.tagmem[g])
            uc.emu_stop()
            return True
        return True

    def _rd(self, n):
        return 0 if n == 31 else self.uc.reg_read(XREG[n])

    def _wr(self, n, v):
        if n != 31:
            self.uc.reg_write(XREG[n], v & ((1 << 64) - 1))

    def _on_code(self, uc, address, size, user):
        enc = int.from_bytes(uc.mem_read(address, 4), "little")
        off = address - CODE_BASE
        # --- MTE data-reg: emulate + skip native ---
        hit, kind, rd, rn, u6, u4 = addg_subg(enc)
        if hit:
            xn = self._rd_sp(rn)
            addr56 = ((xn & ((1 << 56) - 1)) + (u6 << 4) * (1 if kind == "ADDG" else -1)) & ((1 << 56) - 1)
            # ARM ADDG *and* SUBG choose the tag via ChooseNonExcludedTag(start_tag, uimm4, exclude):
            # the uimm4 is ADDED to the tag for BOTH (SUBG subtracts only the address). With
            # GCR_EL1.Exclude == 0 (the executor clears it) this is a plain +uimm4 mod 16.
            tag = (((xn >> 56) & 0xF) + u4) & 0xF
            res = (xn & (0xF << 60)) | (tag << 56) | addr56
            self._wr_sp(rd, res)
            uc.reg_write(PC, address + 4)
            uc.emu_stop()
            return
        shit, two, zero, rt, rn, mode, name = is_stg_family(enc)
        if shit:
            basev = self._rd_sp(rn)
            s9 = (enc >> 12) & 0x1FF
            imm = ((s9 ^ 0x100) - 0x100) * 16
            addr = basev if mode == 1 else (basev + imm)
            tag = (self._rd_sp(rt) >> 56) & 0xF
            for k in range(2 if two else 1):
                g = self._granule(addr + 16 * k)
                if g is not None:
                    self.tagmem[g] = tag
            if mode in (1, 3):
                self._wr_sp(rn, basev + imm)
            uc.reg_write(PC, address + 4)
            uc.emu_stop()
            return
        # --- conditional branch bookkeeping (observe direction) ---
        mn = next(self.cs.disasm(enc.to_bytes(4, "little"), address), None)
        if mn and mn.mnemonic.startswith(("tbz", "tbnz", "b.", "cbz", "cbnz")):
            self._branch_here = (off, address, mn.mnemonic, mn.op_str)
        else:
            self._branch_here = None

    def _rd_sp(self, n):
        return self.uc.reg_read(SP) if n == 31 else self.uc.reg_read(XREG[n])

    def _wr_sp(self, n, v):
        self.uc.reg_write(SP if n == 31 else XREG[n], v & ((1 << 64) - 1))

    def run(self, code, regs, mem):
        uc = self.uc
        uc.mem_map(CODE_BASE, (len(code) + 0xFFF) & ~0xFFF, UC_PROT_ALL)
        uc.mem_write(CODE_BASE, code)
        ctypes.memmove(self.sbuf, bytes(mem[:SANDBOX_SIZE]), min(len(mem), SANDBOX_SIZE))
        for n, v in regs.items():
            self._wr_sp(n, v)
        end = CODE_BASE + len(code)
        pc = CODE_BASE
        steps = 0
        trace = []
        self._branch_here = None
        while steps < 4000:
            steps += 1
            prev_br = None
            try:
                # run one native instruction (or until a hook stops us)
                before = pc
                self._branch_here = None
                uc.emu_start(pc, end, count=1)
            except UcError as e:
                self.fault = self.fault or ("ucerror", str(e), pc - CODE_BASE)
                break
            if self.fault:
                break
            npc = uc.reg_read(PC)
            # branch direction (decode the just-run insn directly)
            benc = int.from_bytes(uc.mem_read(before, 4), "little")
            bm = next(self.cs.disasm(benc.to_bytes(4, "little"), before), None)
            if bm and bm.mnemonic.startswith(("tbz", "tbnz", "b.", "cbz", "cbnz")):
                tk = (npc != before + 4)
                self.taken.append((before - CODE_BASE, bm.mnemonic, bm.op_str,
                                   "TAKEN" if tk else "not-taken", npc - CODE_BASE))
            if (before - CODE_BASE) in (0x50, 0x78, 0x7c, 0x84):
                self.dbg.append((before - CODE_BASE,
                                 self.uc.reg_read(XREG[5]), self.uc.reg_read(XREG[4]),
                                 self.uc.reg_read(XREG[0])))
            trace.append(before - CODE_BASE)
            if npc == before:        # emu_stop by our MTE hook already advanced PC; guard
                npc = before + 4
                uc.reg_write(PC, npc)
            pc = npc
            if pc < CODE_BASE or pc >= end:
                break
        return trace


def load_input_regs_mem(vd):
    from src.config import CONF
    CONF.load(f"{vd}/org-config.yaml")
    from src import factory
    from src.aarch64.aarch64_executor_input_encoder import deserialize
    wd = "/tmp/claude-1000/oracle_wd"; os.makedirs(wd, exist_ok=True); os.chdir(wd)
    fz = factory.get_fuzzer(os.path.join(REPO, "base.json"), wd, f"{vd}/generated.asm", "")
    fz.initialize_modules()
    tc = fz.asm_parser.parse_file(f"{vd}/generated.asm"); ex = fz.executor; ex.load_test_case(tc)
    inp = deserialize(open(glob.glob(f"{vd}/input_0000.*")[0], "rb").read()).input_
    cer = ex._resolve(inp).cer
    g = cer[0].cpu.gpr
    x = {n: g[29 - n] for n in range(30)}
    from src.aarch64.aarch64_executor import _ce_memory_regs
    mem, _ = _ce_memory_regs(inp)
    return x, mem


def main():
    vd = os.path.abspath(sys.argv[1]); enacted = sys.argv[2]
    tag29 = int(sys.argv[sys.argv.index("--tag29") + 1]) if "--tag29" in sys.argv else 15
    x, mem = load_input_regs_mem(vd)
    code = open(f"{vd}/{enacted}", "rb").read()
    regs = {n: x.get(n, 0) for n in range(29)}
    regs[29] = (SANDBOX_UNTAGGED | (tag29 << 56)) & ((1 << 64) - 1)
    regs[31] = (SANDBOX_UNTAGGED + STACK_OFF) | (tag29 << 56)    # SP
    orc = Oracle(tag29)
    trace = orc.run(code, regs, mem)
    print(f"== {enacted}  (x29 tag={tag29}) ==")
    print(f"executed {len(trace)} architectural instructions; last off=0x{trace[-1]:x}" if trace else "no insns")
    for b in orc.taken:
        print(f"  branch @0x{b[0]:x} {b[1]} {b[2]} -> {b[3]} (next 0x{b[4]:x})")
    if orc.fault:
        if orc.fault[0] == "tagfault":
            print(f"  ARCH TAG FAULT at off=0x{orc.fault[1]:x} addr=0x{orc.fault[2]:x} ptr_tag={orc.fault[3]} granule_tag={orc.fault[4]}")
        else:
            print(f"  stop: {orc.fault}")
    else:
        print("  no architectural tag fault")
    print(f"  reached bb_0.1 (0x8c..0xe0) architecturally: {any(0x8c <= o <= 0xe0 for o in trace)}")
    for off, x5, x4, x0 in orc.dbg:
        print(f"    after 0x{off:x}: x5=0x{x5:016x}(t{(x5>>56)&0xF},b59={(x5>>59)&1}) "
              f"x4=0x{x4:016x}(t{(x4>>56)&0xF},b59={(x4>>59)&1})")


if __name__ == "__main__":
    main()
