"""
File: AArch64 executor logging helpers.
"""
import datetime
from typing import Dict, List, Optional, Tuple

from ..util import FuzzLogger
from .aarch64_input_layout import REGISTER_REGION_OFFSET
from .aarch64_disasm import (_CAPSTONE, disassemble_instruction,
                             decode_reg_accesses, is_conditional_branch)

# ANSI color codes for terminal-viewable logs (files get codes too; use `less -R` or `cat`)
_C_RESET  = "\033[0m"
_C_ARCH   = "\033[32m"    # green  — arch execution (nesting 0)
_C_SPEC1  = "\033[33m"    # yellow — speculative (nesting 1)
_C_SPEC2  = "\033[31m"    # red    — deeply speculative (nesting > 1)
_C_TAKEN  = "\033[36m"    # cyan   — branch taken
_C_NTAKEN = "\033[35m"    # magenta — branch not taken


def _fmt_ce_entry(ite, code_base: int) -> str:
    disas = disassemble_instruction(ite.cpu.encoding, ite.cpu.pc) or "<unk>"
    offset = ite.cpu.pc - code_base
    nest = ite.metadata.speculation_nesting
    srcs, _ = decode_reg_accesses(ite.cpu.encoding, ite.cpu.pc)
    reg_parts = []
    for r in srcs[:4]:
        rl = r.lower()
        if rl == "sp":
            reg_parts.append(f"sp={ite.cpu.sp:016x}")
        elif rl.startswith("x") and rl[1:].isdigit():
            n = int(rl[1:])
            if 0 <= n <= 30:
                reg_parts.append(f"x{n}={ite.cpu.gpr[n]:016x}")
    ea = (f"  EA=0x{ite.metadata.memory_access.effective_address:016x}"
          if ite.metadata.has_memory_access else "")
    return f"[{nest}]+{offset:04x}  {disas:<28}  {'  '.join(reg_parts)}{ea}"


def log_start_test_case(log: FuzzLogger, tc_counter: int) -> None:
    sep = "#" * 72
    ts  = datetime.datetime.now().strftime("%H:%M:%S.%f")
    for line in (f"\n{sep}", f"  TEST CASE #{tc_counter}   {ts}", f"{sep}\n"):
        log.w(line, ch="session")
    log.use("session")


def log_input(log: FuzzLogger, inp_idx: int, inp, ch: Optional[str] = None) -> None:
    log.header(f"INPUT  inp={inp_idx}", ch=ch)
    data     = inp.tobytes()
    reg_blob = data[REGISTER_REGION_OFFSET:]
    log.w("  Registers (slots 0-5 = x0..x5, slot 6 = NZCV flags, slot 7 = sp):", ch=ch)
    for slot in range(min(8, len(reg_blob) // 8)):
        val  = int.from_bytes(reg_blob[slot * 8: slot * 8 + 8], "little")
        name = ([f"x{i}" for i in range(6)] + ["nzcv", "sp"])[slot]
        log.w(f"    slot {slot} ({name}): 0x{val:016x}", ch=ch)
    mem = data[:REGISTER_REGION_OFFSET]
    log.w("  Memory (first 128 bytes):", ch=ch)
    for row_off in range(0, min(128, len(mem)), 16):
        chunk    = mem[row_off: row_off + 16]
        hex_part = " ".join(f"{b:02x}" for b in chunk)
        log.w(f"    {row_off:04x}:  {hex_part}", ch=ch)
    log.w("", ch=ch)


def log_ce_trace(log: FuzzLogger, label: str, inp_idx: int, cer_list: List,
                 ch: Optional[str] = None) -> None:
    log.header(f"CE TRACE: {label}  inp={inp_idx}  rows={len(cer_list)}", ch=ch)
    if not cer_list:
        log.w("  <empty>", ch=ch)
        return
    code_base = cer_list[0].cpu.pc
    log.w(f"  code_base=0x{code_base:016x}", ch=ch)
    log.w(f"  {'Tag':<6}  {'Row':>4}  {'Offset':>6}  {'Instruction':<28}  Regs / EA", ch=ch)
    log.w(f"  {'':-<6}  {'':-<4}  {'':-<6}  {'':-<28}  ---------", ch=ch)

    for row, ite in enumerate(cer_list):
        nest = ite.metadata.speculation_nesting
        if nest == 0:
            color, tag = _C_ARCH, "[ARCH]"
        elif nest == 1:
            color, tag = _C_SPEC1, "[SPEC]"
        else:
            color, tag = _C_SPEC2, f"[S{nest}] "

        branch_note = ""
        if row + 1 < len(cer_list) and is_conditional_branch(ite.cpu.encoding):
            nxt = cer_list[row + 1].cpu.pc
            taken = (nxt != ite.cpu.pc + 4)
            if taken:
                branch_note = f"  {_C_TAKEN}→ TAKEN{_C_RESET}"
            else:
                branch_note = f"  {_C_NTAKEN}→ NOT-TAKEN{_C_RESET}"

        entry = _fmt_ce_entry(ite, code_base)
        log.w(f"  {color}{tag}{_C_RESET}  {row:>4}  {color}{entry}{_C_RESET}{branch_note}", ch=ch)
    log.w("", ch=ch)


def log_bb_map(log: FuzzLogger, variant_label: str, inp_idx: int,
               sorted_pcs: List[int], bb_info: Dict[int, Dict],
               pc_to_id: Dict[int, int], code_base: int, entry_x7: int,
               ch: Optional[str] = None) -> None:
    log.header(f"BB MAP: {variant_label}  inp={inp_idx}", ch=ch)
    log.w(f"  x7 before run = 0x{entry_x7:016x}  "
          f"(if panic shows this value, crash is in BB 1 — the entry block)", ch=ch)
    log.w(f"  {'ID':>3}  {'Offset':>7}  {'Status':<6}  Instruction", ch=ch)
    log.w(f"  {'':-<3}  {'':-<7}  {'':-<6}  -----------", ch=ch)
    for pc in sorted_pcs:
        info   = bb_info[pc]
        bb_id  = pc_to_id[pc]
        a, s   = info["arch"], info["spec"]
        status = "both" if a and s else ("arch" if a else "spec")
        note   = "  [entry — not patched]" if pc == code_base else ""
        log.w(f"  {bb_id:>3}  +{info['offset']:04x}    {status:<6}  {info['disas']}{note}", ch=ch)
    log.w("", ch=ch)


def log_tc_binary(log: FuzzLogger, label: str, tc_bytes: bytes,
                  ch: Optional[str] = None) -> None:
    log.header(f"TC BINARY: {label}  ({len(tc_bytes)} bytes)", ch=ch)
    for row_off in range(0, len(tc_bytes), 16):
        chunk    = tc_bytes[row_off: row_off + 16]
        hex_part = " ".join(f"{b:02x}" for b in chunk)
        log.w(f"  {row_off:04x}:  {hex_part}", ch=ch)
    log.w("\n  Disassembly:", ch=ch)
    try:
        for insn in _CAPSTONE.disasm(tc_bytes, 0):
            log.w(f"    +{insn.address:04x}:  {insn.mnemonic} {insn.op_str}", ch=ch)
    except Exception as exc:
        log.w(f"  [disasm error: {exc}]", ch=ch)
    log.w("", ch=ch)


def log_pac_op(log: FuzzLogger, op: str, mnemonic: str,
               ptr: int, ctx: int, result: int) -> None:
    log.w(f"  PAC {op:<6}  {mnemonic:<10}  ptr=0x{ptr:016x}  "
          f"ctx=0x{ctx:016x}  =>  0x{result:016x}", ch="pac_signing")


def log_slot(log: FuzzLogger, inp_idx: int, fp) -> None:
    cs  = f"0x{fp.correct_sig:04x}" if fp.correct_sig is not None else "None"
    alt = ",".join(f"0x{s:04x}" for s in fp.alt_sigs) or "None"
    log.w(f"  slot={fp.slot_id:2d}  spec_nesting={str(fp.spec_nesting):<6}  "
          f"correct_sig={cs}  alt_sigs=[{alt}]", ch="pac_signing")


def log_mistraining(log: FuzzLogger, tc_counter: int, inp_idx: int,
                    entries: List[Tuple[int, bool]], cer,
                    ch: Optional[str] = None) -> None:
    """Log branch mistraining config with arch-flow context."""
    log.header(f"MISTRAINING  tc={tc_counter}  inp={inp_idx}  n_branches={len(entries)}", ch=ch)
    if not entries:
        log.w("  <no trainable branches>", ch=ch)
        return

    # Build offset → (actual_taken, disassembly) from the arch path of the CE trace
    arch_map: Dict[int, Tuple[bool, str]] = {}
    if cer and len(cer) > 0:
        code_base = cer[0].cpu.pc
        for i, ite in enumerate(cer):
            if ite.metadata.speculation_nesting != 0:
                continue
            if not is_conditional_branch(ite.cpu.encoding):
                continue
            if i + 1 >= len(cer):
                continue
            taken = (cer[i + 1].cpu.pc != ite.cpu.pc + 4)
            disas = disassemble_instruction(ite.cpu.encoding, ite.cpu.pc) or "<unk>"
            arch_map[ite.cpu.pc - code_base] = (taken, disas)

    log.w(f"  {'Offset':>6}  {'Instruction':<28}  {'Arch dir':<14}  Train dir", ch=ch)
    log.w(f"  {'':-<6}  {'':-<28}  {'':-<14}  ---------", ch=ch)
    for off, train_taken in entries:
        actual = arch_map.get(off)
        disas = actual[1] if actual else "<unknown>"
        actual_text = ("TAKEN    " if actual and actual[0]
                       else "NOT-TAKEN" if actual
                       else "?        ")
        actual_dir = f"{_C_TAKEN if actual and actual[0] else _C_NTAKEN}{actual_text}{_C_RESET}"
        train_text = "TAKEN    " if train_taken else "NOT-TAKEN"
        train_dir  = f"{_C_TAKEN if train_taken else _C_NTAKEN}{train_text}{_C_RESET}"
        log.w(f"  +{off:04x}   {disas:<28}  {actual_dir}  {train_dir}", ch=ch)
    log.w("", ch=ch)
