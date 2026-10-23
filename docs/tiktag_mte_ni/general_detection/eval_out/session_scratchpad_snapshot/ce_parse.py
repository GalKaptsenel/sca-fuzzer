import re, sys
VD = "/home/gal_k_1_1998/revizor/sca-fuzzer/docs/tiktag_mte_ni/general_detection/eval_out/collect/session_20261023_011712/violation-261023-021502"
CODE_BASE = 0x0000ffff91159000  # PC base of test code in the CE trace (entry 0 PC)

def parse(path):
    txt = open(path).read()
    parts = re.split(r"\n--- Entry (\d+) ---\n", "\n" + txt)
    entries = []
    # parts: [pre, idx, body, idx, body, ...]
    for k in range(1, len(parts), 2):
        idx = int(parts[k]); body = parts[k+1]
        e = {"idx": idx}
        m = re.search(r"PC\s*:\s*0x([0-9a-f]+)", body); e["pc"] = int(m.group(1), 16)
        m = re.search(r"Encoding: 0x([0-9a-f]+) \((.*?)\)", body); e["enc"] = int(m.group(1), 16); e["asm"] = m.group(2)
        m = re.search(r"Speculation nesting: (\d+)", body); e["nest"] = int(m.group(1))
        regs = {}
        for rm in re.finditer(r"^\s{6}x(\d\d):\s*0x([0-9a-f]+)", body, re.M):
            regs[int(rm.group(1))] = int(rm.group(2), 16)
        e["x"] = regs
        mem = []
        for mm in re.finditer(r"MemAccess:\n\s*EA: 0x([0-9a-f]+)\n\s*Before: 0x([0-9a-f]+)\n\s*After: 0x([0-9a-f]+)\n\s*Size: (\d+)\n\s*Type: (\w+)", body):
            ea, b, a, sz, ty = mm.groups()
            mem.append({"ea": int(ea, 16), "before": int(b, 16), "after": int(a, 16), "size": int(sz), "type": ty})
        e["mem"] = mem
        entries.append(e)
    return entries

if __name__ == "__main__":
    n = sys.argv[1]
    ents = parse(f"{VD}/input_{int(n):04d}_trace_log.txt")
    print(f"input {n}: {len(ents)} entries; nesting histogram:",
          {k: sum(1 for e in ents if e['nest']==k) for k in sorted(set(e['nest'] for e in ents))})
    base = None
    for e in ents[:1]:
        base = e["x"][29]
    print(f"x29 at entry0 = {base:#x}")
    for e in ents:
        off = e["pc"] - CODE_BASE
        for ma in e["mem"]:
            tag = (ma["ea"] >> 56) & 0xf
            addr = ma["ea"] & 0x00ffffffffffffff
            sb = e["x"][29] & 0x00ffffffffffffff
            o = addr - sb
            print(f"[{e['idx']:03d}] n{e['nest']} pc+{off:#06x} {ma['type']:5s} sz{ma['size']} EA={ma['ea']:#018x} tag={tag:2d} off={o:#07x} gran={o//16:4d} set={(addr>>6)&63:2d}  asm={e['asm']}")
