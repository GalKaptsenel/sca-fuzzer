#!/usr/bin/env python3
"""Pure text parser of saved CE trace logs. NO device, NO CE execution. Reads the already-saved
input_XXXX_trace_log.txt files and extracts every memory access: idx, pc-offset, nesting, EA,
masked-EA, logical tag (top byte nibble), L1D set, granule index, before/after, type."""
import re, sys, glob, os

VD = sys.argv[1]
MASK = (1 << 56) - 1
GRAN = 16

def parse(path):
    entries = []
    cur = None
    base = None  # x29
    first_pc = None
    with open(path) as f:
        lines = f.readlines()
    # find first_pc from the "[000] 0x....:" listing
    for ln in lines:
        m = re.match(r"\s*\[000\]\s+0x([0-9a-f]+):", ln)
        if m:
            first_pc = int(m.group(1), 16)
            break
    i = 0
    ent = None
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("--- Entry "):
            ent = {"accesses": []}
            entries.append(ent)
        elif ent is not None:
            m = re.match(r"\s*x29:\s+0x([0-9a-f]+)", ln)
            if m: ent["x29"] = int(m.group(1), 16)
            m = re.match(r"\s*PC\s*:\s+0x([0-9a-f]+)", ln)
            if m: ent["pc"] = int(m.group(1), 16)
            m = re.match(r"\s*Encoding:\s+0x([0-9a-f]+)\s+\((.*)\)", ln)
            if m: ent["enc"] = m.group(2)
            m = re.match(r"\s*Index:\s+(\d+)", ln)
            if m: ent["idx"] = int(m.group(1))
            m = re.match(r"\s*Speculation nesting:\s+(\d+)", ln)
            if m: ent["nest"] = int(m.group(1))
            if ln.strip() == "MemAccess:":
                acc = {}
                j = i + 1
                while j < len(lines) and (lines[j].startswith("        ") or lines[j].strip()==""):
                    mm = re.match(r"\s*EA:\s+0x([0-9a-f]+)", lines[j]);     acc["ea"]=int(mm.group(1),16) if mm else acc.get("ea")
                    mm = re.match(r"\s*Before:\s+0x([0-9a-f]+)", lines[j]); acc["before"]=mm.group(1) if mm else acc.get("before")
                    mm = re.match(r"\s*After:\s+0x([0-9a-f]+)", lines[j]);  acc["after"]=mm.group(1) if mm else acc.get("after")
                    mm = re.match(r"\s*Size:\s+(\d+)", lines[j]);           acc["sz"]=int(mm.group(1)) if mm else acc.get("sz")
                    mm = re.match(r"\s*Type:\s+(\w+)", lines[j]);           acc["ty"]=mm.group(1) if mm else acc.get("ty")
                    if lines[j].strip()=="" and "ea" in acc: break
                    j += 1
                if "ea" in acc:
                    ent["accesses"].append(acc)
        i += 1
    return entries, first_pc

def dump(path):
    entries, first_pc = parse(path)
    print(f"\n===== {os.path.basename(path)}  (first_pc=0x{first_pc:x}, {len(entries)} entries) =====")
    print(f"{'idx':>3} {'off':>5} {'nst':>3} {'ty':>5} {'sz':>2} {'set':>3} {'gran':>5} {'tagN':>4} {'EAmasked':>14}  before->after   instr")
    for e in entries:
        if not e.get("accesses"): continue
        off = e["pc"] - first_pc
        base = e.get("x29", 0) & MASK
        for a in e["accesses"]:
            ea = a["ea"] & MASK
            cset = (ea >> 6) & 63
            gran = ((ea & ~(GRAN-1)) - (base & ~(GRAN-1))) // GRAN
            tagN = (a["ea"] >> 56) & 0xf
            mark = ""
            if cset in (46,52): mark += "  <==SET%d"%cset
            if gran == 442: mark += "  <==GRAN442"
            print(f"{e['idx']:>3} 0x{off:>3x} {e['nest']:>3} {a['ty'][:5]:>5} {a['sz']:>2} {cset:>3} {gran:>5} {tagN:>4} 0x{ea:>12x}  {a.get('before','?')}->{a.get('after','?')}   {e.get('enc','')}{mark}")

for p in sorted(glob.glob(f"{VD}/input_000[0]_trace_log.txt") + glob.glob(f"{VD}/input_00[012][048]_trace_log.txt")):
    dump(p)
