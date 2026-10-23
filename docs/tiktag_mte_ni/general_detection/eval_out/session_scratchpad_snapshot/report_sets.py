#!/usr/bin/env python3
"""Parse report.txt HW htraces into per-set residency for the 4 counterexamples. Pure local parse.
Per the skill: report strings are MSB-first, set = 63 - char_index."""
import re, sys
rep = open(sys.argv[1]).read()
blocks = re.split(r"Input #(\d+)\n\* Hardware trace:\n", rep)
# blocks: [pre, id, body, id, body, ...]
res = {}
for i in range(1, len(blocks)-1, 2):
    iid = int(blocks[i]); body = blocks[i+1]
    rows = []
    for ln in body.splitlines():
        m = re.match(r"\s*([\^.]{64})\s*\[(\d+)\]", ln)
        if m: rows.append((m.group(1), int(m.group(2))))
    if not rows: continue
    total = sum(c for _,c in rows)
    setfreq = [0.0]*64
    for s,c in rows:
        for idx,ch in enumerate(s):
            if ch=='^':
                setfreq[63-idx] += c
    setfreq = [x/total for x in setfreq]
    res[iid] = (setfreq, total)

want = [0,8,16,24]
print("per-set HW residency (BATCH data from report.txt; contaminated) for counterexamples")
print("set | " + " | ".join(f"#{i}" for i in want))
for s in range(64):
    vals = [res[i][0][s] for i in want if i in res]
    if max(vals) < 0.03 and min(vals) < 0.03:  # skip near-silent sets
        continue
    mark=""
    if s in (46,52): mark="  <== SET%d"%s
    spread = max(vals)-min(vals)
    if spread>=0.15: mark+="  BIGΔ=%.2f"%spread
    print(f" {s:2d} | " + " | ".join(f"{v:.3f}" for v in vals) + mark)
print("\ntotals:", {i:res[i][1] for i in want if i in res})
