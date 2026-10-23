import re, sys
REPORT = "/home/gal_k_1_1998/revizor/sca-fuzzer/docs/tiktag_mte_ni/general_detection/eval_out/collect/session_20261023_011712/violation-261023-021502/report.txt"
lines = open(REPORT).read().split("\n")
# parse blocks per counterexample input
blocks = {}
cur = None
for i, ln in enumerate(lines):
    m = re.match(r"^Input #(\d+)$", ln)
    if m and i < 220:
        cur = int(m.group(1)); blocks[cur] = []; continue
    if cur is not None and re.match(r"^\s*[\^\.]+\s+\[\d+\]\s*$", ln):
        blocks[cur].append(ln)
    if cur is not None and ln.startswith("* Contract trace"):
        cur = None
freq = {}
for k, rows in blocks.items():
    tot = 0; per = [0]*64
    for r in rows:
        pat, cnt = re.match(r"^\s*(\S+)\s+\[(\d+)\]\s*$", r).groups()
        cnt = int(cnt); tot += cnt
        assert len(pat) == 64, (k, len(pat))
        for ci, ch in enumerate(pat):
            if ch == '^':
                per[63 - ci] += cnt   # set = 63 - char index
    freq[k] = [p / tot for p in per]
    print(f"input #{k}: rows={len(rows)} total_count={tot}")
print()
sets = sorted(range(64))
print("set  " + "  ".join(f"#{k:>2}" for k in sorted(freq)))
for s in sets:
    vals = [freq[k][s] for k in sorted(freq)]
    if max(vals) > 0.0 or min(vals) > 0.0:
        print(f"{s:>3}  " + "  ".join(f"{v:4.2f}" for v in vals))
