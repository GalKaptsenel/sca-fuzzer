import re
import sys
text = open(sys.argv[1]).read().splitlines()
def block(n):
    rows, on = [], False
    for ln in text:
        if re.match(rf"^Input #{n}$", ln.strip()):
            on = True
            continue
        if on and (ln.startswith("* Contract trace") or ln.startswith("Input #")):
            break
        if on:
            m = re.match(r"^\s*([\^\.]{64})\s+\[(\d+)\]\s*$", ln)
            if m:
                rows.append((m.group(1), int(m.group(2))))
    return rows
base = block(0)
bt = sum(c for _, c in base)
def res(rows, s):
    tot = sum(c for _, c in rows)
    return sum(c for st, c in rows if st[63 - s] == "^") / tot
watch = [6, 7, 8, 32, 51, 52, 55, 56, 57, 58]
print("set   " + "  ".join(f"#{n:<5d}" for n in [0, 8, 16, 24]))
rows = {n: block(n) for n in [0, 8, 16, 24]}
for s in watch:
    print(f"{s:3d}   " + "  ".join(f"{res(rows[n], s):.3f}" for n in [0, 8, 16, 24]))
print("rows:", {n: len(rows[n]) for n in rows}, "samples:", {n: sum(c for _, c in rows[n]) for n in rows})
