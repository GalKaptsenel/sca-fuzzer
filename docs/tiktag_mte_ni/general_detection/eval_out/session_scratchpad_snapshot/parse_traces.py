# Parse Revizor htrace strings. Convention (per skill): set = 63 - char_index for a 64-char string.
traces = {
 "#1_dom(874)":  "^......^................................^.^........^....^...^^.^",
 "#1_2nd(309)":  "^......^................................^.^........^....^^..^^.^",
 "#9_dom(1201)": "^......^................................^.^........^^^^^^^^^^^^^",
 "#9_2nd(260)":  "^......^................................^.^........^^^^^^^.^^^^^",
}
for name, s in traces.items():
    L = len(s)
    hot = sorted(63 - i for i,c in enumerate(s) if c=='^')
    print(f"{name}: len={L} hot_sets={hot}")
# diff of dominant #1 vs #9
d1 = set(63 - i for i,c in enumerate(traces["#1_dom(874)"]) if c=='^')
d9 = set(63 - i for i,c in enumerate(traces["#9_dom(1201)"]) if c=='^')
print("\n#1_dom hot:", sorted(d1))
print("#9_dom hot:", sorted(d9))
print("only in #9 (decoy gains):", sorted(d9-d1))
print("only in #1 (baseline gains):", sorted(d1-d9))
