# 155905 — PMU confirmation of the speculative mechanism

Measured INST_RETIRED (0x08), EXC_TAKEN (0x09) and MEM_ACCESS (0x13) on the suspect's match and mismatch
variants, fixed-prefix swap-only protocol, 2000 reps x3, set 9 the leak probe. Raw: eval_out/PMU_155905_pmutest.txt.

| variant  | INST_RETIRED | EXC_TAKEN | set 9 |
|----------|--------------|-----------|-------|
| baseline (match)   | 63 | 0 | 1.00 |
| decoy (mismatch)   | 63 | 0 | 0.02-0.15 |

The tag mismatch changes only the set-9 cache footprint. Retired instructions are identical (63) and no
exception is taken (EXC_TAKEN 0) for match or mismatch. The sealed access executes under speculation and
is suppressed on a mismatch, squashed before retirement, so it leaves no architectural trace beyond the
cache. This is the speculative tag-check signature (TikTag-v1 shrink).

The contrast case is an architectural MTE tag-check fault: under TCF=sync (measurement.c sets
mte_set_sync) a mismatched checked access that retires takes a precise exception, which raises EXC_TAKEN
and commits the handler (INST_RETIRED jumps to ~195k). 072421 showed that architectural signature and was
reclassified; 155905 shows the speculative one.
