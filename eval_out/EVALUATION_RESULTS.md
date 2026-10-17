
### Spectre-v2 (indirect branch) over the real BTB — which-target, NO sealing (N3)
Paper Fig-(a) "what priming cannot find" gadget, generalized to 3 targets. A target returns a distinct
offset (no memory access in the target); the victim loads from the returned value, so which target the BTB
predicts selects the L1D set. Detected by generalized (cross-input) priming; dropped by standard priming.
No branch-target sealing, no PAC — distinct from the canonicality-over-BTB leak.

Key: force **CT-SEQ**. Under [cond,bpas] the modeled bpas speculation taints the target-selecting bits so
boosting holds them fixed (no within-class toggle -> always "false positive"). CT-SEQ makes the speculative
dispatch beyond-contract -> those bits untainted -> boosting varies them -> the which-target toggle appears.

| path | config | result |
|------|--------|--------|
| template | v2_btb_seq.yml + templates/spectre_v2_btb.py, -i100 | CONFIRMED cross-input leaking 0 -> detecting 11, 1 viol / 57s |
| template, deterministic reproduce x3 | pinned prog 716170 / input 2522819617 | 3/3 identical pair 0->11 |
| template under random config (capability check) | v2_btb_seq_rand.yml, -i50 | CONFIRMED leaking 4 -> detecting 5, 1 viol / 13.6s (config is capable) |
| fully-random (no template) | v2_btb_seq_rand2.yml, -i100 | IN PROGRESS (rand gadget rarer than canonicality; needs tries) |
| TTD (R=20) | template | PENDING |
