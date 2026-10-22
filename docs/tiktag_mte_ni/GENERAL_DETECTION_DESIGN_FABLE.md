# A general approach to precondition-gated speculative leaks

TikTag-v1 is one instance of a broader leak class. This document defines the class, then lays out a
ladder of steps from the contrived fixed gadget we have verified toward a random, fuzzer-generated
detector, naming the Revizor pieces each step uses and the condition each step must meet. The
established hardware and framework constraints on this Cortex-X3 are treated as fixed, and every step
that leans on one is marked.

## The leak class

A precondition-gated speculative leak is a speculative memory access whose cache trace is gated by a
precondition that the contract model does not model as affecting the observation. Formally: a program
contains a speculative access A behind a mispredicted control-flow window, A depends on a precondition
P(input), and a downstream probe leaves a cache trace only when A completes inside the window. When P
holds the window stays open long enough for the probe to issue and fill its set, and when P fails the
window shrinks and the set stays absent. Two inputs that the contract calls equivalent, differing only
in P, then produce different hardware traces, which is the non-interference violation.

Instances of P:

- MTE tag match (TikTag-v1, verified).
- A bounds check that passes or fails on a speculative index.
- A page permission or present bit on a speculative address (a PTE precondition).
- Address canonicality on a speculative pointer.
- A PAC authentication that succeeds or faults speculatively.
- An alignment precondition on a speculative access.

The common physics, which no framework change alters: the window shrink needs a trained control-flow
(branch) speculation window, the store-bypass window does not produce the shrink, and the training must
be pollution-free, which on this core means an internal loop inside one test-case execution.

## Detection principle in Revizor terms

The non-interference fuzzer runs one program over an equivalence class of inputs that share a contract
trace and flags a violation when the hardware traces diverge. For this class the two class inputs
differ only in P. The contract does not model P gating the speculative window, so the contract traces
agree, while the hardware diverges because the probe fills its set for the correct P and not for the
incorrect one. The sealer keeps the gated access on a speculative slot, so the architectural path never
exercises a failing P and never faults, while the decoy lane varies P on the speculative slot. The
relevant sealings already exist: `MteSealing`, `PacSealing`, `CanonSealing`, `SignExtSealing`,
`BranchTargetSealing` in `src/aarch64/seal/sealer.py`.

The reload clock is a tunable of the readout, not of the gadget. The L1-residency reload
(`reload_timer_shift = -1`) is the right resolution when P fully gates the probe, which is the TikTag
case. A precondition that only shifts the probe's latency within the cache hierarchy rather than
suppressing it would need a finer cycle threshold (`reload_timer_shift >= 0`, hit iff cycle delta is
below `1 << shift`) or the graded per-set L1 and L2 refill readout in `/sys/executor/reload_refills`.
The detector should sweep the clock across present-or-absent and graded cycle settings so that leaks of
either kind surface.

## The step ladder

### Step 0, the verified fixed harness

The manual detector. One hand-written gadget with an internal training loop, a serial latency gate, and
a probe-free read set, run through the executor as a Flush and Reload measurement harness, with the host
sweeping P (the pointer tag) and computing the verdict.

Revizor pieces: the executor F+R path only.

Holds today: verified genuine on the X3 (see the verification document). This is the ground truth the
later steps must reproduce.

### Step 1, the same gadget as an NI ASM template

Express the fixed gadget as an ASM template passed to `tfuzz` under the non-interference fuzzer, so the
NI machinery produces the verdict instead of the host. The internal training loop becomes a backward
branch in the template. Because the ASM parser rejects `DC CIVAC`, the per-iteration window widening
moves from a cache flush to a compute-delay window, an `SDIV` chain whose latency holds the branch
condition unresolved long enough. The precondition P is varied by the sealer across the two inputs of
one NI class, with `MteSealing` placing the tag-checked load on a speculative slot.

Revizor pieces: `asm_parser`, `tfuzz`, the non-interference fuzzer, the sealer with `MteSealing`.

Must hold: the ASM parser accepts the backward loop, which it does. The `SDIV` compute-delay window
must open the branch window as wide as the flush-widened window did on this core, which has to be
validated because the widening mechanism changes even though the control-flow window physics do not
(constraint on the template API, no flush primitive). The sealer must route the gated load onto a
speculative slot and vary the tag between the two class inputs while the contract holds them equivalent
(the sealer only decoys speculative-and-alts slots, which is the behavior we want).

### Step 2, swap the precondition

Keep the Step 1 skeleton and replace the tag check with another gated access, driving P with the
matching sealing: `CanonSealing` or `SignExtSealing` for canonicality, `PacSealing` for a PAC auth,
`BranchTargetSealing` for a speculative indirect target, or a new sealing for a bounds or PTE
precondition. The genuine lane keeps the correct precondition on architectural slots and never faults,
the decoy lane varies it on the speculative slot.

Revizor pieces: the non-interference fuzzer, the sealer with the chosen sealing, the ASM template from
Step 1.

Must hold: the chosen precondition must gate the effective length of a control-flow speculation window
the same way the tag check does. This is an empirical property of each precondition on this core and
must be confirmed one precondition at a time. The store-bypass window is the known counterexample where
the window opens but no precondition-dependent shrink appears, so a candidate P that only lives on a
store-bypass window will not produce a leak (the verified v4 negative). TikTag falls out of this step as
the `MteSealing` case.

### Step 3, fuzz the body inside a fixed skeleton

Fix the prologue (the internal-loop-trained branch, the latency gate, the probe) and let the fuzzer
generate the instructions between the window opening and the probe, with the sealer free to place the
varied precondition on any eligible speculative slot in that body.

Revizor pieces: a template with a fixed prologue and a fuzzer-filled middle, the non-interference
fuzzer, the sealer.

Must hold: the generated body must not collapse the window or issue the probe unconditionally. The probe
must stay latency-gated behind the precondition-gated access, so the generator needs a constraint that
keeps the probe address data-dependent on the gated access. This is the first step where the fuzzer
contributes structure rather than only input values.

### Step 4, fully random generation

Let the fuzzer assemble the whole structure: an internal-loop-trained branch window, a
precondition-gated speculative access, and a latency-gated probe, with the sealer varying the
precondition across the NI class.

Revizor pieces: the non-interference fuzzer, the sealer, a code generator that can emit an internal loop
and the latency-gate-to-probe dependency, optionally the constant-time sequential contract (CT-SEQ) to
untaint beyond-contract speculative-steering bits.

Must hold, and which constraint each overcomes:

- Pollution-free training requires an internal loop inside one test case. The current generator and the
  object-template path emit forward-only straight-line code, and cross-input or batch training washes
  out because the per-input harness runs branches between inputs that pollute the predictor. This is the
  largest gap. Overcoming it means giving the generator an internal-loop construct, not relying on
  cross-input accumulation.
- The window must come from a genuinely trained branch, not from the deprecated branch-mistraining knob,
  which yields no violations on this core. The internal loop is what supplies the training.
- The sealer varies only speculative slots, which is the behavior we want, so the generator must route
  the precondition onto a speculative slot for the sealer to act on.
- CT-SEQ makes beyond-contract steering bits untainted and drove the Spectre-v2 and BTB leftover with the
  basic fuzzer, but the sequential contract models no speculation and so offers the sealer no speculative
  slot, and the basic path does not tag MTE granules. CT-SEQ can therefore drive a precondition-gated
  leak only if the program self-establishes the precondition, for example tagging a granule with `STG`,
  and a control-flow window still supplies the shrink. This is a partially explored direction and is
  noted as such rather than assumed.

## The most promising concrete next step

Step 1. It converts the verified manual result into a Revizor-native non-interference verdict with the
smallest change, and it depends only on two things that are close at hand: the ASM parser already
accepts the backward training loop, and the compute-delay window needs validating as a replacement for
the flush-widened window on the X3. Concretely: write an ASM template that carries the internal training
loop, an `SDIV` latency window in place of the `DC CIVAC` flush, the serial gate, and the probe, run it
through `tfuzz` under the non-interference fuzzer with `MteSealing`, and confirm that the NI verdict
reproduces the match-versus-mismatch separation the manual detector measures. Sweep the reload clock
during that confirmation so the readout resolution is pinned at the same time. Once Step 1 reproduces
the known leak through the fuzzer, Steps 2 through 4 widen the precondition and then hand structure to
the fuzzer.
