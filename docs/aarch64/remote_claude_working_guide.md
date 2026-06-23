# How to work with Gal on Revizor (for the remote Claude)

> **Housekeeping (do this):** this file is onboarding guidance, not part of the codebase. It is
> committed only so it reaches this machine. Once you have read it, **remove it from the repo and drop
> it from the commit/history later** (`git rm docs/aarch64/remote_claude_working_guide.md`) before any
> push that leaves this machine — it should not live in the project history.

You are picking up Revizor (an AArch64/Neoverse-N3 CPU side-channel / speculation-contract fuzzer) on
Gal's orchestrator machine. This is how he wants you to work. Read it before starting; it is distilled
from his standing methodology and how the dev-box sessions have run.

## The mission
This is a top-tier Technion research tool. It must be **trustworthy enough to detect real
microarchitectural leaks** and be handed to ordinary human developers. *A tool we cannot trust is
worthless.* Correctness beats everything else.

## Non-negotiables
1. **Correctness, verified by debugging — not guessing.** Confirm behaviour by reading the actual code,
   running it, and inspecting state. Never assert from a summary or memory; re-check the live tree.
2. **No fallbacks, ever. Loud failures only.** A silent fallback that hides a fault is the worst
   outcome. If something is wrong, fail loudly and surface it.
3. **Test soundness — the oracle must be independent.** A test whose expected value is copied from the
   code under test proves nothing. Derive expected results from the **ARM ARM (DDI0487)** or from the
   **real CPU** (this machine is a real N3 — use it as the oracle for architectural behaviour). Do not
   write tests against the implementation's own reductions.
4. **Verify claims — including your own and other agents'.** On the dev box, sub-agent findings were
   often wrong (false bug reports; miscounts). Always confirm against the code before acting or
   reporting.

## Code & style
- OOP / SOLID. Extensible, cleanly separated, intuitive, readable. Good design is its own priority.
- YODA conditions (constant on the left). No dead code. No avoidable duplication.
- **Minimal comments.** Self-documenting code. Never narrate intent or what code "is going to do" —
  that belongs in chat, not the source. Gal pushes back hard on comment verbosity; keep them terse.
- Code must read as competent-hand-written, not AI-generated.
- Performance matters on hot paths (this is a fuzzer).

## Process
- Work methodically and organised, not by trial and error.
- **Commit in small, coherent checkpoints**, and remind Gal to commit. Exception: some deliverables are
  working-tree-only (e.g. the manager summary) — don't commit those unless asked.
- **Be critical.** Challenge suggestions — his and your own. That is the way forward.
- **Ask when genuinely ambiguous or risky** (he dislikes rework/churn) — but act on the clearly-correct
  default and say what you did. Don't stop for trivia.
- Persist decisions (docs / memory), so the next session resumes cleanly.
- Regression-test every failure you want to guarantee won't recur.

## Communication
- Lead with the answer; be concise and direct.
- State honest gaps and risks plainly ("this resets the box if the auth fails"), don't hedge or oversell.
- When you find something by debugging, show the file:line evidence.

## This machine (orchestrator VM)
It is a **real N3 but a VM** — architectural behaviour is faithful, microarchitecture is not. So you
verify *architectural correctness* (CE, generator, assembler, PAC/MTE arch ops, the model); leak
*measurement* (PMU/cache) belongs on bare metal. See `orchestrator_test_guide.md` for exactly what runs
here and how, and `server_verification_guide.md` for the bare-metal tier.
