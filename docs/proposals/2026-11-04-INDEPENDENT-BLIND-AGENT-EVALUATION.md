# Gemina — Independent Development and Blinded Agent Evaluation v0.1

**Статус:** ACCEPTED OPTIONAL PROCESS / user-confirmed 2026-11-04. Not a change to the Gemina model, not runtime authorization.  
**Совместимость:** [MVP Baseline](../MVP-BASELINE-v0.1.md), [M0 Passport](../M0-IMPLEMENTATION-PASSPORT-v0.1.md), [PR #5 independent fixtures protocol](https://github.com/LL1yt/Gemina_project/blob/research/2026-11-02-detailed-mvp-plan/docs/plans/2026-11-02-mvp/INDEPENDENT-SAMPLER-TESTS-AND-AGENT-WORKFLOW.md), independent review [PR #8](https://github.com/LL1yt/Gemina_project/pull/8), adopted [difficulty tiers](2026-11-04-DIFFICULTY-AND-AGENT-MODES.md) (PR #10). This protocol adds an **orthogonal optional evaluation mode** and does not change A/B/C risk assignments. See [agent allocation](../AGENT-WORK-ALLOCATION-v0.1.md) and the [leaderboard](../agent-evaluations/LEADERBOARD.md).

## Operational alignment (updated 2026-11-04)

M0 passport and risk-tier proposal have been merged into main through separate, owner-authorized PRs. Updated owner decision: **M1 is not a contest**. Morgana implements the single sampler reference; Virgil prepares independent fixtures and audits it. Parallel implementations are optional **only** for later complex sections when their added value warrants the effort. This protocol is a process choice only, **not** permission to start coding, spend compute or invite new agents automatically. Confirm task card, budget and local toolchain first. Results and winners are recorded only after tests and anonymized judging in [Gemina Agent League](../agent-evaluations/LEADERBOARD.md).

## Purpose

Learn how the current coding agents differ at *the same bounded, real engineering tasks*; obtain independent implementations and tests; detect common silent mistakes; compare code, developer experience and practical effectiveness. This is **not** a ranking of companies or a theorem about which model is generally best. Awarding informal prizes is optional, after results have been locked. 😼

**Non-negotiable priority:** do not delay the first running Gemina kernel or multiply implementation complexity to create a competition. Parallel work is limited to tasks where the extra independence is worth the cost and the scope is frozen.

## Two different comparisons: do not confuse them

- **Independent development (parallel):** two agents receive identical immutable contract and conditions, independently produce implementations in separate isolated workspaces. Neither sees the other's code, tests, prompts, commentary, or intermediate results before freezing the submissions. Both may receive identical preregistered public requirements and shared held-out criteria, but not one another's work.
- **Blind review (evaluation):** an organizer strips author/model identifiers from frozen deliverables, labels them `Candidate A` / `Candidate B`, and sends them to one or more judges, including the owner if desired. Judges do not receive model names or previous opinions until scores are locked. Blinding the judge does **not** imply the coding agents themselves were blind to the project context; define the exact information boundary.
- **Independent audit (not equivalent):** one agent audits another's code against the contract, ideally without the original implementer's self-review. Useful even when no parallel implementation is done.

## Recommended trial tasks and timing

| Task | Mode | Why |
|---|---|---|
| M0 passport | One author + independent contract check | Documentation reconciliation; no need to duplicate code |
| M1 sampler | **One implementer (Morgana) + independent fixtures and audit (Virgil), no contest** | Simple bounded code; use S01–S05 and hand-computable expected values |
| M2 gradients | Independent audits/tests by default; consider parallel implementations only if useful and separately authorized | High consequence and hard-to-detect detach/edge errors; compare analytic vs autodiff vs finite differences |
| M3 Lion+W full step | One implementation + independent review by default; duplicate only if owner explicitly allocates budget | Avoid two large overlapping implementations before first run |
| M4 first training | One reference implementation + independent audit of loss, splits, updates and results | GPU/compute costly; no duplicate training just for contest |
| M5 checkpoint/replay | Ordinary implementer + tests/audit | Usually well captured by round-trip checks |
| M6 scientific comparison | **Blind experimental interpretation** of identical results after protocol freeze; not two expensive training campaigns by default | Reduce confirmation bias; agree on what the results do and do not establish |

M1/M2 retain mandatory technical gates from PR #5 independently of whether a contest is held. A contest never changes S/G/C/T/R/E test acceptance or blocks the main implementation solely for aesthetic preferences.

## Step-by-step protocol

1. **Preregister** one short task card: base SHA, exact document/contract SHAs, intended test IDs, expected API/outputs, implementation language/runtime, precise exclusions, time/token/tool/compute budgets, permitted dependencies, permitted external searches, start/stop conditions, and scoring rubric. Keep the instruction text **identical** for competitors except assignment IDs and access credentials. Fix seed and held-out tests before submissions; no tuning them to favour a candidate.
2. **Isolate** developers: separate working copies/branches or private staging areas, separate agent sessions and no cross-reading. Public GitHub PRs/branches are readable by competitors; a request to “not look” is a procedural promise, **not technical isolation**. For an actually strict blind contest, keep submissions out of the public repo until frozen and evaluated, using access-controlled staging or the owner's private local workspace. Do not copy secrets or repository credentials into submissions.
3. **Precommit independent fixtures** before implementation (especially M1/M2). Preserve signatures/hashes, creation times and sources. The organizer creates or approves a small independent reference suite, with hand-computable fixtures, pathological negative/wrap cases and held-out checks. Changes to tests after seeing submissions require a new logged evaluation round.
4. **Develop in isolation** under the same contract. Each agent records run-time/toolchain/package versions, tool uses, exact prompt and interventions, test commands, test results, limitations, changes from defaults and its own attribution. No access to another candidate's output. No modifying main, another's branch, CI protection or permissions without owner's order.
5. **Freeze** code, manifest, lockfiles, tests, seeds, measured metrics and provenance at a commit/hash. Do not compare moving branches. Strip author metadata, agent signatures, comments revealing model names, Slack handles, unique commit authors, and unrelated stylistic hints as much as possible; preserve separate sealed identity map for the organizer. Perfect anonymity is not guaranteed by formatting alone.
6. **Blind evaluation** of A/B: run both on the **same toolchain/hardware** and preregistered correctness tests. Judges inspect anonymized source and behaviour without being told the author. A neutral coordinator checks correctness and discloses high-severity failures before subjective ranking. A second independent evaluator is preferable for delicate contracts.
7. **Freeze scores** before opening identity map. Reveal authors only after judges have written assessments. Keep a complete non-anonymized provenance audit privately; publish a balanced report with failed tests, costs and constraints, not only the winning code. Owner chooses one implementation, merges compatible portions after independent review, or rejects both.
8. **Avoid automatic conclusions:** passing functional tests is not proof of mathematical correctness, especially gradients; paired differences at one seed are anecdotal. If both candidates fail the same contract, investigate shared assumptions instead of awarding a winner by majority vote.

## Scoring rubric (fixed for each competition)

**First an eligibility gate, not points:** no data leaks, forbidden dependencies, unapproved write operations or known failing safety/correctness invariants; for M1 and M2, required S/G tests must pass before a candidate is eligible for integration.

| Criterion | Points | Evaluation |
|---|---:|---|
| Correctness and robustness on independent/held-out tests | **50** | Required tests, boundary conditions, analytic consistency; failures documented |
| Reproducibility and diagnostic quality | **15** | Setup instructions, seeds, provenance, useful diagnostics, honest NOT_RUN |
| Clarity, maintainability and minimality | **15** | Clear dataflow, explicit contracts, few unnecessary abstractions |
| Computational behaviour | **10** | Real observed latency, memory and build/test friction on same machine; if unmeasured label N/A and do not award speculative points |
| Human developer experience | **10** | Owner's *blind* impression: legibility, usability, ease of explaining code and debugging; score without author labels |

**Fairness rule:** different model budgets, prompting assistance, tool privileges or hardware invalidate strict head-to-head claims unless explicitly matched. Record costs and time separately from quality; a slower or more expensive candidate can still be technically better. If a metric is N/A, rescale only via a preregistered policy, never after identifying a winner. Practical tie/uncertainty is a valid result.

Optional playful honours: `Best Correctness`, `Best Explainability`, `Most Elegant Minimalism` and `Most Interesting Failure`. No winner is required.

## Handling PR #10 difficulty tiers

PR #10 proposes A/B/C difficulty tiers based on the cost of silent errors. The parallel/blind mode here is **orthogonal**, not a fourth difficulty tier: one can choose `A + blind independent development`, `A + independent audit`, or `B + regular tests` depending on risk, budget and purpose. Do **not** hard-code a model brand as universally strongest; user chooses models/effort configurations at each trial. The original strongest-model preference in PR #10 remains a proposal, not a proven ranking.

The owner chooses when a blind competition is worth doing. **M1 is explicitly excluded from competitions** under the 2026-11-04 decision; its ordinary independent code review is not counted as a league event. M2 or later high-risk portions may be considered only if two independent implementations would materially improve confidence. Avoid a full-framework competition and do not route the contest artifacts into the model's W/h/p/r runtime.

## Status

No such independent coding contest or blind evaluation has been performed as of this document. No Swift/MLX implementation has been authorized or tested; all M1–M7 runtime results are **NOT_RUN**. Existing Slack discussions and Morgana's CPU NumPy check are valuable reviews/diagnostics, not entries in a blind implementation competition.
