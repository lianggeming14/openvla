# Instruction-order batch implementation plan

> **For agentic workers:** Use subagent-driven-development for the independent
> prompt transport and report tasks; the controller integrates the runner and
> verifies the complete suite. User has explicitly requested autonomous work.

**Goal:** Prepare a verifiable 240-episode experiment without inventing results.

**Architecture:** Reuse OFT recording, add verified instruction override, and
classify simulator completion events separately from grasp annotations. Keep
each condition in its own directory and preserve the exact official state bank.

**Tech stack:** Python >=3.10, NumPy, pytest, existing OFT/LIBERO interfaces.

**Spec:** `docs/superpowers/specs/2026-11-23-order-batch-design.md`

## Global constraints

- Preserve the standard checkpoint, libero_10, 4-bit loading and one trial/state.
- Twenty states per condition, paired across AB/BA, six groups, 240 episodes.
- No network or filesystem restriction bypass. No synthetic experiment claims.
- `.git` is read-only. Snapshot touched files to `.work` before changes; do not
  attempt a git worktree or local commit requiring prohibited metadata writes.
- Public source publication and private result publication are already allowed.

## Review focus

- Positional and keyword model-query calls must receive the same exact prompt.
- Upstream may swallow inference exceptions; retain an error manifest.
- An initially true or unobserved goal is not a newly completed subtask.
- Final task success does not imply instruction-order compliance or grasp order.
- Interrupted/corrupt/mismatched prior episodes must not be skipped on resume.

## Task 1: Verified instruction transport

Files: `vla_trajectory/oft.py`, `vla_trajectory/cli.py`, `tests/test_oft.py`,
`tests/test_cli.py`.

- [x] Write failing tests for actual prompt override, original text provenance,
  exact query text/hash recording, mismatched query errors and restoration.
- [x] Run the focused tests, observing the expected failures.
- [x] Add keyword-only `instruction_override=None` to `instrument`, and
  `collect --instruction` before the upstream argument separator.
- [x] Persist `queries.json` with `schema_version`, `episode_id`, and `queries`
  rows (`chunk_id`, `instruction`, `instruction_sha256`). Include query text in
  committed query-step context. Validate override input before starting.
- [x] Run focused tests and review changes independently.

## Task 2: Completion-order report

Files: `vla_trajectory/order_report.py`, `tests/test_order_report.py`.

- [x] Write failing tests for AB, BA, ties, missing predicates, initially true,
  interrupted episodes and distinct success/order measures.
- [x] Implement `classify_completion_order(steps, subgoal_keys)` returning
  `actual_completion_order`, `first_A_step`, `first_B_step`, `evidence_status`
  and `reason`. Use contiguous false-to-true evidence only.
- [x] Implement `episode_order_result(episode, *, subgoal_keys,
  expected_order)` returning outcome, completeness and compliance separately.
- [x] Implement a CSV/JSON export of actual recorded episodes only, grouped by
  task and condition; no planned/unrun rows in result exports. Grasp order is
  unknown unless independently supported by existing manual annotations.
- [x] Run focused tests and review changes independently.

## Task 3: Batch orchestration and local acceptance

Files: `scripts/run_order_experiment.py`, `tests/test_order_batch.py`, optional
small package helper if imports warrant it, and usage documentation.

- [x] Inspect Tasks 1/2 interfaces; write failing orchestration tests before code.
- [x] Validate registry/state schedule and create condition plans with stable
  config identity; invoke existing collector through the isolated Python.
- [x] Resume only verified complete episodes matching all experiment identities;
  preserve failed attempts. Fail closed on state bank changes or stale prompts.
- [x] Export classified tables from committed recordings; never label planned
  rows as results. Avoid a broad release until real upstream compatibility is
  tested, and clearly label the CPU-only validation boundary.
- [x] Run full pytest and independent review: 312 passed, 2 optional dependency
  skips. Verify the candidate wheel in an isolated target and the source ZIP.
- [ ] Restore effective network access, fetch the original weights, run a real
  smoke episode and the remaining paired batch, then publish real result tables.

## Execution evidence

- Official OFT query signature verified as `get_action(cfg, model, obs, task_label, ...)`; both official and compatibility aliases have transport tests.
- All twelve cases resolved through the actual local LIBERO reader: 240 planned rows; paired state fingerprints match. Numeric state loading tested on installed PyTorch 2.9.
- Independent review fixes: stable effective checkpoint identity across OFT-generated backups; shared pre-resolved state fingerprints; scoped numeric NumPy allowlist. Re-review PASS.
- One intermediate Windows file-sharing `WinError 32` in atomic recorder replacement did not recur on the focused rerun or full suite. No recorder workaround was introduced; real smoke acceptance must monitor for recurrence.
- Current real policy rollouts: 0. HF/PyPI connections still fail with WinError 10013 despite saved user config. No real result table has been uploaded.
