# OFT trajectory implementation plan

> **For agentic workers:** Use test-driven-development and scoped parallel implementation; root integrates and dispatches independent review.

**Goal:** Record and annotate existing OFT LIBERO evaluations without changing inference.
**Architecture:** Atomic per-step files, evaluation instrumentation, evidence-based labels, offline review export.
**Tech Stack:** Python >=3.10, NumPy, Pillow, pytest; reuse installed OFT/LIBERO.
**Spec:** ../specs/2026-11-03-oft-trajectory-design.md

## Global Constraints
- Existing OFT environment remains untouched; no GPU libraries imported by core package.
- Explicit unknown/failed/interrupted outcomes; never equate arbitrary done with success.
- Data are excluded from git; no credentials in files.
- Original private proposal and A100 runtime are not available for verification.

## Review Focus
- In-place observation/action mutation and action chunk boundaries.
- Incomplete episode files and failed atomic writes.
- Predicate unknown/initial truth/reversal and no causal interpretation.
- Manual annotation lost-update protection and malformed import.
- Unsafe embedded text and huge preview memory usage.

### Task 1: Recorder
Files: vla_trajectory/recorder.py, tests/test_recorder.py.
Interfaces: implement storage contract in spec.
- [x] Write/run failing tests for alignment, metadata JSON validation, finite numeric arrays, duplicate IDs, interrupted/empty episodes and corrupt/incomplete reads.
- [x] Implement atomic compressed NPZ + step JSON then manifest commit; finish semantics and reader validation.
- [x] Run focused tests; report evidence.

### Task 2: Annotations
Files: vla_trajectory/annotations.py, tests/test_annotations.py.
Interfaces: consume recorder read functions; implement annotation contract in spec.
- [x] Write/run failing tests for initial/achieved/reverted/unknown predicates, inclusive ranges, wrong episode, revision conflicts and manual evidence preservation.
- [x] Implement generation, validation, atomic save/locking and summaries.
- [x] Run focused tests; report evidence.

### Task 3: Offline review
Files: vla_trajectory/review.py, tests/test_review.py.
Interfaces: consume recorder + annotations functions; export_review(path, output).
- [x] Write/run failing tests for embedded previews, safe text and sparse/no-camera episodes.
- [x] Build self-contained HTML with scrubber, dual camera, events, segment editor, download JSON and visible save instructions.
- [x] Run focused tests; report evidence.

### Task 4: OFT integration and delivery
Files: vla_trajectory/oft.py, libero.py, cli.py, demo.py, __init__.py, __main__.py; tests/test_oft.py; pyproject.toml; README.md; docs/; scripts/; .github/workflows/.
- [x] Verify upstream functions and write/run failing fake evaluator tests.
- [x] Instrument run_episode, get_action, process_action and env via proxy; capture predicate and simulator state lazily, restore hooks on exit.
- [x] Add CLI/demo, A100 doctor, install/run instructions and opt-in private GitHub creation/upload script.
- [x] Run complete tests, demo -> annotate -> export -> import -> summary; review and fix findings.
- [x] Attempt authorized publication only through available access, otherwise report precise blocker and ready-to-publish files.
