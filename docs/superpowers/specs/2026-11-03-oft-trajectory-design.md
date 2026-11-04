# OpenVLA-OFT + LIBERO trajectory tools

User confirmed OpenVLA-OFT with LIBERO on one A100. The original private proposal is not readable in this session; this implementation is based on that confirmed environment, not claimed to reproduce the inaccessible proposal. No A100 connection is available. The tools do not upgrade model dependencies or edit OFT sources; delegated upstream evaluation retains its own checkpoint initialization side effects.

## Deliverable and decisions

An independent Python package `vla_trajectory` in repository `openvla`, with no torch/CUDA dependencies. NumPy and Pillow only at runtime. A launcher imports the user's existing OFT evaluation module and installs reversible instrumentation, reusing its model, processing, action queue and task loop. The launcher must fail clearly if expected functions are absent. No copied OFT inference implementation.

1. Per-episode append-only transitions: compressed NPZ arrays (no pickle), JSON step records, atomic writes, complete/error/interrupted status. Keep action before processing and action actually executed, raw current/next observations, inference input on chunk queries, simulator state where available, metadata (checkpoint, config, seed, task, initial state), phase (settling/policy), chunk ID and offset. Avoid copying GPU weights or dataset files. Store all outcomes; success is explicit LIBERO success, never inferred merely from termination.
2. Automatic labels are predicate truth transitions with provenance, including initially true goals and reversals. Missing predicate evidence is unknown. Summaries report observed first-achievement order and final truth, never assert causal/task-required order.
3. A standalone offline HTML review page with two camera views, frame navigation, predicate/action display, editable inclusive frame segments, outcome/notes and JSON download. Import validates episode binding, range, strings and revision; originals remain immutable. No web server needed.
4. CPU demo, tests, Chinese installation/integration and A100 acceptance instructions, packaging and a GitHub publishing script. Actual remote creation/push remains contingent on authorized network and GitHub authentication.

## Shared on-disk/API contract

Episode directory contains `manifest.json`, `steps/000000.json`, `steps/000000.npz` ...; immutable committed steps. `manifest` fields: schema_version=1, episode_id, metadata, status (recording/complete/error/interrupted), num_steps, outcome (success/failure/unknown), error (optional). Root may contain many episode directories.

`EpisodeRecorder(root, metadata, episode_id=None)` exposes `.path`, `.manifest`, `append(observation, action, next_observation, *, reward=0.0, terminated=False, truncated=False, context=None, arrays=None)` and `finish(outcome='unknown', status='complete', error=None)`. `append` validates and returns integer step index. `context` JSON contains phase, chunk_id/index, predicates_before/after (mapping string->bool|null), raw action/inference time as needed. `arrays` extra arrays mapping. NPZ keys `obs/<key>`, `next_obs/<key>`, `action`, `extra/<key>`. Step JSON has index, timestamp_ns, reward, terminated, truncated, context, array_file. `.finish` idempotent only for identical result; append after finish is forbidden. `read_manifest(path)`, `read_steps(path)` (list of JSON dicts), `load_arrays(path,index)` (dict of arrays). Manifest count bounds committed steps; a recording status is never silently interpreted complete.

`generate_annotations(episode_path)` returns dict schema_version=1, episode_id, num_steps, source='libero_predicates' (or 'synthetic_demo' when episode metadata.synthetic is true), events list, segments list, outcome, notes, revision=0. Events use frame (step index), predicate, value, kind (initial/achieved/reverted/unknown), source. Segments have start,end,label,notes (inclusive). `save_annotations(path, document, expected_revision=None)` atomically saves `annotations.json`, validates and increments revision with process-safe lock; `load_annotations(path)` reads existing or generates in memory; `summarize_annotations(document)` returns JSON-safe order/final-state summary. Human import must preserve automatic event evidence and set manual fields only (segments,outcome,notes). `export_review(episode_path, output_html)` embeds data and previews and downloads an annotation document with base_revision; import CLI calls `save_annotations(... expected_revision=base_revision)`.

## Verification boundaries

Annotations may be generated or loaded in memory for a stable recording snapshot, but saving requires an episode with terminal status (`complete`, `error`, or `interrupted`). Evidence reads reject a manifest change during the read so annotations never combine different committed snapshots.

Tests must exercise alignment with in-place-mutating fake environment, chunk metadata and warmup, failure/timeout/interruption, missing predicates, event reversals, corruption/invalid inputs, annotation conflict and script injection. CPU fake environment is explicit demo evidence only. Real OFT evaluation + A100 run and private proposal reconciliation remain required acceptance steps.
