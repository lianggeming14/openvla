# Six-task instruction-order experiment

The user has authorized local setup and evaluation with the original
`moojink/openvla-7b-oft-finetuned-libero-10` checkpoint, 4-bit loading,
`libero_10`, and one trial per selected initial state. Six tasks each have
AB and BA instructions, each with twenty distinct official states. The same
state is paired across instructions, for 240 real model episodes in total.

The case registry and resolved state fingerprints have already been verified
against local LIBERO. They live under `.work/order-experiment-2026-11-20/`.
Simulation preflight is not an experiment result. No checkpoint has yet been
downloaded; local outbound connections currently fail with WinError 10013.

## Implementation choices

- Extend the existing reversible OFT instrumentation with a keyword-only
  `instruction_override` option. Replace the bound episode description before
  invoking OFT. Record the original description and exact query instruction,
  with SHA256 hashes. Verify every actual model query matches the episode
  instruction. Record query metadata in a separate atomic JSON file.
- Add optional episode metadata selection/callback only if required by the
  runner; avoid changing existing collection behavior.
- Add an order-report module that reads recorded simulator predicates and
  classifies the first observed false-to-true transition for each of two
  subgoals. Keep ties, missing evidence, initially true goals, interrupted
  recordings, and unobserved transitions explicit. Uncertain transitions must
  never become definite AB or BA. A proven observed sequence can remain visible
  in an interrupted trace, alongside its error status and null trial compliance.
  Official task success remains separate.
- Grasp order requires separate verified evidence/manual video annotation;
  placement predicates cannot establish grasp order. Group six compares stove
  activation with pot placement, and has no two-object grasp comparison.
- Add a batch script using the existing collect interface, with separate
  condition directories and persisted plans. It runs one selected task and
  condition per evaluator invocation, so model memory is released between
  invocations. It must never mark planned rows as completed. Conservative
  resume skips only verified complete recordings, matching protocol/config
  identity, initial-state fingerprint, query instruction, and video evidence.
  Resolve all six state banks before either condition starts, and reject later
  selected-state changes. Fingerprint effective inference configuration while
  excluding OFT's generated backups; the OFT source separately identifies the
  model/configuration modules it copies into the checkpoint directory.

## Delivery and validation

CPU tests prove prompt transport, recording, classification and resume
bookkeeping only. The existing CUDA and six-scene preflight do not prove model
inference. After networking is restored, install compatible inference
dependencies without downgrading the working sm120 PyTorch, fetch the original
weights, validate a real model episode, and then run the batch.

Final real episode and task/condition tables belong in private
`lianggeming14/upload-files`. Code and formal usage documentation belong in
public `lianggeming14/openvla`; private requirements, weights, caches, raw
trajectories and credentials must not be included. Feishu delivery still lacks
a usable connection and recipient.
