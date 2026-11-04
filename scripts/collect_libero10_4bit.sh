#!/usr/bin/env bash
# Run in the same activated conda environment used for the original OFT command.
set -euo pipefail
if (( $# > 2 )); then
  echo "Usage: bash collect_libero10_4bit.sh [OFT_ROOT] [OUTPUT_DIR]" >&2
  exit 2
fi
OFT_ROOT="${1:-$PWD}"
TRACE_OUTPUT="${2:-$OFT_ROOT/trajectory-data/libero10-4bit}"
exec "${VLA_TRACE_PYTHON:-python}" -m vla_trajectory collect \
  --oft-root "$OFT_ROOT" \
  --output "$TRACE_OUTPUT" \
  -- \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --num_trials_per_task 1 \
  --load_in_4bit True
