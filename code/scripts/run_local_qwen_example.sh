#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -m pollution_eval.cli.evaluate \
  --backend local-qwen \
  --input-dir ../datasets \
  --output-dir ../runs/qwen_local \
  --label-space configs/label_space.json \
  --checkpoint-dir "${CHECKPOINT_DIR:?set CHECKPOINT_DIR}" \
  --base-model "${QWEN_BASE_MODEL:-Qwen/Qwen3-8B}" \
  --max-length 1024 \
  --batch-size "${BATCH_SIZE:-1}" \
  --dtype "${DTYPE:-bf16}" \
  --load-in-4bit \
  --trust-remote-code
