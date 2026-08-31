#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -m pollution_eval.cli.evaluate \
  --backend api \
  --input-dir ../datasets \
  --output-dir ../runs/api_model \
  --label-space configs/label_space.json \
  --api-base-url "${API_BASE_URL:-https://api.openai.com/v1}" \
  --api-model "${API_MODEL:?set API_MODEL}" \
  --api-key-env "${API_KEY_ENV:-OPENAI_API_KEY}" \
  --api-workers "${API_WORKERS:-1}"
