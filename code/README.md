# TrainGuard Evaluation Toolkit

This package evaluates hierarchical data-pollution detection models. It provides:

- `local-qwen`: inference with a Qwen backbone, LoRA weights, and four classification heads (L1–L4).
- `api`: evaluation through an OpenAI-compatible `chat/completions` or Responses API.
- `predictions`: rescoring an existing prediction file without model inference.

The released label order and thresholds are in [`configs/label_space.json`](configs/label_space.json). Label values intentionally remain in Chinese (for example, `非污染样本` and `污染样本`).

## Installation

Run the commands below from this directory:

```bash
python -m pip install -e .
```

For the local Qwen backend, install the optional dependencies:

```bash
python -m pip install -e ".[local-qwen]"
```

The API backend only requires the base dependencies. Python 3.10 or newer is supported.

## Dataset format

Input files are UTF-8 JSONL. Each record contains:

```json
{
  "id": "sample_1",
  "messages": [{"role": "user", "content": "..."}],
  "label_l1": "污染样本",
  "label_l2": "偏见类",
  "label_l3": "群体受害者维度",
  "label_l4": "民族偏见"
}
```

`messages` may be replaced by `query`/`answer` or `text`. For `非污染样本`, `label_l2`–`label_l4` are `null` (or empty arrays in model predictions). Pollution labels may be strings or arrays.

## Quick start

The public files in `../datasets` can be prepared into an evaluation manifest:

```bash
python -m pollution_eval.cli.prepare_dataset \
  --input-dir ../datasets \
  --output-dir ../runs/manifest \
  --label-space configs/label_space.json
```

Evaluate an OpenAI-compatible model:

```bash
export OPENAI_API_KEY=YOUR_KEY
python -m pollution_eval.cli.evaluate \
  --backend api \
  --input-dir ../datasets \
  --output-dir ../runs/api_model \
  --label-space configs/label_space.json \
  --api-base-url https://api.openai.com/v1 \
  --api-model YOUR_MODEL \
  --api-workers 4
```

Evaluate a Qwen checkpoint (provide paths available in your environment):

```bash
python -m pollution_eval.cli.evaluate \
  --backend local-qwen \
  --input-dir ../datasets \
  --output-dir ../runs/qwen_local \
  --label-space configs/label_space.json \
  --checkpoint-dir /path/to/checkpoint \
  --base-model Qwen/Qwen3-8B \
  --dtype bf16 \
  --load-in-4bit
```

Rescore existing predictions:

```bash
python -m pollution_eval.cli.evaluate \
  --backend predictions \
  --input-dir ../datasets \
  --output-dir ../runs/rescore \
  --label-space configs/label_space.json \
  --predictions-file /path/to/predictions.jsonl
```

Use `--eval-files` to evaluate a selected JSONL file and `--limit N` for a smoke test. API keys should be supplied through environment variables.

For the `api` backend, the model must return a JSON object with `label_l1`, `label_l2`, `label_l3`, and `label_l4` using the Chinese label strings from the released label space. `reason` is optional and is not scored.

## Outputs and metrics

Each run writes metrics, predictions, error lists, and run metadata to its output directory. L1 reports binary accuracy, precision, recall, F1, FPR, and FNR. L2/L3/L4 report multilabel micro/macro F1 and exact match on pollution samples. `joint_accuracy` requires all applicable levels to match.

Run the test suite with:

```bash
python -m unittest discover -s tests
```
