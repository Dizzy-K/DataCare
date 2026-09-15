# DataCARE-CN

English | [简体中文](README.zh-CN.md)

DataCARE-CN is a project for **hierarchical data pollution detection and evaluation** in text and conversations used by large language models. Its four-level label schema (L1–L4) identifies whether a sample is polluted and describes its pollution family, subcategory, and specific type, providing a shared basis for data quality analysis, detector comparison, and error analysis.

The current release includes **20,000 public test samples, a four-level label schema, and a Python evaluation toolkit**, covering bias and semantic incompleteness. The toolkit supports model APIs, local Qwen models with four classification heads, and rescoring existing predictions. The repository currently includes neither model training code nor pretrained or fine-tuned weights.

## Features

- **Hierarchical, multilabel evaluation**: L1 detects pollution, while L2–L4 describe increasingly specific types. A sample can have multiple pollution labels.
- **Three evaluation backends**: evaluate models through an OpenAI-compatible API, load a local Qwen + LoRA checkpoint with four classification heads, or compute metrics from existing predictions.
- **Inspectable evaluation outputs**: save individual predictions, binary and multilabel metrics, false positives and false negatives, errors at each label level, and run metadata.
- **Batch API evaluation**: concurrent requests, retries, and resumable evaluation keyed by sample ID.

## Repository layout

```text
DataCARE-CN/
├── README.md                       # English documentation
├── README.zh-CN.md                 # Chinese documentation
├── datasets/
│   ├── README.md                    # Dataset documentation
│   ├── white_test.jsonl             # Clean test samples
│   └── pollution_test.jsonl         # Polluted test samples
└── code/
    ├── README.md                    # Detailed toolkit documentation (English)
    ├── pyproject.toml               # Python package configuration and dependencies
    ├── environment-open-eval.yml    # Optional Conda environment configuration
    ├── configs/label_space.json     # Label names, IDs, and default thresholds
    ├── pollution_eval/              # Data processing, inference backends, and metrics
    ├── scripts/                     # API/local inference examples and local server script
    └── tests/                       # Existing tests
```

## Label schema

| Level | Meaning | Current labels |
| --- | --- | --- |
| L1 | Whether pollution is present | `非污染样本` (clean), `污染样本` (polluted) |
| L2 | Pollution family (2 classes) | `偏见类` (bias), `语义缺失类` (semantic incompleteness) |
| L3 | Pollution subcategory (4 classes) | `群体受害者维度`, `价值观偏见`, `结构与语法残缺`, `逻辑与上下文断层` |
| L4 | Specific pollution type (20 classes) | Examples include `民族偏见`, `性别偏见`, `句式截断`, and `逻辑错误` |

See [`label_space.json`](code/configs/label_space.json) for the full label vocabulary and IDs, and [`constants.py`](code/pollution_eval/constants.py) for category definitions. When training or integrating your own model, preserve the Chinese label strings and align classification head output ordering with the label configuration.

L2–L4 are empty for clean samples and may contain multiple labels for polluted samples. Both the binary and multilabel thresholds default to `0.5`; these thresholds decode probabilities in the local Qwen backend.

## Datasets

| File | Samples | Description |
| --- | ---: | --- |
| [`white_test.jsonl`](datasets/white_test.jsonl) | 10,000 | Clean samples with `null` values for L2–L4 |
| [`pollution_test.jsonl`](datasets/pollution_test.jsonl) | 10,000 | Polluted samples with L2–L4 annotations |
| Total | **20,000** | For evaluation and reproducibility |

The current test sets cover 2 L2 classes, 2 L3 classes, and 12 L4 classes, rather than the entire label schema. There are 6,000 bias samples and 4,000 semantic incompleteness samples, with L3 labels `群体受害者维度` and `结构与语法残缺`, respectively. Evaluating the remaining categories requires additional test samples.

Files use UTF-8 JSONL, with one JSON object per line. The following illustrates the format of a clean sample:

```json
{
  "id": "example_clean_001",
  "messages": [
    {"role": "user", "content": "请计算 1 加 1。"},
    {"role": "assistant", "content": "1 加 1 等于 2。"}
  ],
  "label_l1": "非污染样本",
  "label_l2": null,
  "label_l3": null,
  "label_l4": null
}
```

`id` aligns samples with predictions and should be unique. `messages` contains the conversation to inspect; custom inputs can also use `query` / `answer` or `text`. L2–L4 annotations for polluted samples can be strings or arrays of strings.

Both files are test sets. Prepare separate training and validation data for training or tuning, keeping the test sets independent. See [`datasets/README.md`](datasets/README.md) for more field details.

## Quick start

### 1. Install

Requires **Python 3.10 or newer**. The following commands use a Linux / macOS shell:

```bash
git clone https://github.com/Dizzy-K/TrainGuard.git DataCARE-CN
cd DataCARE-CN
python3 -m venv .venv
source .venv/bin/activate
cd code
python -m pip install -e .
```

The base installation supports API evaluation and prediction rescoring. Run all subsequent commands from `DataCARE-CN/code/` using this Python environment.

### 2. Prepare data and check labels

This step does not call a model and requires neither a GPU nor an API key:

```bash
python -m pollution_eval.cli.prepare_dataset \
  --input-dir ../datasets \
  --output-dir ../runs/manifest \
  --label-space configs/label_space.json \
  --unknown-label-policy error
```

Outputs include the combined `eval_all.jsonl`, the label configuration, and a `manifest.json` containing sample counts and label distributions. The `error` policy raises an error if an annotation is outside the label space. This step is useful for checking custom data; evaluation commands can also read the original test files directly.

### 3. Evaluate a model through an API

Set your key in an environment variable and replace `YOUR_MODEL` with a model name available from your provider. Change `--api-base-url` when using another compatible service:

```bash
export OPENAI_API_KEY="YOUR_API_KEY"

python -m pollution_eval.cli.evaluate \
  --backend api \
  --input-dir ../datasets \
  --output-dir ../runs/api_smoke \
  --label-space configs/label_space.json \
  --api-base-url https://api.openai.com/v1 \
  --api-model YOUR_MODEL \
  --api-key-env OPENAI_API_KEY \
  --api-workers 4 \
  --limit 20
```

This example processes only the first 20 loaded samples to check the interface and output format; it does not represent performance on the full test sets. For full evaluation, remove `--limit 20` and use a new output directory, such as `../runs/api_full`.

The toolkit automatically builds detection prompts containing label definitions. The model should return JSON in the following format, using labels from the released Chinese label space:

```json
{
  "label_l1": "污染样本",
  "label_l2": ["偏见类"],
  "label_l3": ["群体受害者维度"],
  "label_l4": ["民族偏见"]
}
```

For clean predictions, L2–L4 should be empty arrays. The optional `reason` field is not scored.

The default endpoint is `chat/completions`; add `--api-type responses` to use the Responses endpoint. If a compatible service does not support `response_format`, add `--no-response-format`. The model must still return parseable JSON.

By default, API evaluation reads `api_raw_predictions.jsonl` in the same output directory and reuses existing results by sample ID. **Use a new output directory when changing the model, data content, or inference settings** to avoid reusing old predictions. By default, the run stops if an API request fails after retries are exhausted.

## Other evaluation modes

### Local Qwen model with four classification heads

Install the optional local inference dependencies first:

```bash
python -m pip install -e ".[local-qwen]"
```

This backend requires a checkpoint matching the code's **Qwen backbone + LoRA + four L1–L4 classification heads** architecture. `--checkpoint-dir` must contain `model.safetensors`. If that directory contains `tokenizer.json`, its tokenizer is used; otherwise, the tokenizer is loaded from the base model. Downloading a standard Qwen chat model alone is insufficient for this backend, and the repository does not currently include the required checkpoint.

Once compatible weights are available, run the following in a GPU environment supporting the selected quantization and precision:

```bash
python -m pollution_eval.cli.evaluate \
  --backend local-qwen \
  --input-dir ../datasets \
  --output-dir ../runs/qwen_local \
  --label-space configs/label_space.json \
  --checkpoint-dir /path/to/checkpoint \
  --base-model Qwen/Qwen3-8B \
  --dtype bf16 \
  --load-in-4bit \
  --batch-size 1 \
  --max-length 4096
```

The base model must match the one used to train the checkpoint. The loader's default LoRA configuration uses `r=16` and `alpha=32`; see [`local_qwen.py`](code/pollution_eval/backends/local_qwen.py) for the architecture. `--max-length` limits the complete input, including task instructions and the conversation, and longer inputs are truncated. Choose this value based on data length and available GPU memory, and record it when reporting results. The 4-bit and 8-bit quantization flags cannot be enabled together.

### Rescore existing predictions

Recompute metrics from a prediction file without calling a model:

```bash
python -m pollution_eval.cli.evaluate \
  --backend predictions \
  --input-dir ../datasets \
  --output-dir ../runs/rescore \
  --label-space configs/label_space.json \
  --predictions-file /path/to/eval_predictions.jsonl
```

Supported formats are JSONL, a JSON array, or a JSON object containing a `records` array. Each prediction must include an `id` matching an evaluation sample. Label fields can use either `label_l1`–`label_l4` or `pred_label_l1`–`pred_label_l4`. A missing prediction for any evaluated ID raises an error. If predictions cover only a subset, select the corresponding inputs with `--eval-files` or the same `--limit`.

### Common options

| Option | Purpose |
| --- | --- |
| `--eval-files FILE [FILE ...]` | Select one or more JSONL files; takes precedence over `--input-dir` |
| `--limit N` | Evaluate only the first N loaded samples, without random or stratified sampling |
| `--unknown-label-policy skip\|error` | Handle unknown L2–L4 annotations; defaults to skipping and recording affected samples |
| `--binary-threshold` / `--multilabel-threshold` | Adjust local Qwen probability decoding thresholds; does not decode existing discrete predictions again |
| `--api-workers` | API concurrency; defaults to `1` |
| `--no-api-resume` | Do not read existing prediction caches for this API run |

Run `python -m pollution_eval.cli.evaluate --help` for the full option list.

## Evaluation outputs and metrics

All results are written to the directory specified by `--output-dir`:

| File | Contents |
| --- | --- |
| `eval_metrics.json` | Binary, multilabel, and joint accuracy metrics |
| `eval_predictions.json` / `eval_predictions.jsonl` | Per-sample annotations, predictions, and correctness; the JSON version also includes metrics and a summary |
| `badcase_summary.json` | Error counts, label distributions, and summaries by input file |
| `false_positives.jsonl` / `false_negatives.jsonl` | False positives and false negatives |
| `l2_errors.jsonl` / `l3_errors.jsonl` / `l4_errors.jsonl` | Samples whose gold and predicted L1 labels are both polluted, but whose labels differ at the corresponding level |
| `joint_errors.jsonl` | All samples that fail the joint correctness criterion |
| `skipped_unknown_labels.jsonl` | Samples skipped because their annotations contain unknown labels |
| `label_space.json` / `run_metadata.json` | Label configuration, thresholds, and backend run information |
| `api_raw_predictions.jsonl` | Raw API responses and cached predictions used to resume evaluation |

Metric definitions:

- **L1 binary classification**: treats `污染样本` as the positive class and reports Accuracy, Precision, Recall, F1, FPR (false positive rate), and FNR (false negative rate).
- **L2–L4 multilabel classification**: computes Micro-F1, Macro-F1, and Exact Match on samples whose gold L1 label is polluted. Macro-F1 averages only over labels with positive gold examples in the evaluated subset. Exact Match requires identical label sets at that level.
- **`joint_accuracy`**: a clean sample requires only a correct L1 prediction; a polluted sample requires a correct L1 prediction and exact label-set matches at L2, L3, and L4.
- **`polluted_joint_accuracy`**: joint accuracy computed only on samples whose gold L1 label is polluted.

To make results comparable, record the model or checkpoint, test files, sample count, thresholds, and either the local inference truncation length or the API inference settings.

## Development and feedback

Run the existing test suite:

```bash
python -m unittest discover -s tests
```

Use [Issues](https://github.com/Dizzy-K/TrainGuard/issues) to report annotation problems, API compatibility issues, or evaluation errors. Pull requests are also welcome. Include a reproducible command, environment versions, and relevant sample IDs, and remove API keys or other credentials from reports.
