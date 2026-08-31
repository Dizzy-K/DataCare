from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..backends.local_qwen import predict_local_qwen
from ..backends.openai_compatible import predict_api
from ..io_utils import api_key_from_env, default_public_data_dir, find_eval_files, iter_jsonl, load_eval_rows, load_label_space, make_artifacts_readable, read_json, write_json, write_jsonl
from ..metrics import build_records, compute_metrics, split_badcases, summarize_records
from ..schema import filter_supported_rows, prediction_from_mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate hierarchical pollution classifiers on JSONL datasets.")
    parser.add_argument("--backend", choices=["local-qwen", "api", "predictions"], required=True)
    parser.add_argument("--eval-files", type=Path, nargs="*", default=None, help="One or more JSONL files to evaluate.")
    parser.add_argument("--input-dir", type=Path, default=None, help="Directory of JSONL files. Defaults to the repo public release and is ignored when --eval-files is set.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label-space", type=Path, default=None, help="Optional label_space.json. Defaults to the released Qwen label space.")
    parser.add_argument("--unknown-label-policy", choices=["skip", "error"], default="skip")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N rows after loading.")
    parser.add_argument("--binary-threshold", type=float, default=None)
    parser.add_argument("--multilabel-threshold", type=float, default=None)

    local = parser.add_argument_group("local-qwen backend")
    local.add_argument("--checkpoint-dir", type=Path, default=None, help="Directory containing model.safetensors and tokenizer files.")
    local.add_argument("--base-model", default=None, help="Base Qwen model path/name used to instantiate the backbone.")
    local.add_argument("--max-length", type=int, default=32000)
    local.add_argument("--batch-size", type=int, default=2)
    local.add_argument("--dtype", choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"], default=None)
    local.add_argument("--load-in-4bit", action="store_true")
    local.add_argument("--load-in-8bit", action="store_true")
    local.add_argument("--trust-remote-code", action="store_true")

    api = parser.add_argument_group("api backend")
    api.add_argument("--api-base-url", default="https://api.openai.com/v1")
    api.add_argument("--api-type", choices=["chat", "responses"], default="chat")
    api.add_argument("--api-model", default=None)
    api.add_argument("--api-key", default=None)
    api.add_argument("--api-key-env", default="OPENAI_API_KEY")
    api.add_argument("--api-timeout", type=float, default=120.0)
    api.add_argument(
        "--api-max-tokens",
        type=int,
        default=4096,
        help="Maximum output/reasoning tokens for the API model.",
    )
    api.add_argument("--api-temperature", type=float, default=0.0)
    api.add_argument(
        "--reasoning-effort",
        choices=["low", "high", "max"],
        default=None,
        help="Optional provider reasoning effort sent as reasoning_effort (low/high/max).",
    )
    api.add_argument(
        "--thinking",
        choices=["enabled", "disabled"],
        default=None,
        help="DeepSeek thinking mode. Defaults to disabled for DeepSeek model names; explicit values override it.",
    )
    api.add_argument("--api-retries", type=int, default=3)
    api.add_argument("--api-workers", type=int, default=1)
    api.add_argument("--no-api-resume", action="store_true")
    api.add_argument("--no-response-format", action="store_true")
    api.add_argument("--on-api-error", choices=["error", "mark-clean"], default="error")

    pred = parser.add_argument_group("predictions backend")
    pred.add_argument("--predictions-file", type=Path, default=None, help="JSONL or eval_predictions.json with prediction records.")
    return parser.parse_args()


def resolve_thinking(model: str, requested: str | None) -> str | None:
    if requested is not None:
        return requested
    return "disabled" if "deepseek" in model.lower() else None


def load_predictions(path: Path, rows: list[dict]) -> list[dict]:
    if path is None:
        raise ValueError("--predictions-file is required for --backend predictions")
    if path.suffix == ".json":
        payload = read_json(path)
        if isinstance(payload, dict) and isinstance(payload.get("records"), list):
            loaded = payload["records"]
        elif isinstance(payload, list):
            loaded = payload
        else:
            raise ValueError(f"Unsupported predictions JSON shape: {path}")
    else:
        loaded = list(iter_jsonl(path))
    by_id = {}
    for item in loaded:
        row_id = str(item.get("id", ""))
        if not row_id:
            continue
        by_id[row_id] = prediction_from_mapping(item.get("prediction") or item)
    predictions = []
    for row in rows:
        row_id = str(row.get("id", ""))
        if row_id not in by_id:
            raise ValueError(f"Missing prediction for id={row_id} in {path}")
        predictions.append(by_id[row_id])
    return predictions


def write_outputs(output_dir: Path, label_space: dict, skipped_rows: list[dict], rows: list[dict], predictions: list[dict], metadata: dict) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = build_records(rows, predictions)
    metrics = compute_metrics(records, label_space)
    summary = summarize_records(records, skipped_rows)
    badcases = split_badcases(records)

    write_json(output_dir / "label_space.json", label_space)
    write_jsonl(output_dir / "skipped_unknown_labels.jsonl", skipped_rows)
    write_json(output_dir / "eval_metrics.json", metrics)
    write_json(output_dir / "badcase_summary.json", summary)
    write_json(output_dir / "eval_predictions.json", {"metrics": metrics, "summary": summary, "records": records})
    write_jsonl(output_dir / "eval_predictions.jsonl", records)
    for name, badcase_rows in badcases.items():
        write_jsonl(output_dir / f"{name}.jsonl", badcase_rows)
    write_json(
        output_dir / "run_metadata.json",
        {
            **metadata,
            "evaluated_count": len(rows),
            "skipped_unknown_label_count": len(skipped_rows),
            "output_files": {
                "metrics": "eval_metrics.json",
                "summary": "badcase_summary.json",
                "predictions_json": "eval_predictions.json",
                "predictions_jsonl": "eval_predictions.jsonl",
                "false_negatives": "false_negatives.jsonl",
                "false_positives": "false_positives.jsonl",
                "l2_errors": "l2_errors.jsonl",
                "l3_errors": "l3_errors.jsonl",
                "l4_errors": "l4_errors.jsonl",
                "joint_errors": "joint_errors.jsonl",
            },
        },
    )
    make_artifacts_readable(output_dir)
    return {"metrics": metrics, "summary": summary, "output_dir": str(output_dir)}


def main() -> None:
    args = parse_args()
    label_space = load_label_space(args.label_space)
    if args.binary_threshold is not None:
        label_space["binary_threshold"] = args.binary_threshold
    if args.multilabel_threshold is not None:
        label_space["multilabel_threshold"] = args.multilabel_threshold

    input_dir = args.input_dir if args.input_dir is not None else default_public_data_dir()
    eval_files = find_eval_files(input_dir, args.eval_files)
    if not eval_files:
        raise SystemExit("No JSONL eval files found.")
    rows = load_eval_rows(eval_files)
    if args.limit is not None:
        rows = rows[:args.limit]
    rows, skipped_rows = filter_supported_rows(rows, label_space, policy=args.unknown_label_policy)
    if not rows:
        raise SystemExit("No supported rows to evaluate after filtering unknown labels.")

    metadata = {
        "backend": args.backend,
        "input_dir": str(input_dir),
        "eval_files": [str(path) for path in eval_files],
        "label_space_source": str(args.label_space) if args.label_space else "built-in-default",
        "binary_threshold": label_space["binary_threshold"],
        "multilabel_threshold": label_space["multilabel_threshold"],
    }

    if args.backend == "local-qwen":
        if args.checkpoint_dir is None:
            raise SystemExit("--checkpoint-dir is required for --backend local-qwen")
        predictions, backend_metadata = predict_local_qwen(
            rows,
            label_space,
            checkpoint_dir=args.checkpoint_dir,
            base_model=args.base_model,
            max_length=args.max_length,
            batch_size=args.batch_size,
            dtype=args.dtype,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            trust_remote_code=args.trust_remote_code,
            binary_threshold=label_space["binary_threshold"],
            multilabel_threshold=label_space["multilabel_threshold"],
        )
        metadata.update({"checkpoint_dir": str(args.checkpoint_dir), "base_model": str(args.base_model), **backend_metadata})
    elif args.backend == "api":
        if not args.api_model:
            raise SystemExit("--api-model is required for --backend api")
        api_key = api_key_from_env(args.api_key_env, args.api_key)
        thinking = resolve_thinking(args.api_model, args.thinking)
        predictions, backend_metadata = predict_api(
            rows,
            label_space,
            output_dir=args.output_dir,
            base_url=args.api_base_url,
            model=args.api_model,
            api_key=api_key,
            api_type=args.api_type,
            timeout=args.api_timeout,
            max_tokens=args.api_max_tokens,
            temperature=args.api_temperature,
            reasoning_effort=args.reasoning_effort,
            thinking=thinking,
            retries=args.api_retries,
            workers=args.api_workers,
            resume=not args.no_api_resume,
            on_error=args.on_api_error,
            response_format=not args.no_response_format,
        )
        metadata.update(backend_metadata)
    else:
        predictions = load_predictions(args.predictions_file, rows)
        metadata.update({"predictions_file": str(args.predictions_file)})

    result = write_outputs(args.output_dir, label_space, skipped_rows, rows, predictions, metadata)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
