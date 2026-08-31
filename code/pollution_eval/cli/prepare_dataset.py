from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from ..io_utils import default_public_data_dir, find_eval_files, load_eval_rows, load_label_space, write_json, write_jsonl
from ..schema import filter_supported_rows, normalize_gold_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a release/evaluation bundle from source JSONL files.")
    parser.add_argument("--input-dir", type=Path, default=None, help="Directory of JSONL files. Defaults to the repo public release.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-files", type=Path, nargs="*", default=None)
    parser.add_argument("--label-space", type=Path, default=None)
    parser.add_argument("--unknown-label-policy", choices=["skip", "error"], default="skip")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label_space = load_label_space(args.label_space)
    input_dir = args.input_dir if args.input_dir is not None else default_public_data_dir()
    files = find_eval_files(input_dir, args.eval_files)
    rows = load_eval_rows(files)
    kept_rows, skipped_rows = filter_supported_rows(rows, label_space, policy=args.unknown_label_policy)

    by_file = defaultdict(Counter)
    counts = {
        "label_l1": Counter(),
        "label_l2": Counter(),
        "label_l3": Counter(),
        "label_l4": Counter(),
    }
    for row in kept_rows:
        gold = normalize_gold_labels(row)
        eval_file = str(row.get("eval_file", "UNKNOWN"))
        counts["label_l1"][gold["label_l1"]] += 1
        by_file[eval_file]["total"] += 1
        by_file[eval_file][gold["label_l1"]] += 1
        for level in ("l2", "l3", "l4"):
            labels = gold[f"label_{level}"]
            if not labels:
                counts[f"label_{level}"][""] += 1
                continue
            for label in labels:
                counts[f"label_{level}"][label] += 1

    manifest = {
        "input_dir": str(input_dir),
        "files": [str(path) for path in files],
        "total_loaded": len(rows),
        "total_supported": len(kept_rows),
        "total_skipped_unknown_labels": len(skipped_rows),
        "counts": {key: dict(value) for key, value in counts.items()},
        "by_file": {key: dict(value) for key, value in sorted(by_file.items())},
        "outputs": {
            "eval_all": "eval_all.jsonl",
            "label_space": "label_space.json",
            "manifest": "manifest.json",
            "skipped_unknown_labels": "skipped_unknown_labels.jsonl",
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "eval_all.jsonl", kept_rows)
    write_json(args.output_dir / "label_space.json", label_space)
    write_json(args.output_dir / "manifest.json", manifest)
    write_jsonl(args.output_dir / "skipped_unknown_labels.jsonl", skipped_rows)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
