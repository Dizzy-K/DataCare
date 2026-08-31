from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .constants import DEFAULT_CLEAN_CLASS_NAME, DEFAULT_POLLUTED_CLASS_NAME
from .schema import normalize_gold_labels, prediction_from_mapping


META_KEYS = (
    "eval_file",
    "eval_source_file",
    "source",
    "label_l2",
    "label_l3",
    "label_l4",
    "notes",
    "context",
    "query",
    "answer",
)


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _as_set(value: list[str] | None) -> set[str]:
    return set(value or [])


def multilabel_scores(gold_sets: list[set[str]], pred_sets: list[set[str]], labels: list[str]) -> dict[str, float]:
    if not gold_sets:
        return {"micro_f1": 0.0, "macro_f1": 0.0, "exact_match": 0.0}

    total_tp = total_fp = total_fn = 0
    class_f1s: list[float] = []
    for label in labels:
        tp = sum(1 for gold, pred in zip(gold_sets, pred_sets) if label in gold and label in pred)
        fp = sum(1 for gold, pred in zip(gold_sets, pred_sets) if label not in gold and label in pred)
        fn = sum(1 for gold, pred in zip(gold_sets, pred_sets) if label in gold and label not in pred)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        if tp + fn == 0:
            continue
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        class_f1s.append(_safe_div(2 * precision * recall, precision + recall))

    micro_precision = _safe_div(total_tp, total_tp + total_fp)
    micro_recall = _safe_div(total_tp, total_tp + total_fn)
    exact_match = sum(1 for gold, pred in zip(gold_sets, pred_sets) if gold == pred) / len(gold_sets)
    return {
        "micro_f1": _safe_div(2 * micro_precision * micro_recall, micro_precision + micro_recall),
        "macro_f1": sum(class_f1s) / len(class_f1s) if class_f1s else 0.0,
        "exact_match": exact_match,
    }


def build_record(row: dict, prediction: dict) -> dict:
    gold = normalize_gold_labels(row)
    pred = prediction_from_mapping(prediction)
    true_polluted = gold["label_l1"] == DEFAULT_POLLUTED_CLASS_NAME
    pred_polluted = pred["pred_label_l1"] == DEFAULT_POLLUTED_CLASS_NAME
    l2_match = _as_set(gold["label_l2"]) == _as_set(pred["pred_label_l2"])
    l3_match = _as_set(gold["label_l3"]) == _as_set(pred["pred_label_l3"])
    l4_match = _as_set(gold["label_l4"]) == _as_set(pred["pred_label_l4"])
    joint_correct = (
        (not true_polluted and not pred_polluted)
        or (true_polluted and pred_polluted and l2_match and l3_match and l4_match)
    )

    record = {
        "id": str(row.get("id", "")),
        "messages": row.get("messages"),
        "true_label_l1": gold["label_l1"],
        "true_label_l2": gold["label_l2"],
        "true_label_l3": gold["label_l3"],
        "true_label_l4": gold["label_l4"],
        "pred_label_l1": pred["pred_label_l1"],
        "pred_label_l2": pred["pred_label_l2"],
        "pred_label_l3": pred["pred_label_l3"],
        "pred_label_l4": pred["pred_label_l4"],
        "binary_correct": gold["label_l1"] == pred["pred_label_l1"],
        "l2_exact_match_when_polluted": None if not true_polluted else l2_match,
        "l3_exact_match_when_polluted": None if not true_polluted else l3_match,
        "l4_exact_match_when_polluted": None if not true_polluted else l4_match,
        "joint_correct": joint_correct,
    }
    for key in ("binary_confidence_polluted", "binary_confidence_clean", "l2_confidences", "l3_confidences", "l4_confidences"):
        if key in prediction:
            record[key] = prediction[key]
    if "raw_response" in prediction:
        record["raw_response"] = prediction["raw_response"]
    if "parse_error" in prediction:
        record["parse_error"] = prediction["parse_error"]
    for key in META_KEYS:
        if key in row:
            record[key] = row[key]
    return record


def build_records(rows: list[dict], predictions: list[dict]) -> list[dict]:
    if len(rows) != len(predictions):
        raise ValueError(f"rows/predictions length mismatch: {len(rows)} vs {len(predictions)}")
    return [build_record(row, prediction) for row, prediction in zip(rows, predictions)]


def compute_metrics(records: list[dict], label_space: dict) -> dict[str, float]:
    if not records:
        return {
            "binary_accuracy": 0.0,
            "binary_precision": 0.0,
            "binary_recall": 0.0,
            "binary_f1": 0.0,
            "binary_fpr": 0.0,
            "binary_fnr": 0.0,
            "l2_micro_f1": 0.0,
            "l2_macro_f1": 0.0,
            "l2_exact_match": 0.0,
            "l3_micro_f1": 0.0,
            "l3_macro_f1": 0.0,
            "l3_exact_match": 0.0,
            "l4_micro_f1": 0.0,
            "l4_macro_f1": 0.0,
            "l4_exact_match": 0.0,
            "joint_accuracy": 0.0,
            "polluted_joint_accuracy": 0.0,
        }

    true_binary = [1 if r["true_label_l1"] == DEFAULT_POLLUTED_CLASS_NAME else 0 for r in records]
    pred_binary = [1 if r["pred_label_l1"] == DEFAULT_POLLUTED_CLASS_NAME else 0 for r in records]
    tp = sum(1 for t, p in zip(true_binary, pred_binary) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(true_binary, pred_binary) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(true_binary, pred_binary) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(true_binary, pred_binary) if t == 0 and p == 0)
    binary_precision = _safe_div(tp, tp + fp)
    binary_recall = _safe_div(tp, tp + fn)
    polluted_records = [record for record in records if record["true_label_l1"] == DEFAULT_POLLUTED_CLASS_NAME]

    metrics: dict[str, float] = {
        "binary_accuracy": sum(1 for t, p in zip(true_binary, pred_binary) if t == p) / len(records),
        "binary_precision": binary_precision,
        "binary_recall": binary_recall,
        "binary_f1": _safe_div(2 * binary_precision * binary_recall, binary_precision + binary_recall),
        "binary_fpr": _safe_div(fp, fp + tn),
        "binary_fnr": _safe_div(fn, fn + tp),
    }
    for level in ("l2", "l3", "l4"):
        labels = [label for label, _ in sorted(label_space[f"{level}_label2id"].items(), key=lambda item: item[1])]
        gold_sets = [_as_set(record[f"true_label_{level}"]) for record in polluted_records]
        pred_sets = [_as_set(record[f"pred_label_{level}"]) for record in polluted_records]
        scores = multilabel_scores(gold_sets, pred_sets, labels)
        metrics[f"{level}_micro_f1"] = scores["micro_f1"]
        metrics[f"{level}_macro_f1"] = scores["macro_f1"]
        metrics[f"{level}_exact_match"] = scores["exact_match"]

    metrics["joint_accuracy"] = sum(1 for record in records if record["joint_correct"]) / len(records)
    metrics["polluted_joint_accuracy"] = (
        sum(1 for record in polluted_records if record["joint_correct"]) / len(polluted_records)
        if polluted_records else 0.0
    )
    return metrics


def summarize_records(records: list[dict], skipped_rows: list[dict]) -> dict[str, Any]:
    false_negatives = [r for r in records if r["true_label_l1"] == DEFAULT_POLLUTED_CLASS_NAME and r["pred_label_l1"] == DEFAULT_CLEAN_CLASS_NAME]
    false_positives = [r for r in records if r["true_label_l1"] == DEFAULT_CLEAN_CLASS_NAME and r["pred_label_l1"] == DEFAULT_POLLUTED_CLASS_NAME]
    l2_errors = [
        r for r in records
        if r["true_label_l1"] == DEFAULT_POLLUTED_CLASS_NAME
        and r["pred_label_l1"] == DEFAULT_POLLUTED_CLASS_NAME
        and not r["l2_exact_match_when_polluted"]
    ]
    l3_errors = [
        r for r in records
        if r["true_label_l1"] == DEFAULT_POLLUTED_CLASS_NAME
        and r["pred_label_l1"] == DEFAULT_POLLUTED_CLASS_NAME
        and not r["l3_exact_match_when_polluted"]
    ]
    l4_errors = [
        r for r in records
        if r["true_label_l1"] == DEFAULT_POLLUTED_CLASS_NAME
        and r["pred_label_l1"] == DEFAULT_POLLUTED_CLASS_NAME
        and not r["l4_exact_match_when_polluted"]
    ]
    joint_errors = [r for r in records if not r["joint_correct"]]

    by_file: dict[str, dict[str, Any]] = {}
    for eval_file, group in groupby(records, "eval_file").items():
        total = len(group)
        file_fn = [
            r for r in group
            if r["true_label_l1"] == DEFAULT_POLLUTED_CLASS_NAME and r["pred_label_l1"] == DEFAULT_CLEAN_CLASS_NAME
        ]
        by_file[eval_file] = {
            "total": total,
            "joint_accuracy": sum(1 for r in group if r["joint_correct"]) / total if total else 0.0,
            "false_negative_count": len(file_fn),
            "false_negative_rate": len(file_fn) / total if total else 0.0,
            "true_l1_counts": dict(Counter(r["true_label_l1"] for r in group)),
            "pred_l1_counts": dict(Counter(r["pred_label_l1"] for r in group)),
            "true_l4_counts": dict(Counter(",".join(r.get("true_label_l4") or []) for r in group)),
            "pred_l4_counts": dict(Counter(",".join(r.get("pred_label_l4") or []) for r in group)),
        }

    return {
        "total": len(records),
        "skipped_unknown_label_count": len(skipped_rows),
        "false_negative_count": len(false_negatives),
        "false_positive_count": len(false_positives),
        "l2_error_count_when_pred_polluted": len(l2_errors),
        "l3_error_count_when_pred_polluted": len(l3_errors),
        "l4_error_count_when_pred_polluted": len(l4_errors),
        "joint_error_count": len(joint_errors),
        "by_eval_file": by_file,
        "true_l1_counts": dict(Counter(r["true_label_l1"] for r in records)),
        "pred_l1_counts": dict(Counter(r["pred_label_l1"] for r in records)),
        "pred_l2_counts": dict(Counter(",".join(r.get("pred_label_l2") or []) for r in records)),
        "pred_l3_counts": dict(Counter(",".join(r.get("pred_label_l3") or []) for r in records)),
        "pred_l4_counts": dict(Counter(",".join(r.get("pred_label_l4") or []) for r in records)),
        "skipped_unknown_label_counts": dict(Counter(",".join(row.get("unknown_labels", [])) for row in skipped_rows)),
    }


def split_badcases(records: list[dict]) -> dict[str, list[dict]]:
    return {
        "false_negatives": [
            r for r in records
            if r["true_label_l1"] == DEFAULT_POLLUTED_CLASS_NAME and r["pred_label_l1"] == DEFAULT_CLEAN_CLASS_NAME
        ],
        "false_positives": [
            r for r in records
            if r["true_label_l1"] == DEFAULT_CLEAN_CLASS_NAME and r["pred_label_l1"] == DEFAULT_POLLUTED_CLASS_NAME
        ],
        "l2_errors": [
            r for r in records
            if r["true_label_l1"] == DEFAULT_POLLUTED_CLASS_NAME
            and r["pred_label_l1"] == DEFAULT_POLLUTED_CLASS_NAME
            and not r["l2_exact_match_when_polluted"]
        ],
        "l3_errors": [
            r for r in records
            if r["true_label_l1"] == DEFAULT_POLLUTED_CLASS_NAME
            and r["pred_label_l1"] == DEFAULT_POLLUTED_CLASS_NAME
            and not r["l3_exact_match_when_polluted"]
        ],
        "l4_errors": [
            r for r in records
            if r["true_label_l1"] == DEFAULT_POLLUTED_CLASS_NAME
            and r["pred_label_l1"] == DEFAULT_POLLUTED_CLASS_NAME
            and not r["l4_exact_match_when_polluted"]
        ],
        "joint_errors": [r for r in records if not r["joint_correct"]],
    }


def groupby(records: list[dict], key: str) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[str(record.get(key, "UNKNOWN"))].append(record)
    return dict(groups)

