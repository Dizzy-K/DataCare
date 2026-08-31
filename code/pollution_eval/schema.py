from __future__ import annotations

import json
import re
from typing import Any

from .constants import DEFAULT_CLEAN_CLASS_NAME, DEFAULT_POLLUTED_CLASS_NAME, LABEL_ALIASES, POLLUTED_CLASS_NAMES


EMPTY_LABELS = {"", "none", "null", "nil", "na", "n/a", "无", "空", "否", "非污染", "非污染样本"}


def canonical_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in EMPTY_LABELS:
        return None
    return LABEL_ALIASES.get(text, LABEL_ALIASES.get(lowered, text))


def normalize_multilabel_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        labels: list[str] = []
        for item in value:
            labels.extend(normalize_multilabel_value(item))
        return dedupe(labels)
    if isinstance(value, dict):
        labels = []
        for key, is_selected in value.items():
            if is_selected:
                labels.extend(normalize_multilabel_value(key))
        return dedupe(labels)

    text = str(value).strip()
    if not text:
        return []
    lowered = text.lower()
    if lowered in EMPTY_LABELS:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            decoded = json.loads(text)
            return normalize_multilabel_value(decoded)
        except json.JSONDecodeError:
            pass

    parts = re.split(r"\s*(?:,|，|;|；|\||、)\s*", text)
    labels = [canonical_label(part) for part in parts]
    return dedupe(label for label in labels if label)


def dedupe(values: Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        label = canonical_label(value)
        if label and label not in seen:
            seen.add(label)
            result.append(label)
    return result


def is_polluted_l1(value: Any) -> bool:
    label = canonical_label(value)
    return label in POLLUTED_CLASS_NAMES or label == DEFAULT_POLLUTED_CLASS_NAME


def normalize_l1(value: Any) -> str:
    label = canonical_label(value)
    if label in POLLUTED_CLASS_NAMES or label == DEFAULT_POLLUTED_CLASS_NAME:
        return DEFAULT_POLLUTED_CLASS_NAME
    return DEFAULT_CLEAN_CLASS_NAME


def normalize_gold_labels(row: dict) -> dict:
    label_l1 = normalize_l1(row.get("label_l1"))
    if label_l1 == DEFAULT_CLEAN_CLASS_NAME:
        return {"label_l1": label_l1, "label_l2": [], "label_l3": [], "label_l4": []}
    return {
        "label_l1": label_l1,
        "label_l2": normalize_multilabel_value(row.get("label_l2")),
        "label_l3": normalize_multilabel_value(row.get("label_l3")),
        "label_l4": normalize_multilabel_value(row.get("label_l4")),
    }


def prediction_from_mapping(payload: dict) -> dict:
    label_l1 = payload.get("pred_label_l1", payload.get("label_l1"))
    label_l2 = payload.get("pred_label_l2", payload.get("label_l2"))
    label_l3 = payload.get("pred_label_l3", payload.get("label_l3"))
    label_l4 = payload.get("pred_label_l4", payload.get("label_l4"))

    l2_labels = normalize_multilabel_value(label_l2)
    l3_labels = normalize_multilabel_value(label_l3)
    l4_labels = normalize_multilabel_value(label_l4)
    if label_l1 is None:
        pred_l1 = DEFAULT_POLLUTED_CLASS_NAME if (l2_labels or l3_labels or l4_labels) else DEFAULT_CLEAN_CLASS_NAME
    else:
        pred_l1 = normalize_l1(label_l1)
    if pred_l1 == DEFAULT_CLEAN_CLASS_NAME:
        l2_labels, l3_labels, l4_labels = [], [], []
    return {
        "pred_label_l1": pred_l1,
        "pred_label_l2": l2_labels,
        "pred_label_l3": l3_labels,
        "pred_label_l4": l4_labels,
        "raw_prediction": payload,
    }


def labels_outside_space(row: dict, label_space: dict) -> list[str]:
    gold = normalize_gold_labels(row)
    if gold["label_l1"] == DEFAULT_CLEAN_CLASS_NAME:
        return []
    unknown: list[str] = []
    for level in ("l2", "l3", "l4"):
        allowed = set(label_space[f"{level}_label2id"])
        for label in gold[f"label_{level}"]:
            if label not in allowed:
                unknown.append(f"label_{level}:{label}")
    return unknown


def filter_supported_rows(rows: list[dict], label_space: dict, policy: str = "skip") -> tuple[list[dict], list[dict]]:
    kept: list[dict] = []
    skipped: list[dict] = []
    for row in rows:
        unknown = labels_outside_space(row, label_space)
        if not unknown:
            kept.append(row)
            continue
        if policy == "error":
            raise ValueError(f"Sample {row.get('id', '')} contains labels outside label space: {unknown}")
        item = dict(row)
        item["skip_reason"] = "unknown_labels"
        item["unknown_labels"] = unknown
        skipped.append(item)
    return kept, skipped


def label_space_counts(label_space: dict) -> tuple[int, int, int]:
    return (
        len(label_space["l2_label2id"]),
        len(label_space["l3_label2id"]),
        len(label_space["l4_label2id"]),
    )

