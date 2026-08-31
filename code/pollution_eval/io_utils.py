from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

from .constants import DEFAULT_LABEL_SPACE, default_label_space


def to_jsonable(value: Any) -> Any:
    try:
        import numpy as np
    except Exception:  # pragma: no cover - numpy is optional for API-only use.
        np = None

    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if np is not None and isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if np is not None and isinstance(value, np.generic):
        return value.item()
    return value


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL line: {exc}") from exc
    return rows


def iter_jsonl(path: str | Path) -> Iterable[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL line: {exc}") from exc


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")
            handle.flush()


def make_artifacts_readable(path: str | Path) -> None:
    root = Path(path)
    if not root.exists():
        return
    for item in [root, *root.rglob("*")]:
        try:
            if item.is_dir():
                item.chmod(0o755)
            elif item.is_file():
                item.chmod(0o644)
        except OSError:
            pass


def load_label_space(path: str | Path | None = None) -> dict:
    if path is None:
        return default_label_space()
    payload = read_json(path)
    label_space = default_label_space()
    label_space.update(payload)
    for level in ("l2", "l3", "l4"):
        label2id = {str(key): int(value) for key, value in label_space[f"{level}_label2id"].items()}
        id2label = {str(int(key)): str(value) for key, value in label_space[f"{level}_id2label"].items()}
        label_space[f"{level}_label2id"] = label2id
        label_space[f"{level}_id2label"] = id2label
    label_space["binary_threshold"] = float(label_space.get("binary_threshold", DEFAULT_LABEL_SPACE["binary_threshold"]))
    label_space["multilabel_threshold"] = float(label_space.get("multilabel_threshold", DEFAULT_LABEL_SPACE["multilabel_threshold"]))
    return label_space


def find_eval_files(input_dir: str | Path | None, eval_files: list[str | Path] | None) -> list[Path]:
    if eval_files:
        return [Path(path) for path in eval_files]
    if input_dir is None:
        raise ValueError("Provide --eval-files or --input-dir.")
    return sorted(Path(input_dir).glob("*.jsonl"))


def default_public_data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "datasets"


def load_eval_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        for row in read_jsonl(path):
            item = dict(row)
            item["eval_file"] = path.name
            item.setdefault("eval_source_file", path.name)
            rows.append(item)
    return rows


def existing_predictions_by_id(path: str | Path) -> dict[str, dict]:
    target = Path(path)
    if not target.exists():
        return {}
    return {str(row.get("id", "")): row for row in iter_jsonl(target)}


def api_key_from_env(env_name: str, explicit_key: str | None = None) -> str:
    if explicit_key:
        return explicit_key
    value = os.environ.get(env_name)
    if not value:
        raise ValueError(f"API key not found. Set {env_name} or pass --api-key.")
    return value
