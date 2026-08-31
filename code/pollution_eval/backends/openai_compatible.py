from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ..constants import DEFAULT_CLEAN_CLASS_NAME
from ..io_utils import append_jsonl, existing_predictions_by_id
from ..rendering import build_api_messages
from ..schema import prediction_from_mapping


def _endpoint_from_base(base_url: str, api_type: str) -> str:
    root = base_url.rstrip("/")
    if root.endswith("/chat/completions") or root.endswith("/responses"):
        return root
    if api_type == "responses":
        return f"{root}/responses"
    return f"{root}/chat/completions"


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str,
        api_type: str = "chat",
        timeout: float = 120.0,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
        thinking: str | None = None,
        retries: int = 3,
        response_format: bool = True,
    ):
        if api_type not in {"chat", "responses"}:
            raise ValueError("--api-type must be chat or responses")
        if reasoning_effort not in {None, "low", "high", "max"}:
            raise ValueError("reasoning_effort must be low, high, max, or None")
        if thinking not in {None, "enabled", "disabled"}:
            raise ValueError("thinking must be enabled, disabled, or None")
        self.endpoint = _endpoint_from_base(base_url, api_type)
        self.model = model
        self.api_key = api_key
        self.api_type = api_type
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.thinking = thinking
        self.retries = retries
        self.response_format = response_format

    def complete(self, messages: list[dict]) -> tuple[str, dict]:
        payload = self._payload(messages)
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = self._post_json(payload)
                return self._extract_text(response), response
            except Exception as exc:  # noqa: BLE001 - retry then re-raise with context.
                last_error = exc
                if attempt >= self.retries:
                    break
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"API request failed after {self.retries + 1} attempts: {last_error}") from last_error

    def _payload(self, messages: list[dict]) -> dict:
        if self.api_type == "responses":
            payload = {
                "model": self.model,
                "input": messages,
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
            }
        else:
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            if self.response_format:
                payload["response_format"] = {"type": "json_object"}
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.thinking is not None:
            payload["thinking"] = {"type": self.thinking}
        return payload

    def _post_json(self, payload: dict) -> dict:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "pollution-eval/0.1",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from {self.endpoint}: {detail[:1000]}") from exc
        return json.loads(response_body)

    def _extract_text(self, response: dict) -> str:
        if self.api_type == "chat":
            return str(response["choices"][0]["message"]["content"])
        if isinstance(response.get("output_text"), str):
            return response["output_text"]
        texts: list[str] = []
        for item in response.get("output", []):
            for content in item.get("content", []):
                if content.get("type") in {"output_text", "text"}:
                    texts.append(str(content.get("text", "")))
        if texts:
            return "\n".join(texts)
        return json.dumps(response, ensure_ascii=False)


def extract_json_object(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        parts = stripped.split("```")
        for part in parts:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{"):
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass
    decoder = json.JSONDecoder()
    for idx, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            payload, _end = decoder.raw_decode(stripped[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError(f"Could not find a JSON object in model response: {stripped[:500]}")


def extract_prediction_fields(text: str) -> dict:
    """Recover label fields when an optional reason string breaks JSON syntax."""
    decoder = json.JSONDecoder()
    fields: dict[str, Any] = {}
    for key in ("label_l1", "label_l2", "label_l3", "label_l4"):
        match = re.search(rf'"{re.escape(key)}"\s*:', text)
        if match is None:
            continue
        value_text = text[match.end():].lstrip()
        try:
            value, _end = decoder.raw_decode(value_text)
        except json.JSONDecodeError:
            continue
        fields[key] = value
    if not fields or ("label_l1" not in fields and not any(key in fields for key in ("label_l2", "label_l3", "label_l4"))):
        raise ValueError(f"Could not recover label fields from model response: {text[:500]}")
    return fields


def prediction_from_api_text(text: str) -> dict:
    parse_warning = None
    try:
        payload = extract_json_object(text)
    except ValueError as exc:
        payload = extract_prediction_fields(text)
        parse_warning = str(exc)
    prediction = prediction_from_mapping(payload)
    prediction["raw_response"] = text
    if parse_warning is not None:
        prediction["parse_warning"] = parse_warning
    return prediction


def prediction_unknown_labels(prediction: dict, label_space: dict) -> list[str]:
    unknown: list[str] = []
    for level in ("l2", "l3", "l4"):
        allowed = set(label_space[f"{level}_label2id"])
        for label in prediction.get(f"pred_label_{level}", []):
            if label not in allowed:
                unknown.append(f"pred_label_{level}:{label}")
    return unknown


def _saved_prediction(saved: dict) -> dict:
    payload = dict(saved.get("prediction") or saved)
    if "raw_response" not in payload and "raw_response" in saved:
        payload["raw_response"] = saved["raw_response"]
    if "parse_error" not in payload and "parse_error" in saved:
        payload["parse_error"] = saved["parse_error"]
    return payload


def _predict_one(row: dict, client: OpenAICompatibleClient, label_space: dict) -> dict:
    response_text, response_payload = client.complete(build_api_messages(row))
    prediction = prediction_from_api_text(response_text)
    unknown = prediction_unknown_labels(prediction, label_space)
    if unknown:
        prediction["prediction_unknown_labels"] = unknown
    return {
        "id": str(row.get("id", "")),
        "prediction": prediction,
        "raw_response": response_text,
        "response_metadata": {
            "model": response_payload.get("model"),
            "usage": response_payload.get("usage"),
        },
    }


def predict_api(
    rows: list[dict],
    label_space: dict,
    *,
    output_dir: str | Path,
    base_url: str,
    model: str,
    api_key: str,
    api_type: str = "chat",
    timeout: float = 120.0,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    reasoning_effort: str | None = None,
    thinking: str | None = None,
    retries: int = 3,
    workers: int = 1,
    resume: bool = True,
    on_error: str = "error",
    response_format: bool = True,
) -> tuple[list[dict], dict]:
    if on_error not in {"error", "mark-clean"}:
        raise ValueError("--on-api-error must be error or mark-clean")
    output_path = Path(output_dir) / "api_raw_predictions.jsonl"
    saved = existing_predictions_by_id(output_path) if resume else {}
    predictions_by_id: dict[str, dict] = {key: _saved_prediction(value) for key, value in saved.items()}
    missing = [row for row in rows if str(row.get("id", "")) not in predictions_by_id]

    client_kwargs = {
        "base_url": base_url,
        "model": model,
        "api_key": api_key,
        "api_type": api_type,
        "timeout": timeout,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "reasoning_effort": reasoning_effort,
        "thinking": thinking,
        "retries": retries,
        "response_format": response_format,
    }

    def handle_error(row: dict, exc: Exception) -> dict:
        if on_error == "error":
            raise RuntimeError(f"API prediction failed for id={row.get('id', '')}: {exc}") from exc
        return {
            "id": str(row.get("id", "")),
            "prediction": {
                "pred_label_l1": DEFAULT_CLEAN_CLASS_NAME,
                "pred_label_l2": [],
                "pred_label_l3": [],
                "pred_label_l4": [],
                "parse_error": str(exc),
            },
            "parse_error": str(exc),
        }

    try:
        from tqdm.auto import tqdm
    except Exception:  # pragma: no cover
        tqdm = lambda value, **_: value

    if workers <= 1:
        client = OpenAICompatibleClient(**client_kwargs)
        for row in tqdm(missing, desc="predict-api"):
            try:
                saved_row = _predict_one(row, client, label_space)
            except Exception as exc:  # noqa: BLE001
                saved_row = handle_error(row, exc)
            predictions_by_id[str(row.get("id", ""))] = _saved_prediction(saved_row)
            append_jsonl(output_path, [saved_row])
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for row in missing:
                client = OpenAICompatibleClient(**client_kwargs)
                futures[pool.submit(_predict_one, row, client, label_space)] = row
            for future in tqdm(as_completed(futures), total=len(futures), desc="predict-api"):
                row = futures[future]
                try:
                    saved_row = future.result()
                except Exception as exc:  # noqa: BLE001
                    saved_row = handle_error(row, exc)
                predictions_by_id[str(row.get("id", ""))] = _saved_prediction(saved_row)
                append_jsonl(output_path, [saved_row])

    ordered_predictions: list[dict] = []
    for row in rows:
        row_id = str(row.get("id", ""))
        if row_id not in predictions_by_id:
            raise RuntimeError(f"Missing prediction for id={row_id}")
        ordered_predictions.append(predictions_by_id[row_id])
    metadata: dict[str, Any] = {
        "api_type": api_type,
        "endpoint": _endpoint_from_base(base_url, api_type),
        "model": model,
        "reasoning_effort": reasoning_effort,
        "thinking": thinking,
        "requested_count": len(rows),
        "loaded_from_resume": len(saved),
        "new_predictions": len(missing),
        "raw_predictions_file": str(output_path),
    }
    return ordered_predictions, metadata
