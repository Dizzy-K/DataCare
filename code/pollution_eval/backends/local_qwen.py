from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ..constants import DEFAULT_CLEAN_CLASS_NAME, DEFAULT_POLLUTED_CLASS_NAME
from ..rendering import build_model_prompt
from ..schema import label_space_counts


def _require_torch_stack():
    try:
        import torch
        import torch.nn as nn
        from peft import LoraConfig, TaskType, get_peft_model
        from safetensors.torch import load_file
        from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
        from transformers.modeling_outputs import SequenceClassifierOutput
    except ImportError as exc:  # pragma: no cover - exercised only when optional deps are missing.
        raise RuntimeError(
            "The local-qwen backend requires torch, transformers, peft, and safetensors. "
            "Install the GPU environment or use --backend api / --backend predictions."
        ) from exc
    return torch, nn, LoraConfig, TaskType, get_peft_model, load_file, AutoModel, AutoTokenizer, BitsAndBytesConfig, SequenceClassifierOutput


def _dtype_from_name(torch, name: str | None):
    if not name:
        return None
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if name not in aliases:
        raise ValueError(f"Unsupported dtype: {name}")
    return aliases[name]


def _model_device(torch, model) -> object:
    for parameter in model.parameters():
        return parameter.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_four_head_model_class():
    torch, nn, _LoraConfig, _TaskType, _get_peft_model, _load_file, AutoModel, _AutoTokenizer, BitsAndBytesConfig, SequenceClassifierOutput = _require_torch_stack()

    class FourHeadQwenForClassification(nn.Module):
        def __init__(
            self,
            model_name_or_path: str,
            num_l2_labels: int,
            num_l3_labels: int,
            num_l4_labels: int,
            trust_remote_code: bool = False,
            load_in_8bit: bool = False,
            load_in_4bit: bool = False,
            torch_dtype=None,
        ):
            super().__init__()
            if load_in_8bit and load_in_4bit:
                raise ValueError("--load-in-8bit and --load-in-4bit cannot be enabled together.")
            model_kwargs = {"trust_remote_code": trust_remote_code}
            if load_in_8bit:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            elif load_in_4bit:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch_dtype or torch.float16,
                )
            if torch_dtype is not None:
                model_kwargs["torch_dtype"] = torch_dtype

            self.backbone = AutoModel.from_pretrained(model_name_or_path, **model_kwargs)
            hidden_size = self.backbone.config.hidden_size
            dropout_p = getattr(self.backbone.config, "classifier_dropout", 0.1) or 0.1
            self.dropout = nn.Dropout(dropout_p)
            self.binary_head = nn.Linear(hidden_size, 2)
            self.l2_head = nn.Linear(hidden_size, num_l2_labels)
            self.l3_head = nn.Linear(hidden_size, num_l3_labels)
            self.l4_head = nn.Linear(hidden_size, num_l4_labels)
            self.config = self.backbone.config

        def forward(self, input_ids, attention_mask=None, **kwargs):
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            hidden_states = outputs.last_hidden_state
            if attention_mask is None:
                pooled = hidden_states[:, -1, :]
            else:
                last_token_indices = attention_mask.sum(dim=1) - 1
                pooled = hidden_states[torch.arange(hidden_states.size(0), device=hidden_states.device), last_token_indices]
            pooled = self.dropout(pooled)
            binary_logits = self.binary_head(pooled)
            l2_logits = self.l2_head(pooled)
            l3_logits = self.l3_head(pooled)
            l4_logits = self.l4_head(pooled)
            return SequenceClassifierOutput(
                logits=binary_logits,
                hidden_states=(binary_logits, l2_logits, l3_logits, l4_logits),
                attentions=getattr(outputs, "attentions", None),
            )

    return FourHeadQwenForClassification


def load_local_qwen(
    checkpoint_dir: str | Path,
    base_model: str | Path | None,
    label_space: dict,
    *,
    dtype: str | None = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    trust_remote_code: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Iterable[str] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
):
    torch, _nn, LoraConfig, TaskType, get_peft_model, load_file, _AutoModel, AutoTokenizer, _BitsAndBytesConfig, _SequenceClassifierOutput = _require_torch_stack()
    checkpoint = Path(checkpoint_dir)
    model_path = Path(base_model) if base_model is not None else checkpoint
    if base_model is None and not (checkpoint / "config.json").exists():
        raise ValueError(
            "--base-model is required when --checkpoint-dir does not contain config.json. "
            "For the released checkpoint, pass the Qwen base model path/name used for training."
        )

    tokenizer_source = checkpoint if (checkpoint / "tokenizer.json").exists() else model_path
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source), trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_l2, num_l3, num_l4 = label_space_counts(label_space)
    torch_dtype = _dtype_from_name(torch, dtype)
    model_cls = build_four_head_model_class()
    model = model_cls(
        model_name_or_path=str(model_path),
        num_l2_labels=num_l2,
        num_l3_labels=num_l3,
        num_l4_labels=num_l4,
        trust_remote_code=trust_remote_code,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        torch_dtype=torch_dtype,
    )

    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=list(target_modules),
    )
    model.backbone = get_peft_model(model.backbone, peft_config)
    state_path = checkpoint / "model.safetensors"
    if not state_path.exists():
        raise FileNotFoundError(f"Checkpoint weights not found: {state_path}")
    state = load_file(str(state_path))
    if load_in_4bit or load_in_8bit:
        # The released checkpoint contains dense copies of frozen backbone weights.
        # bitsandbytes stores those same layers as packed Params4bit/8bit tensors,
        # so the dense base_layer weights must remain loaded from the base model.
        state = {key: value for key, value in state.items() if not key.endswith(".base_layer.weight")}
    missing, unexpected = model.load_state_dict(state, strict=False)

    head_device = _model_device(torch, model.backbone)
    for head in (model.binary_head, model.l2_head, model.l3_head, model.l4_head):
        if torch_dtype is None:
            head.to(device=head_device)
        else:
            head.to(device=head_device, dtype=torch_dtype)
    return model, tokenizer, {"missing": list(missing), "unexpected": list(unexpected)}


def predict_local_qwen(
    rows: list[dict],
    label_space: dict,
    *,
    checkpoint_dir: str | Path,
    base_model: str | Path | None,
    max_length: int = 1024,
    batch_size: int = 2,
    dtype: str | None = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    trust_remote_code: bool = False,
    binary_threshold: float | None = None,
    multilabel_threshold: float | None = None,
) -> tuple[list[dict], dict]:
    torch, _nn, _LoraConfig, _TaskType, _get_peft_model, _load_file, _AutoModel, _AutoTokenizer, _BitsAndBytesConfig, _SequenceClassifierOutput = _require_torch_stack()
    model, tokenizer, load_info = load_local_qwen(
        checkpoint_dir,
        base_model,
        label_space,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        trust_remote_code=trust_remote_code,
    )
    if not (load_in_4bit or load_in_8bit):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    else:
        device = _model_device(torch, model)

    binary_threshold = float(binary_threshold if binary_threshold is not None else label_space.get("binary_threshold", 0.5))
    multilabel_threshold = float(multilabel_threshold if multilabel_threshold is not None else label_space.get("multilabel_threshold", 0.5))
    id2labels = {
        "l2": [label for label, _ in sorted(label_space["l2_label2id"].items(), key=lambda item: item[1])],
        "l3": [label for label, _ in sorted(label_space["l3_label2id"].items(), key=lambda item: item[1])],
        "l4": [label for label, _ in sorted(label_space["l4_label2id"].items(), key=lambda item: item[1])],
    }

    predictions: list[dict] = []
    model.eval()
    try:
        from tqdm.auto import tqdm
    except Exception:  # pragma: no cover
        tqdm = lambda value, **_: value

    prompts = [build_model_prompt(row) for row in rows]
    with torch.inference_mode():
        for start in tqdm(range(0, len(rows), batch_size), desc="predict-local"):
            batch_prompts = prompts[start:start + batch_size]
            encoded = tokenizer(
                batch_prompts,
                truncation=True,
                max_length=max_length,
                padding=True,
                pad_to_multiple_of=8,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items() if torch.is_tensor(value)}
            outputs = model(**encoded)
            binary_logits, l2_logits, l3_logits, l4_logits = outputs.hidden_states
            binary_probs = torch.softmax(binary_logits.float(), dim=-1).cpu()
            level_probs = {
                "l2": torch.sigmoid(l2_logits.float()).cpu(),
                "l3": torch.sigmoid(l3_logits.float()).cpu(),
                "l4": torch.sigmoid(l4_logits.float()).cpu(),
            }
            for offset in range(binary_probs.shape[0]):
                polluted_prob = float(binary_probs[offset, 1])
                clean_prob = float(binary_probs[offset, 0])
                pred_polluted = polluted_prob >= binary_threshold
                prediction = {
                    "pred_label_l1": DEFAULT_POLLUTED_CLASS_NAME if pred_polluted else DEFAULT_CLEAN_CLASS_NAME,
                    "pred_label_l2": [],
                    "pred_label_l3": [],
                    "pred_label_l4": [],
                    "binary_confidence_polluted": polluted_prob,
                    "binary_confidence_clean": clean_prob,
                    "l2_confidences": [],
                    "l3_confidences": [],
                    "l4_confidences": [],
                }
                if pred_polluted:
                    for level in ("l2", "l3", "l4"):
                        selected_labels = []
                        selected_scores = []
                        for label, score_tensor in zip(id2labels[level], level_probs[level][offset]):
                            score = float(score_tensor)
                            if score >= multilabel_threshold:
                                selected_labels.append(label)
                                selected_scores.append(score)
                        prediction[f"pred_label_{level}"] = selected_labels
                        prediction[f"{level}_confidences"] = selected_scores
                predictions.append(prediction)

    load_info["device"] = str(device)
    load_info["binary_threshold"] = binary_threshold
    load_info["multilabel_threshold"] = multilabel_threshold
    return predictions, load_info
