#!/usr/bin/env python3
"""Small OpenAI-compatible chat server for a local Transformers causal LM."""

from __future__ import annotations

import argparse
import time
import uuid

import aiohttp.web
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a local causal LM with an OpenAI-compatible chat API.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3001)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    return parser.parse_args()


def torch_dtype(name: str):
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def build_app(args: argparse.Namespace) -> aiohttp.web.Application:
    dtype = torch_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=True,
        dtype=dtype,
    )
    model.to("cuda")
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_name = args.served_model_name or args.model

    async def health(_request: aiohttp.web.Request) -> aiohttp.web.Response:
        return aiohttp.web.json_response({"status": "ok", "model": model_name})

    async def models(_request: aiohttp.web.Request) -> aiohttp.web.Response:
        return aiohttp.web.json_response(
            {
                "object": "list",
                "data": [
                    {
                        "id": model_name,
                        "object": "model",
                        "owned_by": "local-transformers",
                    }
                ],
            }
        )

    async def chat_completions(request: aiohttp.web.Request) -> aiohttp.web.Response:
        payload = await request.json()
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise aiohttp.web.HTTPBadRequest(text="messages must be a non-empty list")

        prompt_kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:
            prompt = tokenizer.apply_chat_template(messages, enable_thinking=False, **prompt_kwargs)
        except TypeError:
            prompt = tokenizer.apply_chat_template(messages, **prompt_kwargs)
        inputs = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        requested_tokens = payload.get("max_tokens", payload.get("max_completion_tokens", 2048))
        max_new_tokens = max(1, int(requested_tokens))
        temperature = float(payload.get("temperature", 0.0) or 0.0)
        do_sample = temperature > 0.0
        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            if payload.get("top_p") is not None:
                generation_kwargs["top_p"] = float(payload["top_p"])

        with torch.inference_mode():
            generated = model.generate(**generation_kwargs)
        prompt_length = inputs["input_ids"].shape[1]
        output_tokens = generated[0, prompt_length:]
        content = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()

        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "length" if output_tokens.shape[0] >= max_new_tokens else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": int(prompt_length),
                "completion_tokens": int(output_tokens.shape[0]),
                "total_tokens": int(prompt_length + output_tokens.shape[0]),
            },
        }
        return aiohttp.web.json_response(response)

    app = aiohttp.web.Application()
    app.router.add_get("/health", health)
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/chat/completions", chat_completions)
    return app


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the local Qwen API server.")
    aiohttp.web.run_app(build_app(args), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
