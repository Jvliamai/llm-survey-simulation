"""Local batched generation with Hugging Face Transformers."""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline

log = logging.getLogger(__name__)


def _dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def effective_generation_config(model_name_or_path: str, overrides: dict | None = None) -> dict:
    """
    Resolve the sampling parameters that will actually be used.

    :param overrides: values from the YAML config; ``None`` entries are ignored
    :return: dict recorded in the run metadata for reproducibility
    """
    try:
        gc = GenerationConfig.from_pretrained(model_name_or_path)
        resolved = {
            "do_sample": bool(getattr(gc, "do_sample", True)),
            "temperature": getattr(gc, "temperature", None),
            "top_p": getattr(gc, "top_p", None),
            "top_k": getattr(gc, "top_k", None),
            "source": "model generation_config",
        }
    except Exception as exc:  # pragma: no cover - depends on local files
        log.warning("Could not read generation_config for %s: %s", model_name_or_path, exc)
        resolved = {"do_sample": True, "temperature": None, "top_p": None,
                    "top_k": None, "source": "unavailable"}

    for k, v in (overrides or {}).items():
        if v is not None:
            resolved[k] = v
            resolved["source"] = "config override"
    return resolved


def init_pipeline(cfg: dict):
    """Build a text-generation pipeline from a YAML config dict."""
    model_path = cfg["hf_model"]
    adapter_path = cfg.get("adapter")

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_args: dict[str, Any] = {"device_map": "auto", "torch_dtype": _dtype()}
    if cfg.get("flash_attention", False):
        model_args["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_args)

    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
        log.info("Loaded LoRA adapter from %s", adapter_path)

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        batch_size=cfg.get("batch_size", 32),
    )


def _apply_chat_template(tokenizer, prompts, cfg, gen_meta) -> list[str]:
    """Render chat messages into model-specific prompt strings."""
    enable_thinking = cfg.get("enable_thinking", False)
    try:
        out = [
            tokenizer.apply_chat_template(
                p, tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            for p in prompts
        ]
        gen_meta["enable_thinking"] = enable_thinking
        return out
    except TypeError:
        gen_meta["enable_thinking"] = "not supported by template"
        return [
            tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
            for p in prompts
        ]


def get_hf_responses(prompts: list[list[dict]], cfg: dict) -> tuple[list[str], dict]:
    """
    Generate one completion per prompt.

    :param prompts: list of chat message lists
    :param cfg: YAML config; recognised keys are ``hf_model``, ``adapter``,
        ``batch_size``, ``max_new_tokens``, ``temperature``, ``top_p``,
        ``top_k``, ``do_sample``, ``seed``, ``flash_attention``
    :return: ``(completions, generation_metadata)``
    """
    if cfg.get("seed") is not None:
        torch.manual_seed(int(cfg["seed"]))

    gen_meta = effective_generation_config(
        cfg["hf_model"],
        {
            "temperature": cfg.get("temperature"),
            "top_p": cfg.get("top_p"),
            "top_k": cfg.get("top_k"),
            "do_sample": cfg.get("do_sample"),
        },
    )
    log.info("Sampling parameters (%s): %s", gen_meta["source"], gen_meta)

    pipe = init_pipeline(cfg)

    # Only pass parameters that were explicitly overridden; anything else is
    # left to the model's own generation_config.
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": cfg.get("max_new_tokens", 8),
        "pad_token_id": pipe.tokenizer.pad_token_id,
    }
    for key in ("temperature", "top_p", "top_k", "do_sample"):
        if cfg.get(key) is not None:
            gen_kwargs[key] = cfg[key]

    formatted = _apply_chat_template(pipe.tokenizer, prompts, cfg, gen_meta)
    outputs = pipe(formatted, **gen_kwargs)
    completions = [o[0]["generated_text"] for o in outputs]

    gen_meta["max_new_tokens"] = gen_kwargs["max_new_tokens"]
    gen_meta["n_prompts"] = len(prompts)
    return completions, gen_meta
