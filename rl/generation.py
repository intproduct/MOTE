from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch


@contextmanager
def rollout_generation_state(model, *, use_cache: bool) -> Iterator[None]:
    """Temporarily put a generation model in eval mode with a requested cache state."""

    was_training = bool(getattr(model, "training", False))
    config = getattr(model, "config", None)
    has_config_cache = config is not None and hasattr(config, "use_cache")
    original_config_cache = getattr(config, "use_cache", None) if has_config_cache else None
    generation_config = getattr(model, "generation_config", None)
    has_generation_config_cache = generation_config is not None and hasattr(generation_config, "use_cache")
    original_generation_config_cache = (
        getattr(generation_config, "use_cache", None) if has_generation_config_cache else None
    )
    if hasattr(model, "eval"):
        model.eval()
    if has_config_cache:
        config.use_cache = bool(use_cache)
    if has_generation_config_cache:
        generation_config.use_cache = bool(use_cache)
    try:
        yield
    finally:
        if has_generation_config_cache:
            generation_config.use_cache = original_generation_config_cache
        if has_config_cache:
            config.use_cache = original_config_cache
        if hasattr(model, "train"):
            model.train(was_training)


def rollout_grad_context(enabled: bool):
    return torch.inference_mode() if bool(enabled) else torch.no_grad()


def rollout_grad_context_name(enabled: bool) -> str:
    return "inference_mode" if bool(enabled) else "no_grad"


def tokenize_rollout_prompts(tokenizer, prompts, *, max_prompt_tokens: int = 0):
    max_prompt_tokens = int(max_prompt_tokens)
    if max_prompt_tokens <= 0:
        return tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=True)

    had_truncation_side = hasattr(tokenizer, "truncation_side")
    original_truncation_side = getattr(tokenizer, "truncation_side", None) if had_truncation_side else None
    if not had_truncation_side:
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=True)
        for key, value in list(enc.items()):
            if hasattr(value, "shape") and len(value.shape) >= 2 and int(value.shape[1]) > max_prompt_tokens:
                enc[key] = value[:, -max_prompt_tokens:]
        return enc
    if had_truncation_side:
        tokenizer.truncation_side = "left"
    try:
        return tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
            truncation=True,
            max_length=max_prompt_tokens,
        )
    finally:
        if had_truncation_side:
            tokenizer.truncation_side = original_truncation_side


def is_cache_compat_generation_error(exc: Exception) -> bool:
    text = str(exc).lower()
    cache_terms = ("use_cache", "past_key_values", "cache", "gradient checkpoint")
    return any(term in text for term in cache_terms)
