from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import torch

from .generation import (
    is_cache_compat_generation_error,
    rollout_generation_state,
    rollout_grad_context,
    rollout_grad_context_name,
)


@dataclass
class RolloutGenerationConfig:
    max_new_tokens: int
    temperature: float
    top_p: float
    rollout_micro_batch_size: int = 0
    rollout_use_cache: bool = True
    rollout_inference_mode: bool = True
    rollout_log_timing: bool = True
    seed: Optional[int] = None
    do_sample: bool = True
    stop_sequences: tuple[str, ...] = ()
    top_k: Optional[int] = None


@dataclass
class RolloutBatch:
    sequences: torch.Tensor
    original_seq_lens: List[int]
    response_starts: List[int]
    response_token_ids: List[List[int]]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RolloutSyncResult:
    synced: bool
    policy_version: int
    policy_lag_updates: int
    export_dir: Optional[str] = None
    sync_sec: float = 0.0
    engine_rebuild_sec: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class RolloutBackend(Protocol):
    name: str

    def generate(
        self,
        *,
        model,
        tokenizer,
        prompts: List[str],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        generation_config: RolloutGenerationConfig,
        update_step: int,
    ) -> RolloutBatch:
        ...

    def sync_policy(
        self,
        *,
        model,
        tokenizer,
        update_step: int,
        force: bool = False,
    ) -> RolloutSyncResult:
        ...

    def close(self) -> None:
        ...


def rollout_cache_metadata_from_config(config: RolloutGenerationConfig) -> Dict[str, Any]:
    requested = bool(config.rollout_use_cache)
    return {
        "rollout_use_cache": requested,
        "effective_rollout_use_cache": requested,
        "rollout_use_cache_reason": "requested_enabled" if requested else "requested_disabled",
        "rollout_inference_mode": bool(config.rollout_inference_mode),
        "rollout_grad_context": rollout_grad_context_name(bool(config.rollout_inference_mode)),
        "rollout_log_timing": bool(config.rollout_log_timing),
    }


def iter_response_chunks(total_n: int, micro_batch_size: int) -> List[Tuple[int, int]]:
    total_n = int(total_n)
    mb = int(micro_batch_size)
    if total_n <= 0:
        return []
    if mb <= 0 or mb >= total_n:
        return [(0, total_n)]
    return [(start, min(start + mb, total_n)) for start in range(0, total_n, mb)]


def rollout_pad_token_id(tokenizer) -> int:
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        return int(pad_token_id)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        return int(eos_token_id)
    return 0


def pad_generated_to_max_len(generated_chunks: Sequence[torch.Tensor], pad_token_id: int) -> torch.Tensor:
    if not generated_chunks:
        raise RuntimeError("No generated chunks were produced")
    max_len = max(int(chunk.shape[1]) for chunk in generated_chunks)
    padded_chunks = []
    for chunk in generated_chunks:
        pad_len = max_len - int(chunk.shape[1])
        if pad_len <= 0:
            padded_chunks.append(chunk)
            continue
        pad = torch.full(
            (int(chunk.shape[0]), int(pad_len)),
            int(pad_token_id),
            dtype=chunk.dtype,
            device=chunk.device,
        )
        padded_chunks.append(torch.cat([chunk, pad], dim=1))
    return torch.cat(padded_chunks, dim=0)


def build_compact_rollout_sequences(
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    generated_token_ids: Sequence[Sequence[int]],
    pad_token_id: int,
) -> Tuple[torch.Tensor, List[int], List[int], List[List[int]]]:
    """Build canonical ``valid_prompt + response + trailing_pad`` trajectories.

    Rollout engines consume prompts after removing tokenizer padding. Policy,
    reference, and loss forwards must therefore score exactly that same token
    history. Prompt padding must never be retained before or inside the valid
    causal span.
    """
    if input_ids.shape != attention_mask.shape:
        raise ValueError("input_ids and attention_mask must have identical shapes")
    if int(input_ids.shape[0]) != len(generated_token_ids):
        raise ValueError("input_ids and generated_token_ids must have the same batch size")

    rows: List[torch.Tensor] = []
    original_seq_lens: List[int] = []
    response_starts: List[int] = []
    normalized_responses: List[List[int]] = []
    for row, mask, response_ids in zip(input_ids, attention_mask, generated_token_ids):
        valid_prompt = row[mask.to(device=row.device, dtype=torch.bool)]
        if int(valid_prompt.numel()) <= 0:
            raise ValueError("rollout prompts must contain at least one valid token")
        response = [int(value) for value in response_ids]
        suffix = torch.tensor(response, dtype=row.dtype, device=row.device)
        full = torch.cat([valid_prompt, suffix], dim=0)
        rows.append(full)
        response_starts.append(int(valid_prompt.numel()))
        original_seq_lens.append(int(full.numel()))
        normalized_responses.append(response)

    max_len = max(original_seq_lens, default=0)
    padded_rows: List[torch.Tensor] = []
    for row in rows:
        pad_len = int(max_len - row.numel())
        if pad_len > 0:
            tail = torch.full(
                (pad_len,),
                int(pad_token_id),
                dtype=row.dtype,
                device=row.device,
            )
            row = torch.cat([row, tail], dim=0)
        padded_rows.append(row)
    if not padded_rows:
        raise RuntimeError("No rollout rows were produced")
    return torch.stack(padded_rows, dim=0), original_seq_lens, response_starts, normalized_responses


def build_generation_config_from_fit_cfg(fit_cfg) -> RolloutGenerationConfig:
    return RolloutGenerationConfig(
        max_new_tokens=int(fit_cfg.rl.max_new_tokens),
        temperature=float(fit_cfg.rl.temperature),
        top_p=float(fit_cfg.rl.top_p),
        rollout_micro_batch_size=int(getattr(fit_cfg.rl, "rollout_micro_batch_size", 0)),
        rollout_use_cache=bool(getattr(fit_cfg.rl, "rollout_use_cache", True)),
        rollout_inference_mode=bool(getattr(fit_cfg.rl, "rollout_inference_mode", True)),
        rollout_log_timing=bool(getattr(fit_cfg.rl, "rollout_log_timing", True)),
        seed=getattr(fit_cfg.rl, "seed", None),
        stop_sequences=tuple(
            str(value)
            for value in getattr(fit_cfg.rl, "rollout_stop_sequences", ())
            if str(value)
        ),
    )


class HFRolloutBackend:
    name = "hf"

    def __init__(self, logger=None):
        self.logger = logger
        self._last_sync = RolloutSyncResult(synced=True, policy_version=0, policy_lag_updates=0)

    def generate(
        self,
        *,
        model,
        tokenizer,
        prompts: List[str],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        generation_config: RolloutGenerationConfig,
        update_step: int,
    ) -> RolloutBatch:
        del prompts, update_step
        cache_info = rollout_cache_metadata_from_config(generation_config)
        requested_use_cache = bool(cache_info["effective_rollout_use_cache"])
        grad_context_enabled = bool(generation_config.rollout_inference_mode)
        micro_batch_size = int(generation_config.rollout_micro_batch_size)
        pad_token_id = rollout_pad_token_id(tokenizer)
        response_start = int(input_ids.shape[1])

        generate_kwargs = {
            "max_new_tokens": int(generation_config.max_new_tokens),
            "do_sample": bool(generation_config.do_sample),
            "pad_token_id": pad_token_id,
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        }
        if generation_config.do_sample:
            generate_kwargs["temperature"] = float(generation_config.temperature)
            generate_kwargs["top_p"] = float(generation_config.top_p)
            if generation_config.top_k is not None:
                generate_kwargs["top_k"] = int(generation_config.top_k)
        if generation_config.stop_sequences:
            # Modern transformers implements the same text-stop contract used
            # by lm_eval when both stop_strings and tokenizer are supplied.
            generate_kwargs["stop_strings"] = list(generation_config.stop_sequences)
            generate_kwargs["tokenizer"] = tokenizer

        def attempt(use_cache: bool) -> Tuple[torch.Tensor, List[int]]:
            chunks: List[torch.Tensor] = []
            seq_lens: List[int] = []
            for start, end in iter_response_chunks(int(input_ids.shape[0]), micro_batch_size):
                with rollout_grad_context(grad_context_enabled):
                    with rollout_generation_state(model, use_cache=bool(use_cache)):
                        chunk = model.generate(
                            input_ids=input_ids[start:end],
                            attention_mask=attention_mask[start:end],
                            use_cache=bool(use_cache),
                            **generate_kwargs,
                        )
                chunks.append(chunk)
                seq_lens.extend([int(chunk.shape[1])] * int(chunk.shape[0]))
            return pad_generated_to_max_len(chunks, pad_token_id), seq_lens

        try:
            generated, generated_seq_lens = attempt(requested_use_cache)
        except Exception as exc:
            if requested_use_cache and is_cache_compat_generation_error(exc):
                if self.logger is not None:
                    self.logger.warning(
                        "[RLRollout] use_cache=true generate failed; retrying once with use_cache=false: %s",
                        exc,
                    )
                generated, generated_seq_lens = attempt(False)
                cache_info["effective_rollout_use_cache"] = False
                cache_info["rollout_use_cache_reason"] = "fallback_after_error"
                cache_info["rollout_use_cache_error"] = str(exc)
            else:
                raise
        response_token_ids = [
            generated[row, response_start:int(seq_len)].detach().cpu().tolist()
            for row, seq_len in enumerate(generated_seq_lens)
        ]
        sequences, original_seq_lens, response_starts, response_token_ids = build_compact_rollout_sequences(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generated_token_ids=response_token_ids,
            pad_token_id=pad_token_id,
        )
        return RolloutBatch(
            sequences=sequences,
            original_seq_lens=original_seq_lens,
            response_starts=response_starts,
            response_token_ids=response_token_ids,
            metadata={"rollout_backend": self.name, **cache_info},
        )

    def sync_policy(
        self,
        *,
        model,
        tokenizer,
        update_step: int,
        force: bool = False,
    ) -> RolloutSyncResult:
        del model, tokenizer, force
        self._last_sync = RolloutSyncResult(
            synced=True,
            policy_version=int(update_step),
            policy_lag_updates=0,
            metadata={"rollout_backend": self.name},
        )
        return self._last_sync

    def close(self) -> None:
        return None
