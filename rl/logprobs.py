from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _forward_with_optional_logits_to_keep(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    keep_len: int,
    use_logits_to_keep: bool,
):
    if use_logits_to_keep:
        try:
            return model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, logits_to_keep=int(keep_len))
        except TypeError:
            pass
    return model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)


def gather_response_logprobs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
    response_start: Optional[int] = None,
    use_logits_to_keep: bool = True,
) -> torch.Tensor:
    """Return per-token logprobs aligned to input_ids positions.

    For a causal LM, logits at position j-1 predict token input_ids[j]. This
    function therefore gathers logprobs from logits[:, :-1] against
    input_ids[:, 1:], then pads a zero column at j=0. The returned tensor has
    shape [B, T], where logprobs[b, j] is the logprob of input_ids[b, j].

    Position j=0 has no preceding logits and must be excluded by response_mask.
    The effective mask is response_mask & attention_mask, so prompt tokens and
    pad tokens do not contribute to GRPO loss even though their logprobs are
    present in the returned aligned tensor.
    """
    if input_ids.shape != attention_mask.shape or input_ids.shape != response_mask.shape:
        raise ValueError(
            "input_ids, attention_mask, and response_mask must have the same shape; "
            f"got {tuple(input_ids.shape)}, {tuple(attention_mask.shape)}, {tuple(response_mask.shape)}"
        )
    seq_len = int(input_ids.shape[1])
    if response_start is not None:
        response_start = int(response_start)
        if response_start < 1 or response_start > seq_len:
            raise ValueError(f"response_start must be in [1, {seq_len}], got {response_start}")
        keep_len = seq_len - response_start + 1
        outputs = _forward_with_optional_logits_to_keep(model, input_ids, attention_mask, keep_len, bool(use_logits_to_keep))
    else:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits

    if response_start is None:
        if logits.shape[:2] != input_ids.shape:
            raise ValueError(f"logits shape {tuple(logits.shape)} is incompatible with input_ids {tuple(input_ids.shape)}")
        shifted_logits = logits[:, :-1, :].to(torch.float32)
        shifted_labels = input_ids[:, 1:]
        response_offset = 1
    else:
        target_len = seq_len - int(response_start)
        if target_len <= 0:
            aligned = torch.zeros(input_ids.shape, dtype=torch.float32, device=input_ids.device)
            effective_mask = (response_mask.to(dtype=torch.bool) & attention_mask.to(dtype=torch.bool)).to(dtype=aligned.dtype)
            if effective_mask.shape[1] > 0:
                effective_mask[:, 0] = 0.0
            return aligned * effective_mask
        if int(logits.shape[1]) == seq_len:
            shifted_logits = logits[:, int(response_start) - 1 : -1, :].to(torch.float32)
        elif int(logits.shape[1]) >= target_len:
            shifted_logits = logits[:, :target_len, :].to(torch.float32)
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = outputs.logits
            if logits.shape[:2] != input_ids.shape:
                raise ValueError(f"logits shape {tuple(logits.shape)} is incompatible with input_ids {tuple(input_ids.shape)}")
            shifted_logits = logits[:, int(response_start) - 1 : -1, :].to(torch.float32)
        shifted_labels = input_ids[:, int(response_start) :]
        response_offset = int(response_start)

    if shifted_logits.shape[:2] != shifted_labels.shape:
        raise ValueError(
            f"sliced logits shape {tuple(shifted_logits.shape)} is incompatible with labels {tuple(shifted_labels.shape)}"
        )
    shifted_logprobs = F.log_softmax(shifted_logits, dim=-1)
    gathered = shifted_logprobs.gather(dim=-1, index=shifted_labels.unsqueeze(-1)).squeeze(-1)
    aligned = gathered.new_zeros(input_ids.shape, dtype=torch.float32)
    aligned[:, response_offset:] = gathered

    effective_mask = (response_mask.to(dtype=torch.bool) & attention_mask.to(dtype=torch.bool)).to(dtype=aligned.dtype)
    if effective_mask.shape[1] > 0:
        effective_mask[:, 0] = 0.0
    return aligned * effective_mask
