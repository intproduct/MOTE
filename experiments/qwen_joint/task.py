"""Qwen next-target loss with bounded vocabulary-logit memory."""
import torch
from torch.utils.checkpoint import checkpoint

from .model import auxiliary_loss


def next_target_loss(model, input_ids, labels, logit_chunk=64):
    """Labels are already shifted by the data release; never shift them again."""
    if input_ids.shape != labels.shape:
        raise ValueError("Inputs and explicit next-token targets must have equal shapes")
    hidden = model.model(input_ids=input_ids, use_cache=False).last_hidden_state
    flat_hidden, flat_labels = hidden.flatten(0, 1), labels.flatten()

    def chunk_loss(h, y):
        logits = model.lm_head(h)
        return torch.nn.functional.cross_entropy(logits.float(), y, ignore_index=-100, reduction="sum")

    pieces = []
    for start in range(0, len(flat_labels), logit_chunk):
        h, y = flat_hidden[start:start + logit_chunk], flat_labels[start:start + logit_chunk]
        if torch.is_grad_enabled() and h.requires_grad:
            pieces.append(checkpoint(chunk_loss, h, y, use_reentrant=False))
        else:
            pieces.append(chunk_loss(h, y))
    return torch.stack(pieces).sum(), int((labels != -100).sum()), auxiliary_loss(model)
