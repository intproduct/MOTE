from __future__ import annotations

import sys
import types
import json
import importlib
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if "MOTE" not in sys.modules:
    mote_pkg = types.ModuleType("MOTE")
    mote_pkg.__path__ = [str(ROOT)]
    sys.modules["MOTE"] = mote_pkg
if "MOTE.rl" not in sys.modules:
    rl_pkg = types.ModuleType("MOTE.rl")
    rl_pkg.__path__ = [str(ROOT / "rl")]
    sys.modules["MOTE.rl"] = rl_pkg
if "MOTE.cli" not in sys.modules:
    cli_pkg = types.ModuleType("MOTE.cli")
    cli_pkg.__path__ = [str(ROOT / "cli")]
    sys.modules["MOTE.cli"] = cli_pkg

from MOTE.gate import TopKGate
from MOTE.config.defaults import make_default_config
from MOTE.rl import data as rl_data
from MOTE.rl.data import load_gsm8k_rl_records
from MOTE.rl.generation import rollout_generation_state
from MOTE.rl.generation import tokenize_rollout_prompts
from MOTE.rl.grpo import compute_group_advantages, grpo_loss
from MOTE.rl.logprobs import gather_response_logprobs
from MOTE.rl.mgpo import compute_mgpo_weights
from MOTE.rl.reward_shaping import apply_long2short_reward_shift
from MOTE.rl.rewards_gsm8k import gsm8k_reward, normalize_number_answer
from MOTE.rl.runtime import set_trainable_mode_for_rl
from MOTE.train.rl_controller import (
    _cuda_memory_snapshot,
    build_zero_advantage_retry_limit_record,
    build_zero_advantage_skip_record,
    build_rl_prompt_text,
    build_rollout_attention_and_response_mask,
    is_zero_advantage_batch,
    iter_response_chunks,
    generate_rollout_sequences,
    log_cuda_memory,
    maybe_collect_train_forward_router_usage,
    maybe_disable_and_reset_runtime_usage,
    maybe_enable_runtime_usage_for_train_forward,
    resolve_amp_dtype,
    rollout_cache_metadata,
    zero_advantage_retry_exceeded,
)
from MOTE.audit import to_jsonable

rl_controller_module = importlib.import_module("MOTE.train.rl_controller")


class FakeOutput:
    def __init__(self, logits):
        self.logits = logits


class ShiftToyLM(nn.Module):
    def __init__(self, vocab_size=8):
        super().__init__()
        self.vocab_size = vocab_size
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        del attention_mask, use_cache
        bsz, seq_len = input_ids.shape
        logits = torch.full((bsz, seq_len, self.vocab_size), -10.0, device=input_ids.device)
        for b in range(bsz):
            for pos in range(seq_len - 1):
                logits[b, pos, int(input_ids[b, pos + 1])] = 10.0
        return FakeOutput(logits + self.bias)


class LogitsToKeepToyLM(ShiftToyLM):
    def __init__(self, vocab_size=8):
        super().__init__(vocab_size=vocab_size)
        self.last_logits_to_keep = None

    def forward(self, input_ids, attention_mask=None, use_cache=False, logits_to_keep=None):
        full = super().forward(input_ids, attention_mask=attention_mask, use_cache=use_cache).logits
        self.last_logits_to_keep = logits_to_keep
        if logits_to_keep is not None:
            return FakeOutput(full[:, -int(logits_to_keep) :, :])
        return FakeOutput(full)


class NoLogitsToKeepToyLM(ShiftToyLM):
    def forward(self, input_ids, attention_mask=None, use_cache=False, logits_to_keep=None):
        if logits_to_keep is not None:
            raise TypeError("logits_to_keep unsupported")
        return super().forward(input_ids, attention_mask=attention_mask, use_cache=use_cache)


class FakeChatTokenizer:
    chat_template = "fake-template"

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append({"messages": list(messages), "kwargs": dict(kwargs)})
        text = "".join(f"[{msg['role']}]{msg['content']}" for msg in messages)
        if kwargs.get("add_generation_prompt"):
            text += "[assistant]"
        return text


class FakeRolloutTokenizer:
    def __init__(self, pad_token_id=0, eos_token_id=2):
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.truncation_side = "right"
        self.calls = []

    def __call__(self, texts, return_tensors=None, padding=False, add_special_tokens=True, truncation=False, max_length=None):
        self.calls.append(
            {
                "texts": list(texts),
                "return_tensors": return_tensors,
                "padding": padding,
                "add_special_tokens": add_special_tokens,
                "truncation": truncation,
                "max_length": max_length,
                "truncation_side": self.truncation_side,
            }
        )
        rows = []
        token_rows = []
        for text in texts:
            tokens = [ord(ch) % 10 + 3 for ch in text]
            if truncation and max_length is not None and len(tokens) > int(max_length):
                if self.truncation_side == "left":
                    tokens = tokens[-int(max_length) :]
                else:
                    tokens = tokens[: int(max_length)]
            token_rows.append(tokens)
        max_len = max(len(tokens) for tokens in token_rows)
        for tokens in token_rows:
            pad_len = max_len - len(tokens)
            rows.append([int(self.pad_token_id)] * pad_len + tokens)
        input_ids = torch.tensor(rows, dtype=torch.long)
        attention_mask = (input_ids != int(self.pad_token_id)).to(dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class FakeGenerationConfig:
    def __init__(self, use_cache=False):
        self.use_cache = use_cache


class FakeRolloutModel(nn.Module):
    def __init__(self, lengths=None, pad_token_id=0, eos_token_id=2, generation_config=True):
        super().__init__()
        self.param = nn.Parameter(torch.zeros(()))
        self.config = types.SimpleNamespace(use_cache=False)
        if generation_config:
            self.generation_config = FakeGenerationConfig(use_cache=False)
        self.lengths = list(lengths or [])
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.generate_calls = []
        self.forward_use_cache = []

    @property
    def device(self):
        return self.param.device

    def generate(self, input_ids, attention_mask=None, use_cache=False, **kwargs):
        del attention_mask
        self.generate_calls.append(
            {
                "use_cache": bool(use_cache),
                "training": bool(self.training),
                "config_use_cache": bool(getattr(self.config, "use_cache")),
                "generation_config_use_cache": (
                    bool(getattr(self.generation_config, "use_cache")) if hasattr(self, "generation_config") else None
                ),
                "kwargs": dict(kwargs),
            }
        )
        call_idx = len(self.generate_calls) - 1
        extra = self.lengths[call_idx] if call_idx < len(self.lengths) else 2
        suffix = torch.arange(10, 10 + int(extra), dtype=input_ids.dtype, device=input_ids.device).unsqueeze(0)
        suffix = suffix.repeat(int(input_ids.shape[0]), 1)
        if int(extra) > 0:
            suffix[:, -1] = int(self.eos_token_id)
        return torch.cat([input_ids, suffix], dim=1)

    def forward(self, input_ids, attention_mask=None, use_cache=False, logits_to_keep=None):
        del attention_mask, logits_to_keep
        self.forward_use_cache.append(bool(use_cache))
        logits = torch.zeros((*input_ids.shape, 32), dtype=torch.float32, device=input_ids.device)
        return FakeOutput(logits)


def test_group_advantages_are_per_prompt_group_and_safe_for_zero_std():
    rewards = torch.tensor([[1.0, 0.0], [5.0, 5.0]])
    adv = compute_group_advantages(rewards, group_size=2)
    assert torch.allclose(adv[0], torch.tensor([1.0, -1.0]))
    assert torch.allclose(adv[1], torch.tensor([0.0, 0.0]))
    assert torch.isfinite(adv).all()


def test_mgpo_weights_peak_at_half_and_are_finite_near_edges():
    prompt_acc = torch.tensor([0.0, 0.5, 1.0, 1e-9, 1.0 - 1e-9])
    weights = compute_mgpo_weights(prompt_acc, gamma=2.0, weight_min=0.0, weight_max=1.0, eps=1e-6)
    assert weights.shape == prompt_acc.shape
    assert weights[1] > weights[0]
    assert weights[1] > weights[2]
    assert torch.isfinite(weights).all()


def test_mgpo_gamma_zero_gives_unclipped_unit_weights_and_advantage_scaling():
    prompt_acc = torch.tensor([0.0, 0.5, 1.0])
    weights = compute_mgpo_weights(prompt_acc, gamma=0.0, weight_min=0.0, weight_max=2.0)
    assert torch.allclose(weights, torch.ones_like(weights))

    advantages = torch.tensor([[1.0, -1.0], [0.5, -0.5], [2.0, -2.0]])
    weighted = advantages * weights.unsqueeze(1)
    for bidx in range(advantages.shape[0]):
        assert torch.allclose(weighted[bidx], advantages[bidx] * weights[bidx])


def test_long2short_reward_shift_preserves_expected_groups_and_wrong_rewards():
    all_wrong = torch.tensor([[0.0, 0.0, 0.0]])
    assert torch.allclose(apply_long2short_reward_shift(all_wrong, torch.tensor([[3.0, 2.0, 1.0]])), all_wrong)

    single_correct = torch.tensor([[1.0, 0.0, 0.0]])
    assert torch.allclose(apply_long2short_reward_shift(single_correct, torch.tensor([[3.0, 2.0, 1.0]])), single_correct)

    rewards = torch.tensor([[1.0, 1.0, 0.0]])
    shaped = apply_long2short_reward_shift(rewards, torch.tensor([[2.0, 10.0, 1.0]]), lambda_value=0.2)
    assert shaped[0, 0] > shaped[0, 1]
    assert shaped[0, 2] == rewards[0, 2]
    assert torch.allclose((shaped[0, :2] - rewards[0, :2]).sum(), torch.tensor(0.0), atol=1e-6)
    assert torch.isfinite(shaped).all()


def test_grpo_loss_beta_zero_clamps_log_ratio_and_detaches_old():
    new = torch.tensor([[30.0, 0.0, 0.0]], requires_grad=True)
    old = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    adv = torch.tensor([1.0])
    loss, metrics = grpo_loss(new, old, None, adv, mask, beta=0.0)
    loss.backward()
    assert new.grad is not None
    assert old.grad is None
    assert metrics["raw_log_ratio_max"].item() == 30.0
    assert torch.isfinite(loss)


def test_grpo_loss_uses_per_response_mean_and_accepts_bg_advantages():
    new = torch.log(torch.tensor([[2.0, 2.0, 2.0, 2.0], [4.0, 1.0, 1.0, 1.0]]))
    old = torch.zeros_like(new)
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0]])
    advantages = torch.tensor([[1.0, 1.0]])
    loss, metrics = grpo_loss(new, old, None, advantages, mask, eps_clip=10.0, beta=0.0)
    # Per-response ratio mean is mean(mean([2,2,2,2]), mean([4])) = 3.
    # A forbidden global token mean would be mean([2,2,2,2,4]) = 2.4.
    assert torch.allclose(metrics["ratio_mean"], torch.tensor(3.0), atol=1e-6)
    assert torch.allclose(loss, torch.tensor(-3.0), atol=1e-6)


def test_microbatch_weighted_grpo_loss_matches_full_batch():
    new = torch.tensor(
        [[0.1, 0.2, 0.0], [0.3, -0.1, 0.0], [0.2, 0.5, -0.2], [-0.4, 0.1, 0.2]],
        requires_grad=True,
    )
    old = torch.tensor([[0.0, 0.1, 0.0], [0.1, -0.2, 0.0], [0.0, 0.2, -0.1], [-0.2, 0.0, 0.1]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
    advantages = torch.tensor([[1.0, -1.0], [0.5, -0.5]])

    full_loss, full_metrics = grpo_loss(new, old, None, advantages, mask, eps_clip=0.2, beta=0.0)
    weighted = None
    for start, end in iter_response_chunks(total_n=4, micro_batch_size=2):
        chunk_loss, _ = grpo_loss(
            new[start:end],
            old[start:end],
            None,
            advantages.reshape(-1)[start:end],
            mask[start:end],
            eps_clip=0.2,
            beta=0.0,
        )
        item = chunk_loss * ((end - start) / 4.0)
        weighted = item if weighted is None else weighted + item

    assert torch.allclose(weighted, full_loss, atol=1e-6)
    assert torch.isfinite(full_metrics["ratio_mean"])


def test_response_chunk_slicing_keeps_response_order_aligned():
    input_ids = torch.arange(24).reshape(6, 4)
    old_logprobs = input_ids.to(torch.float32) + 100.0
    advantages = torch.arange(6).reshape(2, 3).to(torch.float32)
    mask = torch.ones_like(old_logprobs)

    seen = []
    for start, end in iter_response_chunks(total_n=6, micro_batch_size=2):
        seen.extend(input_ids[start:end, 0].tolist())
        assert torch.equal(old_logprobs[start:end, 0], input_ids[start:end, 0].to(torch.float32) + 100.0)
        assert torch.equal(advantages.reshape(-1)[start:end], torch.arange(start, end).to(torch.float32))
        assert mask[start:end].shape[0] == end - start

    assert seen == [0, 4, 8, 12, 16, 20]


def test_zero_advantage_detection_and_skip_record_are_jsonable():
    cfg = make_default_config()
    cfg.rl.max_zero_advantage_rollout_retries = 3
    zero_adv = compute_group_advantages(torch.tensor([[1.0, 1.0, 1.0, 1.0]]), group_size=4)
    nonzero_adv = compute_group_advantages(torch.tensor([[0.0, 1.0, 0.0, 1.0]]), group_size=4)
    assert is_zero_advantage_batch(zero_adv)
    assert not is_zero_advantage_batch(nonzero_adv)

    record = build_zero_advantage_skip_record(
        fit_cfg=cfg,
        micro_step=7,
        update_step=2,
        next_update_step=3,
        zero_advantage_retry_count=1,
        rewards=[1.0, 1.0, 1.0, 1.0],
        reward_debugs=[
            {"strict_match": True, "fallback_match": True},
            {"strict_match": True, "fallback_match": True},
            {"strict_match": True, "fallback_match": True},
            {"strict_match": True, "fallback_match": True},
        ],
        response_lens=[10.0, 11.0, 12.0, 13.0],
    )
    dumped = json.dumps(to_jsonable(record), ensure_ascii=False)
    assert record["kind"] == "zero_advantage_skip"
    assert record["update_step"] == 2
    assert record["next_update_step"] == 3
    assert record["max_zero_advantage_rollout_retries"] == 3
    assert "tensor" not in dumped.lower()


def test_zero_advantage_retry_cap_predicate():
    assert not zero_advantage_retry_exceeded(7, 8)
    assert zero_advantage_retry_exceeded(8, 8)
    assert zero_advantage_retry_exceeded(9, 8)


def test_zero_advantage_retry_limit_record_warn_continue_is_jsonable():
    cfg = make_default_config()
    cfg.rl.zero_advantage_retry_action = "warn_continue"
    cfg.rl.max_zero_advantage_rollout_retries = 2
    record = build_zero_advantage_retry_limit_record(
        fit_cfg=cfg,
        micro_step=4,
        update_step=0,
        zero_advantage_retry_count=2,
    )
    dumped = json.dumps(to_jsonable(record), ensure_ascii=False)
    assert record["kind"] == "zero_advantage_retry_limit"
    assert record["zero_advantage_retry_action"] == "warn_continue"
    assert record["zero_advantage_retry_count"] == 2
    assert "tensor" not in dumped.lower()


def test_usage_tracking_disabled_does_not_collect_runtime_usage_tensors():
    class ExplodingCore:
        def collect_runtime_usage_tensors(self):
            raise AssertionError("collect_runtime_usage_tensors should not be called")

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.core = ExplodingCore()

    usage, warning = maybe_collect_train_forward_router_usage(Wrapper(), enabled=False)
    assert usage is None
    assert warning == "usage_tracking_disabled"


def test_usage_tracking_helpers_short_circuit_by_default_and_call_when_enabled():
    cfg = make_default_config()
    cfg.rl.enable_usage_tracking = False
    model = nn.Linear(1, 1)
    with patch.object(rl_controller_module, "reset_runtime_usage_buffers") as reset_mock:
        assert maybe_disable_and_reset_runtime_usage(cfg, model) is False
        assert maybe_enable_runtime_usage_for_train_forward(cfg, model) is False
    reset_mock.assert_not_called()

    cfg.rl.enable_usage_tracking = True
    with patch.object(rl_controller_module, "reset_runtime_usage_buffers") as reset_mock, patch.object(
        rl_controller_module, "disable_runtime_usage_tracking"
    ) as disable_mock, patch.object(rl_controller_module, "enable_runtime_usage_tracking") as enable_mock:
        assert maybe_disable_and_reset_runtime_usage(cfg, model) is True
        disable_mock.assert_called_once_with(model)
        reset_mock.assert_called_once_with(model)
        reset_mock.reset_mock()
        assert maybe_enable_runtime_usage_for_train_forward(cfg, model) is True
        reset_mock.assert_called_once_with(model)
        enable_mock.assert_called_once_with(model)


def test_rl_sample_payload_is_jsonable_without_tensors():
    payload = {
        "kind": "samples",
        "groups": [
            {
                "prompt": "q",
                "samples": [{"generated_text": "a", "reward": torch.tensor(1.0), "advantage": torch.tensor(0.5)}],
            }
        ],
    }
    dumped = json.dumps(to_jsonable(payload), ensure_ascii=False)
    assert "tensor" not in dumped.lower()
    assert '"reward": 1.0' in dumped


def test_log_cuda_memory_smoke_no_cuda_safe():
    import logging

    logger = logging.getLogger("test_log_cuda_memory_smoke_no_cuda_safe")
    log_cuda_memory(logger, tag="smoke", update_step=1, micro_step=1)
    snapshot = _cuda_memory_snapshot()
    assert set(snapshot) == {
        "memory_allocated_gib",
        "memory_reserved_gib",
        "memory_max_allocated_gib",
        "memory_max_reserved_gib",
    }


def test_gather_response_logprobs_alignment_masks_j0_and_prompt():
    model = ShiftToyLM()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)
    response_mask = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    logprobs = gather_response_logprobs(model, input_ids, attention_mask, response_mask)
    assert logprobs.shape == input_ids.shape
    assert logprobs[0, 0].item() == 0.0
    assert logprobs[0, 1].item() == 0.0
    assert logprobs[0, 2].item() > -1e-3
    assert logprobs[0, 3].item() > -1e-3


def test_response_only_logprobs_match_full_sequence_and_zero_prompt():
    input_ids = torch.tensor([[1, 2, 3, 4, 5], [1, 3, 4, 5, 6]])
    attention_mask = torch.ones_like(input_ids)
    response_mask = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0, 1.0]])
    response_start = 3

    full = gather_response_logprobs(ShiftToyLM(), input_ids, attention_mask, response_mask)
    kept_model = LogitsToKeepToyLM()
    response_only = gather_response_logprobs(
        kept_model,
        input_ids,
        attention_mask,
        response_mask,
        response_start=response_start,
        use_logits_to_keep=True,
    )
    fallback = gather_response_logprobs(
        NoLogitsToKeepToyLM(),
        input_ids,
        attention_mask,
        response_mask,
        response_start=response_start,
        use_logits_to_keep=True,
    )

    assert kept_model.last_logits_to_keep == input_ids.shape[1] - response_start + 1
    assert torch.allclose(response_only[:, response_start:], full[:, response_start:], atol=1e-6)
    assert torch.allclose(fallback[:, response_start:], full[:, response_start:], atol=1e-6)
    assert torch.equal(response_only[:, :response_start], torch.zeros_like(response_only[:, :response_start]))


def test_rollout_generation_state_restores_training_and_cache_state():
    model = FakeRolloutModel(generation_config=True)
    model.train()
    model.config.use_cache = True
    model.generation_config.use_cache = True
    with rollout_generation_state(model, use_cache=False):
        assert not model.training
        assert model.config.use_cache is False
        assert model.generation_config.use_cache is False
    assert model.training
    assert model.config.use_cache is True
    assert model.generation_config.use_cache is True

    model_without_generation_config = FakeRolloutModel(generation_config=False)
    with rollout_generation_state(model_without_generation_config, use_cache=True):
        assert not model_without_generation_config.training


def test_rollout_cache_metadata_allows_cache_with_gradient_checkpointing():
    cfg = make_default_config()
    cfg.rl.rollout_use_cache = True
    cfg.rl.rollout_inference_mode = True
    cfg.rl.gradient_checkpointing = True
    info = rollout_cache_metadata(cfg)
    assert info["rollout_use_cache"] is True
    assert info["effective_rollout_use_cache"] is True
    assert info["rollout_use_cache_reason"] == "requested_enabled"
    assert info["rollout_grad_context"] == "inference_mode"


def test_rollout_prompt_tokenization_default_and_left_truncation():
    tokenizer = FakeRolloutTokenizer()
    enc = tokenize_rollout_prompts(tokenizer, ["abcdef", "xy"], max_prompt_tokens=0)
    assert tokenizer.calls[-1]["truncation"] is False
    assert tokenizer.calls[-1]["max_length"] is None
    assert enc["input_ids"].shape[1] == 6

    enc = tokenize_rollout_prompts(tokenizer, ["abcdef", "uvwxyz"], max_prompt_tokens=3)
    assert tokenizer.calls[-1]["truncation"] is True
    assert tokenizer.calls[-1]["max_length"] == 3
    assert tokenizer.calls[-1]["truncation_side"] == "left"
    assert tokenizer.truncation_side == "right"
    assert enc["input_ids"].shape[1] == 3


def test_rollout_generate_uses_cache_eval_and_logprobs_stay_no_cache():
    cfg = make_default_config()
    cfg.rl.rollout_use_cache = True
    cfg.rl.rollout_inference_mode = True
    cfg.rl.rollout_micro_batch_size = 0
    cfg.rl.max_new_tokens = 2
    tokenizer = FakeRolloutTokenizer()
    model = FakeRolloutModel(lengths=[2])
    model.train()
    enc = tokenizer(["abc", "de"], return_tensors="pt", padding=True, add_special_tokens=True)

    generated, original_seq_lens, info = generate_rollout_sequences(
        fit_cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
    )

    assert model.training
    assert len(model.generate_calls) == 1
    assert model.generate_calls[0]["use_cache"] is True
    assert model.generate_calls[0]["training"] is False
    assert model.generate_calls[0]["config_use_cache"] is True
    assert model.generate_calls[0]["generation_config_use_cache"] is True
    for unused_key in ("output_scores", "return_dict_in_generate", "output_hidden_states", "output_attentions"):
        assert unused_key not in model.generate_calls[0]["kwargs"]
    assert info["effective_rollout_use_cache"] is True
    assert original_seq_lens == [generated.shape[1], generated.shape[1]]

    response_start = enc["input_ids"].shape[1]
    full_attention, response_mask = build_rollout_attention_and_response_mask(
        generated,
        enc["attention_mask"],
        response_start=response_start,
        original_seq_lens=original_seq_lens,
        eos_token_id=tokenizer.eos_token_id,
    )
    _ = gather_response_logprobs(model, generated, full_attention, response_mask, response_start=response_start)
    assert model.forward_use_cache[-1] is False


def test_rollout_microbatch_padding_masks_manual_tail_and_pad_equals_eos():
    cfg = make_default_config()
    cfg.rl.rollout_use_cache = True
    cfg.rl.rollout_micro_batch_size = 1
    cfg.rl.max_new_tokens = 4
    tokenizer = FakeRolloutTokenizer(pad_token_id=2, eos_token_id=2)
    model = FakeRolloutModel(lengths=[3, 1], pad_token_id=2, eos_token_id=2)
    enc = tokenize_rollout_prompts(tokenizer, ["abcd", "ef"], max_prompt_tokens=3)
    response_start = enc["input_ids"].shape[1]

    generated, original_seq_lens, _ = generate_rollout_sequences(
        fit_cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
    )
    full_attention, response_mask = build_rollout_attention_and_response_mask(
        generated,
        enc["attention_mask"],
        response_start=response_start,
        original_seq_lens=original_seq_lens,
        eos_token_id=tokenizer.eos_token_id,
    )

    assert len(model.generate_calls) == 2
    assert len(tokenizer.calls) == 1
    assert response_start == 3
    assert generated.shape[0] == 2
    assert response_mask.shape == generated.shape
    assert original_seq_lens[0] > original_seq_lens[1]
    assert full_attention[1, original_seq_lens[1] :].sum().item() == 0
    assert response_mask[1, original_seq_lens[1] :].sum().item() == 0
    assert torch.equal(full_attention[:, :response_start], enc["attention_mask"])
    # The valid EOS position is included even when pad_token_id == eos_token_id;
    # only the manually padded tail is masked out by original_seq_lens.
    assert response_mask[1, original_seq_lens[1] - 1].item() == 1.0


def test_resolve_amp_dtype_tracks_model_parameter_dtype():
    assert resolve_amp_dtype(nn.Linear(1, 1).to(dtype=torch.float16)) is torch.float16
    assert resolve_amp_dtype(nn.Linear(1, 1).to(dtype=torch.bfloat16)) is torch.bfloat16
    assert resolve_amp_dtype(nn.Linear(1, 1).to(dtype=torch.float32)) is None


def test_gsm8k_reward_records_strict_fallback_and_reward_acc_inputs():
    reward, debug = gsm8k_reward("We compute it. #### 1,234.0", "#### 1234")
    assert reward == 1.0
    assert debug["strict_match"] is True
    assert debug["fallback_match"] is True
    assert debug["match_type"] == "strict_hash"

    reward, debug = gsm8k_reward("No marker, final answer is $42.", "#### 42")
    assert reward == 1.0
    assert debug["strict_match"] is False
    assert debug["final_answer_match"] is True
    assert debug["fallback_match"] is True
    assert debug["match_type"] == "final_answer"
    assert normalize_number_answer(" -1,234.50 ") == "-1234.50"


def test_gsm8k_reward_supports_chat_final_answer_markers():
    cases = [
        ("#### 72", "strict_hash"),
        ("Work\nFinal Answer: 72", "final_answer"),
        ("After solving, final answer is 72", "final_answer"),
        ("Reasoning with 10.\nAnswer: 72", "answer_marker"),
        ("No marker, numbers 10 then 72", "fallback_last_number"),
    ]
    for text, match_type in cases:
        reward, debug = gsm8k_reward(text, "#### 72")
        assert reward == 1.0
        assert debug["match_type"] == match_type


def test_rl_prompt_format_chat_uses_apply_chat_template_and_raw_does_not():
    cfg = make_default_config()
    cfg.rl.prompt_template = "Question:\n{question}"
    tokenizer = FakeChatTokenizer()

    cfg.rl.prompt_format = "chat"
    cfg.rl.chat_enable_thinking = False
    prompt = build_rl_prompt_text(cfg, tokenizer, "1+1?")
    assert "[assistant]" in prompt
    assert tokenizer.calls[-1]["kwargs"]["add_generation_prompt"] is True
    assert tokenizer.calls[-1]["kwargs"]["enable_thinking"] is False

    cfg.rl.prompt_format = "raw"
    raw_prompt = build_rl_prompt_text(cfg, tokenizer, "1+1?")
    assert raw_prompt == "Question:\n1+1?"
    assert len(tokenizer.calls) == 1


def test_trainable_router_mode_does_not_match_native_gate_proj_by_name():
    class FakeNativeMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(4, 4)

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.native = FakeNativeMLP()
            self.mote_gate = TopKGate(data_dim=4, num_experts=2, k=1, min_capacity=1)

    model = FakeModel()
    names = set_trainable_mode_for_rl(model, "router_only")["trainable_names"]
    assert names
    assert all("native.gate_proj" not in name for name in names)
    assert all(not p.requires_grad for p in model.native.gate_proj.parameters())


def test_trainable_mode_errors_without_mote_params():
    model = nn.Sequential(nn.Linear(4, 4))
    try:
        set_trainable_mode_for_rl(model, "router_only")
    except RuntimeError as exc:
        assert "selected no trainable parameters" in str(exc)
    else:
        raise AssertionError("router_only should fail without MOTE router parameters")


def test_global_only_requires_identifiable_global_block():
    class FakeCore(nn.Module):
        def __init__(self):
            super().__init__()
            self.global_block = nn.Linear(4, 4)

    model = nn.Module()
    model.core = FakeCore()
    info = set_trainable_mode_for_rl(model, "global_only")
    assert info["trainable_names"]
    assert all(name.startswith("core.global_block") for name in info["trainable_names"])


def test_response_mask_uses_padded_response_start():
    from MOTE.cli.train_grpo_gsm8k import _build_response_mask

    seq = torch.tensor([[0, 5, 6, 7, 2, 2], [8, 9, 0, 1, 3, 2]])
    mask = _build_response_mask(seq, response_start=3, eos_token_id=2)
    assert mask[:, :3].sum().item() == 0.0
    assert torch.equal(mask[0], torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 0.0]))
    assert torch.equal(mask[1], torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]))


def test_rl_train_json_records_load_question_and_prompt(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(
        '{"question":"q1","answer":"#### 1"}\n{"prompt":"q2","answer":"2","idx":"custom"}\n',
        encoding="utf-8",
    )
    cfg = make_default_config()
    cfg.rl.enabled = True
    cfg.rl.max_steps = 1
    cfg.rl.train_json = str(path)
    records = load_gsm8k_rl_records(cfg)
    assert [record["question"] for record in records] == ["q1", "q2"]
    assert records[1]["idx"] == "custom"


def test_rl_config_data_records_use_gsm8k_cache_path():
    cfg = make_default_config()
    cfg.rl.enabled = True
    cfg.rl.max_steps = 1
    cfg.rl.train_json = None
    cfg.rl.use_config_data = True
    fake_rows = [{"question": "q", "answer": "#### 7"}]
    with patch.object(rl_data, "load_dataset_auto_cached", return_value=fake_rows) as mocked:
        records = load_gsm8k_rl_records(cfg)
    mocked.assert_called_once()
    assert records == [{"question": "q", "answer": "#### 7", "idx": 0, "source": "gsm8k_train"}]
