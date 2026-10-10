from types import SimpleNamespace

import torch

from fitmotn.rl.grpo import grpo_loss
from fitmotn.rl.logprobs import gather_response_logprobs
from fitmotn.train.rl_controller import compute_new_logprobs_loss_microbatched


class EntropyToyLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 5):
        super().__init__()
        self.vocab_size = vocab_size
        self.scale = torch.nn.Parameter(torch.tensor(0.7))

    def forward(self, input_ids, attention_mask, use_cache=False, logits_to_keep=None):
        del attention_mask, use_cache
        base = torch.arange(self.vocab_size, device=input_ids.device, dtype=torch.float32) * self.scale
        logits = base.view(1, 1, -1).expand(input_ids.shape[0], input_ids.shape[1], -1)
        if logits_to_keep is not None:
            logits = logits[:, -int(logits_to_keep) :, :]
        return SimpleNamespace(logits=logits)


def test_response_entropy_matches_full_distribution_and_mask():
    model = EntropyToyLM()
    input_ids = torch.tensor([[0, 1, 2, 3]])
    attention = torch.ones_like(input_ids)
    response_mask = torch.tensor([[0, 0, 1, 1]])
    default = gather_response_logprobs(model, input_ids, attention, response_mask, response_start=2)
    logprobs, entropy = gather_response_logprobs(
        model,
        input_ids,
        attention,
        response_mask,
        response_start=2,
        return_entropy=True,
    )
    torch.testing.assert_close(logprobs, default)
    distribution_logprobs = torch.log_softmax(
        torch.arange(model.vocab_size, dtype=torch.float32) * model.scale.detach(), dim=-1
    )
    expected = -(distribution_logprobs.exp() * distribution_logprobs).sum()
    torch.testing.assert_close(entropy[0, 2:], expected.expand(2))
    torch.testing.assert_close(entropy[0, :2], torch.zeros(2))


def test_microbatched_grpo_reports_finite_positive_entropy():
    model = EntropyToyLM()
    input_ids = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    attention = torch.ones_like(input_ids)
    response_mask = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])
    old = gather_response_logprobs(model, input_ids, attention, response_mask, response_start=[2, 2]).detach()
    _, metrics = compute_new_logprobs_loss_microbatched(
        model=model,
        input_ids=input_ids,
        attention_mask=attention,
        response_mask=response_mask,
        old_logprobs=old,
        ref_logprobs=None,
        advantages=torch.tensor([1.0, -1.0]),
        micro_batch_size=1,
        grad_accum=1,
        eps_clip=0.2,
        beta=0.0,
        use_amp=False,
        response_start=[2, 2],
    )
    assert torch.isfinite(metrics["entropy_mean"])
    assert float(metrics["entropy_mean"]) > 0.0


def test_beta_zero_reference_reports_kl_without_changing_loss():
    new = torch.tensor([[-0.8, -0.6]])
    old = new.clone()
    ref = torch.tensor([[-1.2, -1.0]])
    advantages = torch.tensor([1.0])
    mask = torch.ones_like(new)
    loss_without_ref, _ = grpo_loss(new, old, None, advantages, mask, beta=0.0)
    loss_with_ref, metrics = grpo_loss(new, old, ref, advantages, mask, beta=0.0)
    torch.testing.assert_close(loss_with_ref, loss_without_ref)
    assert float(metrics["kl_mean"]) > 0.0
