from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fitmotn.patching import patch_qwen_ffn_layers, set_trainable_patch_only
from fitmotn.pretrained_mixt import PretrainedMixTLinear, projection_spec


class PartnerMain(nn.Module):
    def __init__(self, incoming, outgoing, dtype):
        super().__init__()
        self.d = 2
        self.q_number = int(math.log2(incoming))
        self.k_in, self.k_out = self.q_number - 1, int(math.log2(outgoing)) - 1
        self.op_indices = [list(range(i, i + self.k_in)) for i in range(2)]
        self.bricks = nn.Parameter(torch.randn(2, 2**self.k_in, 2**self.k_out, dtype=dtype) * .1)
        self.bias = nn.Parameter(torch.randn(2, 1, 1, dtype=dtype) * .03)

    def forward(self, x):
        shape = x.shape
        digits = x.reshape(-1, *([2] * self.q_number))
        terms = []
        for i, positions in enumerate(self.op_indices):
            unaffected = [p for p in range(self.q_number) if p not in positions]
            a = digits.permute(0, *[p+1 for p in unaffected+positions])
            b = a.reshape(-1, 2**self.k_in) @ (self.bricks[i] + self.bias[i])
            b = b.reshape(digits.shape[0], 2, *([2] * self.k_out))
            if positions[0] == 0:
                b = b.permute(0, *range(2, 2 + self.k_out), 1)
            terms.append(b)
        return ((terms[0] + terms[1]) / 2).reshape(*shape[:-1], -1)


class PartnerProjection(nn.Module):
    is_mixt_projector = True

    def __init__(self, incoming=10, outgoing=18, dtype=torch.float32, dense_edges=False):
        super().__init__()
        self.in_features, self.out_features = incoming, outgoing
        self.main_in_features = 2 ** int(math.log2(incoming))
        self.main_out_features = 2 ** int(math.log2(outgoing))
        self.mixt00 = PartnerMain(self.main_in_features, self.main_out_features, dtype)
        ti, to = incoming - self.main_in_features, outgoing - self.main_out_features
        def edge(a, b):
            if dense_edges:
                return nn.Linear(a, b, bias=False, dtype=dtype)
            return nn.Sequential(nn.Linear(a, 2, bias=False, dtype=dtype), nn.Linear(2, b, bias=False, dtype=dtype))
        self.w10 = edge(ti, self.main_out_features)
        self.w01 = edge(self.main_in_features, to)
        self.w11 = nn.Linear(ti, to, bias=False, dtype=dtype)
        self.alpha10 = nn.Parameter(torch.tensor(.7, dtype=dtype))
        self.alpha01 = nn.Parameter(torch.tensor(1.1, dtype=dtype))
        self.alpha11 = nn.Parameter(torch.tensor(.4, dtype=dtype))
        self.bias = nn.Parameter(torch.randn(outgoing, dtype=dtype) * .1)

    def forward(self, x):
        x0, x1 = x[..., :self.main_in_features], x[..., self.main_in_features:]
        y0 = self.mixt00(x0) + self.alpha10 * self.w10(x1)
        y1 = self.alpha01 * self.w01(x0) + self.alpha11 * self.w11(x1)
        return torch.cat([y0, y1], -1) + self.bias


class PartnerMLP(nn.Module):
    def __init__(self, hidden=10, intermediate=18, dtype=torch.float32):
        super().__init__()
        self.gate_proj = PartnerProjection(hidden, intermediate, dtype)
        self.up_proj = PartnerProjection(hidden, intermediate, dtype)
        self.down_proj = PartnerProjection(intermediate, hidden, dtype)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def cfg(**native):
    return dict(patch_backend="pretrained_mixt", E=2, d=2, k_in=2, topk=2,
                gate_type="softmax", gate_arch="linear", drop_tokens=False,
                global_expert_enabled=False, pretrained_mixt=native)


def model_with_mlp(mlp):
    model = nn.Module()
    model.config = SimpleNamespace(num_hidden_layers=1, mixt_compress_layers=[0])
    model.model = nn.Module()
    layer = nn.Module()
    layer.mlp = mlp
    model.model.layers = nn.ModuleList([layer])
    return model


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("kind", ["softmax", "topk"])
@pytest.mark.parametrize("arch", ["linear", "quantum"])
def test_native_import_exact_forward_including_bf16(dtype, kind, arch):
    torch.manual_seed(41)
    source = PartnerProjection(dtype=dtype)
    config = cfg()
    config.update(gate_type=kind, gate_arch=arch)
    imported = PretrainedMixTLinear(projection_spec(source), config, source=source, device="cpu", dtype=dtype)
    x = torch.randn(2, 3, 10).to(dtype)
    assert torch.equal(imported(x)[0], source(x))
    assert torch.equal(imported.lr01.B.weight, source.w10[0].weight)
    assert torch.equal(imported.lr10.A.weight, source.w01[1].weight)
    assert torch.equal(imported.lr11.weight, source.w11.weight)
    assert torch.equal(imported.core.blocks[0].operator_bias, source.mixt00.bias[0].reshape(()))


@pytest.mark.parametrize("scope", ["gate_only", "gate_experts", "all"])
@pytest.mark.parametrize("arch", ["linear", "quantum"])
def test_training_scope_gradients_and_checkpoint_reconstruction(scope, arch):
    torch.manual_seed(72)
    model = model_with_mlp(PartnerMLP())
    config = cfg(train_scope=scope)
    config["gate_arch"] = arch
    patch_qwen_ffn_layers(model, [0], config, torch.device("cpu"))
    mlp = model.model.layers[0].mlp
    set_trainable_patch_only(model)
    assert mlp.gate_proj.core.gate.router.identity_gain.requires_grad
    assert mlp.gate_proj.core.blocks[0].U.requires_grad == (scope != "gate_only")
    assert mlp.gate_proj.alpha01.requires_grad == (scope == "all")
    assert mlp.gate_proj.lr11.weight.requires_grad == (scope == "all")
    x = torch.randn(2, 4, 10)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    for _ in range(2):
        optimizer.zero_grad()
        mlp(x).square().mean().backward()
        optimizer.step()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for name, p in mlp.gate_proj.core.gate.router.named_parameters() if name != "identity_gain")
    dense = nn.Module()
    dense.act_fn = nn.SiLU()
    dense.gate_proj, dense.up_proj, dense.down_proj = nn.Linear(10, 18), nn.Linear(10, 18), nn.Linear(18, 10)
    restored = model_with_mlp(dense)
    patch_qwen_ffn_layers(restored, [0], config, torch.device("cpu"))
    restored.load_state_dict(model.state_dict(), strict=True)
    assert torch.equal(restored.model.layers[0].mlp(x), mlp(x))


def test_dense_edge_variant_preserves_parameters():
    source = PartnerProjection(dense_edges=True)
    imported = PretrainedMixTLinear(projection_spec(source), cfg(), source=source, device="cpu", dtype=torch.float32)
    x = torch.randn(5, 10)
    assert torch.equal(imported(x)[0], source(x))


def test_import_rejects_unmapped_weights_and_geometry_changes():
    source = PartnerProjection()
    source.register_parameter("unrecognized", nn.Parameter(torch.ones(())))
    with pytest.raises(ValueError, match="Unmapped"):
        PretrainedMixTLinear(projection_spec(source), cfg(), source=source, device="cpu", dtype=torch.float32)
    model = model_with_mlp(PartnerMLP())
    config = cfg()
    patch_qwen_ffn_layers(model, [0], config, torch.device("cpu"))
    changed = model_with_mlp(PartnerMLP())
    config["pretrained_mixt"]["projections"]["0"]["gate_proj"]["k_out"] += 1
    with pytest.raises(ValueError, match="structure differs"):
        patch_qwen_ffn_layers(changed, [0], config, torch.device("cpu"))


def tiny_hybrid(dtype=torch.float32):
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
    base = Qwen3_5TextConfig(
        vocab_size=32, hidden_size=10, intermediate_size=18, num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"], num_attention_heads=2,
        num_key_value_heads=1, head_dim=4, linear_num_key_heads=1, linear_num_value_heads=2,
        linear_key_head_dim=4, linear_value_head_dim=4, tie_word_embeddings=True,
        eos_token_id=2, pad_token_id=0,
        rope_parameters={"rope_type": "default", "rope_theta": 10000., "partial_rotary_factor": 1.,
                         "mrope_section": [1, 1, 0], "mrope_interleaved": True})
    base.mixt_compress_layers = [1]
    base._attn_implementation = "eager"
    base.auto_map = {"AutoModelForCausalLM": "missing_partner_code.CustomModel"}
    original = Qwen3_5ForCausalLM._from_config(base, dtype=dtype).eval()
    original.model.layers[1].mlp = PartnerMLP(dtype=dtype)
    return original


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_hybrid_qwen35_hf_export_reload_and_generation(tmp_path, dtype, attention):
    from transformers import AutoModelForCausalLM
    from fitmotn.export.remote_code.configuration_fitmotn import FitMoTNConfig
    from fitmotn.export.remote_code.modeling_fitmotn import FitMoTNForCausalLM
    from fitmotn.export.hf_export import _copy_remote_code, _load_state_into_wrapper
    original = tiny_hybrid(dtype)
    base = original.config
    original.set_attn_implementation(attention)
    ids = torch.tensor([[1, 4, 5, 6]])
    with torch.no_grad():
        before = original(ids).logits
    config = cfg(train_scope="all")
    patch_qwen_ffn_layers(original, [1], config, torch.device("cpu"))
    with torch.no_grad():
        assert torch.equal(original(ids).logits, before)
    wrapper = FitMoTNForCausalLM(FitMoTNConfig(
        base_model_config=base.to_dict(), fitmotn_patch_config=config,
        layers_to_patch=[1], patch_backend="pretrained_mixt",
        fitmotn_attention_implementation=attention)).eval()
    assert _load_state_into_wrapper(wrapper, original) == ([], [])
    with torch.no_grad():
        torch.testing.assert_close(wrapper(ids).logits, before, rtol=1e-5, atol=1e-7)
        projected_positions = []
        hook = wrapper.lm_head.register_forward_pre_hook(lambda module, inputs: projected_positions.append(inputs[0].shape[1]))
        tail = wrapper(ids, logits_to_keep=3, labels=ids)
        assert projected_positions == [3]
        hook.remove()
        torch.testing.assert_close(tail.logits, before[:, -3:], rtol=1e-5, atol=1e-7)
        expected_loss = torch.nn.functional.cross_entropy(before[:, -3:-1].float().reshape(-1, before.shape[-1]), ids[:, -2:].reshape(-1))
        torch.testing.assert_close(tail.loss, expected_loss)
        inputs = wrapper.prepare_inputs_for_generation(ids, logits_to_keep=1)
        assert inputs["logits_to_keep"] == 1
    wrapper.save_pretrained(tmp_path)
    _copy_remote_code(tmp_path)
    loaded = AutoModelForCausalLM.from_pretrained(
        tmp_path, trust_remote_code=True, dtype=dtype).eval()
    assert loaded.model.config._attn_implementation == attention
    for key, value in wrapper.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value), key
    with torch.no_grad():
        torch.testing.assert_close(loaded(ids).logits, before, rtol=1e-5, atol=1e-7)
        out = loaded.generate(ids, max_new_tokens=2, do_sample=False)
    assert out.shape[1] > ids.shape[1]
    assert out[0, ids.shape[1]] == loaded(ids).logits[0, -1].argmax()
    assert loaded.get_input_embeddings().weight is loaded.get_output_embeddings().weight


def test_trainer_optimizer_checkpoint_exact_resume(tmp_path):
    from transformers import TrainingArguments, default_data_collator
    from fitmotn.train.trainer import FitMoTNTrainer
    from fitmotn.checkpointing import extract_patch_state_dict
    torch.manual_seed(91)
    original = tiny_hybrid()
    config = cfg(train_scope="gate_only")
    patch_qwen_ffn_layers(original, [1], config, torch.device("cpu"))
    set_trainable_patch_only(original)
    original.config.use_cache = False
    rows = [{"input_ids": [1, 4, 5, 6], "labels": [-100, -100, 5, 6]}] * 8
    frozen = {n: p.detach().clone() for n, p in original.named_parameters() if not p.requires_grad}
    initial = copy.deepcopy(original)

    def trainer(model, output):
        args = TrainingArguments(output_dir=str(output), use_cpu=True, per_device_train_batch_size=1,
            max_steps=3, learning_rate=.01, lr_scheduler_type="constant", save_strategy="steps",
            save_steps=1, report_to=[], disable_tqdm=True, dataloader_pin_memory=False,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
        metadata = lambda checkpoint_name=None: dict(patch_state_dict=extract_patch_state_dict(model, [1]),
            patch_cfg=config, motn_cfg=config, patch_backend="pretrained_mixt", layers_to_patch=[1])
        return FitMoTNTrainer(model=model, args=args, train_dataset=rows, data_collator=default_data_collator,
                              fitmotn_metadata_builder=metadata)

    first = trainer(original, tmp_path / "full")
    first.train()
    resumed = trainer(initial, tmp_path / "resumed")
    resumed.train(resume_from_checkpoint=str(tmp_path / "full" / "checkpoint-1"))
    for name, value in first.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[name]), name
        if name in frozen:
            assert torch.equal(value, frozen[name]), name
    assert first.model.model.layers[1].mlp.gate_proj.core.gate.router.identity_gain.abs() > 0
    for key, state in first.optimizer.state_dict()["state"].items():
        for field, value in state.items():
            assert torch.equal(value, resumed.optimizer.state_dict()["state"][key][field]), (key, field)


def test_posttraining_controller_does_not_require_pretraining_sources(monkeypatch):
    from fitmotn.train import controller
    from fitmotn.train.stages import build_stage_plan
    from fitmotn.config.schema import TrainConfig
    def unwanted(*args, **kwargs):
        pytest.fail("Pure SFT must not access unrelated pretraining datasets")
    monkeypatch.setattr(controller, "build_pretrain_tasks", unwanted)
    monkeypatch.setattr(controller, "build_task_mixture_tasks", lambda *a, **k: ["sft"])
    plan = build_stage_plan(TrainConfig(epochs=0, steps=1, stage_a_ratio=0, stage_b_disable_pretrain=True))
    assert controller._build_training_pools(None, plan, None) == ([], ["sft"])


def test_native_rl_loading_and_response_logprob_gradient(tmp_path, monkeypatch):
    from fitmotn.config.defaults import make_default_config
    from fitmotn.rl import runtime
    from fitmotn.rl.logprobs import gather_response_logprobs
    configuration = make_default_config()
    for name, value in cfg().items():
        setattr(configuration.model, name, value)
    configuration.model.device = "cpu"
    configuration.model.model_path = str(tmp_path)
    configuration.model.layers_to_patch = "mixt"
    original = tiny_hybrid()
    tok = SimpleNamespace(pad_token_id=0, eos_token_id=2)
    monkeypatch.setattr(runtime, "load_causal_lm_and_tokenizer", lambda *a, **k: (original, tok, torch.float32))
    policy, _, info = runtime.load_policy_for_rl(configuration)
    assert info.layer_idxs == [1]
    runtime.set_trainable_mode_for_rl(policy, "gate_only")
    policy.train()
    ids = torch.tensor([[1, 4, 5, 6]])
    mask = torch.tensor([[0, 0, 1, 1]])
    logp = gather_response_logprobs(policy, ids, torch.ones_like(ids), mask, response_start=2)
    loss = -(logp * mask).sum()
    loss.backward()
    gain = policy.model.layers[1].mlp.gate_proj.core.gate.router.identity_gain
    assert gain.grad is not None and torch.isfinite(gain.grad) and gain.grad.abs() > 0


def test_native_resume_rejects_silent_router_architecture_change():
    from fitmotn.config.defaults import make_default_config
    from fitmotn.train.controller import _validate_resume_structure
    configuration = make_default_config()
    configuration.model.gate_arch = "quantum"
    configuration._explicit_model_keys = {"gate_arch"}
    with pytest.raises(ValueError, match="gate_arch"):
        _validate_resume_structure(configuration, {"patch_backend": "pretrained_mixt"}, [1], cfg(), 2)


def test_explicit_weight_continuation_can_sparsify_without_reinitializing_tensors():
    from fitmotn.config.defaults import make_default_config
    from fitmotn.train.controller import _validate_resume_structure, _apply_native_resume_options
    original = model_with_mlp(PartnerMLP())
    imported = cfg(uniform_init=True)
    patch_qwen_ffn_layers(original, [0], imported, torch.device("cpu"))
    original.model.layers[0].mlp.gate_proj.core.gate.router.identity_gain.data.fill_(.3)
    saved_state = copy.deepcopy(original.state_dict())
    configuration = make_default_config()
    configuration.model.gate_type = "topk"
    configuration.model.topk = 1
    configuration.model.pretrained_mixt = dict(uniform_init=False, train_scope="gate_experts")
    configuration._explicit_model_keys = {"gate_type", "topk", "pretrained_mixt"}
    configuration.train.resume_weights_from = "saved-soft-routing"
    metadata = {"patch_backend": "pretrained_mixt"}
    _validate_resume_structure(configuration, metadata, [0], imported, 1)
    changed = copy.deepcopy(imported)
    _apply_native_resume_options(configuration, changed)
    assert changed["topk"] == 1 and changed["gate_type"] == "topk"
    sparse = model_with_mlp(PartnerMLP())
    patch_qwen_ffn_layers(sparse, [0], changed, torch.device("cpu"))
    sparse.load_state_dict(saved_state, strict=True)
    assert all(torch.equal(sparse.state_dict()[name], tensor) for name, tensor in saved_state.items())
    assert torch.isfinite(sparse.model.layers[0].mlp(torch.randn(2, 4, 10))).all()
    configuration.train.resume_checkpoint_from = "exact-checkpoint"
    with pytest.raises(ValueError, match="topk"):
        _validate_resume_structure(configuration, metadata, [0], imported, 1)
    configuration.train.resume_checkpoint_from = None
    configuration.model.pretrained_mixt["uniform_init"] = True
    with pytest.raises(ValueError, match="topk"):
        _validate_resume_structure(configuration, metadata, [0], imported, 1)
