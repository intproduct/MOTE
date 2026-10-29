from __future__ import annotations
import argparse
import json
from pathlib import Path
import tempfile
import torch
from .config import load_config, validate, merged
from .models import (
    ImageClassifier,
    BertClassifier,
    shared_state,
    load_shared,
    state_hash,
    auxiliary_loss,
)
from .runner import seed_all, train_one, environment


def tiny_config(cfg, stage):
    c = merged(
        cfg,
        {
            "experts": 4,
            "topk": 2,
            "k_in": 2,
            "rank": 4,
            "sites": 4,
            "bond_dim": 3,
            "bert_layers": 2,
            "eval_every": 1,
            "log_every": 1,
            "precision": cfg["precision"],
            "mlp": {
                "width": 16,
                "layers": 2,
                "expansion": 2,
                "steps": 2,
                "batch_size": 4,
            },
            "bert": {"steps": 2, "batch_size": 4, "warm_init_steps": 0},
        },
    )
    from transformers import BertConfig

    bert = BertConfig(
        vocab_size=40,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=32,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    return c, bert


def smoke(cfg, output=None):
    root = Path(output) if output else Path(tempfile.mkdtemp(prefix="mote_gate_smoke_"))
    for stage in ("mlp", "bert"):
        c, bert = tiny_config(cfg, stage)
        seed_all(7)
        if stage == "mlp":
            features = torch.randn(24, 32)
            tensors = {"features": features, "labels": (features[:, 0] > 0).long()}
            build = lambda arch: ImageClassifier(c, arch, 7, 32, 2)
        else:
            tokens = torch.randint(1, 40, (24, 8))
            mask = torch.ones_like(tokens)
            mask[:, -2:] = 0
            tensors = {
                "input_ids": tokens,
                "attention_mask": mask,
                "labels": (tokens[:, 0] % 2).long(),
            }
            build = lambda arch: BertClassifier(c, arch, 7, 2, tiny_config=bert)
        data = {
            k: {n: t[a:b] for n, t in tensors.items()}
            for k, (a, b) in {
                "train": (0, 16),
                "dev": (16, 20),
                "test": (20, 24),
            }.items()
        }
        common = shared_state(build("mlp"))
        common_hash = state_hash(common)
        for arch in c["gates"]:
            seed_all(7)
            model = build(arch)
            load_shared(model, common)
            train_one(
                model,
                c,
                stage,
                "synthetic",
                arch,
                7,
                data,
                {"synthetic": True},
                root / f"{stage}_{arch}",
                common_hash,
                {},
            )
    print(f"PASS: {2*len(cfg['gates'])} offline training/checkpoint/evaluation runs: {root}")
    return root


def doctor(cfg):
    from ...gate import RouterLogits

    device = torch.device(cfg["device"])
    seed_all(7)
    info = environment()
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        info.update(
            capability=torch.cuda.get_device_capability(device),
            compiled_arches=torch.cuda.get_arch_list(),
        )
    x = torch.randn(8, 64, device=device, requires_grad=True)
    for arch in ("mlp", "compact_born", "mps", "residual_mpo", "amplitude_born"):
        gate = RouterLogits(
            64, 64, gate_arch=arch, gate_hidden_dim=16, gate_learnable_gain=True
        ).to(device)
        output = gate(x)
        output.square().mean().backward()
        if not torch.isfinite(output).all() or not torch.isfinite(x.grad).all():
            raise FloatingPointError(f"{arch} failed device backward")
        x.grad = None
    # Also exercise the repository's expert contraction/backward kernel path.
    c, _ = tiny_config(cfg, "mlp")
    model = ImageClassifier(c, "compact_born", 7, 32, 2).to(device)
    with torch.autocast(
        device.type, dtype=torch.float16, enabled=cfg["precision"] == "fp16"
    ):
        loss = model(
            features=torch.randn(4, 32, device=device)
        ).square().mean() + auxiliary_loss(model)
    loss.backward()
    if not all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    ):
        raise FloatingPointError("expert backward failed")
    print(json.dumps(info, indent=2))
    print("PASS: 64-expert routers and MoTE backward")


def main():
    parser = argparse.ArgumentParser(
        description="Matched MLP / joint Born / MPS MoTE experiments"
    )
    parser.add_argument(
        "command", choices=["doctor", "prepare", "run", "smoke", "summarize"]
    )
    parser.add_argument("--config")
    parser.add_argument("--stage", choices=["all", "mlp", "bert"], default="all")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--device")
    parser.add_argument("--precision", choices=["fp32", "fp16"])
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--gates", nargs="+", help="configured gate architectures or variant names")
    parser.add_argument("--bert-layers", type=int, choices=[2, 3])
    parser.add_argument("--experts", type=int)
    parser.add_argument("--topk", type=int)
    parser.add_argument("--budget", choices=["matched", "rank"])
    parser.add_argument(
        "--output", help="smoke destination, run output root, or summarize root"
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    for key in ("device", "precision", "bert_layers", "experts", "topk", "budget"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    validate(cfg)
    if args.seeds and not set(args.seeds) <= set(cfg["seeds"]):
        parser.error("--seeds selects a subset of seeds in the config")
    if args.gates and not set(args.gates) <= set(cfg["gates"]):
        parser.error("--gates selects a subset of gates in the config")
    if args.tasks and not set(args.tasks) <= set(
        t for s in ("mlp", "bert") if args.stage in {"all", s} for t in cfg[s]["tasks"]
    ):
        parser.error("--tasks must belong to the selected stage and configuration")
    if args.command == "doctor":
        doctor(cfg)
    elif args.command == "smoke":
        smoke(cfg, args.output)
    elif args.command == "prepare":
        from .data import prepare

        prepare(cfg, args.stage, args.tasks)
    elif args.command == "run":
        from .runner import run

        if args.output:
            cfg["output_dir"] = args.output
        run(cfg, args.stage, args.tasks, args.seeds, args.gates)
    else:
        from .report import summarize

        summarize(args.output or cfg["output_dir"])


if __name__ == "__main__":
    main()
