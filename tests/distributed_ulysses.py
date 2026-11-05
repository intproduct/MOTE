"""Real two-process CPU integration acceptance; no downloads or user weights.

Run in the pinned Linux environment: python tests/distributed_ulysses.py --output DIR
Exercises FitMoTNTrainer.train, official DeepSpeed collectives, capacity + aux,
weighted CE, accumulation, atomic checkpoint and mid-epoch resume.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from datetime import timedelta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.update(DS_ACCELERATOR="cpu", CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1",
                  TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")

import torch
import torch.distributed as dist
import torch.nn.functional as F


def model_for(family, mote):
    from transformers import Qwen3Config, Qwen3ForCausalLM, LlamaConfig, LlamaForCausalLM
    from fitmotn.model import MOTNFFNLayer
    cls, config = ((Qwen3ForCausalLM, Qwen3Config) if family == "qwen3" else (LlamaForCausalLM, LlamaConfig))
    kwargs = dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                  max_position_embeddings=64, attention_dropout=0.)
    if family == "llama":
        kwargs.update(rope_theta=500000., max_position_embeddings=131072,
                      rope_scaling=dict(rope_type="llama3", factor=8., low_freq_factor=1.,
                                        high_freq_factor=4., original_max_position_embeddings=8192))
    cfg = config(**kwargs)
    cfg._attn_implementation = "sdpa"
    model = cls(cfg).float().train()
    if mote:
        patch = dict(E=4, d=2, k_in=2, topk=2, gate_type="topk", min_capacity=1,
                     capacity_factor=0.5, drop_tokens=True, global_expert_enabled=True,
                     pos_strategy="sliding", block_init_mode="gamma_normal")
        if mote == "quantum":
            patch.update(gate_arch="quantum", gate_quantum=dict(rank=4, depth=1, block_size=2))
        for index, layer in enumerate(model.model.layers):
            layer.mlp = MOTNFFNLayer(layer.mlp, patch, device=torch.device("cpu"), layer_idx=index)
    return model


def reference_train(model, dataset, batch_size, steps, accumulation, aux_weight, world_size):
    from fitmotn.gate import TopKGate
    from transformers import default_data_collator
    aux = []
    handles = [g.register_forward_hook(lambda m, i, o: aux.append(o[2]["l_aux"]))
               for g in model.modules() if isinstance(g, TopKGate)]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9)
    batches = []
    epoch = 0
    while len(batches) < steps * accumulation:
        order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(epoch)).tolist()
        for start in range(0, len(order), world_size * batch_size):
            for sp_source in range(2):
                groups = []
                for dp in range(world_size // 2):
                    rank = dp * 2 + sp_source
                    groups.append(default_data_collator([
                        dataset[order[start + world_size * j + rank]] for j in range(batch_size)]))
                batches.append(groups)
        epoch += 1
    for step in range(steps):
        optimizer.zero_grad()
        for groups in batches[step * accumulation:(step + 1) * accumulation]:
            for batch in groups:
                aux.clear()
                logits = model(input_ids=batch["input_ids"], use_cache=False).logits
                labels = batch["labels"][:, 1:]
                losses = F.cross_entropy(logits[:, :-1].reshape(-1, 64), labels.reshape(-1), reduction="none").reshape_as(labels)
                weights = batch["loss_weights"][:, 1:] * labels.ne(-100)
                loss = (losses * weights).sum() / weights.sum().clamp_min(1)
                if aux:
                    loss = loss + torch.stack(aux).sum() * aux_weight
                (loss / accumulation / len(groups)).backward()
        optimizer.step()
    for handle in handles:
        handle.remove()
    return copy.deepcopy(model.state_dict())


def worker(rank, rendezvous, output, world_size):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world_size), LOCAL_WORLD_SIZE=str(world_size),
                      MASTER_ADDR="127.0.0.1", MASTER_PORT="29671")
    torch.set_num_threads(1)
    import deepspeed
    # CPU verification only: bypass optional JIT extension, retain real Gloo all-to-all.
    deepspeed.ops.__compatible_ops__["deepspeed_shm_comm"] = False
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world_size,
                            timeout=timedelta(seconds=90))
    from transformers import TrainingArguments, default_data_collator
    from fitmotn.data.tokenized_sequences import TokenizedSequenceDataset
    from fitmotn.train.trainer import FitMoTNTrainer
    from fitmotn.train.sequence_parallel import UlyssesConfig
    from fitmotn.checkpointing import validate_checkpoint
    dataset = TokenizedSequenceDataset(Path(output) / "data.jsonl")
    results = []
    try:
        for family, mote, batch_size, accum, steps in [("qwen3", False, 1, 1, 3), ("llama", False, 2, 2, 3),
                                                      ("qwen3", True, 2, 2, 3), ("qwen3", "quantum", 2, 2, 5)]:
            name = f"{family}_{'quantum' if mote == 'quantum' else 'mote' if mote else 'dense'}"
            torch.manual_seed(1701)
            initial = model_for(family, mote)
            expected = reference_train(copy.deepcopy(initial), dataset, batch_size, steps, accum, 1., world_size)
            continuous = None
            for resume in [False, True]:
                model = copy.deepcopy(initial)
                args = TrainingArguments(
                    output_dir=str(Path(output) / name), use_cpu=True, max_steps=steps,
                    per_device_train_batch_size=batch_size, gradient_accumulation_steps=accum,
                    learning_rate=0.03, max_grad_norm=0., save_steps=1, logging_steps=1,
                    save_safetensors=False, disable_tqdm=True, report_to=[], remove_unused_columns=False,
                    ddp_find_unused_parameters=True, seed=0)
                opt = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9)
                scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.)
                trainer = FitMoTNTrainer(model=model, args=args, train_dataset=dataset,
                    data_collator=default_data_collator, optimizers=(opt, scheduler),
                    sequence_parallel_config=UlyssesConfig(2, 16, 1.))
                checkpoint = Path(output) / name / "checkpoint-1"
                trainer.train(resume_from_checkpoint=str(checkpoint) if resume else None)
                state = model.state_dict()
                parameter_error = max((state[k] - expected[k]).abs().max().item() for k in expected)
                if not resume:
                    continuous = copy.deepcopy(state)
                resume_error = max((state[k] - continuous[k]).abs().max().item() for k in state)
                check = validate_checkpoint(Path(output) / name / f"checkpoint-{steps}", verify_hashes=True)
                case = dict(name=name, resume=resume, batch_size=batch_size, accumulation=accum,
                            reference_parameter_max_abs=parameter_error, resume_parameter_max_abs=resume_error,
                            checkpoint_valid=check["ok"], global_step=trainer.state.global_step, world_size=world_size,
                            sp_size=2, dp_size=world_size // 2)
                results.append(case)
                if rank == 0:
                    print(json.dumps(case), flush=True)
                assert parameter_error < 2e-5, case
                assert resume_error < 2e-6, case
                assert check["ok"], check
                trainer.sequence_parallel.close()
                dist.barrier()
    finally:
        Path(output, f"rank_{rank}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world-size", type=int, choices=[2, 4], default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for i in range(args.world_size * 8):
        ids = ((torch.arange(16) + i) % 60 + 1).tolist()
        # First SP rank has no supervised targets for alternating examples.
        labels = [-100] * (9 if i % 2 else 4) + ids[(9 if i % 2 else 4):]
        records.append(dict(input_ids=ids, labels=labels, loss_weights=[1.] * 12 + [2.] * 4))
    (args.output / "data.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        torch.multiprocessing.spawn(worker, args=((Path(tmp) / "rdz").as_uri(), str(args.output), args.world_size),
                                    nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
