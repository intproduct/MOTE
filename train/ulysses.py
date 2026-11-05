"""torchrun entrypoint for the bounded Ulysses SFT path. See --help.

Reuses FitMoTNTrainer, checkpoint restoration, optimizer groups and transactions.
The existing stage controller remains a separate, single-process entrypoint.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import TrainingArguments, default_data_collator, set_seed

from ..checkpointing import extract_patch_state_dict
from ..data.tokenized_sequences import TokenizedSequenceDataset
from ..eval.restore import restore_fitmotn_model
from ..patching import set_trainable_patch_only
from ..runtime import load_causal_lm_and_tokenizer
from .sequence_parallel import UlyssesConfig
from .trainer import FitMoTNTrainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local HF model or FitMoTN checkpoint")
    parser.add_argument("--data", required=True, help="Fixed-length tokenized JSONL: input_ids, labels, optional loss_weights")
    parser.add_argument("--output", required=True, help="Shared checkpoint directory")
    parser.add_argument("--sp-size", type=int, required=True)
    parser.add_argument("--seq-length", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--aux-weight", type=float, default=0., help="Opt-in multiplier of sum(gate l_aux); default preserves CE-only SFT")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--train-all-parameters", action="store_true", help="By default FitMoTN trains only its patched FFNs")
    parser.add_argument("--resume", help="Committed checkpoint from this entrypoint, same data and topology")
    parser.add_argument("--cpu", action="store_true", help="Small CPU/Gloo verification only")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if not args.resume and output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output directory for a new training run")
    training_args = TrainingArguments(
        output_dir=str(output), max_steps=args.steps, per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.learning_rate,
        save_steps=args.save_steps, save_strategy="steps", save_safetensors=False,
        save_total_limit=3, logging_steps=1, report_to=[], remove_unused_columns=False,
        seed=args.seed, use_cpu=args.cpu, ddp_find_unused_parameters=True,
        bf16=args.dtype == "bfloat16", fp16=args.dtype == "float16", dataloader_num_workers=0)
    set_seed(args.seed)
    source = Path(args.model).resolve()
    metadata = None
    if (source / "fitmotn_state.pt").exists():
        model, tokenizer, metadata = restore_fitmotn_model(
            source, device=str(training_args.device), torch_dtype=args.dtype, use_cache=False)
        if not args.train_all_parameters:
            set_trainable_patch_only(model)
        else:
            raise ValueError("FitMoTN checkpoint entrypoint currently supports patch-only training; "
                             "base-weight changes require a full-model restore adapter")
    else:
        model, tokenizer, _ = load_causal_lm_and_tokenizer(
            source, device=training_args.device, torch_dtype=args.dtype, use_cache=False)
    model.set_attn_implementation("sdpa")
    model.train()
    dataset = TokenizedSequenceDataset(args.data)

    def metadata_builder(checkpoint_name=None):
        return {**metadata, "patch_state_dict": extract_patch_state_dict(model, metadata["layers_to_patch"]),
                "checkpoint_name": checkpoint_name}

    trainer = FitMoTNTrainer(
        model=model, args=training_args, train_dataset=dataset, data_collator=default_data_collator,
        processing_class=tokenizer,
        fitmotn_metadata_builder=metadata_builder if metadata else None,
        sequence_parallel_config=UlyssesConfig(args.sp_size, args.seq_length, args.aux_weight))
    try:
        trainer.train(resume_from_checkpoint=args.resume)
        # Save even when the last update is not a regular save interval.
        final = output / f"checkpoint-{trainer.state.global_step}"
        if not final.is_dir():
            trainer._save_checkpoint(trainer.model_wrapped, trial=None)
    finally:
        trainer.sequence_parallel.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
