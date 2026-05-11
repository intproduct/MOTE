from __future__ import annotations

import argparse

from ..config import load_config
from ..train.controller import run_fitmotn_training


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--seq_len_run", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--eval_every_updates", type=int, default=None)
    parser.add_argument("--save_every_updates", type=int, default=None)
    parser.add_argument("--run_baseline_eval", type=int, choices=[0, 1], default=None)
    parser.add_argument("--layers_to_patch", type=str, default=None)
    parser.add_argument("--dataloader_num_workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--config_json", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = {
        "model": {key: getattr(args, key) for key in ["model_path", "layers_to_patch", "device"] if getattr(args, key) is not None},
        "output": {
            key: value
            for key, value in {"root_dir": args.output_root, "run_name": args.run_name}.items()
            if value is not None
        },
        "train": {
            key: getattr(args, key)
            for key in ["batch_size", "grad_accum", "steps", "epochs", "lr", "eval_every_updates", "save_every_updates"]
            if getattr(args, key) is not None
        },
        "data": {
            key: value
            for key, value in {"seq_len_run": args.seq_len_run, "dataloader_num_workers": args.dataloader_num_workers}.items()
            if value is not None
        },
        "eval": ({"run_baseline_eval": bool(args.run_baseline_eval)} if args.run_baseline_eval is not None else {}),
    }
    cfg = load_config(config_json=args.config_json, overrides=overrides)
    run_fitmotn_training(cfg)


if __name__ == "__main__":
    main()
