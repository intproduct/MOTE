from __future__ import annotations

import argparse

from ..config import load_config
from ..train.rl_controller import run_fitmotn_rl_training


def parse_args():
    parser = argparse.ArgumentParser(description="FitMoTN RL/GRPO trainer")
    parser.add_argument("--config_json", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(config_json=args.config_json)
    run_fitmotn_rl_training(cfg)


if __name__ == "__main__":
    main()
