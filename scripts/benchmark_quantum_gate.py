"""Measure router cost only; run from a checkout with fitmotn installed."""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from fitmotn.gate import RouterLogits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if min(args.tokens, args.dim, args.experts, args.rank, args.iterations, args.threads) < 1:
        parser.error("dimensions, iterations and threads must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(7)
    device = torch.device(args.device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    rows = []
    for arch in ("linear", "mlp", "quantum"):
        router = RouterLogits(args.dim, args.experts, gate_arch=arch, gate_hidden_dim=args.rank,
                              gate_quantum={"rank": args.rank, "depth": args.depth,
                                            "block_size": args.block_size}).to(device)
        x = torch.randn(args.tokens, args.dim, device=device, requires_grad=True)
        target = torch.randn(args.tokens, args.experts, device=device)
        row = {"arch": arch, "parameters": router.router_param_count}
        for backward in (False, True):
            def step():
                router.zero_grad(set_to_none=True)
                x.grad = None
                with torch.set_grad_enabled(backward):
                    y = router(x)
                    if backward:
                        (y - target).square().mean().backward()

            for _ in range(3):
                step()
            sync()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            samples = []
            for _ in range(args.iterations):
                sync()
                start = time.perf_counter()
                step()
                sync()
                samples.append(1000 * (time.perf_counter() - start))
            key = "forward_backward" if backward else "forward"
            row[key + "_ms"] = statistics.median(samples)
            if device.type == "cuda":
                row[key + "_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        rows.append(row)
        del router, x, target
    print(json.dumps({"torch": torch.__version__, "config": vars(args), "results": rows}, indent=2))


if __name__ == "__main__":
    main()
