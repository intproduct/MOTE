"""Reproducible local cost check; no language-model accuracy claim."""
import _bootstrap  # noqa: F401
import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from candidates import BrickwallConeGate, SweepDensityGate, ResidualMPSGate, CompactBornGate
from fitmotn.gate import RouterLogits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--experts", type=int, nargs="+", default=[16, 64])
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    rows = []
    for experts in args.experts:
        factories = {
            "linear": lambda: RouterLogits(args.dim, experts),
            "mlp_r16": lambda: RouterLogits(args.dim, experts, gate_arch="mlp", gate_hidden_dim=16),
            "block2_depth2": lambda: RouterLogits(args.dim, experts, gate_arch="quantum", gate_quantum={"readout": "expectation"}),
            "brickwall_depth2": lambda: BrickwallConeGate(args.dim, experts, depth=2),
            "sweep_two_directions": lambda: SweepDensityGate(args.dim, experts),
            "mps_m16_chi8": lambda: ResidualMPSGate(args.dim, experts, sites=16, bond_dim=8),
            "compact_born_depth2": lambda: CompactBornGate(args.dim, experts, depth=2),
        }
        for name, factory in factories.items():
            torch.manual_seed(71)
            model = factory().to(device)
            x = torch.randn(args.tokens, args.dim, device=device, requires_grad=True)
            target = torch.randn(args.tokens, experts, device=device)
            row = {"name": name, "experts": experts, "parameters": sum(p.numel() for p in model.parameters())}
            if isinstance(model, BrickwallConeGate):
                row.update(max_cone_qubits=model.max_cone_qubits, cone_groups=len(model.cones))
            for backward in [False, True]:
                def step():
                    model.zero_grad(set_to_none=True)
                    x.grad = None
                    with torch.set_grad_enabled(backward):
                        y = model(x)
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
            print(json.dumps(row), flush=True)
            del model, x, target
    result = {"config": {**vars(args), "output": str(args.output) if args.output else None},
              "torch": torch.__version__, "platform": platform.platform(),
              "threads": 1, "dtype": "float32", "warmup": 3, "results": rows}
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
