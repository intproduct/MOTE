"""Small sensitivity probe, not a barren-plateau or accuracy experiment."""
import _bootstrap  # noqa: F401
import json
from pathlib import Path

import torch

from candidates import SweepDensityGate


def main():
    rows = []
    for angle in (0.2, 1.0):
        trials = []
        for seed in range(3):
            torch.manual_seed(seed)
            model = SweepDensityGate(4, 16, rank=3).double()
            with torch.no_grad():
                model.coupling.fill_(angle)
            a = torch.randn(16, dtype=torch.float64, requires_grad=True)
            jac = torch.autograd.functional.jacobian(model.expectations_from_angles, a).abs()
            distance = (torch.arange(16)[:, None] - torch.arange(16)[None, :]).abs()
            trials.append({"seed": seed, "adjacent_mean": jac[distance == 1].mean().item(),
                           "far_mean_distance_ge8": jac[distance >= 8].mean().item(),
                           "diagonal_mean": jac.diag().mean().item()})
        rows.append({"coupling_angle": angle, "trials": trials})
    destination = Path(__file__).with_name("sweep_sensitivity.json")
    destination.write_text(json.dumps(rows, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
