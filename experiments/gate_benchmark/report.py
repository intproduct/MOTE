from __future__ import annotations
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def summarize(root):
    root = Path(root)
    rows, grouped, paired = [], defaultdict(list), {}
    for path in sorted(root.rglob("result.json")):
        r = json.loads(path.read_text())
        # Never pool different configs under one mean or pair.
        experiment = str(path.parent.parent.relative_to(root))
        row = {
            "experiment": experiment,
            **{
                k: r[k]
                for k in (
                    "stage",
                    "task",
                    "gate",
                    "seed",
                    "best_step",
                    "router_parameters",
                    "total_parameters",
                    "training_seconds",
                    "examples_per_second",
                    "amp_skipped_steps",
                )
            },
            **{f"final_{k}": v for k, v in r["final"].items()},
            "run_dir": str(path.parent),
        }
        rows.append(row)
        grouped[(experiment, r["stage"], r["task"], r["gate"])].append(row)
        paired[(experiment, r["stage"], r["task"], r["seed"], r["gate"])] = r
    if not rows:
        raise ValueError(f"no completed results under {root}")
    fields = sorted({k for r in rows for k in r})
    with (root / "runs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for (exp, stage, task, gate), group in sorted(grouped.items()):
        scores = [r["final_score"] for r in group]
        deltas = []
        for row in group:
            baseline = paired.get((exp, stage, task, row["seed"], "mlp"))
            own = paired[(exp, stage, task, row["seed"], gate)]
            if baseline:
                if (baseline["shared_state_sha256"], baseline["data_sha256"]) != (
                    own["shared_state_sha256"],
                    own["data_sha256"],
                ):
                    raise ValueError(
                        "cannot pair runs with different shared initialization or data"
                    )
                deltas.append(own["final"]["score"] - baseline["final"]["score"])
        summary.append(
            {
                "experiment": exp,
                "stage": stage,
                "task": task,
                "gate": gate,
                "n": len(scores),
                "mean_score": statistics.mean(scores),
                "std_score": statistics.stdev(scores) if len(scores) > 1 else None,
                "paired_n": len(deltas),
                "paired_delta_vs_mlp": statistics.mean(deltas) if deltas else None,
                "paired_delta_std": statistics.stdev(deltas)
                if len(deltas) > 1
                else None,
                "mean_examples_per_second": statistics.mean(
                    r["examples_per_second"] for r in group
                ),
            }
        )
    incomplete = [
        str(p.parent)
        for p in root.rglob("manifest.json")
        if not (p.parent / "result.json").exists()
    ]
    missing = []
    for path in root.rglob("config.json"):
        cfg = json.loads(path.read_text())
        if "gates" not in cfg or "seeds" not in cfg:
            continue
        for stage in ("mlp", "bert"):
            for task in cfg[stage]["tasks"]:
                for gate in cfg["gates"]:
                    for seed in cfg["seeds"]:
                        destination = path.parent / f"{stage}_{task}_{gate}_seed{seed}"
                        if not (destination / "result.json").exists():
                            missing.append(str(destination))
    (root / "summary.json").write_text(
        json.dumps(
            {"groups": summary, "incomplete_runs": incomplete, "missing_runs": missing},
            indent=2,
        )
    )
    print(json.dumps(summary, indent=2))
    return summary
