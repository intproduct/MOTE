"""Freeze formal variants using only audited pilot INTERNAL dev scores."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics
from .config import load_config, digest, validate
from .runner import write_json


def select(pilot_config, formal_config):
    cfg = load_config(pilot_config)
    root = Path(cfg['output_dir']) / digest(cfg)[:12]
    audit = json.loads((root/'audit.json').read_text())
    assert audit['passed'] and audit['completed_audited'] == audit['expected']
    # Only best_dev is extracted. No final/test score enters ranking or tie breaks.
    scores = {gate: [] for gate in cfg['gates']}
    for stage in ['mlp','bert']:
        for task in cfg[stage]['tasks']:
            for gate in cfg['gates']:
                for seed in cfg['seeds']:
                    result = json.loads((root/f'{stage}_{task}_{gate}_seed{seed}'/'result.json').read_text())
                    scores[gate].append(result['best_dev']['score'])
    means = {gate:statistics.mean(values) for gate,values in scores.items()}
    selected = []
    decisions = []
    for family, size in [('mpo','bond_dim'),('amp','depth')]:
        for budget in ['full','half']:
            names = [gate for gate in cfg['gates'] if gate.startswith(family+'_') and gate.endswith('_'+budget)]
            best = max(means[gate] for gate in names)
            eligible = [gate for gate in names if means[gate] >= best-.005-1e-12]
            chosen = min(eligible,key=lambda gate:cfg['variants'][gate][size])
            selected.append(chosen)
            decisions.append(dict(family=family,budget=budget,selected=chosen,best_mean=best,
                                  within_0_5pp=eligible,candidates={gate:means[gate] for gate in names}))
    record = dict(pilot_group=str(root),selection_split='internal_dev_only',
                  rule='mean best dev score over tasks; within 0.005 choose smallest chi/depth',
                  internal_dev_means=means,decisions=decisions)
    write_json(root/'selection.json',record)
    formal = load_config('experiments/gate_benchmark/configs/titan_v.json')
    formal['output_dir'] = 'outputs/gate_benchmark_round2/formal'
    formal['gates'] = ['mlp','mlp_half','compact_born']+selected
    formal['variants'] = {gate:cfg['variants'][gate] for gate in ['mlp_half']+selected}
    formal['study_protocol'] = cfg['study_protocol']
    formal['pilot_selection_sha256'] = digest(record)
    validate(formal)
    path = Path(formal_config)
    if path.exists():
        assert json.loads(path.read_text()) == formal, 'refusing to replace a frozen formal config'
    else:
        write_json(path,formal)
    print(json.dumps(dict(decisions=decisions,formal_config=str(path),
                          formal_group=str(Path(formal['output_dir'])/digest(formal)[:12]),
                          formal_runs=7*6*3),indent=2))
    return formal


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--pilot',default='experiments/gate_benchmark/configs/pareto_pilot.json')
    parser.add_argument('--output',default='experiments/gate_benchmark/configs/pareto_formal.json')
    args=parser.parse_args()
    select(args.pilot,args.output)
