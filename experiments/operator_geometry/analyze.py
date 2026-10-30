"""Post-run audit and conditional checks; does not modify source runs."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from ..autoregressive_gate.data import digest,write_json
from .metrics import spearman


def main():
    parser=argparse.ArgumentParser();parser.add_argument('group',type=Path);args=parser.parse_args()
    cfg=json.loads((args.group/'config.json').read_text());rows=[];missing=[];issues=[]
    for seed in cfg['seeds']:
        for variant in cfg['variants']:
            run=args.group/'runs'/f'{variant}_s{seed}'
            if not (run/'result.json').exists():missing.append(run.name);continue
            result=json.loads((run/'result.json').read_text())
            for name,sha in result['files'].items():
                if digest(run/name)!=sha:issues.append(f'{run.name}/{name}: checksum mismatch')
            data=np.load(run/'validation_interventions.npz')
            cache=torch.load(run/'validation_cache.pt',map_location='cpu',weights_only=True)
            n,c=data['kl'].shape
            pair=np.load(run/'train_pair_distance.npy')
            conditioned={}
            # Hold token AND removed expert (therefore its mass) fixed.
            # This prevents a large weight from masquerading as pair geometry.
            for name,p in [('local_output',data['output_distance']),
                           ('train_global_output',pair[data['source'],data['target']]),
                           ('fisher',data['fisher_kl']),('score_gap',data['score_gap'])]:
                rs=[]
                for i in range(n):
                    for expert in np.unique(data['source'][i]):
                        selected=data['source'][i]==expert
                        r=spearman(p[i,selected],data['kl'][i,selected])
                        if r is not None:rs.append(r)
                conditioned[name]=float(np.mean(rs)) if rs else None
            s=result['summaries']['validation']
            row={'variant':variant,'seed':seed,'validation_tokens':n,'interventions':n*c,
                 'baseline_sample_ce':s['baseline_nll']['mean'],
                 'oracle_delta_ce':s['best_sampled_replacement_delta_nll_mean'],
                 'oracle_ci_hi':s['best_sampled_replacement_ci95_windows'][1],
                 'harm_fraction':s['harm_gt_0p1_fraction'],
                 'mass_only_kl_rho':spearman(data['mass'].ravel(),data['kl'].ravel()),
                 **{f'{k}_within_token_source_rho':v for k,v in conditioned.items()},
                 'state_unchanged':result['weights_unchanged']}
            rows.append(row)
    assert not missing and not issues,(missing,issues)
    with (args.group/'diagnostic_summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    eligible={}
    for variant in cfg['variants']:
        subset=[r for r in rows if r['variant']==variant]
        good=sum(r['oracle_ci_hi'] < -cfg['pilot_decision']['oracle_mean_nll_gain_floor'] for r in subset)
        eligible[variant]={'seeds_passing':good,'signal_sufficient_for_small_pilot':good>=cfg['pilot_decision']['required_seeds_with_window_ci_gain']}
    write_json(args.group/'DIAGNOSTIC_AUDIT.json',{'runs':len(rows),'expected_runs':len(cfg['variants'])*len(cfg['seeds']),
        'missing':missing,'issues':issues,'rows':rows,'pilot_eligibility':eligible,
        'scope':'Diagnostic only. Oracle uses realized labels and chooses among 48 routes. It is not a deployable routing improvement. Within-source correlations control removed mass. Validation has been observed in prior studies.'})
    print(json.dumps({'pilot_eligibility':eligible,'rows':rows},indent=2))


if __name__=='__main__':main()
