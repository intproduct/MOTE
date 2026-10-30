"""Gauge-aware raw/centered energy controls for expert-function distances."""
from __future__ import annotations
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import numpy as np
import torch
from ..autoregressive_gate.data import write_json,digest
from .round2_energy import proxies
from .round2_numeric import row_spearman


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    torch.set_num_threads(1);data=Path(args.data);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((data/'config.json').read_text(encoding='utf8'));rows=[];repo=Path(__file__).resolve().parents[2]
    root=repo/'outputs/operator_geometry_round2'
    sources={f'experiments/operator_geometry/{name}':digest(Path(__file__).with_name(name)) for name in
             ['round2_energy_decomposition.py','round2_energy.py','round2_numeric.py']}
    write_json(out/'source_hashes.json',sources)
    for path in sources:
        dest=out/'source_snapshot'/path;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/path,dest)
    status={'state':'running','pid':os.getpid(),'group':str(out.resolve()),'completed':0,'total':len(cfg['seeds'])}
    write_json(root/'ENERGY_STATUS.json',status)
    for seed in cfg['seeds']:
        src=data/f's{seed}';record=json.loads((src/'result.json').read_text(encoding='utf8'))
        for name,sha in record['files'].items():assert digest(src/name)==sha
        cache=torch.load(src/'train.pt',map_location='cpu',weights_only=True)
        outputs=cache['outputs'];contrast=outputs-outputs.mean(1,keepdim=True)
        energy={'raw':outputs.square().sum(-1).mean(0).numpy().astype(np.float64),
                'centered':contrast.square().sum(-1).mean(0).numpy().astype(np.float64)}
        del cache,outputs,contrast
        distance=np.load(src/'train_pair_distance.npy')
        np.savez_compressed(out/f's{seed}_train_geometry.npz',distance=distance,**{k+'_energy':v for k,v in energy.items()})
        for split in ['dev','validation']:
            cache=torch.load(src/f'{split}.pt',map_location='cpu',weights_only=True)
            exact=dict(np.load(src/f'{split}_exact.npz'));original=cache['outputs'].numpy();full=None
            for gauge in ['raw','centered']:
                output=original if gauge=='raw' else original-original.mean(1,keepdims=True)
                estimates,mass,per_token_energy=proxies(output,cache['weights'].numpy(),exact,distance,energy[gauge])
                if full is None:full=estimates['local_full_distance'].copy()
                else:np.testing.assert_allclose(full,estimates['local_full_distance'],atol=1e-4,rtol=1e-5)
                result={'seed':seed,'split':split,'gauge':gauge,'tokens':len(output),'estimates':{}}
                for name,values in estimates.items():
                    shape=(-1,12)
                    result['estimates'][name]={
                        'rho_vs_local_full_distance':float(np.nanmean(row_spearman(values.reshape(shape),full.reshape(shape)))),
                        'rho_vs_abs_delta_nll':float(np.nanmean(row_spearman(values.reshape(shape),abs(exact['delta_nll']).reshape(shape)))),
                        'nearest_target_agreement_with_full_distance':float((values.reshape(shape).argmin(1)==full.reshape(shape).argmin(1)).mean())}
                result['cross_abs_over_energy_mean']=float((abs(estimates['local_cross_inner'])/np.maximum(estimates['local_energy_only'],1e-12)).mean())
                result['global_energy_cv']=float(energy[gauge].std()/energy[gauge].mean())
                np.savez_compressed(out/f's{seed}_{split}_{gauge}_proxies.npz',**estimates,mass=mass,energy=per_token_energy)
                rows.append(result)
            del cache,original,output,estimates
        status.update(completed=status['completed']+1,current_seed=seed);write_json(root/'ENERGY_STATUS.json',status)
        print('ENERGY_DECOMPOSITION',seed,flush=True)
    assert all(digest(repo/path)==sha for path,sha in sources.items())
    write_json(out/'summary.json',{'rows':rows,'full_distance_common_shift_invariance_passed':True,
        'identity':'D_ij=E_i+E_j-2 inner_ij. On a regular basis graph the total node-energy term is assignment-invariant.',
        'scope':'Report both raw and centered gauges; pair distance is invariant to common shifts, energy and cross terms separately are not. Expert-family compatibility requires evidence beyond per-expert magnitude.'})
    write_json(out/'COMPLETE.json',{'seeds':cfg['seeds'],'finished':datetime.now().astimezone().isoformat()})
    status.update(state='ready_for_review',current_seed=None);write_json(root/'ENERGY_STATUS.json',status)


if __name__=='__main__':main()
