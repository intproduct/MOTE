"""CPU readout-boundary sensitivity on frozen first-cycle checkpoints.

The 4th/5th fixed pair is a local diagnostic, not an exhaustive certificate for
all 4-by-12 crossings. Input roundtrips do not emulate native low-precision GEMM.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import traceback
import numpy as np
import torch
from ..autoregressive_gate.data import digest,write_json
from ..routing_stability.runner import setup,timestamp
from ..routing_stability.model import full_state_hash
from .routers import ControlledRouter
from .round2_boundary import hamming
from .round2_train import train_hashes
from .metrics import bootstrap_window_mean


def read(path):return json.loads(path.read_text(encoding='utf8'))


def local_boundary(router,x):
    x=x.detach().requires_grad_(True)
    p=router(x);indices=p.topk(5,dim=-1).indices
    pair=indices[:,3:5]
    logp=p.log();margin=logp.gather(1,pair[:,0:1])-logp.gather(1,pair[:,1:2])
    gradient=torch.autograd.grad(margin.sum(),x)[0]
    return p.detach(),pair.detach(),margin[:,0].detach(),gradient.detach()


def measure(router,x,window,batch):
    arrays={key:[] for key in ['margin','gradient_norm','linear_l2_radius','boundary_hamming']}
    conditions={name:{key:[] for key in ['projected_delta_margin','actual_delta_margin','pair_crossed','predicted_pair_crossed','set_flip','multiple_swap','input_l2']}
                for name in ['bf16_input_roundtrip','fp16_input_roundtrip']}
    for start in range(0,len(x),batch):
        a=x[start:start+batch];p,pair,margin,gradient=local_boundary(router,a)
        norm=gradient.norm(dim=-1)
        arrays['margin'].append(margin.numpy());arrays['gradient_norm'].append(norm.numpy())
        arrays['linear_l2_radius'].append((margin/norm.clamp_min(1e-12)).numpy())
        mapping=router.inverse_assignment.cpu().numpy()
        arrays['boundary_hamming'].append(hamming(mapping[pair[:,0]],mapping[pair[:,1]]))
        for name,values in conditions.items():
            b=a.to(torch.bfloat16 if name.startswith('bf16') else torch.float16).float()
            with torch.no_grad():q=router(b);lq=q.log()
            actual=lq.gather(1,pair[:,0:1])[:,0]-lq.gather(1,pair[:,1:2])[:,0]
            linear=margin+((b-a)*gradient).sum(-1)
            old=torch.zeros_like(p,dtype=torch.bool).scatter(1,p.topk(4,-1).indices,True)
            new=torch.zeros_like(q,dtype=torch.bool).scatter(1,q.topk(4,-1).indices,True)
            removed=(old&~new).sum(-1)
            updates={'projected_delta_margin':linear-margin,'actual_delta_margin':actual-margin,
                     'pair_crossed':actual<0,'predicted_pair_crossed':linear<0,'set_flip':removed>0,
                     'multiple_swap':removed>1,'input_l2':(b-a).norm(dim=-1)}
            for key,value in updates.items():values[key].append(value.numpy())
    arrays={key:np.concatenate(value) for key,value in arrays.items()}
    summary={'tokens':len(x),'margin_quantiles':np.quantile(arrays['margin'],[0,.01,.1,.5,.9,1]).tolist(),
             'gradient_norm_quantiles':np.quantile(arrays['gradient_norm'],[0,.01,.1,.5,.9,1]).tolist(),
             'linear_l2_radius_quantiles':np.quantile(arrays['linear_l2_radius'],[0,.01,.1,.5,.9,1]).tolist(),
             'hamming_one_fraction':float((arrays['boundary_hamming']==1).mean()),'conditions':{}}
    for name,values in conditions.items():
        values={key:np.concatenate(value) for key,value in values.items()}
        arrays.update({name+'__'+key:value for key,value in values.items()})
        actual=values['pair_crossed'];predicted=values['predicted_pair_crossed'];flips=values['set_flip']
        error=abs(values['projected_delta_margin']-values['actual_delta_margin'])
        summary['conditions'][name]={'set_flips':int(flips.sum()),'set_flip_rate':float(flips.mean()),
            'set_flip_rate_ci95_windows':bootstrap_window_mean(flips.astype(float),window,20261033,repeats=1000),
            'pair_crossings':int(actual.sum()),'predicted_pair_crossings':int(predicted.sum()),
            'prediction_true_positive':int((actual&predicted).sum()),'prediction_false_positive':int((~actual&predicted).sum()),
            'prediction_false_negative':int((actual&~predicted).sum()),'set_flips_without_pair_crossing':int((flips&~actual).sum()),
            'multiple_swaps':int(values['multiple_swap'].sum()),'linearization_abs_error_quantiles':np.quantile(error,[.5,.9,.99,1]).tolist()}
    arrays['window']=window
    for key,value in arrays.items():
        if not np.isfinite(value).all():raise FloatingPointError(key)
    return summary,arrays


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();repo=Path(__file__).resolve().parents[2];cfg=read(repo/args.config);dest=repo/args.output
    setup();dest.mkdir(parents=True,exist_ok=False);write_json(dest/'config.json',cfg)
    source=repo/cfg['source_group'];data=repo/cfg['data_group'];pcfg=read(source/'config.json')
    sources=train_hashes();sources.update({f'experiments/operator_geometry/{name}':digest(Path(__file__).with_name(name))
        for name in ['round2_boundary.py','round2_boundary_gradient.py']})
    write_json(dest/'source_hashes.json',sources)
    for relative in sources:
        path=dest/'source_snapshot'/relative;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/relative,path)
    status={'state':'running','pid':os.getpid(),'device':'cpu','completed':0,'total':len(cfg['seeds'])*len(cfg['methods'])*len(cfg['splits']),'started':timestamp()}
    write_json(dest/'STATUS.json',status)
    try:
        rows=[]
        for seed in cfg['seeds']:
            geometry=source/'data'/f's{seed}';geo=read(geometry/'geometry.json');codes=np.load(geometry/'codes.npy')
            manifest=read(data/f's{seed}/result.json')
            for split in cfg['splits']:
                path=data/f's{seed}/{split}.pt';assert digest(path)==manifest['files'][path.name]
                cache=torch.load(path,map_location='cpu',weights_only=True)
                x=cache['x'];window=cache['window'].numpy();del cache
                for method in cfg['methods']:
                    assert all(digest(repo/path)==sha for path,sha in sources.items())
                    checkpoint=source/'runs'/f'{method}_s{seed}'/'final.pt'
                    assert digest(checkpoint)==read(checkpoint.parent/'result.json')['files']['final.pt']
                    router=ControlledRouter(x.shape[-1],16,pcfg['rank'],method,np.asarray(geo['expert_for_basis']),codes,np.asarray(geo['random_expert_for_basis'])).eval()
                    router.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model'],strict=True);router.requires_grad_(False)
                    before=full_state_hash(router.state_dict());summary,arrays=measure(router,x,window,cfg['batch'])
                    assert full_state_hash(router.state_dict())==before
                    out=dest/f'{method}_s{seed}_{split}.npz';np.savez_compressed(out,**arrays)
                    rows.append({'seed':seed,'method':method,'split':split,'checkpoint_sha256':digest(checkpoint),'arrays_sha256':digest(out),**summary})
                    status['completed']=len(rows);write_json(dest/'STATUS.json',status);print('BOUNDARY_GRADIENT_DONE',seed,split,method,flush=True)
        write_json(dest/'raw_summary.json',rows);status.update(state='complete',finished=timestamp());write_json(dest/'STATUS.json',status)
        write_json(dest/'COMPLETE.json',status)
    except BaseException:
        status.update(state='failed',error=traceback.format_exc());write_json(dest/'STATUS.json',status);raise


if __name__=='__main__':main()
