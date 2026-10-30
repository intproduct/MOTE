"""CPU-only mechanism audit of existing saved angle-router pilots.

Near-boundary basis adjacency and actual perturbation swaps are separate tests.
Gaussian input stress is explicitly not hardware floating-point nondeterminism.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import numpy as np
import torch
from torch.nn import functional as F
from ..autoregressive_gate.data import digest,write_json
from ..routing_stability.runner import setup,read_json,canonical_hash,timestamp
from ..routing_stability.model import CausalLM
from .runner import REPO
from .routers import ControlledRouter,sparse_weights
from .metrics import head


def hamming(a,b):
    a=np.asarray(a,dtype=np.int64);b=np.asarray(b,dtype=np.int64)
    return sum(((a^b)>>k)&1 for k in range(4))


@torch.no_grad()
def measure(model,router,cache,distance,seed):
    rows=[];events=[];n=len(cache['x']);ranks=[];normed=[]
    mapping=router.inverse_assignment.cpu().numpy()
    for start in range(0,n,256):
        ix=torch.arange(start,min(start+256,n));x=cache['x'][ix];p=router(x)
        top=p.topk(5,-1).indices;a=top[:,3].numpy();b=top[:,4].numpy()
        ranks.extend(hamming(mapping[a],mapping[b]).tolist());normed.extend(distance[a,b].tolist())
    rows.append({'condition':'boundary_4_5','tokens':n,'single_swaps':n,'hamming_one_fraction':float(np.mean(np.array(ranks)==1)),
                 'mean_hamming':float(np.mean(ranks)),'mean_train_function_distance':float(np.mean(normed)),
                 'uniform_unordered_pair_hamming_one_fraction':4/15})
    for condition in ['bf16_input_roundtrip','fp16_input_roundtrip','gaussian_rms_001_r0','gaussian_rms_001_r1','gaussian_rms_001_r2']:
        gen=torch.Generator().manual_seed(seed+20261103+(int(condition[-1]) if condition.startswith('gaussian') else 0))
        count=0;multi=0;hs=[];ds=[];deltas=[];harm=0
        for start in range(0,n,256):
            ix=torch.arange(start,min(start+256,n));x=cache['x'][ix];p=router(x)
            if condition.startswith('gaussian'):
                xp=x+.001*x.square().mean(-1,keepdim=True).sqrt()*torch.randn(x.shape,generator=gen)
            else:xp=x.to(torch.bfloat16 if condition.startswith('bf16') else torch.float16).float()
            q=router(xp);w=sparse_weights(p,4);v=sparse_weights(q,4)
            removed=(w>0)&(v==0);added=(v>0)&(w==0);changed=removed.any(-1)
            count+=int(changed.sum());multi+=int((removed.sum(-1)>1).sum())
            single=removed.sum(-1)==1
            if single.any():
                a=removed[single].long().argmax(-1).numpy();b=added[single].long().argmax(-1).numpy()
                hs.extend(hamming(mapping[a],mapping[b]).tolist());ds.extend(distance[a,b].tolist())
            if changed.any():
                j=ix[changed];out=cache['outputs'][j];h=cache['hidden'][j];old=cache['weights'][j]
                ref=head(model,h+((w[changed]-old)[:,:,None]*out).sum(1))
                alt=head(model,h+((v[changed]-old)[:,:,None]*out).sum(1))
                d=(F.cross_entropy(alt,cache['labels'][j],reduction='none')-F.cross_entropy(ref,cache['labels'][j],reduction='none')).numpy()
                deltas.extend(d.tolist());harm+=int((d>.1).sum())
                for k,value in zip(j.tolist(),d.tolist()):events.append({'condition':condition,'token_row':k,'delta_nll_given_set_flip':value})
        rows.append({'condition':condition,'tokens':n,'set_flips':count,'multi_swaps':multi,'single_swaps':len(hs),
                     'hamming_one_fraction':float(np.mean(np.array(hs)==1)) if hs else None,
                     'mean_hamming':float(np.mean(hs)) if hs else None,
                     'mean_train_function_distance':float(np.mean(ds)) if ds else None,
                     'set_flip_harm_gt_01':harm,'set_flip_mean_delta_nll':float(np.mean(deltas)) if deltas else None,
                     'set_flip_max_abs_delta_nll':float(np.max(np.abs(deltas))) if deltas else None})
    return rows,events


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='outputs/operator_geometry_round2');args=p.parse_args()
    setup();root=REPO/args.root;lock=root/'BOUNDARY_LOCK.json'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w') as f:json.dump({'pid':os.getpid(),'device':'cpu','started':timestamp()},f)
    try:
        source=REPO/'outputs/operator_geometry_round1/router_pilot/da5325087d6b'
        cfg=read_json(source/'config.json');methods=['angle_natural','angle_random','angle_functional','product_functional']
        protocol={'source_group':str(source),'methods':methods,'seeds':cfg['seeds'],'device':'cpu','scope':'post-hoc mechanism diagnostics, not new candidate selection'}
        dest=root/'boundary'/canonical_hash(protocol)[:12];dest.mkdir(parents=True,exist_ok=False)
        write_json(dest/'protocol.json',protocol);shutil.copy2(__file__,dest/'source.py')
        rows=[];events=[]
        for seed in cfg['seeds']:
            data=source/'data'/f's{seed}';geo=read_json(data/'geometry.json');distance=np.load(data/'train_pair_distance.npy')
            model=CausalLM(read_json(REPO/cfg['source_config']),'mlp_half',seed).eval()
            model.load_state_dict(torch.load(REPO/cfg['source_group']/'runs'/f'mlp_half_s{seed}'/'best.pt',map_location='cpu',weights_only=False)['model'],strict=True)
            model.requires_grad_(False)
            for split in ['train','validation']:
                cache=torch.load(data/f'{split}.pt',map_location='cpu',weights_only=True)
                for method in methods:
                    checkpoint=source/'runs'/f'{method}_s{seed}'/'final.pt'
                    assert digest(checkpoint)==read_json(checkpoint.parent/'result.json')['files']['final.pt']
                    router=ControlledRouter(cache['x'].shape[-1],16,cfg['rank'],method,np.array(geo['expert_for_basis']),
                         np.load(data/'codes.npy'),np.array(geo['random_expert_for_basis'])).eval()
                    router.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model'],strict=True)
                    r,e=measure(model,router,cache,distance,seed)
                    rows.extend({'seed':seed,'method':method,'split':split,**x} for x in r)
                    events.extend({'seed':seed,'method':method,'split':split,**x} for x in e)
                    print('BOUNDARY_DONE',seed,split,method,flush=True)
            del model,cache
        write_json(dest/'rows.json',rows);write_json(dest/'events.json',events)
        write_json(dest/'COMPLETE.json',{'finished':timestamp(),'rows':len(rows),'event_rows':len(events),'source_sha256':digest(Path(__file__)),
                    'scope':'Post-hoc CPU replay. NLL damage measured conditional on a set flip; continuous weight-only changes not included. Gaussian input stress is not native GPU precision.'})
        print(dest,flush=True)
    finally:lock.unlink()


if __name__=='__main__':main()
