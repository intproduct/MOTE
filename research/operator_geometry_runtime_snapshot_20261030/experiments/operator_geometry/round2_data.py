"""Disjoint cached states, input-conditional utility, and exact replacement targets.

The first-order cost of expert e is grad_hidden CE dot expert_output_e.
It depends on a realized training label and is never supplied at inference.
All exact probes preserve the original four-expert budget and removed mass.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F
from ..autoregressive_gate.data import digest, write_json
from ..routing_stability.runner import setup, read_json, canonical_hash, timestamp
from ..routing_stability.model import CausalLM, full_state_hash
from .runner import REPO, collect, source_hashes, train_pair_geometry
from .metrics import head, substitution_indices, fixed_mass_delta


def data_hashes():
    values=source_hashes()
    values['experiments/operator_geometry/round2_data.py']=digest(Path(__file__))
    return values


def directional_cost(model, hidden, outputs, labels):
    """Exact derivative of summed token CE with respect to independent h rows."""
    with torch.enable_grad():
        h=hidden.detach().requires_grad_(True)
        loss=F.cross_entropy(head(model,h),labels,reduction='sum')
        gradient,=torch.autograd.grad(loss,h)
    costs=torch.einsum('nh,neh->ne',gradient.detach(),outputs)
    return costs-costs.mean(-1,keepdim=True)


@torch.no_grad()
def exact_replacements(model,cache,batch):
    a,b=substitution_indices(cache['weights'],4)
    count=len(a);device=next(model.parameters()).device
    n=torch.arange(count)[:,None].expand_as(a).flatten()
    source,target=a.flatten(),b.flatten()
    baseline=cache['baseline_nll']
    values=[]
    for start in range(0,len(n),batch):
        ix=n[start:start+batch];s=source[start:start+batch].to(device);t=target[start:start+batch].to(device)
        h,w,out=[cache[k][ix].to(device) for k in ['hidden','weights','outputs']]
        logits=head(model,h+fixed_mass_delta(out,w,s,t))
        ce=F.cross_entropy(logits,cache['labels'][ix].to(device),reduction='none')
        values.append((ce.cpu()-baseline[ix]).numpy())
        if start and start%(batch*128)==0:print('EXACT',start,len(n),flush=True)
    return {'source':a.numpy(),'target':b.numpy(),'delta_nll':np.concatenate(values).reshape(count,48)}


def prepare_split(model,source_tokens,cfg,split,offset=0):
    calls=[0];started=time.perf_counter()
    def progress(*args):
        calls[0]+=1
        if calls[0]%128==0:print('CACHE_PROGRESS',split,'forwards',calls[0],
                                'seconds',round(time.perf_counter()-started,1),flush=True)
    handle=model.register_forward_hook(progress)
    try:cache,checks=collect(model,source_tokens,{**cfg,'windows_per_split':cfg[f'{split}_windows']},split)
    finally:handle.remove()
    cache['window']+=offset
    device=next(model.parameters()).device;target=[];base=[];teacher=[]
    gate=model.get_submodule(cfg['module']).mote.core.gate
    for start in range(0,len(cache['x']),cfg['target_batch']):
        h,out,y=[cache[k][start:start+cfg['target_batch']].to(device) for k in ['hidden','outputs','labels']]
        target.append(directional_cost(model,h,out,y).cpu())
        with torch.no_grad():
            base.append(F.cross_entropy(head(model,h),y,reduction='none').cpu())
            teacher.append(gate._softmax_temp(cache['scores'][start:start+cfg['target_batch']].to(device)).cpu())
    cache['utility_cost']=torch.cat(target);cache['baseline_nll']=torch.cat(base);cache['teacher']=torch.cat(teacher)
    if not all(torch.isfinite(t).all() for t in cache.values()):raise FloatingPointError(split)
    return cache,checks


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    p.add_argument('--root',default='outputs/operator_geometry_round2');args=p.parse_args()
    setup();cfg=read_json(args.config);root=(REPO/args.root).resolve();root.mkdir(parents=True,exist_ok=True)
    lock=root/'PIPELINE_LOCK.json';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w') as f:json.dump({'pid':os.getpid(),'stage':'data','started':timestamp()},f)
    status={'state':'starting','pid':os.getpid(),'stage':'data','completed':0,'total':len(cfg['seeds']),
            'started':timestamp(),'config_hash':canonical_hash(cfg)}
    try:
        if cfg['device']=='cuda':torch.cuda.set_per_process_memory_fraction(cfg['memory_fraction'])
        group=root/cfg['stage']/canonical_hash(cfg)[:12]
        group.mkdir(parents=True,exist_ok=False)
        frozen=data_hashes();write_json(group/'config.json',cfg);write_json(group/'source_hashes.json',frozen)
        for path in frozen:
            dst=group/'source_snapshot'/path;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/path,dst)
        status.update(state='running',group=str(group),log='logs/data_v1.log',err='logs/data_v1.err')
        write_json(root/'ACTIVE_PROCESS.json',{**status,'active':True});write_json(root/'STATUS.json',status)
        train=np.load(REPO/cfg['data_root']/'train.npy');validation=np.load(REPO/cfg['data_root']/'validation.npy')
        cut=int((len(train)-1)//cfg['context']*cfg['train_fraction'])*cfg['context']
        arrays={'train':train[:cut],'dev':train[cut:],'validation':validation}
        write_json(group/'split_manifest.json',{'train_token_sha256':digest(REPO/cfg['data_root']/'train.npy'),
                  'validation_token_sha256':digest(REPO/cfg['data_root']/'validation.npy'),
                  'cut_token_index':cut,'train_range':[0,cut],'dev_range':[cut,len(train)],
                  'policy':'Train range is half-open at cut; no train inputs or targets reach dev. First dev context begins at cut. Test samples are not read.'})
        source_cfg=read_json(REPO/cfg['source_config']);seeds=[]
        for seed in cfg['seeds']:
            assert data_hashes()==frozen
            run=group/f's{seed}';run.mkdir(exist_ok=False)
            checkpoint=REPO/cfg['source_group']/'runs'/f'mlp_half_s{seed}'/'best.pt'
            expected=read_json(checkpoint.parent/'result.json')['checkpoint_sha256'];assert digest(checkpoint)==expected
            model=CausalLM(source_cfg,'mlp_half',seed).to(cfg['device']).eval()
            model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model'],strict=True)
            model.requires_grad_(False);before=full_state_hash(model.state_dict());checks={};windows={}
            for split,tokens in arrays.items():
                status.update(current=f'{split}_s{seed}',updated=timestamp());write_json(root/'STATUS.json',status)
                cache,check=prepare_split(model,tokens,cfg,split,cut if split=='dev' else 0)
                checks[split]=check;windows[split]=cache['window'].unique().tolist()
                torch.save(cache,run/f'{split}.pt')
                if split=='train':
                    np.save(run/'train_pair_distance.npy',train_pair_geometry(cache))
                    probe={k:v[:cfg['train_exact_probe_tokens']] for k,v in cache.items()}
                else:probe=cache
                exact=exact_replacements(model,probe,cfg['exact_batch'])
                exact['window']=probe['window'].numpy();exact['position']=probe['position'].numpy()
                np.savez_compressed(run/f'{split}_exact.npz',**exact)
                print('SPLIT_DONE',seed,split,len(cache['x']),flush=True)
                del cache,probe
            assert not set(windows['train'])&set(windows['dev'])
            assert full_state_hash(model.state_dict())==before and digest(checkpoint)==expected
            result={'seed':seed,'config_hash':canonical_hash(cfg),'checkpoint_sha256':expected,
                    'backbone_state_hash':before,'frozen_state_unchanged':True,'disjoint_train_dev':True,
                    'identity_checks':checks,'windows':windows,
                    'files':{p.name:digest(p) for p in run.iterdir() if p.is_file()}}
            write_json(run/'result.json',result);seeds.append(seed)
            status['completed']=len(seeds);write_json(root/'STATUS.json',status)
            del model;torch.cuda.empty_cache()
        assert data_hashes()==frozen
        write_json(group/'DATA_COMPLETE.json',{'seeds':seeds,'finished':timestamp(),'source_unchanged':True})
        status.update(state='data_ready',finished=timestamp(),current=None)
        write_json(root/'STATUS.json',status);write_json(root/'ACTIVE_PROCESS.json',{**status,'active':False})
    except BaseException:
        status.update(state='failed',error=traceback.format_exc(),finished=timestamp())
        write_json(root/'STATUS.json',status);write_json(root/'ACTIVE_PROCESS.json',{**status,'active':False});raise
    finally:lock.unlink()


if __name__=='__main__':main()
