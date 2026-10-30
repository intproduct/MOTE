"""Frozen-checkpoint functional geometry. No training or test-set access."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import numpy as np
import torch

from ..autoregressive_gate.data import digest, write_json
from ..routing_stability.model import CausalLM, full_state_hash
from ..routing_stability.runner import setup, read_json, canonical_hash, timestamp, SOURCE_PATHS
from ..routing_stability.routers import SharedBornRouter
from .metrics import (substitution_indices, fixed_mass_delta, head, head_jvp,
                      compare_outputs, spearman, bootstrap_window_mean)

REPO = Path(__file__).resolve().parents[2]


def source_hashes():
    paths = set(SOURCE_PATHS) | {
        'experiments/operator_geometry/runner.py',
        'experiments/operator_geometry/metrics.py',
        'experiments/routing_stability/configs/titan_v.json',
    }
    return {p:digest(REPO/p) for p in sorted(paths)}


def tensor_state_hash(model):
    return full_state_hash(model.state_dict())


def scalar_metrics(values):
    values=np.asarray(values, dtype=float)
    return {'mean':float(values.mean()), 'median':float(np.median(values)),
            'p95':float(np.quantile(values,.95)), 'p99':float(np.quantile(values,.99)),
            'min':float(values.min()), 'max':float(values.max())}


@torch.no_grad()
def collect(model, tokens, cfg, split):
    module = model.get_submodule(cfg['module'])
    gate = module.mote.core.gate
    experts = module.mote.core.num_blocks
    assert cfg['module'] == f'blocks.{len(model.blocks)-1}.down', 'exact local continuation requires final down'
    device=next(model.parameters()).device
    length=cfg['context']; count=cfg['windows_per_split']
    # Distinct nonoverlapping windows; same starts across models and seeds.
    rng=np.random.default_rng(cfg['sampling_seed'] + (0 if split=='train' else 1))
    blocks=np.sort(rng.choice((len(tokens)-1)//length,count,replace=False))
    starts=blocks*length
    chunks={k:[] for k in ['x','hidden','outputs','weights','scores','labels','window','position']}
    max_reconstruction=max_head=max_forced=0.
    for offset in range(0,count,cfg['window_batch']):
        chosen=starts[offset:offset+cfg['window_batch']]
        ids=torch.tensor(np.stack([tokens[s:s+length] for s in chosen]),device=device,dtype=torch.long)
        target=torch.tensor(np.stack([tokens[s+1:s+length+1] for s in chosen]),device=device,dtype=torch.long)
        seen={}
        def save_down(mod, inputs, output):
            seen['x']=inputs[0].detach();seen['down']=output.detach()
        def save_gate(mod, inputs, output):
            seen['weights']=output[0].detach()
        def save_scores(mod, inputs, output):
            seen['scores']=output.detach()
        def save_hidden(mod, inputs):
            seen['hidden']=inputs[0].detach()
        handles=[module.register_forward_hook(save_down),gate.register_forward_hook(save_gate),
                 gate.router.register_forward_hook(save_scores),model.norm.register_forward_pre_hook(save_hidden)]
        try:
            logits=model(ids)
        finally:
            for h in handles:h.remove()
        idx=(torch.arange(len(ids),device=device)[:,None]*length+
             torch.tensor(cfg['positions'],device=device)[None,:]).flatten()
        x=seen['x'].reshape(-1,module.in_features)[idx]
        weights=seen['weights'].reshape(-1,experts)[idx]
        hidden=seen['hidden'].reshape(-1,module.out_features)[idx]
        outputs=[]
        for expert in range(experts):
            one=x.new_zeros(len(x),experts);one[:,expert]=1
            value,_=module.mote(x,probs=one,mask=one)
            outputs.append(value+module.bias)
        outputs=torch.stack(outputs,dim=1)
        reconstructed=(weights[:,:,None]*outputs).sum(1)
        actual=seen['down'].reshape(-1,module.out_features)[idx]
        max_reconstruction=max(max_reconstruction,float((reconstructed-actual).abs().max()))
        max_head=max(max_head,float((head(model,hidden)-logits.reshape(-1,logits.shape[-1])[idx]).abs().max()))
        if offset==0:
            # Full-model injected replay tests the shortcut independently.
            a,b=substitution_indices(weights,gate.k)
            changed=seen['weights'].clone().reshape(-1,experts)
            mass=weights.gather(1,a[:,:1]).squeeze(1)
            changed[idx,a[:,0]]=0;changed[idx,b[:,0]]=mass
            def force(mod,inputs,result):
                return changed.reshape_as(result[0]),(changed>0).to(changed.dtype).reshape_as(result[1]),result[2]
            handle=gate.register_forward_hook(force)
            try:
                replay=model(ids).reshape(-1,logits.shape[-1])[idx]
            finally:
                handle.remove()
            exact=head(model,hidden+fixed_mass_delta(outputs,weights,a[:,0],b[:,0]))
            max_forced=max(max_forced,float((replay-exact).abs().max()))
        cpu={'x':x,'hidden':hidden,'outputs':outputs,'weights':weights,
             'scores':seen['scores'].reshape(-1,experts)[idx],
             'labels':target.flatten()[idx],
             'window':torch.tensor(np.repeat(chosen,len(cfg['positions']))),
             'position':torch.tensor(np.tile(cfg['positions'],len(chosen)))}
        for k,value in cpu.items():chunks[k].append(value.cpu())
    checks={'mixture_max_abs_error':max_reconstruction,'head_max_abs_error':max_head,
            'forced_replay_max_abs_error':max_forced}
    if max(checks.values())>cfg['identity_atol']:
        raise AssertionError(f'counterfactual identity failed: {checks}')
    return {k:torch.cat(v) for k,v in chunks.items()},checks


@torch.no_grad()
def interventions(model, cache, cfg):
    device=next(model.parameters()).device
    w=cache['weights']; a,b=substitution_indices(w,int((w>0).sum(1)[0]))
    n,c=a.shape
    repeated=torch.arange(n)[:,None].expand(n,c).flatten()
    source,target=a.flatten(),b.flatten()
    fields={k:[] for k in ['delta_nll','kl','fisher_kl','first_order_nll','argmax_changed',
                          'output_distance','weighted_output_distance','mass','score_gap']}
    router=model.get_submodule(cfg['module']).mote.core.gate.router
    score=cache['scores'].clamp_min(1e-30).log() if isinstance(router,SharedBornRouter) else cache['scores']
    score=(score-score.mean(-1,keepdim=True))/(score.std(-1,unbiased=False,keepdim=True)+1e-12)
    for begin in range(0,len(repeated),cfg['intervention_batch']):
        end=begin+cfg['intervention_batch']; r=repeated[begin:end]
        s,t=source[begin:end].to(device),target[begin:end].to(device)
        outputs,weights,h=[cache[k][r].to(device) for k in ['outputs','weights','hidden']]
        dh=fixed_mass_delta(outputs,weights,s,t)
        stats=compare_outputs(head(model,h),head(model,h+dh),head_jvp(model,h,dh),cache['labels'][r].to(device))
        j=torch.arange(len(r),device=device)
        stats.update(output_distance=(outputs[j,t]-outputs[j,s]).square().sum(-1),
                     weighted_output_distance=dh.square().sum(-1),mass=weights[j,s],
                     score_gap=(score[r,source[begin:end]]-score[r,target[begin:end]]).to(device))
        for k,v in stats.items():
            if not torch.isfinite(v).all():raise FloatingPointError(k)
            fields[k].append(v.cpu().numpy())
    return {'source':a.numpy(),'target':b.numpy(),**{k:np.concatenate(v).reshape(n,c) for k,v in fields.items()}}


@torch.no_grad()
def rounded_router(model,cache,cfg):
    device=next(model.parameters()).device
    gate=model.get_submodule(cfg['module']).mote.core.gate
    x,h,outputs,w=[cache[k].to(device) for k in ['x','hidden','outputs','weights']]
    result={}
    for dtype,name in [(torch.float16,'fp16_input_roundtrip'),(torch.bfloat16,'bf16_input_roundtrip')]:
        changed,_,_=gate(x.to(dtype).float())
        dh=((changed-w)[:,:,None]*outputs).sum(1)
        stats=compare_outputs(head(model,h),head(model,h+dh),head_jvp(model,h,dh),cache['labels'].to(device))
        flip=((changed>0)!=(w>0)).any(-1).cpu().numpy()
        delta=stats['delta_nll'].cpu().numpy()
        result[name]={'tokens':len(x),'flip_count':int(flip.sum()),'flip_rate':float(flip.mean()),
            'delta_nll':scalar_metrics(delta),'abs_delta_nll':scalar_metrics(abs(delta)),
            'harm_gt_0p1_count':int((delta>.1).sum()),
            'kl':scalar_metrics(stats['kl'].cpu().numpy())}
    return result


def train_pair_geometry(cache):
    out=cache['outputs'].numpy().astype(np.float64)
    d=np.zeros((out.shape[1],out.shape[1]))
    for i in range(len(d)):
        for j in range(len(d)):
            d[i,j]=np.square(out[:,i]-out[:,j]).sum(-1).mean()
    return d


def summarize(events,cache,pair,cfg):
    d=events['delta_nll']; k=events['kl']; ids=cache['window'].numpy()
    global_proxy=pair[events['source'],events['target']]*events['mass']**2
    predictors={'input_conditioned_output':events['weighted_output_distance'],
                'train_global_output':global_proxy,'score_gap':events['score_gap'],
                'local_output_fisher':events['fisher_kl']}
    correlations={}
    for name,p in predictors.items():
        within=[spearman(a,b) for a,b in zip(p,k)]
        good=[v for v in within if v is not None]
        correlations[name]={'spearman_kl_pooled':spearman(p.ravel(),k.ravel()),
            'spearman_abs_nll_pooled':spearman(p.ravel(),abs(d).ravel()),
            'spearman_kl_within_token_mean':float(np.mean(good)) if good else None}
    best=np.minimum(d.min(1),0)
    return {'tokens':len(d),'interventions':d.size,'baseline_nll':scalar_metrics(cache['baseline_nll'].numpy()),
            'substitution_delta_nll':scalar_metrics(d),'substitution_kl':scalar_metrics(k),
            'best_sampled_replacement_delta_nll_mean':float(best.mean()),
            'best_sampled_replacement_ci95_windows':bootstrap_window_mean(best,ids,cfg['sampling_seed'],cfg['bootstrap_repeats']),
            'harm_gt_0p1_fraction':float((d>.1).mean()),'benefit_lt_minus_0p1_fraction':float((d<-.1).mean()),
            'predictor_correlations':correlations,
            'first_order_nll_spearman':spearman(events['first_order_nll'].ravel(),d.ravel())}


def run_one(cfg,source_cfg,variant,seed,group):
    dest=group/'runs'/f'{variant}_s{seed}'
    if dest.exists():raise RuntimeError(f'Existing run retained; refusing overwrite: {dest}')
    dest.mkdir(parents=True)
    ckpt=REPO/cfg['source_group']/'runs'/dest.name/'best.pt'
    expected=read_json(ckpt.parent/'result.json')['checkpoint_sha256']
    assert digest(ckpt)==expected
    state=torch.load(ckpt,map_location='cpu',weights_only=False)
    assert state['variant']==variant and state['seed']==seed
    assert read_json(ckpt.parent/'result.json')['config_hash']==canonical_hash(source_cfg)
    model=CausalLM(source_cfg,variant,seed).to(cfg['device']).eval()
    model.load_state_dict(state['model'],strict=True)
    model.requires_grad_(False)
    before=tensor_state_hash(model); start=time.perf_counter()
    write_json(dest/'manifest.json',{'config':cfg,'checkpoint':str(ckpt),'checkpoint_sha256':expected,
        'variant':variant,'seed':seed,'initial_state_hash':before,'started':timestamp(),
        'source_hashes':source_hashes()})
    cache={};events={};checks={};summaries={};rounded={}
    for split in cfg['splits']:
        tokens=np.load(REPO/cfg['data_root']/f'{split}.npy')
        cache[split],checks[split]=collect(model,tokens,cfg,split)
        with torch.no_grad():
            ref=head(model,cache[split]['hidden'].to(cfg['device'])).cpu()
            cache[split]['baseline_nll']=-ref.double().log_softmax(-1).gather(1,cache[split]['labels'][:,None]).squeeze(1)
        torch.save(cache[split],dest/f'{split}_cache.pt')
        events[split]=interventions(model,cache[split],cfg)
        np.savez_compressed(dest/f'{split}_interventions.npz',**events[split])
        rounded[split]=rounded_router(model,cache[split],cfg)
        print(f'PROBE {dest.name} {split} {events[split]["delta_nll"].size} substitutions',flush=True)
    pair=train_pair_geometry(cache['train'])
    np.save(dest/'train_pair_distance.npy',pair)
    for split in cfg['splits']:summaries[split]=summarize(events[split],cache[split],pair,cfg)
    after=tensor_state_hash(model)
    assert before==after,'frozen model changed'
    assert digest(ckpt)==expected,'source checkpoint changed'
    result={'variant':variant,'seed':seed,'state':'complete','finished':timestamp(),
        'checkpoint_sha256':expected,'weights_unchanged':before==after,'identity_checks':checks,
        'summaries':summaries,'roundtrip':rounded,'elapsed_seconds':time.perf_counter()-start,
        'memory_peak_allocated_mb':torch.cuda.max_memory_allocated()/2**20 if cfg['device']=='cuda' else None,
        'files':{p.name:digest(p) for p in dest.iterdir() if p.is_file()}}
    write_json(dest/'result.json',result)
    del model
    if cfg['device']=='cuda':torch.cuda.empty_cache()
    print('DONE',dest.name,round(result['elapsed_seconds'],1),'seconds',flush=True)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--root',default='outputs/operator_geometry_round1');args=parser.parse_args()
    setup();cfg=read_json(args.config);root=(REPO/args.root).resolve()
    root.mkdir(parents=True,exist_ok=True)
    lock=root/'PIPELINE_LOCK.json'
    fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL)
    with os.fdopen(fd,'w') as f:json.dump({'pid':os.getpid(),'started':timestamp()},f)
    status={'pid':os.getpid(),'started':timestamp(),'state':'starting','completed':0,'total':len(cfg['variants'])*len(cfg['seeds'])}
    try:
        assert cfg['splits']==['train','validation']
        assert cfg['device']!='cuda' or torch.cuda.is_available()
        if cfg['device']=='cuda':torch.cuda.set_per_process_memory_fraction(cfg['memory_fraction'])
        group=root/cfg['stage']/canonical_hash(cfg)[:12]
        if group.exists():raise RuntimeError(f'group already exists, retained: {group}')
        group.mkdir(parents=True);frozen=source_hashes()
        status.update(state='running',group=str(group),config_hash=canonical_hash(cfg))
        write_json(root/'ACTIVE_PROCESS.json',status)
        write_json(group/'config.json',cfg)
        write_json(group/'environment.json',{'source_hashes':frozen,'torch':torch.__version__,
            'gpu':torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
            'nvidia_smi':subprocess.check_output(['nvidia-smi'],text=True,encoding='utf-8',errors='replace'),
            'timing_scope':'shared workstation diagnostic timing, not an isolated efficiency benchmark'})
        for path in frozen:
            target=group/'source_snapshot'/path;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(REPO/path,target)
        source_cfg=read_json(REPO/cfg['source_config']); results=[]
        for seed in cfg['seeds']:
            for variant in cfg['variants']:
                assert source_hashes()==frozen,'source changed during run'
                status.update(current=f'{variant}_s{seed}',updated=timestamp())
                write_json(root/'STATUS.json',status)
                result=run_one(cfg,source_cfg,variant,seed,group);results.append(result)
                status['completed']=len(results);write_json(root/'STATUS.json',status)
        assert source_hashes()==frozen
        write_json(group/'raw_summary.json',results)
        status.update(state='ready_for_review',finished=timestamp(),current=None)
        write_json(root/'STATUS.json',status)
        write_json(root/'ACTIVE_PROCESS.json',{**status,'active':False})
    except BaseException:
        status.update(state='failed',error=traceback.format_exc(),finished=timestamp())
        write_json(root/'STATUS.json',status)
        write_json(root/'ACTIVE_PROCESS.json',{**status,'active':False})
        raise
    finally:
        lock.unlink()


if __name__=='__main__':main()
