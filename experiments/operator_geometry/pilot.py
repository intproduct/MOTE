"""Real GPU router-only training over a frozen, exact final-layer expert bank."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F
from ..autoregressive_gate.data import write_json,digest
from ..routing_stability.runner import setup,read_json,canonical_hash,timestamp,append_json
from ..routing_stability.model import CausalLM,full_state_hash
from .runner import REPO,collect,train_pair_geometry,source_hashes
from .metrics import head
from .routers import ControlledRouter,optimize_assignment,spectral_codes,sparse_weights


def pilot_hashes():
    result=source_hashes()
    for path in ['experiments/operator_geometry/pilot.py','experiments/operator_geometry/routers.py']:
        result[path]=digest(REPO/path)
    return result


def soft_teacher(cache,gate):
    return gate._softmax_temp(cache['scores'])


def local_logits(model,router,cache,idx,ste=False):
    p=router(cache['x'][idx]);w=sparse_weights(p,4,ste)
    delta=((w-cache['weights'][idx])[:,:,None]*cache['outputs'][idx]).sum(1)
    return head(model,cache['hidden'][idx]+delta),p


@torch.no_grad()
def evaluate(model,router,cache):
    router.eval();losses=[];kl=[];load=torch.zeros(16,device=cache['x'].device)
    for begin in range(0,len(cache['x']),128):
        idx=torch.arange(begin,min(begin+128,len(cache['x'])),device=cache['x'].device)
        logits,p=local_logits(model,router,cache,idx)
        losses.append(F.cross_entropy(logits,cache['labels'][idx],reduction='none').cpu())
        teacher=cache['teacher'][idx]
        kl.append((teacher*(teacher.clamp_min(1e-12).log()-p.clamp_min(1e-12).log())).sum(-1).cpu())
        load+=torch.bincount(p.topk(4,-1).indices.flatten(),minlength=16)
    values=torch.cat(losses).numpy();loads=load.cpu().numpy()/len(values)
    return {'ce':float(values.mean()),'ppl':float(np.exp(values.mean())),
            'teacher_kl':float(torch.cat(kl).mean()),'load':loads.tolist(),
            'load_cv':float(loads.std()/loads.mean()),'unused_experts':int((loads==0).sum())},values


@torch.no_grad()
def rounding(model,router,cache):
    router.eval();n=len(cache['x']);counts={'flips':0,'harm':0,'abs_gt_01':0};deltas=[]
    for begin in range(0,n,128):
        ix=torch.arange(begin,min(begin+128,n),device=cache['x'].device)
        ref,p=local_logits(model,router,cache,ix)
        q=router(cache['x'][ix].bfloat16().float());w=sparse_weights(q,4)
        change=((w-cache['weights'][ix])[:,:,None]*cache['outputs'][ix]).sum(1)
        out=head(model,cache['hidden'][ix]+change)
        a=F.cross_entropy(ref,cache['labels'][ix],reduction='none')
        b=F.cross_entropy(out,cache['labels'][ix],reduction='none');delta=b-a
        counts['flips']+=int((torch.sort(p.topk(4,-1).indices,-1).values!=torch.sort(q.topk(4,-1).indices,-1).values).any(-1).sum())
        counts['harm']+=int((delta>.1).sum());counts['abs_gt_01']+=int((delta.abs()>.1).sum())
        deltas.append(delta.cpu())
    delta=torch.cat(deltas)
    return {'tokens':n,**counts,'delta_ce':float(delta.mean()),'abs_delta_p99':float(torch.quantile(delta.abs(),.99))}


def train_router(cfg,model,train,val,assignment,codes,random_perm,method,seed,group):
    run=group/'runs'/f'{method}_s{seed}'
    if run.exists():raise RuntimeError(f'refusing overwrite {run}')
    run.mkdir(parents=True)
    torch.manual_seed(seed+900000)
    router=ControlledRouter(train['x'].shape[-1],16,cfg['rank'],method,assignment,codes,random_perm).to(cfg['device'])
    optimizer=torch.optim.AdamW(router.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    generator=torch.Generator().manual_seed(seed+910000)
    initial=full_state_hash(router.state_dict()); t0=time.perf_counter()
    write_json(run/'manifest.json',{'config_hash':canonical_hash(cfg),'method':method,'seed':seed,
        'initial_router_hash':initial,'router_parameters':sum(p.numel() for p in router.parameters()),
        'expert_pool_sha256':digest(REPO/cfg['source_group']/'runs'/f'mlp_half_s{seed}'/'best.pt'),
        'initialization_seed':seed+900000,'sampling_seed':seed+910000})
    first,_=evaluate(model,router,val);append_json(run/'curves.jsonl',{'step':0,**first})
    samples=[];finite=0
    for step in range(1,cfg['steps']+1):
        router.train();idx=torch.randint(len(train['x']),(cfg['batch_size'],),generator=generator).to(cfg['device'])
        samples.extend(idx.cpu().tolist());optimizer.zero_grad(set_to_none=True)
        p=router(train['x'][idx]);teacher=train['teacher'][idx]
        kl=(teacher*(teacher.clamp_min(1e-12).log()-p.clamp_min(1e-12).log())).sum(-1).mean()
        if step<=cfg['imitation_steps']:
            loss=kl; ce=kl.new_zeros(())
        else:
            w=sparse_weights(p,4,True)
            delta=((w-train['weights'][idx])[:,:,None]*train['outputs'][idx]).sum(1)
            logits=head(model,train['hidden'][idx]+delta)
            ce=F.cross_entropy(logits,train['labels'][idx]);loss=ce+cfg['anchor_kl_weight']*kl
        if not torch.isfinite(loss):raise FloatingPointError('training loss')
        loss.backward()
        grads=[p.grad for p in router.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):raise FloatingPointError('router gradients')
        grad=torch.nn.utils.clip_grad_norm_(router.parameters(),cfg['gradient_clip'])
        if not float(grad)>0:raise RuntimeError('zero router gradient')
        optimizer.step();finite+=1
        if step%cfg['eval_every']==0 or step==cfg['steps']:
            metrics,_=evaluate(model,router,val)
            append_json(run/'curves.jsonl',{'step':step,'train_loss':float(loss.detach()),'train_ce':float(ce.detach()),
                'teacher_kl_train':float(kl.detach()),'gradient_norm':float(grad),'seconds':time.perf_counter()-t0,**metrics})
            print('TRAIN',method,seed,step,'validation CE',round(metrics['ce'],5),flush=True)
    metrics,values=evaluate(model,router,val);train_metrics,_=evaluate(model,router,train)
    np.save(run/'validation_nll.npy',values)
    np.save(run/'sample_indices.npy',np.asarray(samples,dtype=np.int32))
    # Includes optimizer and random states; no resume implementation is claimed.
    torch.save({'model':router.state_dict(),'optimizer':optimizer.state_dict(),'config':cfg,'method':method,
        'seed':seed,'step':cfg['steps'],'torch_rng':torch.get_rng_state(),'sampling_rng':generator.get_state(),
        'cuda_rng':torch.cuda.get_rng_state_all()},run/'final.pt')
    result={'method':method,'seed':seed,'validation':metrics,'train':train_metrics,
        'rounding':rounding(model,router,val),'finite_steps':finite,'router_changed':full_state_hash(router.state_dict())!=initial,
        'router_parameters':sum(p.numel() for p in router.parameters()),'seconds':time.perf_counter()-t0,
        'files':{p.name:digest(p) for p in run.iterdir() if p.is_file()}}
    write_json(run/'result.json',result)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--root',default='outputs/operator_geometry_round1');args=parser.parse_args()
    setup();cfg=read_json(args.config);root=(REPO/args.root).resolve();root.mkdir(parents=True,exist_ok=True)
    lock=root/'PIPELINE_LOCK.json';fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL)
    with os.fdopen(fd,'w') as f:json.dump({'pid':os.getpid(),'started':timestamp(),'stage':'router_pilot'},f)
    status={'pid':os.getpid(),'state':'starting','completed':0,'total':len(cfg['methods'])*len(cfg['seeds']),'started':timestamp()}
    try:
        eligibility=read_json(REPO/cfg['diagnostic_group']/'DIAGNOSTIC_AUDIT.json')['pilot_eligibility']['mlp_half']
        assert eligibility['signal_sufficient_for_small_pilot']
        if cfg['device']=='cuda':torch.cuda.set_per_process_memory_fraction(cfg['memory_fraction'])
        group=root/cfg['stage']/canonical_hash(cfg)[:12]
        if group.exists():raise RuntimeError(f'group exists, retained: {group}')
        group.mkdir(parents=True);frozen=pilot_hashes();write_json(group/'config.json',cfg)
        write_json(group/'source_hashes.json',frozen)
        import shutil,subprocess
        for path in frozen:
            target=group/'source_snapshot'/path;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/path,target)
        write_json(group/'environment.json',{'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
             'nvidia_smi':subprocess.check_output(['nvidia-smi'],text=True,encoding='utf-8',errors='replace'),
             'scope':'Shared workstation wall time; not an isolated latency comparison'})
        status.update(state='running',group=str(group),config_hash=canonical_hash(cfg))
        write_json(root/'ACTIVE_PROCESS.json',{**status,'active':True});write_json(root/'STATUS.json',status)
        source_cfg=read_json(REPO/cfg['source_config']);results=[]
        for seed in cfg['seeds']:
            checkpoint=REPO/cfg['source_group']/'runs'/f'mlp_half_s{seed}'/'best.pt'
            expected=read_json(checkpoint.parent/'result.json')['checkpoint_sha256'];assert digest(checkpoint)==expected
            ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
            model=CausalLM(source_cfg,'mlp_half',seed).to(cfg['device']).eval();model.load_state_dict(ckpt['model'],strict=True)
            model.requires_grad_(False);before=full_state_hash(model.state_dict())
            cache={};checks={};data_dir=group/'data'/f's{seed}';data_dir.mkdir(parents=True)
            for split,windows in [('train',cfg['train_windows']),('validation',cfg['validation_windows'])]:
                cache[split],checks[split]=collect(model,np.load(REPO/cfg['data_root']/f'{split}.npy'),
                    {**cfg,'windows_per_split':windows},split)
                cache[split]['teacher']=soft_teacher(cache[split],model.get_submodule(cfg['module']).mote.core.gate)
                torch.save(cache[split],data_dir/f'{split}.pt')
                print('CACHE',seed,split,len(cache[split]['x']),flush=True)
            pair=train_pair_geometry(cache['train']);assignment,graph=optimize_assignment(pair,seed+920000)
            codes=spectral_codes(pair);random_perm=np.random.default_rng(seed+930000).permutation(16)
            np.save(data_dir/'train_pair_distance.npy',pair);np.save(data_dir/'codes.npy',codes)
            write_json(data_dir/'geometry.json',{**graph,'random_expert_for_basis':random_perm.tolist(),'identity_checks':checks,
                'checkpoint_sha256':expected,'frozen_state_hash':before})
            for split in cache:
                cache[split]={k:v.to(cfg['device']) for k,v in cache[split].items()}
            with torch.no_grad():
                base=head(model,cache['validation']['hidden'])
                baseline=F.cross_entropy(base,cache['validation']['labels'],reduction='none').cpu().numpy()
                np.save(data_dir/'baseline_validation_nll.npy',baseline)
            for method in cfg['methods']:
                assert pilot_hashes()==frozen
                status.update(current=f'{method}_s{seed}',updated=timestamp());write_json(root/'STATUS.json',status)
                results.append(train_router(cfg,model,cache['train'],cache['validation'],assignment,codes,random_perm,method,seed,group))
                assert full_state_hash(model.state_dict())==before,'frozen backbone changed'
                status['completed']=len(results);write_json(root/'STATUS.json',status)
            assert digest(checkpoint)==expected
            del model,cache;torch.cuda.empty_cache()
        assert pilot_hashes()==frozen
        write_json(group/'raw_summary.json',results)
        status.update(state='ready_for_review',finished=timestamp(),current=None)
        write_json(root/'STATUS.json',status);write_json(root/'ACTIVE_PROCESS.json',{**status,'active':False})
    except BaseException:
        status.update(state='failed',error=traceback.format_exc(),finished=timestamp())
        write_json(root/'STATUS.json',status);write_json(root/'ACTIVE_PROCESS.json',{**status,'active':False});raise
    finally:lock.unlink()


if __name__=='__main__':main()
