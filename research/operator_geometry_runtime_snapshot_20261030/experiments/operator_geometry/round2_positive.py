"""Input-determined teacher calibration after utility and risk experiments.

This is an optimizer/representation positive control, not LM quality training.
It learns the frozen original router's scores using the same pre-expert inputs.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
import torch
from ..autoregressive_gate.data import digest,write_json
from ..routing_stability.runner import setup,canonical_hash,timestamp,append_json
from ..routing_stability.model import full_state_hash
from .round2_train import UtilityPredictor,predict,train_hashes
from .round2_numeric import row_spearman


def read(path):return json.loads(path.read_text(encoding='utf8'))


def teacher_cost(probability):
    value=-probability.clamp_min(1e-12).log()
    return value-value.mean(-1,keepdim=True)


def measures(pred,target,constant):
    mse=float(np.square(pred-target).mean());base=float(np.square(target-constant).mean())
    ps=np.sort(np.argsort(pred,axis=1)[:,:4],axis=1);ts=np.sort(np.argsort(target,axis=1)[:,:4],axis=1)
    return {'mse':mse,'constant_mse':base,'r2_vs_constant':1-mse/base,
            'within_token_rho':float(np.nanmean(row_spearman(pred,target))),
            'top4_set_agreement':float((ps==ts).all(1).mean())}


def train_one(cfg,group,seed,method,caches,geometry):
    run=group/'runs'/f'{method}_s{seed}';run.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(seed+960000)
    model=UtilityPredictor(caches['train']['x'].shape[-1],cfg['rank'],method,*geometry).to(cfg['device'])
    initial=full_state_hash(dict(model.named_parameters()));scale=float(caches['train']['target'].std())
    constant=caches['train']['target'].mean(0).numpy()
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    generator=torch.Generator().manual_seed(seed+970000);sampled=[];best=float('inf');started=time.perf_counter()
    x=caches['train']['x'].to(cfg['device']);y=caches['train']['target'].to(cfg['device'])/scale
    for step in range(1,cfg['steps']+1):
        if step%cfg['eval_every']==1 and datetime.now().astimezone()>=datetime.fromisoformat(cfg['stop_training_by']):
            torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step-1,'sampling_rng':generator.get_state()},run/'deadline_partial.pt')
            raise TimeoutError('Positive control deadline reached; partial checkpoint preserved')
        model.train();idx=torch.randint(len(x),(cfg['batch_size'],),generator=generator).to(cfg['device']);sampled.extend(idx.cpu().tolist())
        opt.zero_grad(set_to_none=True);loss=(model(x[idx])-y[idx]).square().mean()
        if not torch.isfinite(loss):raise FloatingPointError('positive loss')
        loss.backward();grads=[p.grad for p in model.parameters() if p.grad is not None]
        if not all(torch.isfinite(g).all() for g in grads):raise FloatingPointError('positive gradients')
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'])
        if not torch.isfinite(norm) or norm<=0:raise FloatingPointError('positive gradient norm')
        opt.step()
        if step%cfg['eval_every']==0 or step==cfg['steps']:
            prediction=predict(model,caches['dev'],scale,cfg['device']);metrics=measures(prediction,caches['dev']['target'].numpy(),constant)
            append_json(run/'curves.jsonl',{'step':step,'train_loss_normalized':float(loss.detach()),'gradient_norm':float(norm),**metrics})
            if metrics['mse']<best:
                best=metrics['mse'];torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step,
                    'target_scale':scale,'seed':seed,'method':method,'config':cfg,'sampling_rng':generator.get_state(),
                    'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if cfg['device']=='cuda' else []},run/'best_dev.pt')
            print('POSITIVE',seed,method,step,'dev R2',round(metrics['r2_vs_constant'],4),flush=True)
    torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':cfg['steps'],'target_scale':scale,
        'seed':seed,'method':method,'config':cfg,'sampling_rng':generator.get_state(),'torch_rng':torch.get_rng_state(),
        'cuda_rng':torch.cuda.get_rng_state_all() if cfg['device']=='cuda' else []},run/'last.pt')
    ck=torch.load(run/'best_dev.pt',map_location=cfg['device'],weights_only=False);model.load_state_dict(ck['model'],strict=True)
    result={'seed':seed,'method':method,'parameters':sum(p.numel() for p in model.parameters()),'selected_step':ck['step'],
            'initial_parameter_hash':initial,'parameter_changed':initial!=full_state_hash(dict(model.named_parameters())),
            'seconds_shared_gpu':time.perf_counter()-started,'target_scale':scale,'splits':{},'positive_control_only':True}
    for split,cache in caches.items():
        prediction=predict(model,cache,scale,cfg['device']);result['splits'][split]=measures(prediction,cache['target'].numpy(),constant)
        np.savez_compressed(run/f'{split}_predictions.npz',cost=prediction)
    np.save(run/'sample_indices.npy',np.asarray(sampled,dtype=np.int32))
    result['files']={p.name:digest(p) for p in run.iterdir() if p.is_file()};write_json(run/'result.json',result)
    del model,opt,x,y
    if cfg['device']=='cuda':torch.cuda.empty_cache()
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--root',default='outputs/operator_geometry_round2');args=p.parse_args()
    repo=Path(__file__).resolve().parents[2];cfg=read(repo/args.config);root=repo/args.root;root.mkdir(parents=True,exist_ok=True)
    data=repo/cfg['data_group'];utility=repo/cfg['utility_group'];risk=repo/cfg['risk_group']
    group=root/'positive'/canonical_hash(cfg)[:12];group.mkdir(parents=True,exist_ok=False)
    sources=train_hashes();sources.update({f'experiments/operator_geometry/{name}':digest(Path(__file__).with_name(name)) for name in ['round2_positive.py','round2_numeric.py']})
    write_json(group/'config.json',cfg);write_json(group/'source_hashes.json',sources)
    for relative in sources:
        dest=group/'source_snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/relative,dest)
    status={'state':'waiting_for_risk','pid':os.getpid(),'stage':'positive','group':str(group),'completed':0,
            'total':len(cfg['seeds'])*len(cfg['methods']),'started':timestamp(),'log':'logs/positive_v1.log','err':'logs/positive_v1.err'}
    write_json(root/'POSITIVE_QUEUE_STATUS.json',status);lock=root/'PIPELINE_LOCK.json';acquired=False
    try:
        if cfg['wait_for_risk']:
            while not ((risk/'COMPLETE.json').exists() and not lock.exists()):
                if read(root/'RISK_QUEUE_STATUS.json')['state']=='failed':raise RuntimeError('Risk stage failed; positive control will not start')
                if datetime.now().astimezone()>=datetime.fromisoformat(cfg['latest_start']):raise TimeoutError('Positive control start deadline')
                time.sleep(20)
        with os.fdopen(os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY),'w') as f:json.dump({'pid':os.getpid(),'stage':'positive','started':timestamp()},f)
        acquired=True;setup()
        if cfg['device']=='cuda':torch.cuda.set_per_process_memory_fraction(.35)
        if (root/'STATUS.json').exists():write_json(group/'PREVIOUS_STATUS.json',read(root/'STATUS.json'))
        status.update(state='positive_running',running_started=timestamp());write_json(root/'STATUS.json',status)
        write_json(root/'ACTIVE_PROCESS.json',{**status,'active':True});write_json(root/'POSITIVE_QUEUE_STATUS.json',status)
        rows=[]
        for seed in cfg['seeds']:
            manifest=read(data/f's{seed}/result.json')
            for name,sha in manifest['files'].items():assert digest(data/f's{seed}'/name)==sha
            caches={}
            for split in ['train','dev','validation']:
                cached=torch.load(data/f's{seed}/{split}.pt',map_location='cpu',weights_only=True)
                caches[split]={'x':cached['x'],'target':teacher_cost(cached['teacher'])};del cached
            geo=read(utility/f'geometry/s{seed}/assignment.json');codes=np.load(utility/f'geometry/s{seed}/codes.npy')
            geometry=(np.asarray(geo['expert_for_basis']),codes,np.asarray(geo['random_expert_for_basis']))
            for method in cfg['methods']:
                assert all(digest(repo/path)==sha for path,sha in sources.items())
                status.update(current=f'{method}_s{seed}');write_json(root/'STATUS.json',status)
                row=train_one(cfg,group,seed,method,caches,geometry)
                original=utility/f'runs/{method}_s{seed}'
                if cfg['steps']==read(utility/'config.json')['steps']:
                    assert digest(group/f'runs/{method}_s{seed}/sample_indices.npy')==digest(original/'sample_indices.npy')
                row['same_initial_parameters_as_utility']=row['initial_parameter_hash']==read(original/'manifest.json')['initial_parameter_hash']
                assert row['same_initial_parameters_as_utility']
                rows.append(row);status['completed']=len(rows);write_json(root/'STATUS.json',status)
        write_json(group/'raw_summary.json',rows);write_json(group/'COMPLETE.json',{'models':len(rows),'finished':timestamp(),'positive_control_only':True})
        status.update(state='positive_ready_for_review',current=None,finished=timestamp());write_json(root/'STATUS.json',status)
        write_json(root/'ACTIVE_PROCESS.json',{**status,'active':False});write_json(root/'POSITIVE_QUEUE_STATUS.json',status)
    except BaseException:
        status.update(state='failed',error=traceback.format_exc(),finished=timestamp());write_json(root/'POSITIVE_QUEUE_STATUS.json',status)
        if acquired:write_json(root/'STATUS.json',status);write_json(root/'ACTIVE_PROCESS.json',{**status,'active':False})
        raise
    finally:
        if acquired:lock.unlink()


if __name__=='__main__':main()
