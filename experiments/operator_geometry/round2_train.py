"""Input-only utility learning with train-only development and frozen experts.

All architectures receive identical signed first-order supervision. Exact,
equal-budget replacement costs assess whether prediction yields useful actions.
The realized label is teacher data and is unavailable to the deployed predictor.
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
from torch import nn
from ..autoregressive_gate.data import digest,write_json
from ..routing_stability.runner import setup,read_json,canonical_hash,timestamp,append_json
from ..routing_stability.model import full_state_hash
from .runner import REPO
from .routers import ControlledRouter,optimize_assignment,spectral_codes
from .metrics import bootstrap_window_mean,spearman


class UtilityPredictor(nn.Module):
    def __init__(self,dim,rank,method,assignment,codes,random_perm):
        super().__init__()
        self.router=ControlledRouter(dim,16,rank,method,assignment,codes,random_perm)
        self.log_scale=nn.Parameter(torch.zeros(()))

    def forward(self,x):
        cost=-self.router(x).clamp_min(1e-12).log()
        return (cost-cost.mean(-1,keepdim=True))*self.log_scale.clamp(-3,3).exp()


def action_arrays(cost,cache,exact):
    """Select one replacement or retain the original budget-four route."""
    source,target=exact['source'],exact['target'];n=len(source)
    rows=np.arange(n)[:,None];w=cache['weights'][:n].cpu().numpy()
    predicted=w[rows,source]*(cost[:n][rows,target]-cost[:n][rows,source])
    chosen=predicted.argmin(1)
    return predicted[np.arange(n),chosen],exact['delta_nll'][np.arange(n),chosen],chosen


def calibrate_threshold(cost,cache,exact,thresholds):
    predicted,actual,_=action_arrays(cost,cache,exact);choices=[]
    for threshold in thresholds:
        limit=np.inf if threshold is None else threshold
        selected=predicted < -limit
        delta=np.where(selected,actual,0.)
        choices.append({'threshold':threshold,'delta_ce':float(delta.mean()),'changed':int(selected.sum())})
    # On exact ties prefer fewer interventions; threshold is chosen only on dev.
    best=min(choices,key=lambda r:(r['delta_ce'],r['changed']))
    return best['threshold'],choices


def action_metrics(cost,cache,exact,threshold,seed):
    predicted,actual,chosen=action_arrays(cost,cache,exact)
    selected=predicted < -(np.inf if threshold is None else threshold)
    delta=np.where(selected,actual,0.)
    oracle=np.minimum(exact['delta_nll'].min(1),0.)
    corr=spearman(predicted,actual)
    return {'tokens':len(delta),'threshold':threshold,'changed':int(selected.sum()),
            'mean_delta_ce':float(delta.mean()),'delta_ce_ci95_windows':bootstrap_window_mean(delta,exact['window'],seed,repeats=1000),
            'regret_to_label_oracle':float((delta-oracle).mean()),'label_oracle_mean_delta_ce':float(oracle.mean()),
            'harm_gt_01':int((delta>.1).sum()),'benefit_lt_minus_01':int((delta<-.1).sum()),
            'predicted_vs_actual_selected_candidate_rho':corr},delta,chosen,selected


@torch.no_grad()
def predict(model,cache,scale,device):
    model.eval();return torch.cat([model(cache['x'][i:i+512].to(device)).cpu()*scale
                                   for i in range(0,len(cache['x']),512)]).numpy()


def prediction_metrics(cost,cache,constant):
    target=cache['utility_cost'].cpu().numpy();mse=float(np.square(cost-target).mean())
    base=float(np.square(constant-target).mean())
    rs=[spearman(a,b) for a,b in zip(cost,target)];rs=[r for r in rs if r is not None]
    return {'mse':mse,'constant_mse':base,'r2_vs_train_mean':1-mse/base,
            'within_token_expert_rho':float(np.mean(rs)) if rs else None}


def train_one(cfg,group,seed,method,caches,exacts,geometry):
    run=group/'runs'/f'{method}_s{seed}';run.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(seed+960000)
    model=UtilityPredictor(caches['train']['x'].shape[-1],cfg['rank'],method,*geometry).to(cfg['device'])
    initial=full_state_hash(dict(model.named_parameters()));scale=float(caches['train']['utility_cost'].std())
    constant=caches['train']['utility_cost'].mean(0).numpy();opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    gen=torch.Generator().manual_seed(seed+970000);best=float('inf');sampled=[];started=time.perf_counter()
    manifest={'config_hash':canonical_hash(cfg),'seed':seed,'method':method,'initial_parameter_hash':initial,
              'parameters':sum(p.numel() for p in model.parameters()),'target_scale':scale,
              'selection':'minimum internal-dev utility MSE among fixed evaluation steps; validation not accessed until selection is complete'}
    write_json(run/'manifest.json',manifest)
    x=caches['train']['x'].to(cfg['device']);y=caches['train']['utility_cost'].to(cfg['device'])/scale
    for step in range(1,cfg['steps']+1):
        model.train();idx=torch.randint(len(x),(cfg['batch_size'],),generator=gen).to(cfg['device']);sampled.extend(idx.cpu().tolist())
        opt.zero_grad(set_to_none=True);loss=(model(x[idx])-y[idx]).square().mean()
        if not torch.isfinite(loss):raise FloatingPointError('utility training loss')
        loss.backward();grads=[p.grad for p in model.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):raise FloatingPointError('utility gradients')
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'])
        if not torch.isfinite(norm) or norm<=0:raise FloatingPointError('utility gradient norm')
        opt.step()
        if step%cfg['eval_every']==0 or step==cfg['steps']:
            cost=predict(model,caches['dev'],scale,cfg['device']);metrics=prediction_metrics(cost,caches['dev'],constant)
            record={'step':step,'train_loss_normalized':float(loss.detach()),'gradient_norm':float(norm),
                    'seconds':time.perf_counter()-started,**metrics}
            append_json(run/'curves.jsonl',record)
            if metrics['mse']<best:
                best=metrics['mse']
                torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step,'config':cfg,
                      'method':method,'seed':seed,'target_scale':scale,'torch_rng':torch.get_rng_state(),
                      'cuda_rng':torch.cuda.get_rng_state_all() if cfg['device']=='cuda' else [],'sampling_rng':gen.get_state()},run/'best_dev.pt')
            print('UTILITY',method,seed,step,'dev R2',round(metrics['r2_vs_train_mean'],5),flush=True)
    torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':cfg['steps'],'config':cfg,
                'method':method,'seed':seed,'target_scale':scale,'torch_rng':torch.get_rng_state(),
                'cuda_rng':torch.cuda.get_rng_state_all() if cfg['device']=='cuda' else [],'sampling_rng':gen.get_state()},run/'last.pt')
    selected=torch.load(run/'best_dev.pt',map_location=cfg['device'],weights_only=False);model.load_state_dict(selected['model'],strict=True)
    dev_cost=predict(model,caches['dev'],scale,cfg['device'])
    threshold,calibration=calibrate_threshold(dev_cost,caches['dev'],exacts['dev'],cfg['thresholds'])
    result={'method':method,'seed':seed,'steps':cfg['steps'],'selected_step':selected['step'],
            'parameters':manifest['parameters'],'parameter_changed':full_state_hash(dict(model.named_parameters()))!=initial,
            'threshold':threshold,'threshold_calibration_dev':calibration,'splits':{},'seconds_shared_gpu':time.perf_counter()-started}
    for split in ['dev','validation']:
        cost=dev_cost if split=='dev' else predict(model,caches[split],scale,cfg['device'])
        prediction=prediction_metrics(cost,caches[split],constant)
        action,delta,chosen,changed=action_metrics(cost,caches[split],exacts[split],threshold,seed+42)
        np.savez_compressed(run/f'{split}_predictions.npz',cost=cost,action_delta_nll=delta,chosen_index=chosen,changed=changed)
        result['splits'][split]={'prediction':prediction,'action':action}
    np.save(run/'sample_indices.npy',np.asarray(sampled,dtype=np.int32))
    result['files']={p.name:digest(p) for p in run.iterdir() if p.is_file()}
    write_json(run/'result.json',result)
    del model,x,y,opt
    if cfg['device']=='cuda':torch.cuda.empty_cache()
    return result


def train_hashes():
    from .round2_data import data_hashes
    values=data_hashes()
    for name in ['round2_train.py','routers.py']:
        path=Path('experiments/operator_geometry')/name;values[str(path).replace('\\','/')]=digest(REPO/path)
    return values


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--root',default='outputs/operator_geometry_round2');args=p.parse_args()
    setup();cfg=read_json(args.config);root=REPO/args.root;root.mkdir(parents=True,exist_ok=True)
    data=REPO/cfg['data_group'];assert (data/'DATA_COMPLETE.json').exists(),'data collection must finish first'
    lock=root/'PIPELINE_LOCK.json';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w') as f:json.dump({'pid':os.getpid(),'stage':'utility','started':timestamp()},f)
    status={'state':'starting','pid':os.getpid(),'stage':'utility','completed':0,'total':len(cfg['seeds'])*len(cfg['methods']),
            'started':timestamp(),'config_hash':canonical_hash(cfg)}
    try:
        if cfg['device']=='cuda':torch.cuda.set_per_process_memory_fraction(cfg['memory_fraction'])
        group=root/cfg['stage']/canonical_hash(cfg)[:12];group.mkdir(parents=True,exist_ok=False)
        frozen=train_hashes();write_json(group/'config.json',cfg);write_json(group/'source_hashes.json',frozen)
        for path in frozen:
            target=group/'source_snapshot'/path;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/path,target)
        status.update(state='running',group=str(group),log='logs/utility_v1.log',err='logs/utility_v1.err')
        write_json(root/'STATUS.json',status);write_json(root/'ACTIVE_PROCESS.json',{**status,'active':True})
        results=[];baselines=[]
        for seed in cfg['seeds']:
            src=data/f's{seed}';record=read_json(src/'result.json')
            for name,sha in record['files'].items():assert digest(src/name)==sha,f'data checksum {name}'
            caches={s:torch.load(src/f'{s}.pt',map_location='cpu',weights_only=True) for s in ['train','dev','validation']}
            exacts={s:dict(np.load(src/f'{s}_exact.npz')) for s in ['dev','validation']}
            distance=np.load(src/'train_pair_distance.npy');assignment,g=optimize_assignment(distance,seed+920000)
            codes=spectral_codes(distance);random=np.random.default_rng(seed+930000).permutation(16)
            geometry=(assignment,codes,random)
            gd=group/'geometry'/f's{seed}';gd.mkdir(parents=True);write_json(gd/'assignment.json',{**g,'random_expert_for_basis':random.tolist()});np.save(gd/'codes.npy',codes)
            mean=caches['train']['utility_cost'].mean(0).numpy()
            for kind in ['constant_train_mean','true_first_order_label_oracle']:
                c={s:(np.broadcast_to(mean,(len(caches[s]['x']),16)) if kind.startswith('constant') else caches[s]['utility_cost'].numpy()) for s in ['dev','validation']}
                threshold,cal=calibrate_threshold(c['dev'],caches['dev'],exacts['dev'],cfg['thresholds'])
                for split in c:
                    metrics,*_=action_metrics(c[split],caches[split],exacts[split],threshold,seed+42)
                    baselines.append({'seed':seed,'kind':kind,'split':split,'action':metrics,'scope':'true_first_order_label_oracle uses realized labels and is diagnostic only'})
            for method in cfg['methods']:
                assert train_hashes()==frozen
                status.update(current=f'{method}_s{seed}',updated=timestamp());write_json(root/'STATUS.json',status)
                results.append(train_one(cfg,group,seed,method,caches,exacts,geometry))
                status['completed']=len(results);write_json(root/'STATUS.json',status)
            del caches,exacts
        assert train_hashes()==frozen
        write_json(group/'raw_summary.json',results);write_json(group/'baselines.json',baselines)
        status.update(state='utility_ready_for_review',finished=timestamp(),current=None)
        write_json(root/'STATUS.json',status);write_json(root/'ACTIVE_PROCESS.json',{**status,'active':False})
    except BaseException:
        status.update(state='failed',error=traceback.format_exc(),finished=timestamp());write_json(root/'STATUS.json',status)
        write_json(root/'ACTIVE_PROCESS.json',{**status,'active':False});raise
    finally:lock.unlink()


if __name__=='__main__':main()
