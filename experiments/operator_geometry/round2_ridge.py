"""Strong input-only classical reduced-rank ridge utility controls on CPU.

Hyperparameters and the optional replacement threshold use internal dev only.
The reduced-rank solution minimizes ridge-penalized squared error globally.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from ..autoregressive_gate.data import digest,write_json
from .round2_train import calibrate_threshold,action_metrics,prediction_metrics


def reduced_rank_solution(eigenvalues,eigenvectors,cross,ridge,rank):
    projected=eigenvectors.T@cross
    full=eigenvectors@(projected/(eigenvalues[:,None]+ridge))
    gram=cross.T@full
    _,v=np.linalg.eigh((gram+gram.T)/2)
    v=v[:,-rank:]
    return full@v@v.T


def load_small(path):
    cache=torch.load(path,map_location='cpu',weights_only=True)
    return {k:cache[k] for k in ['x','utility_cost','weights','window']}


def run_seed(source,out,seed,cfg):
    expected=json.loads((source/'result.json').read_text(encoding='utf8'))
    for name,sha in expected['files'].items():assert digest(source/name)==sha
    run=out/f's{seed}';run.mkdir(exist_ok=False);started=time.perf_counter()
    caches={s:load_small(source/f'{s}.pt') for s in ['train','dev','validation']}
    exacts={s:dict(np.load(source/f'{s}_exact.npz')) for s in ['dev','validation']}
    tx=caches['train']['x'].numpy().astype(np.float64)
    mean=tx.mean(0);std=tx.std(0);std=np.maximum(std,1e-5)
    tx-=mean;tx/=std
    y=caches['train']['utility_cost'].numpy().astype(np.float64);ym=y.mean(0);y-=ym
    gram=tx.T@tx/len(tx);cross=tx.T@y/len(tx)
    eigenvalues,eigenvectors=np.linalg.eigh((gram+gram.T)/2)
    eigenvalues=np.maximum(eigenvalues,0)
    dx=(caches['dev']['x'].numpy()-mean)/std
    dy=caches['dev']['utility_cost'].numpy()
    records=[]
    for rank in cfg['ranks']:
        candidates=[];best=None
        for ridge in cfg['ridge_grid']:
            w=reduced_rank_solution(eigenvalues,eigenvectors,cross,ridge,rank)
            mse=float(np.mean((dx@w+ym-dy)**2))
            candidates.append({'rank':rank,'ridge':ridge,'internal_dev_mse':mse})
            if best is None or mse<best[0]:best=(mse,ridge,w)
        _,ridge,w=best
        dev=dx@w+ym
        threshold,cal=calibrate_threshold(dev,caches['dev'],exacts['dev'],cfg['thresholds'])
        row={'seed':seed,'method':f'ridge_rank{rank}','rank':rank,'ridge':ridge,
             'rank_factor_parameter_count':rank*(tx.shape[1]+y.shape[1])+y.shape[1],
             'stored_dense_weight_values':int(w.size),'normalization_statistics':int(2*len(mean)),
             'hyperparameter_candidates':candidates,'threshold':threshold,'threshold_calibration_dev':cal,
             'train_mse':float(np.mean((tx@w-y)**2)),'splits':{}}
        for split in ['dev','validation']:
            cost=dev if split=='dev' else ((caches[split]['x'].numpy()-mean)/std)@w+ym
            action,delta,chosen,changed=action_metrics(cost,caches[split],exacts[split],threshold,seed+42)
            row['splits'][split]={'prediction':prediction_metrics(cost,caches[split],ym),'action':action}
            np.savez_compressed(run/f'rank{rank}_{split}.npz',cost=cost,action_delta_nll=delta,chosen_index=chosen,changed=changed)
        # Store both factors for efficient deployment and the dense audit form.
        u,s,vh=np.linalg.svd(w,full_matrices=False)
        np.savez_compressed(run/f'rank{rank}_model.npz',weight=w,left=u[:,:rank]*s[:rank],right=vh[:rank],
                            input_mean=mean,input_std=std,target_mean=ym)
        records.append(row)
        print('RIDGE_DONE',seed,rank,'dev R2',row['splits']['dev']['prediction']['r2_vs_train_mean'],flush=True)
    result={'seed':seed,'results':records,'seconds_cpu':time.perf_counter()-started,
            'source_checkpoint_sha256':expected['checkpoint_sha256'],
            'files':{p.name:digest(p) for p in run.iterdir() if p.is_file()}}
    write_json(run/'result.json',result)
    return records


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args()
    torch.set_num_threads(1)
    repo=Path(__file__).resolve().parents[2];cfg=json.loads((repo/args.config).read_text(encoding='utf8'))
    root=repo/'outputs/operator_geometry_round2';source=repo/cfg['data_group']
    hash_=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    out=root/'ridge'/hash_[:12];out.mkdir(parents=True,exist_ok=False)
    write_json(out/'config.json',cfg);write_json(out/'source_hashes.json',{str(Path(__file__).relative_to(repo)):digest(Path(__file__))})
    status={'pid':os.getpid(),'state':'waiting_for_seed','group':str(out),'completed':0,'total':len(cfg['seeds'])}
    write_json(root/'RIDGE_STATUS.json',status)
    results=[]
    try:
        for seed in cfg['seeds']:
            while not (source/f's{seed}/result.json').exists():
                if datetime.now().astimezone()>=datetime.fromisoformat(cfg['deadline']):raise TimeoutError('Data not ready before control deadline')
                time.sleep(20)
            status.update(state='running',current_seed=seed);write_json(root/'RIDGE_STATUS.json',status)
            results.extend(run_seed(source/f's{seed}',out,seed,cfg))
            status['completed']+=1;write_json(root/'RIDGE_STATUS.json',status)
        write_json(out/'raw_summary.json',results)
        status.update(state='ready_for_review',current_seed=None,finished=datetime.now().astimezone().isoformat())
        write_json(root/'RIDGE_STATUS.json',status)
    except BaseException:
        status.update(state='failed',error=traceback.format_exc());write_json(root/'RIDGE_STATUS.json',status);raise


if __name__=='__main__':main()
