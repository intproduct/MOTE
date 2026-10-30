"""Label-free input-conditional operator geometry and quantum-ray diagnostics.

The targets are shared random sketches of expert-output contrasts. A small
classical linear factor predicts these from pre-expert input. Quantum fidelity
on the same features is assessed as a geometry, not a deployed quantum gate.
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
from .round2_ridge import reduced_rank_solution
from .metrics import spearman
from .round2_numeric import row_spearman


def pair_scores(features,source,target,anchor=None):
    n=np.arange(len(features))[:,None];a=features[n,source];b=features[n,target]
    if anchor is None:return np.square(a-b).sum(-1)
    dot=(a*b).sum(-1)+anchor**2
    aa=np.square(a).sum(-1)+anchor**2;bb=np.square(b).sum(-1)+anchor**2
    return np.clip(1-dot**2/np.maximum(aa*bb,1e-30),0,1)


def conditional_correlations(proxy,target):
    return float(np.nanmean(row_spearman(proxy.reshape(-1,12),target.reshape(-1,12))))


def geometry_metrics(features,outputs,exact,anchor,global_pair):
    source,target=exact['source'],exact['target'];idx=np.arange(len(features))[:,None]
    actual=np.square(outputs[idx,source]-outputs[idx,target]).sum(-1)
    estimates={'predicted_euclidean':pair_scores(features,source,target),
               'unanchored_ray':pair_scores(features,source,target,0.),
               'anchored_ray':pair_scores(features,source,target,anchor),
               'anchored_ray_x4':pair_scores(features,source,target,4*anchor),
               'train_global_euclidean':global_pair[source,target]}
    rows={}
    for name,values in estimates.items():
        rows[name]={'within_token_fixed_source_rho_vs_true_output_distance':conditional_correlations(values,actual),
                    'within_token_fixed_source_rho_vs_abs_nll':conditional_correlations(values,abs(exact['delta_nll'])),
                    'nearest_target_agreement':float((values.reshape(-1,12).argmin(1)==actual.reshape(-1,12).argmin(1)).mean())}
    rows['predicted_euclidean']['relative_mse']=float(np.mean((estimates['predicted_euclidean']-actual)**2)/np.mean(actual**2))
    return rows,estimates,actual


def load_split(path,sketch):
    cache=torch.load(path,map_location='cpu',weights_only=True)
    outputs=cache['outputs'].numpy()
    contrasts=outputs-outputs.mean(1,keepdims=True)
    projected=(contrasts@sketch).reshape(len(outputs),-1).astype(np.float64)
    return cache['x'].numpy().astype(np.float64),projected,outputs


def run_seed(src,out,seed,cfg):
    manifest=json.loads((src/'result.json').read_text(encoding='utf8'))
    for name,sha in manifest['files'].items():assert digest(src/name)==sha
    run=out/f's{seed}';run.mkdir(exist_ok=False);started=time.perf_counter()
    peek=torch.load(src/'dev.pt',map_location='cpu',weights_only=True);h=peek['outputs'].shape[-1];del peek
    rng=np.random.default_rng(cfg['sketch_seed'])
    sketch=(rng.normal(size=(h,cfg['sketch_dim']))/np.sqrt(cfg['sketch_dim'])).astype(np.float32)
    tx,ty,unused=load_split(src/'train.pt',sketch);del unused
    mean=tx.mean(0);std=np.maximum(tx.std(0),1e-5);tx-=mean;tx/=std
    ym=ty.mean(0);ty-=ym
    gram=tx.T@tx/len(tx);cross=tx.T@ty/len(tx);ev,vec=np.linalg.eigh((gram+gram.T)/2);ev=np.maximum(ev,0)
    dx,dy,doutputs=load_split(src/'dev.pt',sketch);dx-=mean;dx/=std
    dev_exact=dict(np.load(src/'dev_exact.npz'))
    distance=np.load(src/'train_pair_distance.npy')
    # Fixed anchor statistic derives only from training sketches, before model selection.
    anchor=float(np.quantile(np.linalg.norm((ty+ym).reshape(len(ty),16,-1),axis=-1),.95))
    rows=[];selected=[]
    for rank in cfg['ranks']:
        best=None;candidates=[]
        for ridge in cfg['ridge_grid']:
            w=reduced_rank_solution(ev,vec,cross,ridge,rank);mse=float(np.square(dx@w+ym-dy).mean())
            candidates.append({'ridge':ridge,'internal_dev_sketch_mse':mse})
            if best is None or mse<best[0]:best=(mse,ridge,w)
        mse,ridge,w=best;selected.append((rank,ridge,w,candidates))
    # Validation is loaded after all choices have been fixed on internal dev.
    vx,vy,voutputs=load_split(src/'validation.pt',sketch);vx-=mean;vx/=std
    vex=dict(np.load(src/'validation_exact.npz'))
    for rank,ridge,w,candidates in selected:
        row={'seed':seed,'rank':rank,'ridge':ridge,'parameters_factorized':rank*(tx.shape[1]+ty.shape[1])+ty.shape[1],
             'normalization_statistics':2*len(mean),'sketch_matrix_values':int(sketch.size),'anchor_train_p95':anchor,
             'selection_candidates':candidates,'splits':{}}
        for split,x,y,outputs,exact in [('dev',dx,dy,doutputs,dev_exact),('validation',vx,vy,voutputs,vex)]:
            pred=x@w+ym;features=pred.reshape(len(pred),16,-1)
            metrics,estimates,actual=geometry_metrics(features,outputs,exact,anchor,distance)
            oracle_metrics,oracle_estimates,_=geometry_metrics(y.reshape(len(y),16,-1),outputs,exact,anchor,distance)
            mse=float(np.square(pred-y).mean());base=float(np.square(y-ym).mean())
            row['splits'][split]={'sketch_mse':mse,'sketch_r2_vs_constant':1-mse/base,'geometry':metrics,
                                 'oracle_sketch_geometry':oracle_metrics}
            np.savez_compressed(run/f'rank{rank}_{split}.npz',features=features,true_distance=actual,
                                **estimates,**{'oracle_'+k:v for k,v in oracle_estimates.items()})
        u,s,vh=np.linalg.svd(w,full_matrices=False)
        np.savez_compressed(run/f'rank{rank}_model.npz',left=u[:,:rank]*s[:rank],right=vh[:rank],
                            input_mean=mean,input_std=std,target_mean=ym,sketch=sketch,anchor=anchor)
        rows.append(row)
        print('GEOMETRY_DONE',seed,rank,'dev R2',row['splits']['dev']['sketch_r2_vs_constant'],flush=True)
    result={'seed':seed,'results':rows,'seconds_cpu':time.perf_counter()-started,
            'checkpoint_sha256':manifest['checkpoint_sha256'],'files':{p.name:digest(p) for p in run.iterdir() if p.is_file()}}
    write_json(run/'result.json',result);return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args()
    torch.set_num_threads(1);repo=Path(__file__).resolve().parents[2];cfg=json.loads((repo/args.config).read_text(encoding='utf8'))
    root=repo/'outputs/operator_geometry_round2';out=root/'geometry'/cfg['study_id'];out.mkdir(parents=True,exist_ok=False)
    source_hashes={
        str(Path(__file__).relative_to(repo)):digest(Path(__file__)),
        'experiments/operator_geometry/round2_ridge.py':digest(Path(__file__).with_name('round2_ridge.py')),
        'experiments/operator_geometry/round2_numeric.py':digest(Path(__file__).with_name('round2_numeric.py'))}
    write_json(out/'config.json',cfg);write_json(out/'source_hashes.json',source_hashes)
    for relative in source_hashes:
        dest=out/'source_snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/relative,dest)
    status={'state':'waiting_for_data','pid':os.getpid(),'group':str(out),'completed':0,'total':len(cfg['seeds'])}
    write_json(root/'GEOMETRY_STATUS.json',status);rows=[]
    try:
        for seed in cfg['seeds']:
            assert all(digest(repo/path)==sha for path,sha in source_hashes.items())
            source=repo/cfg['data_group']/f's{seed}'
            while not (source/'result.json').exists():
                if datetime.now().astimezone()>=datetime.fromisoformat(cfg['deadline']):raise TimeoutError('Geometry data not ready before deadline')
                time.sleep(20)
            status.update(state='running',current_seed=seed);write_json(root/'GEOMETRY_STATUS.json',status)
            rows.extend(run_seed(source,out,seed,cfg));status['completed']+=1;write_json(root/'GEOMETRY_STATUS.json',status)
        write_json(out/'raw_summary.json',rows)
        status.update(state='ready_for_review',current_seed=None,finished=datetime.now().astimezone().isoformat());write_json(root/'GEOMETRY_STATUS.json',status)
    except BaseException:
        status.update(state='failed',error=traceback.format_exc());write_json(root/'GEOMETRY_STATUS.json',status);raise


if __name__=='__main__':main()
