"""Serial label-free KL/Fisher enrichment after utility GPU training finishes."""
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
from ..routing_stability.runner import setup,canonical_hash,timestamp
from ..routing_stability.model import CausalLM,full_state_hash
from .round2_data import data_hashes
from .round2_numeric import row_spearman
from .metrics import head,head_jvp,compare_outputs


def load(path):return json.loads(path.read_text(encoding='utf8'))


@torch.no_grad()
def measure(model,cache,exact,batch,device):
    source,target=exact['source'],exact['target'];n,c=source.shape
    index=torch.arange(n)[:,None].expand(n,c).flatten()
    s=torch.from_numpy(source.flatten());t=torch.from_numpy(target.flatten())
    values={k:[] for k in ['delta_nll','kl','fisher_kl','first_order_nll','argmax_changed']}
    for start in range(0,len(index),batch):
        ix=index[start:start+batch];a=s[start:start+batch];b=t[start:start+batch]
        mass=cache['weights'][ix,a]
        delta=(mass[:,None]*(cache['outputs'][ix,b]-cache['outputs'][ix,a])).to(device)
        h=cache['hidden'][ix].to(device);labels=cache['labels'][ix].to(device)
        stats=compare_outputs(head(model,h),head(model,h+delta),head_jvp(model,h,delta),labels)
        for key,value in stats.items():
            if not torch.isfinite(value).all():raise FloatingPointError(key)
            values[key].append(value.cpu().numpy())
    values={k:np.concatenate(v).reshape(n,c) for k,v in values.items()}
    np.testing.assert_allclose(values['delta_nll'],exact['delta_nll'],atol=2e-4,rtol=1e-4)
    return values


def compare_proxy(proxy,risk,mass):
    return {'rho_kl_fixed_source':float(np.nanmean(row_spearman(proxy.reshape(-1,12),risk['kl'].reshape(-1,12)))),
            'rho_abs_nll_fixed_source':float(np.nanmean(row_spearman(proxy.reshape(-1,12),abs(risk['delta_nll']).reshape(-1,12)))),
            'rho_kl_within_token_mass_weighted':float(np.nanmean(row_spearman(proxy*mass**2,risk['kl']))),
            'nearest_target_agreement_with_minimum_kl':float((proxy.reshape(-1,12).argmin(1)==risk['kl'].reshape(-1,12).argmin(1)).mean())}


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--root',default='outputs/operator_geometry_round2');args=p.parse_args()
    repo=Path(__file__).resolve().parents[2];cfg=load(repo/args.config);root=repo/args.root;root.mkdir(parents=True,exist_ok=True)
    data=repo/cfg['data_group'];energy=repo/cfg['energy_group'];geometry=repo/cfg['geometry_group']
    group=root/'risk'/canonical_hash(cfg)[:12];group.mkdir(parents=True,exist_ok=False)
    sources=data_hashes();sources.update({f'experiments/operator_geometry/{name}':digest(Path(__file__).with_name(name)) for name in ['round2_risk.py','round2_numeric.py']})
    write_json(group/'config.json',cfg);write_json(group/'source_hashes.json',sources)
    for path in sources:
        dest=group/'source_snapshot'/path;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/path,dest)
    status={'state':'waiting_for_utility','pid':os.getpid(),'stage':'risk','completed':0,'total':len(cfg['seeds']),
            'started':timestamp(),'group':str(group),'log':'logs/risk_v1.log','err':'logs/risk_v1.err'}
    write_json(root/'RISK_QUEUE_STATUS.json',status);acquired=False;lock=root/'PIPELINE_LOCK.json'
    try:
        if cfg['wait_for_utility']:
            while True:
                q=load(root/'QUEUE_STATUS.json')
                if q['state']=='failed':raise RuntimeError('Utility queue failed; risk stage must not proceed')
                if q['state']=='ready_for_agent_review' and (energy/'COMPLETE.json').exists() and not lock.exists():break
                if datetime.now().astimezone()>=datetime.fromisoformat(cfg['latest_start']):raise TimeoutError('Risk start deadline passed')
                time.sleep(20)
        with os.fdopen(os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY),'w') as f:
            json.dump({'pid':os.getpid(),'stage':'risk','started':timestamp()},f)
        acquired=True
        assert all(digest(repo/path)==sha for path,sha in sources.items())
        if (root/'STATUS.json').exists():write_json(group/'PREVIOUS_STATUS.json',load(root/'STATUS.json'))
        status.update(state='risk_running',running_started=timestamp());write_json(root/'STATUS.json',status)
        write_json(root/'ACTIVE_PROCESS.json',{**status,'active':True});write_json(root/'RISK_QUEUE_STATUS.json',status)
        setup()
        if cfg['device']=='cuda':torch.cuda.set_per_process_memory_fraction(.35)
        source_cfg=load(repo/load(data/'config.json')['source_config']);dcfg=load(data/'config.json');rows=[]
        for seed in cfg['seeds']:
            assert all(digest(repo/path)==sha for path,sha in sources.items())
            src=data/f's{seed}';manifest=load(src/'result.json')
            for name,sha in manifest['files'].items():assert digest(src/name)==sha
            checkpoint=repo/dcfg['source_group']/'runs'/f'mlp_half_s{seed}'/'best.pt'
            assert digest(checkpoint)==manifest['checkpoint_sha256']
            model=CausalLM(source_cfg,'mlp_half',seed).to(cfg['device']).eval()
            model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model'],strict=True);model.requires_grad_(False)
            before=full_state_hash(model.state_dict());run=group/f's{seed}';run.mkdir(exist_ok=False)
            for split in ['dev','validation']:
                cache=torch.load(src/f'{split}.pt',map_location='cpu',weights_only=True);exact=dict(np.load(src/f'{split}_exact.npz'))
                risk=measure(model,cache,exact,cfg['batch'],cfg['device'])
                np.savez_compressed(run/f'{split}_risk.npz',**risk,window=exact['window'],source=exact['source'],target=exact['target'])
                row={'seed':seed,'split':split,'tokens':len(exact['source']),'replacements':int(risk['kl'].size),
                     'mean_kl':float(risk['kl'].mean()),'mean_fisher_kl':float(risk['fisher_kl'].mean()),
                     'fisher_kl_rmse':float(np.sqrt(np.mean((risk['fisher_kl']-risk['kl'])**2))),
                     'rho_fisher_kl_fixed_source':float(np.nanmean(row_spearman(risk['fisher_kl'].reshape(-1,12),risk['kl'].reshape(-1,12)))),
                     'exact_nll_replay_max_abs':float(abs(risk['delta_nll']-exact['delta_nll']).max()),'energy_controls':{},'state_geometry':{}}
                for gauge in ['raw','centered']:
                    proxies=dict(np.load(energy/f's{seed}_{split}_{gauge}_proxies.npz'));mass=proxies.pop('mass');proxies.pop('energy')
                    row['energy_controls'][gauge]={name:compare_proxy(v,risk,mass) for name,v in proxies.items()}
                for rank in cfg['geometry_ranks']:
                    predictions=dict(np.load(geometry/f's{seed}/rank{rank}_{split}.npz'))
                    row['state_geometry'][str(rank)]={name:compare_proxy(v,risk,mass) for name,v in predictions.items() if name not in ['features','true_distance']}
                rows.append(row);print('RISK_DONE',seed,split,flush=True);del cache,risk
            assert full_state_hash(model.state_dict())==before and digest(checkpoint)==manifest['checkpoint_sha256']
            write_json(run/'audit.json',{'seed':seed,'frozen_backbone_unchanged':True,'checkpoint_sha256':manifest['checkpoint_sha256'],
                       'files':{p.name:digest(p) for p in run.iterdir() if p.is_file()}})
            del model
            if cfg['device']=='cuda':torch.cuda.empty_cache()
            status.update(completed=status['completed']+1,current_seed=seed);write_json(root/'STATUS.json',status)
        write_json(group/'summary.json',{'rows':rows,'scope':'Frozen final-layer single-expert fixed-mass substitutions. KL is model-distribution risk; realized-label CE can have either sign. No new training or test samples.'})
        write_json(group/'COMPLETE.json',{'seeds':cfg['seeds'],'finished':timestamp(),'source_unchanged':True})
        status.update(state='risk_ready_for_review',finished=timestamp(),current_seed=None);write_json(root/'STATUS.json',status)
        write_json(root/'ACTIVE_PROCESS.json',{**status,'active':False});write_json(root/'RISK_QUEUE_STATUS.json',status)
    except BaseException:
        status.update(state='failed',error=traceback.format_exc(),finished=timestamp());write_json(root/'RISK_QUEUE_STATUS.json',status)
        if acquired:write_json(root/'STATUS.json',status);write_json(root/'ACTIVE_PROCESS.json',{**status,'active':False})
        raise
    finally:
        if acquired:lock.unlink()


if __name__=='__main__':main()
