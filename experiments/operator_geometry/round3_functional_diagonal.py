"""Prespecified diagonal controls reusing frozen training-only functional Grams."""
from pathlib import Path
import argparse,json,os,shutil,traceback
import numpy as np
import torch
from ..autoregressive_gate.data import digest,write_json
from ..routing_stability.runner import setup,timestamp,canonical_hash
from ..routing_stability.model import CausalLM,full_state_hash
from .round3_functional import fit,read,ROOT

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);cfg=read(ROOT/ap.parse_args().config);setup()
    parent=ROOT/cfg['functional_group'];assert read(parent/'AUDIT.json')['passed']
    assert read(ROOT/'outputs/operator_geometry_round3/reviews/functional_v1/AUDIT.json')['passed']
    root=ROOT/cfg['output_root'];root.mkdir(parents=True,exist_ok=True);group=root/canonical_hash(cfg)[:12];group.mkdir(exist_ok=False)
    sources=read(parent/'source_hashes.json');sources['experiments/operator_geometry/round3_functional_diagonal.py']=digest(Path(__file__))
    for rel,h in sources.items():
        assert digest(ROOT/rel)==h
        p=group/'source_snapshot'/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,p)
    write_json(group/'config.json',cfg);write_json(group/'source_hashes.json',sources)
    state={'state':'preparing','stage':'functional_diagonal_control','pid':os.getpid(),'active':True,'started':timestamp(),'group':str(group),'completed':0,'total':len(cfg['methods'])*len(cfg['seeds'])*len(cfg['objectives'])}
    lock=ROOT/'outputs/operator_geometry_round2/PIPELINE_LOCK.json';acquired=False
    try:
        with os.fdopen(os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY),'w') as f:json.dump(state,f)
        acquired=True
        if cfg['device']=='cuda':torch.cuda.set_per_process_memory_fraction(.35)
        rows=[];geometry_checks=[];dcfg=read(ROOT/cfg['data_group']/'config.json')
        for seed in cfg['seeds']:
            state.update(state='preparing_geometry',current=f's{seed}');write_json(root/'STATUS.json',state)
            data=ROOT/cfg['data_group']/f's{seed}';mf=read(data/'result.json')
            for name,h in mf['files'].items():assert digest(data/name)==h
            ckpath=ROOT/dcfg['source_group']/f'runs/mlp_half_s{seed}/best.pt';assert digest(ckpath)==mf['checkpoint_sha256']
            lm=CausalLM(read(ROOT/dcfg['source_config']),'mlp_half',seed).to(cfg['device']).eval();lm.load_state_dict(torch.load(ckpath,map_location='cpu',weights_only=False)['model']);lm.requires_grad_(False);lmhash=full_state_hash(lm.state_dict())
            caches={s:torch.load(data/f'{s}.pt',map_location='cpu',weights_only=True) for s in ['train','dev']}
            gm_path=parent/f'geometry_s{seed}.pt';h=next(r['sha256'] for r in read(parent/'GEOMETRY_CHECKS.json') if r['seed']==seed);assert digest(gm_path)==h
            geometry=torch.load(gm_path,map_location='cpu',weights_only=True)
            if cfg.get('smoke'):
                caches={s:{k:v[:cfg['smoke_tokens']] for k,v in c.items()} for s,c in caches.items()}
                geometry={k:v[:cfg['smoke_tokens']] for k,v in geometry.items()}
            grams={};checks={}
            for name,g in geometry.items():
                diag=g.diagonal(dim1=1,dim2=2);assert torch.isfinite(diag).all() and diag.min()>=0
                gd=torch.diag_embed(diag);assert torch.equal(gd.diagonal(dim1=1,dim2=2),diag)
                grams[name+'_diag']=gd;checks[name]={'same_trace_normalizer':float(diag.mean()),'off_diagonal_zero':bool(torch.equal(gd,torch.diag_embed(gd.diagonal(dim1=1,dim2=2))))}
            geometry_checks.append({'seed':seed,'source':str(gm_path),'sha256':h,'checks':checks});del geometry
            geo=ROOT/cfg['geometry_group']/f's{seed}';a=read(geo/'assignment.json');assignment=np.asarray(a['expert_for_basis']);perm=np.asarray(a['random_expert_for_basis']);codes=np.load(geo/'codes.npy')
            for method in cfg['methods']:
                for objective in cfg['objectives']:
                    assert all(digest(ROOT/rel)==h for rel,h in sources.items())
                    state.update(state='training',current=f'{method}_{objective}_s{seed}');write_json(root/'STATUS.json',state)
                    rows.append(fit(cfg,group,seed,method,objective,caches,grams,lm,assignment,codes,perm));state['completed']=len(rows);write_json(root/'STATUS.json',state)
            assert full_state_hash(lm.state_dict())==lmhash and digest(ckpath)==mf['checkpoint_sha256'];del caches,grams,lm
        write_json(group/'raw_summary.json',rows);write_json(group/'GEOMETRY_CHECKS.json',geometry_checks)
        write_json(group/'AUDIT.json',{'passed':True,'runs':len(rows),'post_selection_quality_rows':2*len(rows),'roundtrip_records':4*len(rows),'sources_unchanged':True,'frozen_experts':True,'missing':[],'failed':[]})
        state.update(state='ready_for_review',active=False,current=None,finished=timestamp());write_json(root/'STATUS.json',state);write_json(group/'COMPLETE.json',state)
    except BaseException:
        state.update(state='failed',active=False,error=traceback.format_exc(),finished=timestamp());write_json(root/'STATUS.json',state);raise
    finally:
        if acquired:lock.unlink()

if __name__=='__main__':main()
