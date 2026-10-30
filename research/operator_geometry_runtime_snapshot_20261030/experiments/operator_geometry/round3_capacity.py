"""Predeclared eight-coordinate re-encoding calibration, with paired controls.

The teacher is a frozen input-determined MLP router. No language labels enter
training. Training and validation artifacts from earlier stages are read-only.
"""
from pathlib import Path
import argparse
import json
import math
import os
import shutil
import time
import traceback
import numpy as np
import torch
from torch import nn
from ...quantum_gate import apply_ry
from ..autoregressive_gate.data import digest,write_json
from ..routing_stability.runner import setup,canonical_hash,timestamp,append_json
from ..routing_stability.model import full_state_hash
from .round2_train import train_hashes,predict,UtilityPredictor
from .round2_positive import teacher_cost,measures

ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads(p.read_text(encoding='utf8'))

class ReencodingPredictor(nn.Module):
    def __init__(self,dim,rank,method,assignment):
        super().__init__();self.method=method
        self.norm=nn.LayerNorm(dim);self.input_proj=nn.Linear(dim,rank,bias=False)
        self.output=nn.Linear(rank,16 if method=='affine_born' else 8)
        nn.init.normal_(self.output.weight,std=.01);nn.init.zeros_(self.output.bias)
        self.log_scale=nn.Parameter(torch.zeros(()))
        bits=(torch.arange(16)[:,None]>>torch.arange(3,-1,-1))&1
        self.register_buffer('cz',((1-2*bits[:,:-1]*bits[:,1:]).prod(1)).float())
        self.register_buffer('inverse_assignment',torch.tensor(np.argsort(assignment)))

    def state(self,angles):
        state=angles.new_zeros(len(angles),16);state[:,0]=1
        for wire in range(4):state=apply_ry(state,angles[:,wire],wire,4)
        if self.method=='reencode_cz':state=state*self.cz
        for wire in range(4):state=apply_ry(state,angles[:,wire+4],wire,4)
        return state

    def probability(self,x):
        v=self.output(torch.tanh(self.input_proj(self.norm(x))))
        if self.method=='affine_born':
            p=(1+v).square()
        else:
            offset=v.new_tensor([math.pi/2]*4+[.35]*4)
            p=self.state(offset+(math.pi/2)*torch.tanh(v)).square()[:,self.inverse_assignment]
        p=p/p.sum(-1,keepdim=True).clamp_min(1e-12)
        return (p+1e-6)/(1+16e-6)

    def forward(self,x):
        cost=-self.probability(x).log()
        return (cost-cost.mean(-1,keepdim=True))*self.log_scale.clamp(-3,3).exp()

def mathematical_checks():
    torch.manual_seed(733);rows=[]
    for method in ['reencode_cz','reencode_product']:
        model=ReencodingPredictor(12,8,method,np.arange(16)).double()
        for i in range(12):
            angles=torch.rand(8,dtype=torch.float64)*2+.2
            jac=torch.autograd.functional.jacobian(lambda a:model.state(a[None]).square()[0],angles)
            rank=int(torch.linalg.matrix_rank(jac,atol=1e-8));rows.append({'method':method,'point':i,'probability_jacobian_rank':rank})
            state=model.state(angles[None]);assert abs(float(state.square().sum())-1)<1e-12
            if method=='reencode_product':
                collapsed=angles.new_zeros(1,16);collapsed[:,0]=1
                for w in range(4):collapsed=apply_ry(collapsed,(angles[w]+angles[w+4])[None],w,4)
                torch.testing.assert_close(collapsed,state,atol=1e-12,rtol=1e-12);assert rank==4
            else:assert rank==8
    return rows

def train(cfg,group,seed,method,caches,assignment):
    path=group/'runs'/f'{method}_s{seed}';path.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(seed+960000);model=ReencodingPredictor(caches['train']['x'].shape[1],8,method,assignment).to(cfg['device'])
    initial=full_state_hash(dict(model.named_parameters()));encoder=full_state_hash({k:v for k,v in model.state_dict().items() if k.startswith(('norm.','input_proj.'))})
    reference=torch.load(ROOT/cfg['reference_group']/f'runs/mlp_s{seed}/best_dev.pt',map_location='cpu',weights_only=False)
    # Reconstruct the original initialization (the trained checkpoint is not an initialization).
    torch.manual_seed(seed+960000);old=UtilityPredictor(caches['train']['x'].shape[1],8,'mlp',np.arange(16),np.zeros((16,4)),np.arange(16))
    old_encoder=full_state_hash({k.removeprefix('router.'):v for k,v in old.state_dict().items() if k.startswith(('router.norm.','router.input_proj.'))})
    assert encoder==old_encoder
    scale=float(caches['train']['target'].std());constant=caches['train']['target'].mean(0).numpy()
    gen=torch.Generator().manual_seed(seed+970000);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
    x=caches['train']['x'].to(cfg['device']);y=caches['train']['target'].to(cfg['device'])/scale
    best=float('inf');sampled=[];start=time.perf_counter()
    for step in range(1,cfg['steps']+1):
        model.train();idx=torch.randint(len(x),(cfg['batch_size'],),generator=gen).to(cfg['device']);sampled.extend(idx.cpu().tolist())
        opt.zero_grad(set_to_none=True);loss=(model(x[idx])-y[idx]).square().mean();assert torch.isfinite(loss)
        loss.backward();grads=[p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm) and norm>0;opt.step()
        if step%cfg['eval_every']==0 or step==cfg['steps']:
            pred=predict(model,caches['dev'],scale,cfg['device']);m=measures(pred,caches['dev']['target'].numpy(),constant)
            append_json(path/'curves.jsonl',{'step':step,'loss':float(loss.detach()),'gradient_norm':float(norm),**m})
            if m['mse']<best:
                best=m['mse'];torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step,'scale':scale,'config':cfg,'seed':seed,'method':method,
                    'sample_rng':gen.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if cfg['device']=='cuda' else []},path/'best_dev.pt')
            print('CAPACITY',seed,method,step,'dev_R2',round(m['r2_vs_constant'],5),flush=True)
    torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step,'sample_rng':gen.get_state(),'torch_rng':torch.get_rng_state(),
        'cuda_rng':torch.cuda.get_rng_state_all() if cfg['device']=='cuda' else []},path/'last.pt')
    bestck=torch.load(path/'best_dev.pt',map_location=cfg['device'],weights_only=False);model.load_state_dict(bestck['model'],strict=True)
    row={'seed':seed,'method':method,'parameters':sum(p.numel() for p in model.parameters()),'selected_step':bestck['step'],'seconds_shared_gpu':time.perf_counter()-start,
        'initial_parameter_hash':initial,'parameter_changed':initial!=full_state_hash(dict(model.named_parameters())),'shared_encoder_initial_hash':encoder,'splits':{},'replay':{}}
    for split,cache in caches.items():
        prediction=predict(model,cache,scale,cfg['device']);row['splits'][split]=measures(prediction,cache['target'].numpy(),constant)
        np.savez_compressed(path/f'{split}_predictions.npz',cost=prediction)
    np.save(path/'sample_indices.npy',np.asarray(sampled,dtype=np.int32))
    if cfg['steps']==reference['config']['steps']:
        assert digest(path/'sample_indices.npy')==digest(ROOT/cfg['reference_group']/f'runs/mlp_s{seed}/sample_indices.npy')
    model.cpu().eval()
    for split in ['dev','validation']:
        saved=np.load(path/f'{split}_predictions.npz')['cost'];cpu=predict(model,caches[split],scale,'cpu')
        error=float(abs(cpu-saved).max());assert error<5e-4;row['replay'][split]=error
    assert row['parameter_changed'];row['files']={p.name:digest(p) for p in path.iterdir() if p.is_file()}
    write_json(path/'result.json',row);del model,opt,x,y
    if cfg['device']=='cuda':torch.cuda.empty_cache()
    return row

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);args=ap.parse_args();cfg=read(ROOT/args.config)
    root=ROOT/cfg['output_root'];root.mkdir(parents=True,exist_ok=True);group=root/canonical_hash(cfg)[:12];group.mkdir(exist_ok=False)
    lock=ROOT/'outputs/operator_geometry_round2/PIPELINE_LOCK.json';acquired=False
    state={'stage':'capacity_calibration','state':'preparing','pid':os.getpid(),'active':True,'group':str(group),'started':timestamp(),'completed':0,'total':len(cfg['seeds'])*len(cfg['methods'])}
    sources=train_hashes();sources['experiments/operator_geometry/round2_positive.py']=digest(Path(__file__).with_name('round2_positive.py'));sources['experiments/operator_geometry/round3_capacity.py']=digest(Path(__file__))
    write_json(group/'config.json',cfg);write_json(group/'source_hashes.json',sources)
    for rel in sources:
        dest=group/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
    try:
        with os.fdopen(os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY),'w') as f:json.dump(state,f)
        acquired=True;setup()
        if cfg['device']=='cuda':torch.cuda.set_per_process_memory_fraction(.35)
        write_json(group/'MATHEMATICAL_CHECKS.json',mathematical_checks());rows=[]
        for seed in cfg['seeds']:
            data=ROOT/cfg['data_group']/f's{seed}';manifest=read(data/'result.json')
            for name,h in manifest['files'].items():assert digest(data/name)==h
            caches={}
            for split in ['train','dev','validation']:
                c=torch.load(data/f'{split}.pt',map_location='cpu',weights_only=True)
                caches[split]={'x':c['x'],'target':teacher_cost(c['teacher'])};del c
                if cfg.get('smoke'):caches[split]={k:v[:256] for k,v in caches[split].items()}
            assignment=np.asarray(read(ROOT/cfg['geometry_group']/f's{seed}/assignment.json')['expert_for_basis'])
            for method in cfg['methods']:
                assert all(digest(ROOT/rel)==h for rel,h in sources.items())
                state.update(state='running',current=f'{method}_s{seed}');write_json(root/'STATUS.json',state)
                rows.append(train(cfg,group,seed,method,caches,assignment));state['completed']=len(rows)
                write_json(root/'STATUS.json',state)
        write_json(group/'raw_summary.json',rows)
        state.update(state='ready_for_review',active=False,current=None,finished=timestamp());write_json(root/'STATUS.json',state)
        write_json(group/'AUDIT.json',{'passed':True,'runs':len(rows),'cpu_replays':2*len(rows),'same_sampling_as_reference':not cfg.get('smoke',False),
            'same_encoder_initialization':True,'finite_gradients':True,'sources_unchanged':True,'missing':[],'failed':[]})
        write_json(group/'COMPLETE.json',state)
    except BaseException:
        state.update(state='failed',active=False,error=traceback.format_exc(),finished=timestamp());write_json(root/'STATUS.json',state);raise
    finally:
        if acquired:lock.unlink()

if __name__=='__main__':main()
