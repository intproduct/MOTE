"""Matched functional-metric adaptation; frozen experts and label-free targets."""
from pathlib import Path
import argparse,json,os,shutil,time,traceback
import numpy as np
import torch
from ..autoregressive_gate.data import digest,write_json
from ..routing_stability.runner import setup,timestamp,canonical_hash,append_json
from ..routing_stability.model import CausalLM,full_state_hash
from .round3_capacity import ReencodingPredictor
from .round3_replay_v2 import verify_run,reweight,compare_head,quality_stats,route_stats,predict_output,full_model_probe
from .metrics import head,head_jvp

ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads(p.read_text(encoding='utf8'))

def probability(model,x):
    return model.probability(x) if isinstance(model,ReencodingPredictor) else model.router(x)

def weights(model,x,beta):
    p=probability(model,x)
    return reweight(p,beta,p.topk(4,-1).indices)

def gram_matrices(model,hidden,outputs):
    """The output/Fisher pullbacks on centered expert coefficients."""
    v=outputs-outputs.mean(1,keepdim=True)
    output=v@v.transpose(1,2)/v.shape[-1]
    j=head_jvp(model,hidden[:,None,:],v)
    p=head(model,hidden).softmax(-1)
    j=j-(j*p[:,None,:]).sum(-1,keepdim=True)
    a=j*p[:,None,:].sqrt()
    fisher=.5*(a@a.transpose(1,2))
    return output,fisher

def quadratic(d,g):return torch.einsum('bi,bij,bj->b',d,g,d)

@torch.no_grad()
def create_geometry(lm,c,device,path,batch=32):
    result={'output':[],'fisher':[]};checks=[]
    for start in range(0,len(c['x']),batch):
        h=c['hidden'][start:start+batch].to(device);v=c['outputs'][start:start+batch].to(device)
        out,fish=gram_matrices(lm,h,v)
        for name,g in [('output',out),('fisher',fish)]:
            assert torch.isfinite(g).all()
            assert (g-g.transpose(1,2)).abs().max()<1e-4
            result[name].append(g.cpu())
        if start==0:
            torch.manual_seed(8431);d=torch.randn(len(h),16,device=device);d-=d.mean(1,keepdim=True)
            direction=(d[:,:,None]*v).sum(1);j=head_jvp(lm,h,direction).double();p=head(lm,h).double().softmax(-1)
            direct=.5*(p*(j-(p*j).sum(-1,keepdim=True)).square()).sum(-1)
            rel=float(((quadratic(d,fish).double()-direct).abs()/(1+direct)).max());assert rel<2e-4
            checks.append({'fisher_direct_relative_error':rel})
    result={k:torch.cat(v) for k,v in result.items()}
    torch.save(result,path)
    return result,checks

@torch.no_grad()
def evaluate(model,beta,lm,c,device,include_labels=False):
    w=torch.cat([weights(model,c['x'][i:i+256].to(device),beta).cpu() for i in range(0,len(c['x']),256)])
    h=c['hidden']+((w-c['weights'])[:,:,None]*c['outputs']).sum(1)
    # Labels are accessed only in post-selection reporting, never in checkpoint selection.
    if include_labels:
        stats=compare_head(lm,c['hidden'],h,c['labels'])
        return w,stats
    kl=[]
    for i in range(0,len(h),256):
        lp=head(lm,c['hidden'][i:i+256].to(device)).double().log_softmax(-1)
        lq=head(lm,h[i:i+256].to(device)).double().log_softmax(-1)
        kl.append((lp.exp()*(lp-lq)).sum(-1).clamp_min(0).cpu())
    return float(torch.cat(kl).mean())

def fit(cfg,group,seed,method,objective,caches,gram,lm,assignment,codes,perm):
    run=group/'runs'/f'{method}_{objective}_s{seed}';run.mkdir(parents=True,exist_ok=False)
    isnew=method!='mlp';source=ROOT/cfg['capacity_group' if isnew else 'reference_group']
    model,beta,prior=verify_run(source,seed,method,assignment,codes,perm,isnew)
    model.requires_grad_(True);model.log_scale.requires_grad_(False);model.to(cfg['device'])
    init=full_state_hash(model.state_dict());opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['lr'],weight_decay=1e-4)
    x=caches['train']['x'].to(cfg['device']);teacher=caches['train']['weights'].to(cfg['device'])
    g=None if objective=='coefficient' else gram[objective].to(cfg['device'])
    normalizer=1. if g is None else float(g.diagonal(dim1=1,dim2=2).mean());assert normalizer>0
    gen=torch.Generator().manual_seed(seed+980000);sampled=[];start=time.perf_counter();best=evaluate(model,beta,lm,caches['dev'],cfg['device'])
    initial_kl=best;append_json(run/'curves.jsonl',{'step':0,'dev_exact_kl':best})
    def save(step):
        torch.save({'model':model.state_dict(),'step':step,'beta':beta,'optimizer':opt.state_dict(),'sample_rng':gen.get_state(),
                    'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if cfg['device']=='cuda' else [],'config':cfg,'method':method,'objective':objective,'seed':seed},run/'best_dev.pt')
    save(0)
    for step in range(1,cfg['steps']+1):
        idx=torch.randint(len(x),(cfg['batch_size'],),generator=gen);sampled.extend(idx.tolist());idx=idx.to(cfg['device'])
        model.train();opt.zero_grad(set_to_none=True);d=weights(model,x[idx],beta)-teacher[idx]
        loss=(d.square().sum(1) if g is None else quadratic(d,g[idx])).mean()/normalizer
        assert torch.isfinite(loss) and loss>=-1e-5;loss.backward()
        grads=[p.grad for p in model.parameters() if p.grad is not None];assert grads and all(torch.isfinite(v).all() for v in grads)
        gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(gn) and gn>0;opt.step()
        if step%cfg['eval_every']==0 or step==cfg['steps']:
            model.eval();score=evaluate(model,beta,lm,caches['dev'],cfg['device']);assert np.isfinite(score)
            append_json(run/'curves.jsonl',{'step':step,'loss':float(loss.detach()),'gradient_norm':float(gn),'dev_exact_kl':score})
            if score<best:best=score;save(step)
            print('FUNCTIONAL',seed,method,objective,step,'KL',round(score,7),flush=True)
    changed=full_state_hash(model.state_dict())!=init;assert changed
    torch.save({'model':model.state_dict(),'step':step,'beta':beta,'optimizer':opt.state_dict(),'sample_rng':gen.get_state(),
                'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if cfg['device']=='cuda' else []},run/'last.pt')
    ck=torch.load(run/'best_dev.pt',map_location=cfg['device'],weights_only=False);model.load_state_dict(ck['model']);model.eval()
    # Final validation loading happens only after selection, and cannot alter it.
    if 'validation' not in caches:
        c=torch.load(ROOT/cfg['data_group']/f's{seed}/validation.pt',map_location='cpu',weights_only=True)
        caches['validation']={k:v[:cfg['smoke_tokens']] for k,v in c.items()} if cfg.get('smoke') else c
    rows=[];checks={};dcfg=read(ROOT/cfg['data_group']/'config.json');gate=lm.get_submodule(dcfg['module']).mote.core.gate
    for split in ['dev','validation']:
        c=caches[split];w,stats=evaluate(model,beta,lm,c,cfg['device'],True);p,_=predict_output(model,c['x'],cfg['device'])
        np.savez_compressed(run/f'{split}_clean.npz',**stats,weights=w.numpy(),window=c['window'].numpy())
        row={'seed':seed,'method':method,'objective':objective,'split':split,'quality':quality_stats(stats,c['window'].numpy(),seed,c['baseline_nll'].numpy()),
             'routing':route_stats(w,p),'top4_set_agreement_original':float(((w>0)==(c['weights']>0)).all(1).float().mean()),'roundtrip':{}}
        h=c['hidden']+((w-c['weights'])[:,:,None]*c['outputs']).sum(1)
        for dtype,name in [(torch.float16,'fp16'),(torch.bfloat16,'bf16')]:
            with torch.no_grad():
                wr=torch.cat([weights(model,c['x'][i:i+256].to(dtype).float().to(cfg['device']),beta).cpu() for i in range(0,len(w),256)])
            hr=c['hidden']+((wr-c['weights'])[:,:,None]*c['outputs']).sum(1);st=compare_head(lm,h,hr,c['labels']);flip=((wr>0)!=(w>0)).any(1).numpy()
            np.savez_compressed(run/f'{split}_{name}.npz',**st,flip=flip,weights=wr.numpy(),window=c['window'].numpy())
            row['roundtrip'][name]={'flip_count':int(flip.sum()),'harmful_flip_gt_01':int((flip&(st['delta_nll']>.1)).sum()),'mean_kl':float(st['kl'].mean()),
                                   'absolute_logprob_p99':float(np.quantile(abs(st['delta_nll']),.99)),'absolute_logprob_max':float(abs(st['delta_nll']).max())}
        tokens=np.load(ROOT/dcfg['data_root']/('train.npy' if split=='dev' else 'validation.npy'))
        checks[split+'_full_model_max_abs']=full_model_probe(lm,model,gate,c,tokens,'calibrated',beta,cfg['device'])
        rows.append(row)
    seconds=time.perf_counter()-start;np.save(run/'sample_indices.npy',np.asarray(sampled,dtype=np.int32))
    model.cpu().eval()
    for split in ['dev','validation']:
        c=caches[split];cpu=torch.cat([weights(model,c['x'][i:i+256],beta).detach() for i in range(0,len(c['x']),256)])
        gpu=np.load(run/f'{split}_clean.npz')['weights'];same=((cpu.numpy()>0)==(gpu>0)).all(1)
        # Different device tie behavior is recorded. Continuous weights must agree away from membership changes.
        err=float(abs(cpu.numpy()[same]-gpu[same]).max()) if same.any() else None
        assert err is not None and err<2e-4
        checks[split+'_cpu_gpu_set_changes']=int((~same).sum());checks[split+'_same_set_weight_error']=err
    np.testing.assert_array_equal(np.asarray(sampled),torch.randint(len(x),(cfg['steps'],cfg['batch_size']),generator=torch.Generator().manual_seed(seed+980000)).numpy().flatten())
    result={'seed':seed,'method':method,'objective':objective,'prior':prior,'beta_fixed':beta,'normalizer_train_only':normalizer,'initial_state_hash':init,
            'parameters_total':sum(p.numel() for p in model.parameters()),'parameters_updated':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'parameter_changed':changed,'selected_step':ck['step'],'initial_dev_kl':initial_kl,'selected_dev_kl':best,'seconds_including_checks':seconds,'rows':rows,'checks':checks,
            'files':{p.name:digest(p) for p in run.iterdir() if p.is_file()}}
    write_json(run/'result.json',result);del model,opt,x,teacher,g
    if cfg['device']=='cuda':torch.cuda.empty_cache()
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);cfg=read(ROOT/ap.parse_args().config);setup()
    root=ROOT/cfg['output_root'];root.mkdir(parents=True,exist_ok=True);group=root/canonical_hash(cfg)[:12];group.mkdir(exist_ok=False)
    sources={}
    for parent in [cfg['capacity_group'],cfg['reference_group'],cfg['data_group']]:
        for rel,h in read(ROOT/parent/'source_hashes.json').items():assert digest(ROOT/rel)==h;sources[rel]=h
    for name in ['round3_functional.py','round3_replay_v2.py','metrics.py']:sources['experiments/operator_geometry/'+name]=digest(Path(__file__).with_name(name))
    write_json(group/'config.json',cfg);write_json(group/'source_hashes.json',sources)
    for rel in sources:
        p=group/'source_snapshot'/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,p)
    state={'state':'preparing','stage':'functional_metric_adaptation','pid':os.getpid(),'active':True,'started':timestamp(),'group':str(group),'completed':0,
           'total':len(cfg['seeds'])*len(cfg['methods'])*len(cfg['objectives'])};lock=ROOT/'outputs/operator_geometry_round2/PIPELINE_LOCK.json';acquired=False
    try:
        with os.fdopen(os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY),'w') as f:json.dump(state,f)
        acquired=True
        if cfg['device']=='cuda':torch.cuda.set_per_process_memory_fraction(.35)
        rows=[];geometry_checks=[];dcfg=read(ROOT/cfg['data_group']/'config.json')
        for seed in cfg['seeds']:
            state.update(state='geometry',current=f's{seed}');write_json(root/'STATUS.json',state)
            data=ROOT/cfg['data_group']/f's{seed}';mf=read(data/'result.json')
            for name,h in mf['files'].items():assert digest(data/name)==h
            ckpath=ROOT/dcfg['source_group']/f'runs/mlp_half_s{seed}/best.pt';assert digest(ckpath)==mf['checkpoint_sha256']
            lm=CausalLM(read(ROOT/dcfg['source_config']),'mlp_half',seed).to(cfg['device']).eval();lm.load_state_dict(torch.load(ckpath,map_location='cpu',weights_only=False)['model']);lm.requires_grad_(False);lmhash=full_state_hash(lm.state_dict())
            caches={s:torch.load(data/f'{s}.pt',map_location='cpu',weights_only=True) for s in ['train','dev']}
            if cfg.get('smoke'):caches={s:{k:v[:cfg['smoke_tokens']] for k,v in c.items()} for s,c in caches.items()}
            geometry,checks=create_geometry(lm,caches['train'],cfg['device'],group/f'geometry_s{seed}.pt');geometry_checks.append({'seed':seed,'checks':checks,'sha256':digest(group/f'geometry_s{seed}.pt')})
            geo=ROOT/cfg['geometry_group']/f's{seed}';a=read(geo/'assignment.json');assignment=np.asarray(a['expert_for_basis']);perm=np.asarray(a['random_expert_for_basis']);codes=np.load(geo/'codes.npy')
            for method in cfg['methods']:
                for objective in cfg['objectives']:
                    assert all(digest(ROOT/rel)==h for rel,h in sources.items())
                    state.update(state='training',current=f'{method}_{objective}_s{seed}');write_json(root/'STATUS.json',state)
                    rows.append(fit(cfg,group,seed,method,objective,caches,geometry,lm,assignment,codes,perm));state['completed']=len(rows);write_json(root/'STATUS.json',state)
            assert full_state_hash(lm.state_dict())==lmhash and digest(ckpath)==mf['checkpoint_sha256'];del caches,geometry,lm
        write_json(group/'raw_summary.json',rows);write_json(group/'GEOMETRY_CHECKS.json',geometry_checks)
        write_json(group/'AUDIT.json',{'passed':True,'runs':len(rows),'post_selection_quality_rows':2*len(rows),'roundtrip_records':4*len(rows),'sources_unchanged':True,'frozen_experts':True,'missing':[],'failed':[]})
        state.update(state='ready_for_review',active=False,current=None,finished=timestamp());write_json(root/'STATUS.json',state);write_json(group/'COMPLETE.json',state)
    except BaseException:
        state.update(state='failed',active=False,error=traceback.format_exc(),finished=timestamp());write_json(root/'STATUS.json',state);raise
    finally:
        if acquired:lock.unlink()

if __name__=='__main__':main()
