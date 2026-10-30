"""Frozen sparse reweighting, gate-isolated roundtrip, and serial gate timing.

No training or validation-based selection. Native probabilities and separately
temperature-calibrated teacher scores are both reported, with identical top-k
indices selected from native scores. Experts and upstream activations stay fixed.
"""
from pathlib import Path
import argparse,json,os,shutil,subprocess,time,traceback
import numpy as np
import torch
from torch.nn import functional as F
from ..autoregressive_gate.data import digest,write_json
from ..routing_stability.runner import setup,timestamp,canonical_hash
from ..routing_stability.model import CausalLM,full_state_hash
from .round2_train import UtilityPredictor
from .round3_capacity import ReencodingPredictor
from .metrics import head,bootstrap_window_mean

ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads(p.read_text(encoding='utf8'))

@torch.no_grad()
def native_output(model,x,original_gate=None):
    if original_gate is not None:
        logits=original_gate.router(x)
        return original_gate._softmax_temp(logits),logits.topk(4,-1).indices
    p=model.probability(x) if isinstance(model,ReencodingPredictor) else model.router(x)
    return p,p.topk(4,-1).indices

def reweight(prob,beta=1.,indices=None):
    if indices is None:indices=prob.topk(4,-1).indices
    selected=prob.gather(1,indices).clamp_min(1e-12).log().mul(beta).softmax(-1)
    return torch.zeros_like(prob).scatter(1,indices,selected)

@torch.no_grad()
def predict_output(model,x,device,original_gate=None,batch=256):
    outputs=[native_output(model,x[i:i+batch].to(device),original_gate) for i in range(0,len(x),batch)]
    return torch.cat([p.cpu() for p,i in outputs]),torch.cat([i.cpu() for p,i in outputs])

@torch.no_grad()
def compare_head(model,h0,h1,labels,batch=256):
    device=next(model.parameters()).device;out={k:[] for k in ['delta_nll','kl','argmax_changed']}
    for start in range(0,len(h0),batch):
        a=head(model,h0[start:start+batch].to(device));b=head(model,h1[start:start+batch].to(device))
        lp=a.double().log_softmax(-1);lq=b.double().log_softmax(-1);y=labels[start:start+batch,None].to(device)
        values={'delta_nll':(lp-lq).gather(1,y).squeeze(1),'kl':(lp.exp()*(lp-lq)).sum(1).clamp_min(0),'argmax_changed':a.argmax(1)!=b.argmax(1)}
        for k,v in values.items():assert torch.isfinite(v).all();out[k].append(v.cpu().numpy())
    return {k:np.concatenate(v) for k,v in out.items()}

def route_stats(w,p):
    mask=w>0;loads=mask.float().mean(0);share=loads/loads.sum()
    return {'load_cv':float(loads.std(unbiased=False)/loads.mean()),'unused_experts':int((loads==0).sum()),
        'load_entropy':float(-(share*share.clamp_min(1e-12).log()).sum()),'mean_probability_entropy':float(-(p*p.clamp_min(1e-12).log()).sum(1).mean()),
        'unique_top4_sets':int(torch.unique(mask,dim=0).shape[0]),'selection_frequency':loads.tolist()}

def quality_stats(stats,window,seed,baseline_nll):
    delta=stats['delta_nll'];return {'tokens':len(delta),'mean_delta_ce':float(delta.mean()),'ci95_windows':bootstrap_window_mean(delta,window,seed),
        'sampled_ce':float(np.asarray(baseline_nll).mean()+delta.mean()),'sampled_ppl':float(np.exp(np.asarray(baseline_nll).mean()+delta.mean())),
        'mean_kl':float(stats['kl'].mean()),'kl_p99':float(np.quantile(stats['kl'],.99)),
        'absolute_target_logprob_error_p99':float(np.quantile(abs(delta),.99)), 'absolute_target_logprob_error_max':float(abs(delta).max()),
        'harm_gt_01':int((delta>.1).sum()),'benefit_lt_minus_01':int((delta<-.1).sum()),'argmax_changed':int(stats['argmax_changed'].sum())}

def verify_run(group,seed,method,assignment,codes,permutation,isnew):
    run=group/f'runs/{method}_s{seed}';r=read(run/'result.json');cfg=read(group/'config.json')
    for name,h in r['files'].items():assert digest(run/name)==h
    ck=torch.load(run/'best_dev.pt',map_location='cpu',weights_only=False)
    curves=[json.loads(x) for x in (run/'curves.jsonl').read_text().splitlines()]
    assert [x['step'] for x in curves]==list(range(cfg['eval_every'],cfg['steps']+1,cfg['eval_every']))
    assert all(np.isfinite(x['gradient_norm']) and x['gradient_norm']>0 for x in curves)
    assert min(curves,key=lambda x:x['mse'])['step']==ck['step']==r['selected_step']
    sample=torch.randint(32768,(cfg['steps'],cfg['batch_size']),generator=torch.Generator().manual_seed(seed+970000)).numpy().flatten()
    np.testing.assert_array_equal(sample,np.load(run/'sample_indices.npy'))
    torch.manual_seed(seed+960000)
    model=ReencodingPredictor(1024,8,method,assignment) if isnew else UtilityPredictor(1024,8,method,assignment,codes,permutation)
    assert full_state_hash(dict(model.named_parameters()))==r['initial_parameter_hash']
    model.load_state_dict(ck['model'],strict=True);model.eval();model.requires_grad_(False)
    assert all(torch.isfinite(v).all() for v in ck['model'].values())
    assert sum(p.numel() for p in model.parameters())==r['parameters']
    scale=ck['scale'] if isnew else ck['target_scale'];beta=float(model.log_scale.clamp(-3,3).exp()*scale)
    return model,beta,{'checkpoint_sha256':digest(run/'best_dev.pt'),'selected_step':ck['step'],'parameter_count_with_calibration_scalar':r['parameters'],
        'effective_calibration_power':beta,'samples_verified':True,'curves_verified':True,'finite_weights':True,'result_path':str(run/'result.json')}

@torch.no_grad()
def full_model_probe(lm,gate_model,gate,cache,tokens,mode,beta,device):
    starts=torch.unique_consecutive(cache['window'])[:2].numpy();ids=torch.tensor(np.stack([tokens[int(s):int(s)+128] for s in starts]),device=device,dtype=torch.long)
    def hook(module,inputs,result):
        p,idx=native_output(gate_model,inputs[0].reshape(-1,1024));w=reweight(p,beta if mode=='calibrated' else 1.,idx)
        w=w.reshape_as(result[0]);return w,(w>0).to(w.dtype),result[2]
    handle=gate.register_forward_hook(hook)
    try:logits=lm(ids).reshape(-1,lm.embedding.num_embeddings)
    finally:handle.remove()
    n=len(starts)*4;idx=(torch.arange(len(starts),device=device)[:,None]*128+torch.tensor([31,63,95,127],device=device)[None,:]).flatten()
    p,indices=predict_output(gate_model,cache['x'][:n],device);w=reweight(p,beta if mode=='calibrated' else 1.,indices)
    changed=cache['hidden'][:n]+((w-cache['weights'][:n])[:,:,None]*cache['outputs'][:n]).sum(1)
    error=float((logits[idx]-head(lm,changed.to(device))).abs().max());assert error<2e-4,error
    return error

@torch.no_grad()
def timing(model,x,device,beta,original_gate=None):
    if device!='cuda':return {'measured':False,'reason':'CPU smoke excludes GPU timing'}
    def forward():
        p,indices=native_output(model,x,original_gate)
        return reweight(p,beta,indices)
    for _ in range(20):forward()
    torch.cuda.synchronize();baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();samples=[]
    for _ in range(50):
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();forward();b.record();b.synchronize();samples.append(a.elapsed_time(b))
    med=float(np.median(samples));return {'measured':True,'rows':len(x),'warmup':20,'repeats':50,'median_ms':med,'p95_ms':float(np.quantile(samples,.95)),
        'rows_per_second':1000*len(x)/med,'incremental_peak_allocated_bytes':torch.cuda.max_memory_allocated()-baseline,'samples_ms':samples,
        'scope':'Serial router+top4 only; shared GPU system, fixed state-vector Python kernels; not isolated system or end-to-end LM'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);args=ap.parse_args();cfg=read(ROOT/args.config)
    setup();root=ROOT/cfg['output_root'];root.mkdir(parents=True,exist_ok=True);group=root/canonical_hash(cfg)[:12];group.mkdir(exist_ok=False)
    new=ROOT/cfg['capacity_group'];old=ROOT/cfg['reference_group'];data=ROOT/cfg['data_group'];geo=ROOT/cfg['geometry_group']
    for g in [new,old,data]:
        for rel,h in read(g/'source_hashes.json').items():assert digest(ROOT/rel)==h and digest(g/'source_snapshot'/rel)==h
    sources={f'experiments/operator_geometry/{p}':digest(Path(__file__).with_name(p)) for p in ['round3_replay_v2.py','round3_capacity.py','round2_train.py','metrics.py']}
    write_json(group/'config.json',cfg);write_json(group/'source_hashes.json',sources)
    for rel in sources:
        dst=group/'source_snapshot'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dst)
    state={'state':'preparing','stage':'sparse_replay','pid':os.getpid(),'active':True,'group':str(group),'completed':0,'total':len(cfg['seeds'])*(len(cfg['new_methods'])+len(cfg['old_methods'])+1),'started':timestamp()}
    lock=ROOT/'outputs/operator_geometry_round2/PIPELINE_LOCK.json';acquired=False
    try:
        with os.fdopen(os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY),'w') as f:json.dump(state,f)
        acquired=True
        if cfg['device']=='cuda':torch.cuda.set_per_process_memory_fraction(.35)
        gpu=subprocess.run(['nvidia-smi'],capture_output=True,text=True,check=False);(group/'gpu_before.txt').write_text(gpu.stdout+gpu.stderr,encoding='utf8')
        rows=[];checks=[];times=[];dcfg=read(data/'config.json');lmcfg=read(ROOT/dcfg['source_config'])
        for seed in cfg['seeds']:
            mf=read(data/f's{seed}/result.json')
            for name,h in mf['files'].items():assert digest(data/f's{seed}'/name)==h
            ckpath=ROOT/dcfg['source_group']/f'runs/mlp_half_s{seed}/best.pt';assert digest(ckpath)==mf['checkpoint_sha256']
            lm=CausalLM(lmcfg,'mlp_half',seed).to(cfg['device']).eval();lm.load_state_dict(torch.load(ckpath,map_location='cpu',weights_only=False)['model']);lm.requires_grad_(False)
            lmhash=full_state_hash(lm.state_dict());gate=lm.get_submodule(dcfg['module']).mote.core.gate
            caches={split:torch.load(data/f's{seed}/{split}.pt',map_location='cpu',weights_only=True) for split in ['dev','validation']}
            if cfg.get('smoke'):caches={s:{k:v[:32] for k,v in c.items()} for s,c in caches.items()}
            toks={s:np.load(ROOT/dcfg['data_root']/('train.npy' if s=='dev' else 'validation.npy')) for s in caches}
            geometry=read(geo/f's{seed}/assignment.json');assignment=np.asarray(geometry['expert_for_basis']);perm=np.asarray(geometry['random_expert_for_basis']);codes=np.load(geo/f's{seed}/codes.npy')
            for method in ['original']+cfg['old_methods']+cfg['new_methods']:
                assert all(digest(ROOT/rel)==h for rel,h in sources.items())
                state.update(state='running',current=f'{method}_s{seed}');write_json(root/'STATUS.json',state)
                if method=='original':model=None;beta=1.;check={'seed':seed,'method':method,'checkpoint_sha256':mf['checkpoint_sha256']};og=gate
                else:
                    model,beta,check=verify_run(new if method in cfg['new_methods'] else old,seed,method,assignment,codes,perm,method in cfg['new_methods']);model.to(cfg['device']);og=None;check.update(seed=seed,method=method)
                run=group/f'{method}_s{seed}';run.mkdir()
                for mode in (['native'] if method=='original' else ['native','calibrated']):
                    power=beta if mode=='calibrated' else 1.
                    for split,c in caches.items():
                        p,indices=predict_output(model,c['x'],cfg['device'],og);w=reweight(p,power,indices);window=c['window'].numpy()
                        if method=='original':torch.testing.assert_close(w,c['weights'],atol=3e-6,rtol=1e-4)
                        else:
                            saved=np.load((new if method in cfg['new_methods'] else old)/f'runs/{method}_s{seed}/{split}_predictions.npz')['cost'][:len(p)]
                            reproduced=-p.clamp_min(1e-12).log()*beta;reproduced-=reproduced.mean(1,keepdim=True)
                            err=float(abs(reproduced.numpy()-saved).max());assert err<5e-4;check[f'{split}_score_replay_max_abs']=err
                        h=c['hidden']+((w-c['weights'])[:,:,None]*c['outputs']).sum(1)
                        stats=compare_head(lm,c['hidden'],h,c['labels'])
                        arr={**stats,'weights':w.numpy(),'probabilities':p.numpy(),'window':window};np.savez_compressed(run/f'{split}_{mode}_clean.npz',**arr)
                        row={'seed':seed,'method':method,'split':split,'mode':mode,'power':power,'quality':quality_stats(stats,window,seed,c['baseline_nll'].numpy()),
                            'routing':route_stats(w,p),'top4_set_agreement_original':float(((w>0)==(c['weights']>0)).all(1).float().mean()),'roundtrip':{}}
                        if method!='original':check[f'{split}_{mode}_full_model_max_abs']=full_model_probe(lm,model,gate,c,toks[split],mode,beta,cfg['device'])
                        for dtype,name in [(torch.float16,'fp16_input_roundtrip'),(torch.bfloat16,'bf16_input_roundtrip')]:
                            pr,ir=predict_output(model,c['x'].to(dtype).float(),cfg['device'],og);wr=reweight(pr,power,ir)
                            hr=c['hidden']+((wr-c['weights'])[:,:,None]*c['outputs']).sum(1);pert=compare_head(lm,h,hr,c['labels']);flip=((wr>0)!=(w>0)).any(1).numpy()
                            np.savez_compressed(run/f'{split}_{mode}_{name}.npz',**pert,flip=flip,weights=wr.numpy(),window=window)
                            delta=pert['delta_nll'];row['roundtrip'][name]={'flip_count':int(flip.sum()),'flip_rate':float(flip.mean()),'flip_ci95_windows':bootstrap_window_mean(flip,window,seed),
                                'mean_delta_ce':float(delta.mean()),'delta_ce_ci95_windows':bootstrap_window_mean(delta,window,seed),'harmful_flip_gt_01':int((flip&(delta>.1)).sum()),
                                'absolute_logprob_p99':float(np.quantile(abs(delta),.99)),'absolute_logprob_max':float(abs(delta).max()),'mean_kl':float(pert['kl'].mean()),'kl_p99':float(np.quantile(pert['kl'],.99))}
                        rows.append(row)
                    for size in cfg['timing_rows']:
                        x=caches['dev']['x'][:size].to(cfg['device']);times.append({'seed':seed,'method':method,'mode':mode,**timing(model,x,cfg['device'],power,og)})
                check['files']={p.name:digest(p) for p in run.iterdir() if p.is_file()};write_json(run/'audit.json',check);checks.append(check)
                state['completed']=len(checks);write_json(root/'STATUS.json',state);print('REPLAY_DONE',seed,method,flush=True)
                if model is not None:del model
                if cfg['device']=='cuda':torch.cuda.empty_cache()
            assert full_state_hash(lm.state_dict())==lmhash and digest(ckpath)==mf['checkpoint_sha256'];del lm,caches
            if cfg['device']=='cuda':torch.cuda.empty_cache()
        write_json(group/'summary.json',{'rows':rows,'timings':times,'scope':cfg['scope']})
        write_json(group/'AUDIT.json',{'passed':True,'models_including_original':len(checks),'quality_rows':len(rows),'checks':checks,'missing':[],'failed':[],'frozen_backbones_unchanged':True})
        state.update(state='ready_for_review',active=False,current=None,finished=timestamp());write_json(root/'STATUS.json',state);write_json(group/'COMPLETE.json',state)
        gpu=subprocess.run(['nvidia-smi'],capture_output=True,text=True,check=False);(group/'gpu_after.txt').write_text(gpu.stdout+gpu.stderr,encoding='utf8')
    except BaseException:
        state.update(state='failed',active=False,error=traceback.format_exc(),finished=timestamp());write_json(root/'STATUS.json',state);raise
    finally:
        if acquired:lock.unlink()

if __name__=='__main__':main()
