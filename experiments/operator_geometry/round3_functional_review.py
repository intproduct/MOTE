"""CPU-only independent audit/statistics for the completed functional stage."""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from ..autoregressive_gate.data import digest,write_json
from ..routing_stability.model import full_state_hash,CausalLM
from ..routing_stability.runner import setup,timestamp
from .round3_replay_review import csvwrite,ci
from .round3_functional import gram_matrices,read,ROOT
from .round3_capacity import ReencodingPredictor
from .round2_train import UtilityPredictor

def run(group,out):
    out.mkdir(parents=True,exist_ok=False);cfg=read(group/'config.json');src=read(group/'source_hashes.json')
    for rel,h in src.items():assert digest(ROOT/rel)==h and digest(group/'source_snapshot'/rel)==h
    records=read(group/'raw_summary.json');assert len(records)==36;assert read(group/'AUDIT.json')['passed']
    seen=set();rows=[];raw={};samples={};initial={};geometry=[];checks=[];dcfg=read(ROOT/cfg['data_group']/'config.json')
    for seed in cfg['seeds']:
        lm=CausalLM(read(ROOT/dcfg['source_config']),'mlp_half',seed).eval();lm.load_state_dict(torch.load(ROOT/dcfg['source_group']/f'runs/mlp_half_s{seed}/best.pt',map_location='cpu',weights_only=False)['model']);lm.requires_grad_(False)
        data=torch.load(ROOT/cfg['data_group']/f's{seed}/train.pt',map_location='cpu',weights_only=True)
        gm=torch.load(group/f'geometry_s{seed}.pt',map_location='cpu',weights_only=True)
        gmf=next(x for x in read(group/'GEOMETRY_CHECKS.json') if x['seed']==seed);assert digest(group/f'geometry_s{seed}.pt')==gmf['sha256']
        chosen=torch.tensor([0,23,521,3045,15673,32767]);outg,fg=gram_matrices(lm,data['hidden'][chosen],data['outputs'][chosen])
        for name,g in [('output',outg),('fisher',fg)]:
            assert gm[name].shape==(32768,16,16) and torch.isfinite(gm[name]).all()
            diff=float(((g-gm[name][chosen]).abs()/(1+g.abs())).max());assert diff<5e-4
            eigen=torch.linalg.eigvalsh(gm[name][chosen].double());assert float(eigen.min())>-1e-4
            geometry.append({'seed':seed,'metric':name,'independent_cpu_replay_relative_error':diff,'minimum_sample_eigenvalue':float(eigen.min())})
        del data,lm
        geom=read(ROOT/cfg['geometry_group']/f's{seed}/assignment.json');assignment=np.asarray(geom['expert_for_basis']);perm=np.asarray(geom['random_expert_for_basis']);codes=np.load(ROOT/cfg['geometry_group']/f's{seed}/codes.npy')
        for r in [x for x in records if x['seed']==seed]:
            method,loss=r['method'],r['objective'];key=(seed,method,loss);assert key not in seen;seen.add(key);run=group/'runs'/f'{method}_{loss}_s{seed}'
            for name,h in r['files'].items():assert digest(run/name)==h
            prior=ROOT/cfg['reference_group' if method=='mlp' else 'capacity_group']/f'runs/{method}_s{seed}/best_dev.pt'
            assert digest(prior)==r['prior']['checkpoint_sha256'];pck=torch.load(prior,map_location='cpu',weights_only=False)
            assert full_state_hash(pck['model'])==r['initial_state_hash'];initial.setdefault((seed,method),set()).add(r['initial_state_hash'])
            ck=torch.load(run/'best_dev.pt',map_location='cpu',weights_only=False);curves=[json.loads(line) for line in (run/'curves.jsonl').read_text().splitlines()]
            assert [c['step'] for c in curves]==list(range(0,2001,200)) and min(curves,key=lambda c:c['dev_exact_kl'])['step']==ck['step']==r['selected_step']
            assert ck['beta']==r['beta_fixed'] and r['parameter_changed']
            assert all(np.isfinite(c['gradient_norm']) and c['gradient_norm']>0 for c in curves[1:])
            last=torch.load(run/'last.pt',map_location='cpu',weights_only=False);assert full_state_hash(last['model'])!=r['initial_state_hash']
            np.testing.assert_array_equal(np.load(run/'sample_indices.npy'),torch.randint(32768,(2000,256),generator=torch.Generator().manual_seed(seed+980000)).numpy().flatten())
            samples.setdefault(seed,set()).add(digest(run/'sample_indices.npy'))
            norm=1. if loss=='coefficient' else float(gm[loss].diagonal(dim1=1,dim2=2).mean());assert abs(norm-r['normalizer_train_only'])<max(1e-6,norm*2e-6)
            model=UtilityPredictor(1024,8,'mlp',assignment,codes,perm) if method=='mlp' else ReencodingPredictor(1024,8,method,assignment)
            model.load_state_dict(ck['model']);assert sum(p.numel() for p in model.parameters())==r['parameters_total'];assert r['parameters_updated']==r['parameters_total']-1
            for split in ['dev','validation']:
                a=dict(np.load(run/f'{split}_clean.npz'));raw[(seed,method,loss,split)]=a
                for name,v in a.items():assert np.isfinite(v).all()
                assert a['weights'].shape==(4096,16);assert ((a['weights']>0).sum(1)==4).all();np.testing.assert_allclose(a['weights'].sum(1),1,atol=3e-7)
                row=next(x for x in r['rows'] if x['split']==split);q=row['quality'];assert abs(q['mean_kl']-a['kl'].mean())<1e-12 and abs(q['mean_delta_ce']-a['delta_nll'].mean())<1e-12
                record={'seed':seed,'method':method,'objective':loss,'split':split,'selected_step':r['selected_step'],'initial_dev_kl':r['initial_dev_kl'],'parameters':r['parameters_total'],
                        'seconds':r['seconds_including_checks'],'delta_ce':q['mean_delta_ce'],'sampled_ce':q['sampled_ce'],'sampled_ppl':q['sampled_ppl'],'kl':q['mean_kl'],'kl_p99':q['kl_p99'],
                        'logprob_error_p99':q['absolute_target_logprob_error_p99'],'top4_agreement':row['top4_set_agreement_original'],**{k:row['routing'][k] for k in ['load_cv','unused_experts','unique_top4_sets']}}
                for dtype in ['fp16','bf16']:
                    z=dict(np.load(run/f'{split}_{dtype}.npz'));raw[(seed,method,loss,split,dtype)]=z
                    for v in z.values():assert np.isfinite(v).all()
                    np.testing.assert_array_equal(z['flip'],((a['weights']>0)!=(z['weights']>0)).any(1))
                    record[dtype+'_flips']=int(z['flip'].sum());record[dtype+'_harmful']=int((z['flip']&(z['delta_nll']>.1)).sum())
                rows.append(record)
            checks.append({'seed':seed,'method':method,'objective':loss,'selected_step':ck['step'],'files_verified':len(r['files']),**r['checks']});del model,pck,ck,last
        del gm
    assert all(len(s)==1 for s in samples.values()) and all(len(s)==1 for s in initial.values())
    paired=[];reference=ROOT/'outputs/operator_geometry_round3/replay_v2/ff4f146e8d29';oldsummary=read(reference/'summary.json')
    for seed in cfg['seeds']:
        for split in ['dev','validation']:
            for method in cfg['methods']:
                base=dict(np.load(reference/f'{method}_s{seed}/{split}_calibrated_clean.npz'))
                raw[(seed,method,'initial',split)]=base
                if split=='dev':
                    for r in records:
                        if r['seed']==seed and r['method']==method:assert abs(r['initial_dev_kl']-base['kl'].mean())<1e-7
            pairs=[(m,l,m,'initial') for m in cfg['methods'] for l in cfg['objectives']]
            pairs += [(m,l,m,'coefficient') for m in cfg['methods'] for l in ['output','fisher']]
            pairs += [(m,'fisher',m,'output') for m in cfg['methods']]
            pairs += [('reencode_cz',l,b,l) for l in cfg['objectives'] for b in ['reencode_product','mlp','affine_born']]
            for m,l,b,bl in pairs:
                a=raw[(seed,m,l,split)];z=raw[(seed,b,bl,split)];np.testing.assert_array_equal(a['window'],z['window'])
                metrics={'delta_ce':a['delta_nll']-z['delta_nll'],'delta_kl':a['kl']-z['kl']}
                if bl!='initial':
                    for dtype in ['fp16','bf16']:
                        ar=raw[(seed,m,l,split,dtype)];br=raw[(seed,b,bl,split,dtype)]
                        metrics[dtype+'_flip_rate']=ar['flip'].astype(float)-br['flip'].astype(float)
                        metrics[dtype+'_harmful_rate']=(ar['flip']&(ar['delta_nll']>.1)).astype(float)-(br['flip']&(br['delta_nll']>.1)).astype(float)
                for metric,v in metrics.items():
                    lo,hi=ci(v,a['window'],seed);paired.append({'seed':seed,'split':split,'method':m,'objective':l,'reference':b,'reference_objective':bl,'metric':metric,'difference':float(v.mean()),'ci95_low':lo,'ci95_high':hi})
    csvwrite(out/'quality.csv',rows);csvwrite(out/'paired.csv',paired);write_json(out/'CHECKPOINT_CHECKS.json',checks);write_json(out/'GEOMETRY_REPLAY.json',geometry)
    result={'passed':True,'runs':len(records),'quality_rows':len(rows),'roundtrip_records':4*len(records),'paired_rows':len(paired),'all_same_samples_per_seed':True,'all_same_start_per_method_seed':True,
            'common_dev_kl_checkpoint_selection':True,'no_missing_or_failed':True,'completed':timestamp(),'group':str(group),'source_hashes_verified':len(src),'checks_sha256':digest(out/'CHECKPOINT_CHECKS.json')}
    write_json(out/'AUDIT.json',result);write_json(out/'SUMMARY.json',{'quality':rows,'paired':paired,'audit':result});print(json.dumps(result),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--wait',action='store_true');args=ap.parse_args();setup()
    root=ROOT/'outputs/operator_geometry_round3';status=root/'functional/STATUS.json';queue=root/'FUNCTIONAL_REVIEW_QUEUE.json'
    start=time.monotonic();write_json(queue,{'active':True,'state':'waiting','started':timestamp()})
    try:
        while True:
            s=read(status)
            if s['state']=='ready_for_review':break
            if s['state']=='failed':raise RuntimeError(s.get('error','functional training failed'))
            if not args.wait:raise RuntimeError('functional stage not complete')
            if time.monotonic()-start>7200:raise TimeoutError('CPU review wait exceeded2h')
            time.sleep(30)
        write_json(queue,{'active':True,'state':'auditing','started':timestamp()});run(Path(s['group']),root/'reviews/functional_v1')
        write_json(queue,{'active':False,'state':'ready_for_review','finished':timestamp(),'output':str(root/'reviews/functional_v1')})
    except BaseException as e:
        write_json(queue,{'active':False,'state':'failed','error':repr(e),'finished':timestamp()});raise

if __name__=='__main__':main()
