"""Independent CPU audit and predefined full-versus-diagonal comparisons."""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from ..autoregressive_gate.data import digest,write_json
from ..routing_stability.runner import setup,timestamp
from ..routing_stability.model import full_state_hash
from .round3_functional import ROOT,read,weights
from .round3_capacity import ReencodingPredictor
from .round2_train import UtilityPredictor
from .round3_replay_review import csvwrite,ci

def run(group,out):
    cfg=read(group/'config.json');records=read(group/'raw_summary.json');assert len(records)==24
    assert read(group/'AUDIT.json')['passed']
    full=ROOT/cfg['functional_group']; assert read(ROOT/'outputs/operator_geometry_round3/reviews/functional_v1/AUDIT.json')['passed']
    out.mkdir(parents=True,exist_ok=False);checks=[];rows=[];paired=[];seen=set()
    for rel,h in read(group/'source_hashes.json').items():assert digest(ROOT/rel)==digest(group/'source_snapshot'/rel)==h
    for seed in cfg['seeds']:
        gmpath=full/f'geometry_s{seed}.pt';gm=torch.load(gmpath,map_location='cpu',weights_only=True)
        assert digest(gmpath)==next(x['sha256'] for x in read(group/'GEOMETRY_CHECKS.json') if x['seed']==seed)
        geo=read(ROOT/cfg['geometry_group']/f's{seed}/assignment.json');assignment=np.asarray(geo['expert_for_basis']);perm=np.asarray(geo['random_expert_for_basis']);codes=np.load(ROOT/cfg['geometry_group']/f's{seed}/codes.npy')
        caches={s:torch.load(ROOT/cfg['data_group']/f's{seed}/{s}.pt',map_location='cpu',weights_only=True) for s in ['dev','validation']}
        for r in [x for x in records if x['seed']==seed]:
            m,l=r['method'],r['objective'];base_loss=l.removesuffix('_diag');key=(seed,m,l);assert key not in seen;seen.add(key)
            run=group/'runs'/f'{m}_{l}_s{seed}';fr=full/'runs'/f'{m}_{base_loss}_s{seed}';frec=read(fr/'result.json')
            for name,h in r['files'].items():assert digest(run/name)==h
            prior=ROOT/cfg['reference_group' if m=='mlp' else 'capacity_group']/f'runs/{m}_s{seed}/best_dev.pt'
            assert digest(prior)==r['prior']['checkpoint_sha256'];pck=torch.load(prior,map_location='cpu',weights_only=False)
            assert full_state_hash(pck['model'])==r['initial_state_hash']==frec['initial_state_hash']
            assert r['beta_fixed']==frec['beta_fixed'];assert r['parameter_changed']
            assert abs(r['normalizer_train_only']-frec['normalizer_train_only'])<2e-6*max(1.,r['normalizer_train_only'])
            assert abs(r['normalizer_train_only']-float(gm[base_loss].diagonal(dim1=1,dim2=2).mean()))<2e-6*max(1.,r['normalizer_train_only'])
            np.testing.assert_array_equal(np.load(run/'sample_indices.npy'),np.load(fr/'sample_indices.npy'))
            ck=torch.load(run/'best_dev.pt',map_location='cpu',weights_only=False);last=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
            curves=[json.loads(x) for x in (run/'curves.jsonl').read_text().splitlines()];assert [x['step'] for x in curves]==list(range(0,2001,200))
            assert min(curves,key=lambda x:x['dev_exact_kl'])['step']==ck['step']==r['selected_step'];assert full_state_hash(last['model'])!=r['initial_state_hash']
            assert all(np.isfinite(x['gradient_norm']) and x['gradient_norm']>0 for x in curves[1:])
            model=UtilityPredictor(1024,8,'mlp',assignment,codes,perm) if m=='mlp' else ReencodingPredictor(1024,8,m,assignment)
            model.load_state_dict(ck['model']);model.eval();assert sum(p.numel() for p in model.parameters())==r['parameters_total']==frec['parameters_total']
            for split,c in caches.items():
                a=dict(np.load(run/f'{split}_clean.npz'));np.testing.assert_array_equal(a['window'],c['window'].numpy())
                assert all(np.isfinite(v).all() for v in a.values());assert ((a['weights']>0).sum(1)==4).all()
                np.testing.assert_allclose(a['weights'].sum(1),1,atol=3e-7)
                with torch.no_grad():cpu=torch.cat([weights(model,c['x'][i:i+256],ck['beta']) for i in range(0,len(c['x']),256)]).numpy()
                same=((cpu>0)==(a['weights']>0)).all(1);err=float(abs(cpu[same]-a['weights'][same]).max());assert err<2e-4
                rr=next(x for x in r['rows'] if x['split']==split);q=rr['quality']
                assert abs(a['kl'].mean()-q['mean_kl'])<1e-12 and abs(a['delta_nll'].mean()-q['mean_delta_ce'])<1e-12
                rec={'seed':seed,'method':m,'objective':l,'split':split,'selected_step':r['selected_step'],'parameters':r['parameters_total'],'seconds':r['seconds_including_checks'],
                     'kl':q['mean_kl'],'delta_ce':q['mean_delta_ce'],'sampled_ce':q['sampled_ce'],'sampled_ppl':q['sampled_ppl'],'top4_agreement':rr['top4_set_agreement_original'],
                     'unused_experts':rr['routing']['unused_experts'],'load_cv':rr['routing']['load_cv'],'unique_top4_sets':rr['routing']['unique_top4_sets']}
                for dtype in ['fp16','bf16']:
                    ar=dict(np.load(run/f'{split}_{dtype}.npz'));assert all(np.isfinite(v).all() for v in ar.values());np.testing.assert_array_equal(ar['flip'],((ar['weights']>0)!=(a['weights']>0)).any(1))
                    rec[dtype+'_flips']=int(ar['flip'].sum());rec[dtype+'_harmful']=int((ar['flip']&(ar['delta_nll']>.1)).sum())
                rows.append(rec);checks.append({'seed':seed,'method':m,'objective':l,'split':split,'cpu_weight_error':err,'cpu_gpu_set_changes':int((~same).sum())})
                for reference_loss in [base_loss,'coefficient']:
                    b=dict(np.load(full/'runs'/f'{m}_{reference_loss}_s{seed}'/f'{split}_clean.npz'));np.testing.assert_array_equal(a['window'],b['window'])
                    # Full/coeff minus diagonal: negative means reference loss gives lower measured risk.
                    for metric,field in [('delta_kl','kl'),('delta_ce','delta_nll')]:
                        delta=b[field]-a[field];lo,hi=ci(delta,a['window'],seed)
                        paired.append({'seed':seed,'split':split,'method':m,'objective':reference_loss,'reference_objective':l,'metric':metric,'difference':float(delta.mean()),'ci95_low':lo,'ci95_high':hi})
            del ck,last,pck,model
        del gm,caches
    csvwrite(out/'quality.csv',rows);csvwrite(out/'paired.csv',paired);write_json(out/'CHECKPOINT_CHECKS.json',checks)
    audit={'passed':True,'runs':24,'cpu_checkpoint_split_replays':48,'quality_rows':48,'roundtrip_records':96,'paired_rows':len(paired),'same_warm_start_samples_and_trace_normalizer':True,'missing':[],'failed':[],'completed':timestamp()}
    write_json(out/'AUDIT.json',audit);write_json(out/'SUMMARY.json',{'quality':rows,'paired':paired,'audit':audit});print(json.dumps(audit),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--wait',action='store_true');args=ap.parse_args();setup();root=ROOT/'outputs/operator_geometry_round3';start=time.monotonic()
    queue=root/'DIAGONAL_REVIEW_QUEUE.json';write_json(queue,{'active':True,'state':'waiting','started':timestamp()})
    try:
        while True:
            s=read(root/'functional_diagonal/STATUS.json')
            if s['state']=='ready_for_review':break
            if s['state']=='failed':raise RuntimeError(s.get('error'))
            if not args.wait or time.monotonic()-start>3600:raise TimeoutError('diagonal review wait bound')
            time.sleep(30)
        write_json(queue,{'active':True,'state':'auditing','started':timestamp()});run(Path(s['group']),root/'reviews/functional_diagonal_v1')
        write_json(queue,{'active':False,'state':'ready_for_review','finished':timestamp()})
    except BaseException as e:write_json(queue,{'active':False,'state':'failed','error':repr(e),'finished':timestamp()});raise

if __name__=='__main__':main()
