"""Independent raw-array integrity and paired statistics for corrected replay."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
from ..autoregressive_gate.data import digest,write_json

def read(p):return json.loads(p.read_text(encoding='utf8'))
def csvwrite(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def ci(v,win,seed):
    groups,inv=np.unique(win,return_inverse=True);m=np.bincount(inv,weights=v)/np.bincount(inv)
    rng=np.random.default_rng(seed);z=m[rng.integers(len(m),size=(2000,len(m)))].mean(1)
    return [float(x) for x in np.quantile(z,[.025,.975])]

def main():
    base=Path('outputs/operator_geometry_round3');g=base/'replay_v2/ff4f146e8d29';out=base/'reviews/sparse_replay_v2';out.mkdir(parents=True,exist_ok=False)
    audit=read(g/'AUDIT.json');s=read(g/'summary.json');assert audit['passed'] and len(audit['checks'])==21 and len(s['rows'])==78 and len(s['timings'])==78
    for rel,h in read(g/'source_hashes.json').items():assert digest(Path(rel))==h and digest(g/'source_snapshot'/rel)==h
    verified=0
    for a in audit['checks']:
        run=g/f"{a['method']}_s{a['seed']}"
        for name,h in a['files'].items():assert digest(run/name)==h;verified+=1
    rows=[];rounds=[];raw={}
    for r in s['rows']:
        seed,m,mode,split=[r[k] for k in ['seed','method','mode','split']];run=g/f'{m}_s{seed}'
        a=dict(np.load(run/f'{split}_{mode}_clean.npz'));raw[(seed,m,mode,split)]=a
        for k,v in a.items():assert np.isfinite(v).all(),(run,k)
        assert a['weights'].shape==(4096,16) and ((a['weights']>0).sum(1)==4).all()
        np.testing.assert_allclose(a['weights'].sum(1),1,atol=2e-7)
        q=r['quality'];assert abs(a['delta_nll'].mean()-q['mean_delta_ce'])<1e-12 and abs(a['kl'].mean()-q['mean_kl'])<1e-12
        row={k:r[k] for k in ['seed','method','mode','split','power','top4_set_agreement_original']}
        row.update({k:q[k] for k in ['mean_delta_ce','sampled_ce','sampled_ppl','mean_kl','kl_p99','absolute_target_logprob_error_p99','absolute_target_logprob_error_max']})
        row.update({k:r['routing'][k] for k in ['load_cv','unused_experts','load_entropy','unique_top4_sets']});rows.append(row)
        for dtype in ['fp16','bf16']:
            z=dict(np.load(run/f'{split}_{mode}_{dtype}_input_roundtrip.npz'));raw[(seed,m,mode,split,dtype)]=z
            for k,v in z.items():assert np.isfinite(v).all(),(run,k)
            flip=((z['weights']>0)!=(a['weights']>0)).any(1);np.testing.assert_array_equal(flip,z['flip'])
            rr=r['roundtrip'][dtype+'_input_roundtrip'];assert int(flip.sum())==rr['flip_count']
            assert int((flip&(z['delta_nll']>.1)).sum())==rr['harmful_flip_gt_01']
            rounds.append({'seed':seed,'method':m,'mode':mode,'split':split,'dtype':dtype,**{k:v for k,v in rr.items() if not isinstance(v,list)}})
    pairs=[('reencode_cz','angle_functional'),('reencode_cz','reencode_product'),('affine_born','mlp'),('reencode_cz','mlp'),('reencode_cz','affine_born')];paired=[]
    for seed in [17,29,43]:
        for split in ['dev','validation']:
            for mode in ['native','calibrated']:
                for m,b in pairs:
                    a=raw[(seed,m,mode,split)];z=raw[(seed,b,mode,split)];np.testing.assert_array_equal(a['window'],z['window'])
                    metrics={'delta_ce':a['delta_nll']-z['delta_nll'],'delta_kl':a['kl']-z['kl']}
                    for dtype in ['fp16','bf16']:
                        ar=raw[(seed,m,mode,split,dtype)];br=raw[(seed,b,mode,split,dtype)]
                        metrics[dtype+'_flip_rate']=ar['flip'].astype(float)-br['flip'].astype(float)
                        metrics[dtype+'_harmful_flip_rate']=(ar['flip']&(ar['delta_nll']>.1)).astype(float)-(br['flip']&(br['delta_nll']>.1)).astype(float)
                    for metric,v in metrics.items():
                        low,high=ci(v,a['window'],seed)
                        paired.append({'seed':seed,'split':split,'mode':mode,'method':m,'reference':b,'metric':metric,'difference':float(v.mean()),'ci95_low':low,'ci95_high':high})
            for m in ['mlp','angle_functional','product_functional','reencode_cz','reencode_product','affine_born']:
                a=raw[(seed,m,'calibrated',split)];b=raw[(seed,m,'native',split)]
                assert ((a['weights']>0)==(b['weights']>0)).all()
                for metric,k in [('calibration_delta_ce','delta_nll'),('calibration_delta_kl','kl')]:
                    v=a[k]-b[k];lo,hi=ci(v,a['window'],seed);paired.append({'seed':seed,'split':split,'mode':'calibrated_minus_native','method':m,'reference':m,'metric':metric,'difference':float(v.mean()),'ci95_low':lo,'ci95_high':hi})
    means=[]
    for split in ['dev','validation']:
        for mode in ['native','calibrated']:
            for m in ['original','mlp','angle_functional','product_functional','reencode_cz','reencode_product','affine_born']:
                block=[r for r in rows if r['split']==split and r['mode']==mode and r['method']==m]
                if not block:continue
                assert len(block)==3
                for metric in ['mean_delta_ce','mean_kl','sampled_ce','sampled_ppl','top4_set_agreement_original','load_cv','unique_top4_sets']:
                    v=np.array([r[metric] for r in block]);means.append({'split':split,'mode':mode,'method':m,'metric':metric,'mean':float(v.mean()),'seed_sd':float(v.std(ddof=1))})
    csvwrite(out/'quality.csv',rows);csvwrite(out/'roundtrip.csv',rounds);csvwrite(out/'paired.csv',paired);csvwrite(out/'mean_sd.csv',means)
    times=[{k:v for k,v in t.items() if k!='samples_ms'} for t in s['timings']];csvwrite(out/'timing.csv',times)
    checks={'passed':True,'models':21,'quality_rows':len(rows),'roundtrip_records':len(rounds),'timings':len(times),'paired_rows':len(paired),'raw_files_verified':verified,
            'all_calibrations_keep_top4':True,'full_model_shortcut_max_abs':max(v for a in audit['checks'] for k,v in a.items() if 'full_model_max_abs' in k),
            'missing':[],'failed':[],'source_audit_sha256':digest(g/'AUDIT.json'),'scope':'Exploratory 2000-resample paired window intervals, conditional on3frozen seeds, no multiplicity correction. Sampled final-module replay; no test data.'}
    write_json(out/'AUDIT.json',checks);write_json(out/'SUMMARY.json',{'means':means,'paired':paired,'audit':checks});print(json.dumps(checks),flush=True)

if __name__=='__main__':main()
