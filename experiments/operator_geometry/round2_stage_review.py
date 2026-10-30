"""Independent CPU stage review; preserves all frozen experiment artifacts."""
from pathlib import Path
import csv
import hashlib
import json
from datetime import datetime
import numpy as np
import torch
from .round2_numeric import row_spearman

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/operator_geometry_round2'
OUT = BASE / 'reviews/stage_20261030_v1'

def read(p): return json.loads(p.read_text(encoding='utf8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(name,data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def table(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def groups(v,window):
    _,ix=np.unique(window,return_inverse=True)
    return np.bincount(ix,weights=v)/np.bincount(ix)
def interval(values,seed=801):
    values=np.asarray(values,dtype=float);rng=np.random.default_rng(seed)
    boot=np.array([values[rng.integers(len(values),size=len(values))].mean() for _ in range(2000)])
    return [float(x) for x in np.quantile(boot,[.025,.975])]
def mean_sd(v):return {'mean':float(np.mean(v)),'sd':float(np.std(v,ddof=1))}

def main():
    torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=False)
    ug=BASE/'utility/3a0386d9a187';pg=BASE/'positive/d4ee19e23787';rg=BASE/'risk/3a17e2fefbb6'
    dg=BASE/'data/86a2ba9f5aa2';eg=BASE/'energy/decomposition_v1';gg=BASE/'geometry/input_geometry_v1'
    utility=read(ug/'raw_summary.json');positive=read(pg/'raw_summary.json');risks=read(rg/'summary.json')['rows']
    ridge=read(BASE/'ridge/01c1341f7586/raw_summary.json');geometry=read(gg/'raw_summary.json')
    boundary=read(BASE/'boundary_gradient/input_sensitivity_v1/raw_summary.json')
    assert len(utility)==27 and len(positive)==9 and len(risks)==6 and len(ridge)==9 and len(geometry)==9 and len(boundary)==24
    manifests=[]
    for path in [ug,pg,rg,dg,eg,gg,BASE/'ridge/01c1341f7586',BASE/'boundary_gradient/input_sensitivity_v1']:
        hashes=read(path/'source_hashes.json')
        for relative,expected in hashes.items():
            actual=sha(ROOT/relative);assert actual==expected,(path,relative)
            snap=path/'source_snapshot'/relative
            if snap.exists():assert sha(snap)==expected
        manifests.append({'group':str(path.relative_to(ROOT)),'source_files_verified':len(hashes)})
    audits=[]
    for name in ['utility_action_v1/UTILITY_AUDIT.json','utility_prediction_v1/PREDICTION_AUDIT.json','positive_formal_v1/PREDICTION_AUDIT.json','controls_v1/AUDIT.json']:
        p=BASE/'reviews'/name;a=read(p);assert a['passed'];audits.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p),'passed':True})
    metric_rows=[];paired=[];risk_checks=[];baselines=[];positive_paired=[]
    for seed in [17,29,43]:
        manifest=read(rg/f's{seed}/audit.json');assert manifest['frozen_backbone_unchanged']
        checkpoint=ROOT/f'outputs/routing_stability_round1/formal/756b04397b2d/runs/mlp_half_s{seed}/best.pt'
        assert sha(checkpoint)==manifest['checkpoint_sha256']
        for split in ['dev','validation']:
            path=rg/f's{seed}/{split}_risk.npz';assert sha(path)==manifest['files'][path.name]
            risk=dict(np.load(path));exact=dict(np.load(dg/f's{seed}/{split}_exact.npz'))
            assert risk['kl'].shape==(4096,48)
            assert all(np.isfinite(a).all() for a in risk.values())
            for key in ['source','target','window']:np.testing.assert_array_equal(risk[key],exact[key])
            np.testing.assert_allclose(risk['delta_nll'],exact['delta_nll'],atol=2e-4,rtol=1e-4)
            row=next(x for x in risks if x['seed']==seed and x['split']==split)
            rho=lambda x:row_spearman(x.reshape(-1,12),risk['kl'].reshape(-1,12)).reshape(4096,4).mean(1)
            fs=rho(risk['fisher_kl']);np.testing.assert_allclose(fs.mean(),row['rho_fisher_kl_fixed_source'],atol=1e-12)
            np.testing.assert_allclose(risk['kl'].mean(),row['mean_kl'])
            proxies={};conditions={}
            for gauge in ['raw','centered']:
                p=dict(np.load(eg/f's{seed}_{split}_{gauge}_proxies.npz'))
                for name in ['local_full_distance','local_energy_only','global_full_distance','global_energy_only']:
                    v=rho(p[name]);np.testing.assert_allclose(v.mean(),row['energy_controls'][gauge][name]['rho_kl_fixed_source'],atol=1e-12)
                    proxies[f'{gauge}_{name}']=v
                for local in ['local','global']:
                    conditions[f'{gauge}_{local}_distance_minus_energy']=proxies[f'{gauge}_{local}_full_distance']-proxies[f'{gauge}_{local}_energy_only']
            pred=dict(np.load(gg/f's{seed}/rank8_{split}.npz'))
            for name in ['predicted_euclidean','unanchored_ray','anchored_ray_x4','train_global_euclidean']:
                v=rho(pred[name]);proxies[name]=v
                np.testing.assert_allclose(v.mean(),row['state_geometry']['8'][name]['rho_kl_fixed_source'],atol=1e-12)
            conditions['ray_minus_euclidean']=proxies['unanchored_ray']-proxies['predicted_euclidean']
            conditions['anchor_x4_minus_euclidean']=proxies['anchored_ray_x4']-proxies['predicted_euclidean']
            conditions['input_geometry_minus_train_global']=proxies['predicted_euclidean']-proxies['train_global_euclidean']
            conditions['fisher_minus_local_distance']=fs-proxies['raw_local_full_distance']
            for name,v in conditions.items():
                ci=interval(groups(v,exact['window']),seed)
                paired.append({'seed':seed,'split':split,'comparison':name,'paired_rho_difference':float(v.mean()),'ci95_low':ci[0],'ci95_high':ci[1]})
            error=abs(risk['fisher_kl']-risk['kl']);quant=np.quantile(error,[.5,.9,.99,1])
            mr={'seed':seed,'split':split,'mean_kl':float(risk['kl'].mean()),'mean_fisher':float(risk['fisher_kl'].mean()),'fisher_rmse':float(np.sqrt(np.mean(error**2))),
                'fisher_abs_error_p50':float(quant[0]),'fisher_abs_error_p90':float(quant[1]),'fisher_abs_error_p99':float(quant[2]),'fisher_abs_error_max':float(quant[3]),'fisher_rho':float(fs.mean())}
            mr.update({k:float(v.mean()) for k,v in proxies.items()});metric_rows.append(mr)
            risk_checks.append({'seed':seed,'split':split,'sha256':sha(path),'replacements':risk['kl'].size,'finite':True,'summary_recomputed':True,
                'nll_replay_max_abs':float(abs(risk['delta_nll']-exact['delta_nll']).max())})
            cache=torch.load(dg/f's{seed}/{split}.pt',map_location='cpu',weights_only=True)
            nll=cache['baseline_nll'].numpy();ce=float(nll.mean());ci=interval(groups(nll,exact['window']),seed)
            baselines.append({'seed':seed,'split':split,'tokens':len(nll),'sampled_ce':ce,'sampled_bpe_ppl':float(np.exp(ce)),'ce_ci95_low':ci[0],'ce_ci95_high':ci[1]})
            target=-cache['teacher'].clamp_min(1e-12).log();target=(target-target.mean(-1,keepdim=True)).numpy()
            ref=np.load(pg/f'runs/mlp_s{seed}/{split}_predictions.npz')['cost'];referr=np.square(ref-target).mean(1)
            refset=np.sort(ref.argsort(1)[:,:4],1);tset=np.sort(target.argsort(1)[:,:4],1)
            refmatch=(refset==tset).all(1)
            base=next(p for p in positive if p['method']=='mlp' and p['seed']==seed)['splits'][split]['constant_mse']
            for method in ['angle_functional','product_functional']:
                prediction=np.load(pg/f'runs/{method}_s{seed}/{split}_predictions.npz')['cost']
                err=np.square(prediction-target).mean(1);match=(np.sort(prediction.argsort(1)[:,:4],1)==tset).all(1)
                for name,v in [('r2_difference',(referr-err)/base),('top4_agreement_difference',match.astype(float)-refmatch)]:
                    ci=interval(groups(v,exact['window']),seed)
                    positive_paired.append({'seed':seed,'split':split,'method':method,'reference':'mlp','metric':name,'paired_difference':float(v.mean()),'ci95_low':ci[0],'ci95_high':ci[1]})
            del cache
        print('REVIEW_SEED',seed,flush=True)
    urows=[];prows=[];grows=[];brows=[]
    for study,items in [('utility',utility),('ridge',ridge)]:
        for r in items:
            for split,d in r['splits'].items():
                a=d['action'];b=next(x for x in baselines if x['seed']==r['seed'] and x['split']==split)
                urows.append({'study':study,'method':r['method'],'seed':r['seed'],'split':split,'parameters':r.get('parameters',r.get('rank_factor_parameter_count')),
                    'selected_step':r.get('selected_step','closed_form'),'r2':d['prediction']['r2_vs_train_mean'],'changed':a['changed'],'delta_ce':a['mean_delta_ce'],
                    'ci95_low':a['delta_ce_ci95_windows'][0],'ci95_high':a['delta_ce_ci95_windows'][1],'regret':a['regret_to_label_oracle'],'harm_gt_01':a['harm_gt_01'],
                    'sampled_ce':b['sampled_ce']+a['mean_delta_ce'],'sampled_bpe_ppl':float(np.exp(b['sampled_ce']+a['mean_delta_ce'])),
                    'shared_gpu_seconds':r.get('seconds_shared_gpu','not_measured')})
    for r in positive:
        for split,d in r['splits'].items():prows.append({'method':r['method'],'seed':r['seed'],'split':split,'parameters':r['parameters'],'selected_step':r['selected_step'],'r2':d['r2_vs_constant'],'top4_agreement':d['top4_set_agreement'],'shared_gpu_seconds':r['seconds_shared_gpu']})
    for r in geometry:
        for split,d in r['splits'].items():grows.append({'seed':r['seed'],'rank':r['rank'],'split':split,'parameters':r['parameters_factorized'],'sketch_r2':d['sketch_r2_vs_constant'],**{k:v['within_token_fixed_source_rho_vs_true_output_distance'] for k,v in d['geometry'].items()}})
    for r in boundary:
        for name,c in r['conditions'].items():brows.append({'seed':r['seed'],'method':r['method'],'split':r['split'],'condition':name,'tokens':r['tokens'],'hamming_one_fraction':r['hamming_one_fraction'],**{k:c[k] for k in ['set_flips','pair_crossings','predicted_pair_crossings','prediction_true_positive','prediction_false_positive','prediction_false_negative','set_flips_without_pair_crossing']}})
    table('utility_and_ridge.csv',urows);table('positive_control.csv',prows);table('risk_metrics.csv',metric_rows);table('risk_paired.csv',paired)
    table('positive_paired.csv',positive_paired);table('geometry.csv',grows);table('boundary.csv',brows);table('sampled_ce_ppl.csv',baselines)
    table('utility_paired_vs_mlp.csv',read(BASE/'reviews/utility_action_v1/UTILITY_AUDIT.json')['paired_against_mlp'])
    aggregates={}
    for method in [r['method'] for r in positive if r['seed']==17]:
        rows=[r for r in prows if r['method']==method and r['split']=='validation']
        aggregates[method]={k:mean_sd([r[k] for r in rows]) for k in ['r2','top4_agreement','shared_gpu_seconds']}
    write('SUMMARY.json',{'positive_validation':aggregates,'utility_all_dev_r2_negative':all(r['splits']['dev']['prediction']['r2_vs_train_mean']<0 for r in utility),
        'risk':metric_rows,'risk_paired':paired,'sampled_baselines':baselines,'utility':urows,'positive':prows,'geometry':grows,'boundary':brows,
        'bootstrap':'2000 paired window resamples per seed; four positions and four removed sources kept within window; descriptive intervals, no multiplicity correction; fixed train-mean denominator for positive R2 differences.'})
    write('AUDIT.json',{'passed':True,'finished':datetime.now().astimezone().isoformat(),'counts':{'utility':27,'positive':9,'ridge':9,'geometry':9,'risk_checkpoints':3,'risk_replacements':sum(x['replacements'] for x in risk_checks),'boundary_replays':24},
        'missing':[],'failed':[],'source_checks':manifests,'prior_audits':audits,'risk_checks':risk_checks,
        'not_performed':['conditional readout adaptation','isolated deployment timing','full-validation or test CE/PPL evaluation of new routers','end-to-end backbone retraining'],
        'scope':'Stage completion only. All calculations CPU; no training, no changes to prior results. Stage budget may be extended after network interruption per user instruction.'})
    print('STAGE_REVIEW_PASSED',OUT,flush=True)

if __name__=='__main__':main()
