"""CPU saved-checkpoint replay and independent review of the utility study."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from ..autoregressive_gate.data import digest,write_json
from .round2_train import UtilityPredictor,predict,calibrate_threshold,action_metrics
from .metrics import spearman,bootstrap_window_mean
from .round2_numeric import row_spearman


def load(path):return json.loads(path.read_text(encoding='utf8'))


def small_cache(path):
    c=torch.load(path,map_location='cpu',weights_only=True)
    return {k:c[k] for k in ['x','weights','utility_cost','window']}


def review(group,out):
    cfg=load(group/'config.json');repo=Path(__file__).resolve().parents[2]
    data=repo/cfg['data_group'];checks=[];comparison=[];missing=[]
    # This review is read-only except its new review directory.
    for path,sha in load(group/'source_hashes.json').items():
        assert digest(group/'source_snapshot'/path)==sha
        assert digest(repo/path)==sha
    all_samples={};data_windows={}
    for seed in cfg['seeds']:
        src=data/f's{seed}';manifest=load(src/'result.json')
        for name,sha in manifest['files'].items():assert digest(src/name)==sha
        caches={s:small_cache(src/f'{s}.pt') for s in ['dev','validation']}
        exacts={s:dict(np.load(src/f'{s}_exact.npz')) for s in caches}
        data_windows[seed]={s:caches[s]['window'].numpy() for s in caches}
        geometry=load(group/'geometry'/f's{seed}/assignment.json')
        assignment=np.asarray(geometry['expert_for_basis']);random=np.asarray(geometry['random_expert_for_basis'])
        codes=np.load(group/'geometry'/f's{seed}/codes.npy')
        per_method={}
        for method in cfg['methods']:
            run=group/'runs'/f'{method}_s{seed}'
            if not (run/'result.json').exists():missing.append(f'{method}_s{seed}');continue
            result=load(run/'result.json');manifest=load(run/'manifest.json')
            for name,sha in result['files'].items():assert digest(run/name)==sha
            sample_sha=digest(run/'sample_indices.npy')
            if seed in all_samples:assert all_samples[seed]==sample_sha
            else:all_samples[seed]=sample_sha
            curves=[json.loads(line) for line in (run/'curves.jsonl').read_text().splitlines()]
            assert len(curves)==cfg['steps']//cfg['eval_every']
            assert all(np.isfinite(c['train_loss_normalized']) and np.isfinite(c['gradient_norm']) and c['gradient_norm']>0 for c in curves)
            chosen=min(curves,key=lambda c:c['mse'])['step'];assert chosen==result['selected_step']
            checkpoint=torch.load(run/'best_dev.pt',map_location='cpu',weights_only=False)
            assert checkpoint['step']==chosen and checkpoint['method']==method and checkpoint['seed']==seed
            model=UtilityPredictor(caches['dev']['x'].shape[-1],cfg['rank'],method,assignment,codes,random)
            model.load_state_dict(checkpoint['model'],strict=True)
            assert all(torch.isfinite(t).all() for t in checkpoint['model'].values())
            assert sum(p.numel() for p in model.parameters())==result['parameters']==manifest['parameters']
            assert result['parameter_changed']
            row={'method':method,'seed':seed,'files_verified':len(result['files']),'selected_step':chosen,'replays':{}}
            per_method[method]={}
            for split in caches:
                saved=dict(np.load(run/f'{split}_predictions.npz'))
                replay=predict(model,caches[split],checkpoint['target_scale'],'cpu')
                error=float(abs(replay-saved['cost']).max())
                assert error<5e-4,(method,seed,split,error)
                if split=='dev':
                    threshold,cal=calibrate_threshold(saved['cost'],caches[split],exacts[split],cfg['thresholds'])
                    assert threshold==result['threshold']
                metrics,delta,chosen_idx,changed=action_metrics(saved['cost'],caches[split],exacts[split],result['threshold'],seed+42)
                np.testing.assert_array_equal(changed,saved['changed']);np.testing.assert_array_equal(chosen_idx,saved['chosen_index'])
                np.testing.assert_allclose(delta,saved['action_delta_nll'],atol=0,rtol=0)
                assert abs(metrics['mean_delta_ce']-result['splits'][split]['action']['mean_delta_ce'])<1e-10
                row['replays'][split]={'cpu_vs_saved_cost_max_abs':error,'action_recomputed':True,'tokens':len(delta)}
                per_method[method][split]=delta
            checks.append(row)
        for method in cfg['methods']:
            if method not in per_method or 'mlp' not in per_method:continue
            for split in caches:
                delta=per_method[method][split]-per_method['mlp'][split]
                comparison.append({'method':method,'seed':seed,'split':split,'reference':'mlp',
                                   'paired_delta_ce':float(delta.mean()),'ci95_windows':bootstrap_window_mean(delta,data_windows[seed][split],seed+1024)})
    expected=len(cfg['methods'])*len(cfg['seeds'])
    result={'passed':len(checks)==expected and not missing,'expected_runs':expected,'verified_runs':len(checks),
            'missing':missing,'checks':checks,'same_samples_per_seed':all_samples,'paired_against_mlp':comparison,
            'scope':'CPU replay independently verifies saved GPU predictions and exact action accounting; CPU/GPU tolerance5e-4. No additional training or validation selection.'}
    write_json(out/'UTILITY_AUDIT.json',result)
    return result


def first_order_diagnostics(data,out):
    cfg=load(data/'config.json');rows=[]
    for seed in cfg['seeds']:
        src=data/f's{seed}'
        for split in ['train','dev','validation']:
            cache=small_cache(src/f'{split}.pt');exact=dict(np.load(src/f'{split}_exact.npz'))
            n=len(exact['source']);idx=np.arange(n)[:,None];s,t=exact['source'],exact['target']
            c=cache['utility_cost'].numpy()[:n];w=cache['weights'].numpy()[:n]
            linear=w[idx,s]*(c[idx,t]-c[idx,s]);actual=exact['delta_nll']
            cond=row_spearman(linear,actual)
            conditional_source=row_spearman(linear.reshape(-1,12),actual.reshape(-1,12))
            chosen=linear.argmin(1);oracle=actual.argmin(1)
            predicted_improve=linear<0
            row={'seed':seed,'split':split,'tokens':n,'replacements':int(actual.size),
                 'mean_linear':float(linear.mean()),'mean_actual':float(actual.mean()),
                 'mean_residual':float((actual-linear).mean()),'rmse':float(np.sqrt(np.mean((actual-linear)**2))),
                 'within_token_rho':float(np.nanmean(cond)),
                 'within_token_fixed_source_rho':float(np.nanmean(conditional_source)),
                 'predicted_negative_count':int(predicted_improve.sum()),
                 'actual_positive_given_predicted_negative':float((actual[predicted_improve]>0).mean()),
                 'linear_argmin_is_exact_argmin_rate':float((chosen==oracle).mean()),
                 'label_linear_oracle_selected_delta_ce':float(actual[np.arange(n),chosen].mean()),
                 'label_exact_oracle_with_nochange_delta_ce':float(np.minimum(actual.min(1),0).mean()),
                 'scope':'Realized-label teacher diagnostics; prediction is measured elsewhere using input only.'}
            rows.append(row)
    write_json(out/'FIRST_ORDER_DIAGNOSTICS.json',rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--group');p.add_argument('--data');p.add_argument('--output',required=True);args=p.parse_args()
    torch.set_num_threads(1);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    if args.group:
        result=review(Path(args.group),out);print('UTILITY_AUDIT',result['passed'],result['verified_runs'],flush=True)
    if args.data:first_order_diagnostics(Path(args.data),out);print('FIRST_ORDER_DIAGNOSTICS_DONE',flush=True)
    write_json(out/'review_source_hashes.json',{'review':digest(Path(__file__)),'numeric':digest(Path(__file__).with_name('round2_numeric.py'))})


if __name__=='__main__':main()
