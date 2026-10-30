"""CPU calibration audit and train/held-out prediction diagnostics.

Read-only source/result use; writes to a new directory. No checkpoint selection
or new fitting is performed here.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from ..autoregressive_gate.data import digest,write_json
from ..routing_stability.model import full_state_hash
from .round2_train import UtilityPredictor,predict
from .round2_positive import teacher_cost,measures
from .round2_numeric import row_spearman


def read(path):return json.loads(path.read_text(encoding='utf8'))


def review(group,output,kind):
    repo=Path(__file__).resolve().parents[2];cfg=read(group/'config.json');data=repo/cfg['data_group']
    geometry_group=repo/cfg['utility_group'] if kind=='positive' else group
    for relative,sha in read(group/'source_hashes.json').items():
        assert digest(repo/relative)==sha and digest(group/'source_snapshot'/relative)==sha
    rows=[];checks=[];missing=[]
    for seed in cfg['seeds']:
        src=data/f's{seed}';manifest=read(src/'result.json');caches={}
        for split in ['train','dev','validation']:
            path=src/f'{split}.pt';assert digest(path)==manifest['files'][path.name]
            cached=torch.load(path,map_location='cpu',weights_only=True)
            target=teacher_cost(cached['teacher']) if kind=='positive' else cached['utility_cost']
            caches[split]={'x':cached['x'],'target':target};del cached
        constant=caches['train']['target'].mean(0).numpy();scale=float(caches['train']['target'].std())
        geo=read(geometry_group/f'geometry/s{seed}/assignment.json');codes=np.load(geometry_group/f'geometry/s{seed}/codes.npy')
        geometry=(np.asarray(geo['expert_for_basis']),codes,np.asarray(geo['random_expert_for_basis']))
        samples=torch.randint(len(caches['train']['x']),(cfg['steps'],cfg['batch_size']),generator=torch.Generator().manual_seed(seed+970000)).numpy().reshape(-1)
        for method in cfg['methods']:
            run=group/f'runs/{method}_s{seed}'
            if not (run/'result.json').exists():missing.append(run.name);continue
            result=read(run/'result.json')
            for name,sha in result['files'].items():assert digest(run/name)==sha
            np.testing.assert_array_equal(np.load(run/'sample_indices.npy'),samples)
            torch.manual_seed(seed+960000);model=UtilityPredictor(caches['train']['x'].shape[-1],cfg['rank'],method,*geometry)
            initial=full_state_hash(dict(model.named_parameters()))
            original=read((geometry_group/f'runs/{method}_s{seed}')/'manifest.json')
            assert initial==original['initial_parameter_hash']
            if kind=='positive':assert initial==result['initial_parameter_hash']
            curves=[json.loads(line) for line in (run/'curves.jsonl').read_text().splitlines()]
            assert [row['step'] for row in curves]==list(range(cfg['eval_every'],cfg['steps']+1,cfg['eval_every']))
            assert all(np.isfinite(row['mse']) and np.isfinite(row['gradient_norm']) and row['gradient_norm']>0 for row in curves)
            selected=min(curves,key=lambda row:row['mse'])['step']
            checkpoint=torch.load(run/'best_dev.pt',map_location='cpu',weights_only=False)
            assert selected==result['selected_step']==checkpoint['step'] and abs(scale-checkpoint['target_scale'])<1e-8
            model.load_state_dict(checkpoint['model'],strict=True);assert result['parameter_changed']
            assert initial!=full_state_hash(dict(model.named_parameters()))
            assert sum(p.numel() for p in model.parameters())==result['parameters']
            assert all(torch.isfinite(value).all() for value in checkpoint['model'].values())
            check={'seed':seed,'method':method,'initial_and_sampling_match':True,'files_verified':len(result['files']),'replays':{}}
            for split,cache in caches.items():
                prediction=predict(model,cache,scale,'cpu');target=cache['target'].numpy()
                metrics=measures(prediction,target,constant)
                saved_path=run/f'{split}_predictions.npz'
                if saved_path.exists():
                    saved=np.load(saved_path)['cost'];error=float(abs(prediction-saved).max());assert error<5e-4
                    check['replays'][split]=error
                    if kind=='positive':
                        for name in ['mse','r2_vs_constant']:
                            assert abs(measures(saved,target,constant)[name]-result['splits'][split][name])<1e-8
                rows.append({'seed':seed,'method':method,'split':split,'target_kind':kind,'parameters':result['parameters'],
                    'selected_step':selected,'tokens':len(target),**metrics})
            checks.append(check)
    expected=len(cfg['seeds'])*len(cfg['methods'])
    write_json(output/'PREDICTION_AUDIT.json',{'passed':len(checks)==expected and not missing,'kind':kind,'expected':expected,
        'verified':len(checks),'missing':missing,'checks':checks,'rows':rows,
        'scope':'Input-only prediction diagnostics. Rank/top4 target agreement for utility concerns realized-label first-order costs, not expert identity labels or deployed routing.'})
    print('PREDICTION_AUDIT',kind,len(checks),'/',expected,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--group',required=True);p.add_argument('--kind',choices=['utility','positive'],required=True);p.add_argument('--output',required=True)
    args=p.parse_args();torch.set_num_threads(1);output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    review(Path(args.group),output,args.kind)
    write_json(output/'review_source_hashes.json',{'review':digest(Path(__file__)),'numeric':digest(Path(__file__).with_name('round2_numeric.py'))})


if __name__=='__main__':main()
