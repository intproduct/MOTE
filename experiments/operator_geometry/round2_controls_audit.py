"""Independent CPU factor replay for completed ridge and geometry controls."""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import torch
from ..autoregressive_gate.data import digest,write_json
from .round2_train import calibrate_threshold,action_arrays


def read(path):return json.loads(path.read_text(encoding='utf8'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='outputs/operator_geometry_round2');p.add_argument('--output',required=True);args=p.parse_args()
    torch.set_num_threads(1);root=Path(args.root);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    ridge=root/'ridge/01c1341f7586';geometry=root/'geometry/input_geometry_v1';data=root/'data/86a2ba9f5aa2'
    rcfg=read(ridge/'config.json');gcfg=read(geometry/'config.json');checks=[]
    for seed in rcfg['seeds']:
        dm=read(data/f's{seed}/result.json')
        for name,sha in dm['files'].items():assert digest(data/f's{seed}'/name)==sha
        caches={s:torch.load(data/f's{seed}/{s}.pt',map_location='cpu',weights_only=True) for s in ['dev','validation']}
        exacts={s:dict(np.load(data/f's{seed}/{s}_exact.npz')) for s in caches}
        for family,group in [('ridge',ridge),('geometry',geometry)]:
            run=group/f's{seed}';record=read(run/'result.json')
            for name,sha in record['files'].items():assert digest(run/name)==sha
            for row in record['results']:
                rank=row['rank'];model=dict(np.load(run/f'rank{rank}_model.npz'))
                w=model['left']@model['right']
                if 'weight' in model:np.testing.assert_allclose(w,model['weight'],atol=1e-10,rtol=1e-10)
                result={'family':family,'seed':seed,'rank':rank,'replays':{}}
                for split,cache in caches.items():
                    x=(cache['x'].numpy()-model['input_mean'])/model['input_std']
                    pred=x@w+model['target_mean'];saved=dict(np.load(run/f'rank{rank}_{split}.npz'))
                    if family=='ridge':
                        err=float(abs(pred-saved['cost']).max());assert err<1e-10
                        if split=='dev':
                            threshold,_=calibrate_threshold(saved['cost'],cache,exacts[split],rcfg['thresholds'])
                            assert threshold==row['threshold']
                        predicted,actual,chosen=action_arrays(saved['cost'],cache,exacts[split])
                        selected=predicted<-(np.inf if row['threshold'] is None else row['threshold'])
                        np.testing.assert_array_equal(selected,saved['changed']);np.testing.assert_array_equal(chosen,saved['chosen_index'])
                        np.testing.assert_array_equal(np.where(selected,actual,0),saved['action_delta_nll'])
                        candidates=row['hyperparameter_candidates'];key='internal_dev_mse'
                    else:
                        err=float(abs(pred.reshape(saved['features'].shape)-saved['features']).max());assert err<1e-9
                        outputs=cache['outputs'].numpy();target=((outputs-outputs.mean(1,keepdims=True))@model['sketch']).reshape(len(outputs),-1)
                        mse=float(np.square(pred-target).mean());base=float(np.square(target-model['target_mean']).mean())
                        assert abs(mse-row['splits'][split]['sketch_mse'])<1e-7
                        assert abs(1-mse/base-row['splits'][split]['sketch_r2_vs_constant'])<1e-7
                        candidates=row['selection_candidates'];key='internal_dev_sketch_mse'
                    assert min(candidates,key=lambda r:r[key])['ridge']==row['ridge']
                    result['replays'][split]={'tokens':len(pred),'factor_prediction_max_abs_error':err,'selection_recomputed':True}
                checks.append(result)
        print('CONTROL_AUDIT',seed,flush=True)
    assert len(checks)==18
    write_json(out/'AUDIT.json',{'passed':True,'models_verified':len(checks),'split_replays':2*len(checks),'checks':checks,
        'scope':'All saved factors, data/result hashes, predictions, internal-dev regularizer selection, geometry targets, and calibrated utility actions checked on CPU.'})
    write_json(out/'source_hash.json',{'audit':digest(Path(__file__))})


if __name__=='__main__':main()
