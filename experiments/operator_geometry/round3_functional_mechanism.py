"""CPU-only decomposition of saved functional-adaptation routing errors."""
from pathlib import Path
import argparse, json
import numpy as np
import torch
from .metrics import spearman
from ..autoregressive_gate.data import digest, write_json
from ..routing_stability.runner import setup, timestamp
from .round3_replay_review import csvwrite

ROOT = Path(__file__).resolve().parents[2]

def read(path):
    return json.loads(path.read_text(encoding='utf8'))

def decompose(w, original, experts):
    d = w - original
    v = experts - experts.mean(1, keepdims=True)
    displacement = np.einsum('ne,ned->nd', d, v)
    full = (displacement ** 2).mean(1)
    individual = (d ** 2 * (v ** 2).mean(2)).sum(1)
    coefficient = (d ** 2).sum(1)
    return dict(coefficient_l2=coefficient,
                coefficient_l1=np.abs(d).sum(1),
                omitted_original_mass=(original * (w == 0)).sum(1),
                original_top1_retained=w[np.arange(len(w)), original.argmax(1)] > 0,
                top4_agreement=((w > 0) == (original > 0)).all(1),
                output_squared=full, individual_centered_energy=individual,
                cross_term=full-individual,
                output_rayleigh=np.divide(full, coefficient, out=np.zeros_like(full), where=coefficient > 1e-15))

def self_check():
    rng=np.random.default_rng(12)
    w=rng.dirichlet(np.ones(16),32); a=rng.dirichlet(np.ones(16),32)
    v=rng.normal(size=(32,16,256)); shift=rng.normal(size=(32,1,256))
    z=decompose(w,a,v); zz=decompose(w,a,v+shift)
    for name in z: np.testing.assert_allclose(z[name],zz[name],rtol=1e-10,atol=1e-12)
    d=w-a; centered=v-v.mean(1,keepdims=True)
    gram=np.einsum('ned,nfd->nef',centered,centered)/256
    np.testing.assert_allclose(z['output_squared'],np.einsum('ne,nef,nf->n',d,gram,d),rtol=1e-12,atol=1e-12)
    return True

def run():
    setup(); self_check()
    root=ROOT/'outputs/operator_geometry_round3'
    status=read(root/'functional/STATUS.json'); assert status['state']=='ready_for_review'
    group=Path(status['group']); cfg=read(group/'config.json')
    assert read(root/'reviews/functional_v1/AUDIT.json')['passed']
    out=root/'reviews/functional_mechanism_v1';out.mkdir(parents=True,exist_ok=False)
    rows=[]; hashes={}; checks=[]
    for seed in cfg['seeds']:
        for split in ['dev','validation']:
            cachepath=ROOT/cfg['data_group']/f's{seed}/{split}.pt'
            c=torch.load(cachepath,map_location='cpu',weights_only=True)
            original=c['weights'].numpy().astype('float64');experts=c['outputs'].numpy().astype('float64')
            hashes[str(cachepath.relative_to(ROOT))]=digest(cachepath)
            for method in cfg['methods']:
                for loss in ['initial']+cfg['objectives']:
                    path=(root/'replay_v2/ff4f146e8d29'/f'{method}_s{seed}'/f'{split}_calibrated_clean.npz' if loss=='initial' else group/'runs'/f'{method}_{loss}_s{seed}'/f'{split}_clean.npz')
                    raw=dict(np.load(path)); hashes[str(path.relative_to(ROOT))]=digest(path)
                    np.testing.assert_array_equal(raw['window'],c['window'].numpy())
                    w=raw['weights'].astype('float64'); z=decompose(w,original,experts)
                    assert all(np.isfinite(a).all() for a in z.values())
                    # Gram identity and common-shift invariance on independent sample positions.
                    ix=np.array([0,73,521,2095,4095]); d=w[ix]-original[ix];v=experts[ix]-experts[ix].mean(1,keepdims=True)
                    gram=np.einsum('ned,nfd->nef',v,v)/256
                    err=float(np.max(np.abs(z['output_squared'][ix]-np.einsum('ne,nef,nf->n',d,gram,d))))
                    assert err<1e-8
                    changed=decompose(w[ix],original[ix],experts[ix]+np.arange(256)[None,None,:]*.03)
                    assert np.max(np.abs(z['output_squared'][ix]-changed['output_squared']))<1e-8
                    record={'seed':seed,'split':split,'method':method,'objective':loss,**{k:float(a.mean()) for k,a in z.items()},
                            'kl':float(raw['kl'].mean()),'delta_ce':float(raw['delta_nll'].mean()),
                            'cancellation_fraction_aggregate':float(1-z['output_squared'].sum()/z['individual_centered_energy'].sum()),
                            'rayleigh_aggregate':float(z['output_squared'].sum()/z['coefficient_l2'].sum())}
                    for name in ['coefficient_l2','omitted_original_mass','output_squared','individual_centered_energy']:
                        record[name+'_spearman_kl']=spearman(z[name],raw['kl'])
                    for key in record:
                        if isinstance(record[key],float): assert np.isfinite(record[key]),key
                    rows.append(record); checks.append({'seed':seed,'split':split,'method':method,'objective':loss,'gram_abs_error':err})
                    np.savez_compressed(out/f'{method}_{loss}_s{seed}_{split}.npz',**z,kl=raw['kl'],delta_nll=raw['delta_nll'],window=raw['window'])
            del c,experts,original
    assert len(rows)==96
    csvwrite(out/'decomposition.csv',rows);write_json(out/'SUMMARY.json',rows);write_json(out/'INPUT_HASHES.json',hashes)
    write_json(out/'AUDIT.json',{'passed':True,'records':len(rows),'toy_identity_and_gauge_checks':True,'checks':checks,'completed':timestamp(),
                               'script_sha256':digest(Path(__file__)),'plan_sha256':digest(root/'FUNCTIONAL_MECHANISM_PLAN.json')})
    print(json.dumps({'passed':True,'records':len(rows),'output':str(out)}),flush=True)

if __name__=='__main__':run()
