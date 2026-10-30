"""Disentangle expert energy from pair compatibility in functional distances."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from ..autoregressive_gate.data import write_json,digest
from .round2_numeric import row_spearman


def proxies(outputs,weights,exact,global_distance,global_energy):
    gram=outputs@outputs.transpose(0,2,1)
    energy=np.diagonal(gram,axis1=1,axis2=2)
    s,t=exact['source'],exact['target'];ix=np.arange(len(outputs))[:,None]
    normsum=energy[ix,s]+energy[ix,t];inner=gram[ix,s,t]
    mass=weights[ix,s]
    gs=global_energy[s]+global_energy[t]
    ginner=(gs-global_distance[s,t])/2
    return {
        'local_full_distance':np.maximum(normsum-2*inner,0),
        'local_energy_only':normsum,
        'local_target_energy':energy[ix,t],
        'local_cross_inner':-2*inner,
        'local_cosine_distance':1-inner/np.maximum(np.sqrt(energy[ix,s]*energy[ix,t]),1e-12),
        'global_full_distance':global_distance[s,t],
        'global_energy_only':gs,
        'global_cross_inner':-2*ginner,
        'global_cosine_distance':1-ginner/np.maximum(np.sqrt(global_energy[s]*global_energy[t]),1e-12),
    },mass,energy


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    torch.set_num_threads(1);data=Path(args.data);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((data/'config.json').read_text(encoding='utf8'));rows=[]
    for seed in cfg['seeds']:
        src=data/f's{seed}';record=json.loads((src/'result.json').read_text(encoding='utf8'))
        for name,sha in record['files'].items():assert digest(src/name)==sha
        cache=torch.load(src/'train.pt',map_location='cpu',weights_only=True)
        train_energy=cache['outputs'].square().sum(-1).mean(0).numpy().astype(np.float64)
        del cache
        distance=np.load(src/'train_pair_distance.npy')
        global_gram=(train_energy[:,None]+train_energy[None,:]-distance)/2
        np.savez_compressed(out/f's{seed}_train_geometry.npz',energy=train_energy,gram=global_gram,distance=distance)
        for split in ['dev','validation']:
            cache=torch.load(src/f'{split}.pt',map_location='cpu',weights_only=True)
            exact=dict(np.load(src/f'{split}_exact.npz'))
            estimates,mass,energy=proxies(cache['outputs'].numpy(),cache['weights'].numpy(),exact,distance,train_energy)
            result={'seed':seed,'split':split,'tokens':len(energy),'estimates':{}}
            for name,values in estimates.items():
                shape=(-1,12)
                result['estimates'][name]={
                    'rho_vs_local_full_distance':float(np.nanmean(row_spearman(values.reshape(shape),estimates['local_full_distance'].reshape(shape)))),
                    'rho_vs_abs_delta_nll':float(np.nanmean(row_spearman(values.reshape(shape),abs(exact['delta_nll']).reshape(shape)))),
                    'nearest_target_agreement_with_full_distance':float((values.reshape(shape).argmin(1)==estimates['local_full_distance'].reshape(shape).argmin(1)).mean())}
            result['cross_term_abs_to_normsum_mean']=float((abs(estimates['local_cross_inner'])/np.maximum(estimates['local_energy_only'],1e-12)).mean())
            result['global_energy_cv_across_experts']=float(train_energy.std()/train_energy.mean())
            np.savez_compressed(out/f's{seed}_{split}_proxies.npz',**estimates,mass=mass,energy=energy)
            rows.append(result);print('ENERGY_DONE',seed,split,flush=True)
            del cache,estimates
    write_json(out/'summary.json',{'rows':rows,
        'identity':'D_ij(x)=||f_i(x)||^2+||f_j(x)||^2-2<f_i(x),f_j(x)>',
        'interpretation':'Correlation of full distance with replacement effects may reflect individual expert magnitude rather than pair-specific compatibility. Fixed-source comparisons still retain target-energy variation.',
        'regular_graph_identity':'For any d-regular basis graph, sum_edges(||f_i||^2+||f_j||^2)=d*sum_i||f_i||^2, independent of the expert-to-basis permutation. Only cross-inner-product terms can improve its assignment objective.',
        'scope':'Target includes sampled absolute signed NLL changes; label-free KL/Fisher enrichment should be assessed independently when GPU is free.'})
    write_json(out/'source_hashes.json',{str(Path(__file__)):digest(Path(__file__))})


if __name__=='__main__':main()
