"""Per-seed paired stage3 calibration comparison, no selection/refitting."""
from pathlib import Path
import csv,json
import numpy as np
import torch
from .metrics import bootstrap_window_mean

def main():
    torch.set_num_threads(1);base=Path('outputs/operator_geometry_round3');out=base/'reviews/capacity_comparison_v1';out.mkdir(parents=True,exist_ok=False)
    old=Path('outputs/operator_geometry_round2/positive/d4ee19e23787');new=base/'capacity/481aeb0bc8dd'
    data=Path('outputs/operator_geometry_round2/data/86a2ba9f5aa2')
    pairs=[('reencode_cz','angle_functional'),('reencode_cz','reencode_product'),('affine_born','mlp'),('affine_born','reencode_cz')]
    newmethods=['reencode_cz','reencode_product','affine_born'];rows=[]
    for seed in [17,29,43]:
        for split in ['dev','validation']:
            c=torch.load(data/f's{seed}/{split}.pt',map_location='cpu',weights_only=True);y=-c['teacher'].clamp_min(1e-12).log();y=(y-y.mean(1,keepdim=True)).numpy();truth=np.sort(y.argsort(1)[:,:4],1)
            preds={m:np.load((new if m in newmethods else old)/f'runs/{m}_s{seed}/{split}_predictions.npz')['cost'] for m in set(sum(([a,b] for a,b in pairs),[]))}
            den=json.loads((old/f'runs/mlp_s{seed}/result.json').read_text())['splits'][split]['constant_mse']
            for a,b in pairs:
                va,vb=preds[a],preds[b]
                diffs={'r2_difference':(np.square(vb-y).mean(1)-np.square(va-y).mean(1))/den,
                       'top4_agreement_difference':(np.sort(va.argsort(1)[:,:4],1)==truth).all(1).astype(float)-(np.sort(vb.argsort(1)[:,:4],1)==truth).all(1)}
                for metric,v in diffs.items():
                    ci=bootstrap_window_mean(v,c['window'].numpy(),seed,repeats=2000)
                    rows.append({'seed':seed,'split':split,'method':a,'reference':b,'metric':metric,'paired_difference':float(v.mean()),'ci95_low':ci[0],'ci95_high':ci[1]})
            del c
    with (out/'paired.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={'pairs':rows,'scope':'Paired window bootstrap with2000resamples, train-mean fixeddenominator. Descriptive exploratory intervals, no multiplicity correction. No new model selection.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8');print('PAIRED_COMPLETE',len(rows),flush=True)

if __name__=='__main__':main()
