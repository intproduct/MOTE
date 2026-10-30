"""Serial deployment-path timing of all60 fixed functional checkpoints."""
import argparse,json,os,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
from ..autoregressive_gate.data import digest,write_json
from ..routing_stability.runner import setup,timestamp
from ..routing_stability.model import full_state_hash
from .round3_functional import ROOT,read
from .round3_capacity import ReencodingPredictor
from .round2_train import UtilityPredictor
from .round3_replay_v2 import timing
from .round3_replay_review import csvwrite

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--wait',action='store_true');args=ap.parse_args();setup()
    root=ROOT/'outputs/operator_geometry_round3';queue=root/'FUNCTIONAL_TIMING_QUEUE.json';start=time.monotonic()
    lock=ROOT/'outputs/operator_geometry_round2/PIPELINE_LOCK.json';acquired=False
    state={'state':'waiting','active':True,'pid':os.getpid(),'started':timestamp(),'completed':0,'total':60};write_json(queue,state)
    try:
        while True:
            q=read(root/'DIAGONAL_REVIEW_QUEUE.json')
            if q['state']=='failed':raise RuntimeError(q.get('error'))
            if q['state']=='ready_for_review' and not lock.exists():break
            if not args.wait or time.monotonic()-start>5400:raise TimeoutError('timing queue wait bound')
            time.sleep(30)
        for audit in ['functional_v1','functional_diagonal_v1']:assert read(root/f'reviews/{audit}/AUDIT.json')['passed']
        out=root/'timing/functional_v1';out.mkdir(parents=True,exist_ok=False);state.update(state='timing',output=str(out))
        with os.fdopen(os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY),'w') as f:json.dump(state,f)
        acquired=True;torch.cuda.set_per_process_memory_fraction(.35)
        def gpu(name):
            p=subprocess.run(['nvidia-smi'],capture_output=True,text=True,check=False);(out/name).write_text(p.stdout+p.stderr,encoding='utf8')
        gpu('gpu_before.txt');rows=[];checks=[]
        groups=[Path(read(root/f'{stage}/STATUS.json')['group']) for stage in ['functional','functional_diagonal']]
        for group in groups:
            cfg=read(group/'config.json');sources=read(group/'source_hashes.json')
            for rel,h in sources.items():assert digest(ROOT/rel)==h
            for seed in cfg['seeds']:
                geo=read(ROOT/cfg['geometry_group']/f's{seed}/assignment.json');assignment=np.asarray(geo['expert_for_basis']);perm=np.asarray(geo['random_expert_for_basis']);codes=np.load(ROOT/cfg['geometry_group']/f's{seed}/codes.npy')
                x=torch.load(ROOT/cfg['data_group']/f's{seed}/dev.pt',map_location='cpu',weights_only=True)['x'].cuda()
                for m in cfg['methods']:
                    for loss in cfg['objectives']:
                        run=group/'runs'/f'{m}_{loss}_s{seed}';r=read(run/'result.json');p=run/'best_dev.pt';h=digest(p);assert h==r['files']['best_dev.pt']
                        ck=torch.load(p,map_location='cpu',weights_only=False)
                        model=UtilityPredictor(1024,8,'mlp',assignment,codes,perm) if m=='mlp' else ReencodingPredictor(1024,8,m,assignment)
                        model.load_state_dict(ck['model']);model.cuda().eval();mh=full_state_hash(model.state_dict())
                        for n in [1,256,4096]:
                            row={'seed':seed,'method':m,'objective':loss,'checkpoint_sha256':h,**timing(model,x[:n],'cuda',ck['beta'])}
                            assert row['median_ms']>0 and np.isfinite(row['samples_ms']).all();rows.append(row)
                        assert full_state_hash(model.state_dict())==mh and digest(p)==h;checks.append({'seed':seed,'method':m,'objective':loss,'checkpoint_sha256':h,'state_unchanged':True})
                        state['completed']+=1;write_json(queue,state);del model,ck;torch.cuda.empty_cache()
                del x
        assert len(rows)==180 and len(checks)==60;gpu('gpu_after.txt')
        write_json(out/'raw_timing.json',rows);csvwrite(out/'timing.csv',[{k:v for k,v in r.items() if k!='samples_ms'} for r in rows]);write_json(out/'CHECKPOINT_CHECKS.json',checks)
        write_json(out/'AUDIT.json',{'passed':True,'checkpoints':60,'timing_rows':180,'samples_per_row':50,'warmup_per_row':20,'missing':[],'failed':[],'script_sha256':digest(Path(__file__)),'completed':timestamp()})
        state.update(state='ready_for_review',active=False,finished=timestamp());write_json(queue,state)
    except BaseException:
        state.update(state='failed',active=False,error=traceback.format_exc(),finished=timestamp());write_json(queue,state);raise
    finally:
        if acquired:lock.unlink()

if __name__=='__main__':main()
