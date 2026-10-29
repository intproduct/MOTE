"""Owned, logged subprocess pipeline; finish with review-ready, never fabricate success."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/gate_benchmark_round2'
os.chdir(ROOT)
os.environ.update(PYTHONUTF8='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                  HF_HOME=str(ROOT/'outputs/hf_cache'),HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1')
os.environ['PATH']=str(ROOT/'.venv-titan/Library/bin')+os.pathsep+os.environ['PATH']
CFG='experiments/gate_benchmark/configs/pareto_formal.json'
MODULE=['-m','fitmotn.experiments.gate_benchmark']


def now():return datetime.datetime.now().astimezone().isoformat()


def phase(name,args):
    logfile=OUT/'logs'/f'{name}_{int(time.time())}.log'
    with logfile.open('w',encoding='utf-8') as handle:
        command=[sys.executable,'-u',*args]
        handle.write(json.dumps({'command':command,'started':now()})+'\n');handle.flush()
        proc=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,
                              creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        state={'phase':name,'driver_pid':os.getpid(),'pid':proc.pid,'started':now(),
               'log':str(logfile),'config':CFG}
        (OUT/'ACTIVE_PROCESS.json').write_text(json.dumps(state,indent=2))
        print(json.dumps(state),flush=True)
        code=proc.wait()
        handle.write(f'\nEXIT_CODE={code}\n')
    state.update(ended=now(),exit_code=code)
    with (OUT/'phase_history.jsonl').open('a') as handle:handle.write(json.dumps(state)+'\n')
    if code:
        (OUT/'ACTIVE_PROCESS.json').write_text(json.dumps({**state,'needs_attention':True},indent=2))
        raise SystemExit(code)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--max-workers',type=int,choices=range(1,7),default=6)
    args=parser.parse_args()
    assert Path(CFG).exists(),'formal config must be selected and frozen first'
    snapshot=OUT/'source_snapshot_formal'
    sources=['advanced_routers.py','gate.py','structured_routers.py','quantum_gate.py','ADTN.py','sparse_mixt.py',
             'experiments/gate_benchmark/models.py','experiments/gate_benchmark/config.py',
             'experiments/gate_benchmark/runner.py','experiments/gate_benchmark/data.py',CFG]
    hashes={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in sources}
    if (snapshot/'files_sha256.json').exists():
        assert json.loads((snapshot/'files_sha256.json').read_text())==hashes,'frozen training source changed'
    else:
        for name in sources:
            dest=snapshot/name;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(name,dest)
        (snapshot/'files_sha256.json').write_text(json.dumps(hashes,indent=2))
    phase('formal_prepare',MODULE+['prepare','--config',CFG])
    phase('historical_controls',['-m','fitmotn.experiments.gate_benchmark.reference_pareto'])
    phase('formal_train',['outputs/gate_benchmark/concurrent_matrix.py','--config',CFG,'--max-workers',str(args.max_workers)])
    phase('formal_audit',['-m','fitmotn.experiments.gate_benchmark.audit_pareto','--config',CFG])
    phase('formal_summarize',MODULE+['summarize','--config',CFG])
    phase('serial_efficiency',['outputs/gate_benchmark_round2/serial_efficiency.py'])
    phase('analysis',['outputs/gate_benchmark_round2/analyze.py'])
    state={'phase':'ready_for_review','time':now(),'driver_pid':os.getpid(),
           'note':'training/independent evaluation/plots complete; report conclusions and visual QA still required'}
    (OUT/'ACTIVE_PROCESS.json').write_text(json.dumps(state,indent=2))
    print(json.dumps(state),flush=True)


if __name__=='__main__':main()
