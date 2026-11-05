"""Hold already-running controller process handles; read-only, no GPU/signals."""
import json
import os
from pathlib import Path
import sys
import time
import traceback
R=Path('E:/AI/MOTE-quantum');V12=R/'runtime/formal-supervision-v12-release-20261031'
sys.path.insert(0,str(V12))
from watch_pro_product_exit_v9 import HeldProcess,atomic_json,sha
OUT=R/'runtime/diagnostic-controller-exits-v15-20261031'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def main():
    assert Path.cwd().resolve()==Path('D:/AI/MOTE').resolve()
    # Verify the complete published dependency inventory before attaching.
    assert sha(V12/'MANIFEST.json')=='54d23c18597285fc9d35f8bce3c5dde7e34c1bf978da6933b832ace0b3e5f1f2'
    for name,h in read(V12/'MANIFEST.json')['files'].items():assert sha(V12/name)==h
    assert not OUT.exists();OUT.mkdir();held={};completed=set()
    try:
        specs=[('bf16',R/'diagnostics/bf16-followup-v6r1-pro4000-20261031/CONTROLLER.json',R/'queue-20261030/bf16-recovery-v13-launch-20261031.json','wrapper_pid'),
               ('fp16_initial_local',R/'diagnostics/initial-local-fp16-v14-20261031/CONTROLLER.json',R/'queue-20261030/initial-local-v14-launch-20261031.json','launcher_pid')]
        for label,status,receipt,wrapper_key in specs:
            state=read(status);launch=read(receipt);pair=[];held[label]=pair
            for pid in (launch[wrapper_key],state['pid']):
                handle=HeldProcess(pid);pair.append(handle)
                assert handle.record['command'][1:]==launch['command'][1:]
            atomic_json(OUT/(label+'.ATTACHED.json'),dict(phase='observing_existing_controller',observer_pid=os.getpid(),workers=[h.record for h in pair],
                status=str(status),launch_receipt=str(receipt),watcher_sha256=sha(__file__),time=time.time(),no_gpu_use=True,no_signals=True))
        started=time.monotonic()
        while len(completed)<len(held):
            for label,pair in held.items():
                if label in completed:continue
                codes=[h.poll() for h in pair]
                if all(code is not None for code in codes):
                    atomic_json(OUT/(label+'.RESULT.json'),dict(phase='exited',passed=all(code==0 for code in codes),
                        workers=[dict(**h.record,exit_code=code) for h,code in zip(pair,codes)],time=time.time(),no_gpu_use=True,no_signals=True))
                    completed.add(label)
            assert time.monotonic()-started<6*3600,'Observation timeout; no process was signalled'
            time.sleep(2)
    except BaseException:
        atomic_json(OUT/'FAILURE.json',dict(error=traceback.format_exc(),time=time.time()));raise
    finally:
        for pair in held.values():
            for handle in pair:handle.close()

if __name__=='__main__':main()
