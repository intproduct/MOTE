"""Read-only held handles for the authorized long-running formal controllers."""
import json
import os
from pathlib import Path
import sys
import time
import traceback
R=Path('E:/AI/MOTE-quantum');B=R/'runtime/formal-supervision-v12-release-20261031'
sys.path.insert(0,str(B))
from watch_pro_product_exit_v9 import HeldProcess,atomic_json,sha
O=R/'runtime/formal-controller-exits-v15-20261031'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def main():
    assert Path.cwd().resolve()==Path('D:/AI/MOTE').resolve()
    assert sha(B/'MANIFEST.json')=='54d23c18597285fc9d35f8bce3c5dde7e34c1bf978da6933b832ace0b3e5f1f2'
    for name,h in read(B/'MANIFEST.json')['files'].items():assert sha(B/name)==h
    assert not O.exists();O.mkdir();held={};done=set()
    try:
        receipt=read(R/'queue-20261030/formal-matrix-v15-launch-20261031.json')
        for label,spec in receipt['controllers'].items():
            pair=[];held[label]=pair
            for pid in (spec['launcher_pid'],spec['native_pid']):
                h=HeldProcess(pid);pair.append(h);assert h.record['command'][1:]==spec['command'][1:]
            atomic_json(O/(label+'.ATTACHED.json'),dict(observer_pid=os.getpid(),workers=[h.record for h in pair],
                receipt_sha256=sha(R/'queue-20261030/formal-matrix-v15-launch-20261031.json'),watcher_sha256=sha(__file__),time=time.time(),no_gpu_use=True,no_signals=True))
        started=time.monotonic()
        while len(done)<len(held):
            for label,pair in held.items():
                if label in done:continue
                codes=[h.poll() for h in pair]
                if all(c is not None for c in codes):
                    atomic_json(O/(label+'.RESULT.json'),dict(phase='exited',passed=all(c==0 for c in codes),workers=[dict(**h.record,exit_code=c) for h,c in zip(pair,codes)],time=time.time(),no_gpu_use=True,no_signals=True))
                    done.add(label)
            assert time.monotonic()-started<30*86400,'Thirty-day observation timeout; no process signalled'
            time.sleep(5)
    except BaseException:
        atomic_json(O/'FAILURE.json',dict(error=traceback.format_exc(),time=time.time()));raise
    finally:
        for pair in held.values():
            for h in pair:h.close()

if __name__=='__main__':main()
