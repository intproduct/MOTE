"""CPU-only follow-on audits; does not acquire or use the GPU lock."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from ..autoregressive_gate.data import digest,write_json
from ..routing_stability.runner import timestamp


def read(path):return json.loads(path.read_text(encoding='utf8'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='outputs/operator_geometry_round2');args=p.parse_args()
    repo=Path(__file__).resolve().parents[2];root=repo/args.root;lock=root/'AUDIT_QUEUE_LOCK.json'
    with os.fdopen(os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY),'w') as f:json.dump({'pid':os.getpid(),'device':'cpu'},f)
    modules=['round2_review.py','round2_prediction_review.py','round2_audit_queue.py']
    hashes={name:digest(Path(__file__).with_name(name)) for name in modules}
    status={'state':'waiting_for_utility','pid':os.getpid(),'device':'cpu','started':timestamp(),'source_hashes':hashes,'completed_audits':[]}
    write_json(root/'AUDIT_QUEUE_STATUS.json',status)
    def wait_for(queue,expected):
        while True:
            q=read(root/queue)
            if q['state']=='failed':raise RuntimeError(f'{queue} failed')
            if q['state']==expected:return q
            if datetime.now().astimezone()>=datetime.fromisoformat('2026-10-30T08:00:00+08:00'):raise TimeoutError('CPU audit wait deadline')
            time.sleep(15)
    def run(module,arguments,name):
        assert all(digest(Path(__file__).with_name(key))==sha for key,sha in hashes.items())
        status.update(state='running',current=name);write_json(root/'AUDIT_QUEUE_STATUS.json',status)
        subprocess.run([sys.executable,'-u','-m','fitmotn.experiments.operator_geometry.'+module,*arguments],cwd=repo,check=True)
        status['completed_audits'].append(name);write_json(root/'AUDIT_QUEUE_STATUS.json',status)
    try:
        wait_for('QUEUE_STATUS.json','ready_for_agent_review')
        utility=root/'utility/3a0386d9a187'
        run('round2_review',['--group',str(utility),'--output',str(root/'reviews/utility_action_v1')],'utility_action_v1')
        run('round2_prediction_review',['--group',str(utility),'--kind','utility','--output',str(root/'reviews/utility_prediction_v1')],'utility_prediction_v1')
        status.update(state='waiting_for_positive',current=None);write_json(root/'AUDIT_QUEUE_STATUS.json',status)
        positive=wait_for('POSITIVE_QUEUE_STATUS.json','positive_ready_for_review')
        run('round2_prediction_review',['--group',positive['group'],'--kind','positive','--output',str(root/'reviews/positive_formal_v1')],'positive_formal_v1')
        status.update(state='complete',current=None,finished=timestamp());write_json(root/'AUDIT_QUEUE_STATUS.json',status)
    except BaseException:
        status.update(state='failed',finished=timestamp(),error=traceback.format_exc());write_json(root/'AUDIT_QUEUE_STATUS.json',status);raise
    finally:lock.unlink()


if __name__=='__main__':main()
