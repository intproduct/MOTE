"""Serialize the already-running data collection and fixed utility study.

This launcher never resumes, overwrites, or kills an experiment. It contains no
torch import and can wait while the sole GPU data process continues.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def save(path, value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--deadline',required=True,help='Latest allowed utility start, ISO with timezone')
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[2]
    root=repo/'outputs/operator_geometry_round2'
    config=repo/args.config
    cfg=read(config);data=repo/cfg['data_group']
    lock=root/'QUEUE_LOCK.json'
    with os.fdopen(os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY),'w') as f:
        json.dump({'pid':os.getpid(),'started':datetime.now().astimezone().isoformat()},f)
    protected=[config,Path(__file__),Path(__file__).with_name('round2_train.py')]
    frozen={str(p):sha(p) for p in protected}
    status={'pid':os.getpid(),'state':'waiting_for_data','config':str(config),'data_group':str(data),
            'deadline':args.deadline,'source_hashes':frozen,'started':datetime.now().astimezone().isoformat()}
    save(root/'QUEUE_STATUS.json',status)
    try:
        while True:
            now=datetime.now().astimezone()
            if now>=datetime.fromisoformat(args.deadline):
                raise TimeoutError('Latest utility start passed; leave data/results intact for review')
            if {str(p):sha(p) for p in protected}!=frozen:
                raise RuntimeError('Queued source/config changed; refusing mixed configuration')
            current=read(root/'STATUS.json')
            if current.get('state')=='failed':
                raise RuntimeError('Preceding data pipeline failed; inspect original error')
            if (data/'DATA_COMPLETE.json').exists() and not (root/'PIPELINE_LOCK.json').exists():
                if current.get('state')!='data_ready' or Path(current['group'])!=data:
                    raise RuntimeError('Unexpected preceding pipeline state/group')
                break
            time.sleep(20)
        status.update(state='utility_running',utility_started=datetime.now().astimezone().isoformat())
        with (root/'logs/utility_v1.log').open('x',encoding='utf8') as out, (root/'logs/utility_v1.err').open('x',encoding='utf8') as err:
            child=subprocess.Popen([sys.executable,'-u','-m','fitmotn.experiments.operator_geometry.round2_train',
                                    '--config',str(config)],cwd=repo,stdout=out,stderr=err)
            status['child_pid']=child.pid;save(root/'QUEUE_STATUS.json',status)
            code=child.wait()
        if code:raise RuntimeError(f'Utility child returned {code}; preserve all outputs')
        if read(root/'STATUS.json').get('state')!='utility_ready_for_review':
            raise RuntimeError('Utility child exited without review-ready state')
        status.update(state='ready_for_agent_review',finished=datetime.now().astimezone().isoformat())
        save(root/'QUEUE_STATUS.json',status)
    except BaseException:
        status.update(state='failed',error=traceback.format_exc(),finished=datetime.now().astimezone().isoformat())
        save(root/'QUEUE_STATUS.json',status)
        raise
    finally:
        lock.unlink()


if __name__=='__main__':main()
