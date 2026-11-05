"""Recheck the frozen global reservation; no imports of training or GPU code."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('E:/AI/MOTE-quantum')
BUNDLE = ROOT/'runtime/formal-supervision-v12-release-20261031'
sys.path.insert(0, str(BUNDLE))
import formal_runtime_v12 as core


def observe(label):
    if not label or any(c not in '0123456789TZ' for c in label):
        raise ValueError('UTC timestamp label required')
    assert Path.cwd().resolve() == core.WORK.resolve()
    core.bound(BUNDLE/'formal_runtime_v12.py', '3dd08d0dd25c6fb33cd6176bb6dced6e5ef5dc2c4e8848273eb2345cc636a2e2')
    acceptance = core.bound(core.ACCEPTANCE, '443cffa69bc28e743d6cd4d5552a9b751559315b1ec57087260ec9d013ac77e3')
    rows, _ = core.matrix_rows()
    capacity = core.capacity_snapshot(rows, core.read(core.ACCEPTANCE))
    out = ROOT/'queue-20261030'/('resource-capacity-'+label+'.json')
    result = dict(observed_epoch=time.time(), capacity=capacity, acceptance=acceptance,
                  observer=core.bound(__file__),
                  shortfall_bytes=max(0, capacity['pending_all_categories_plus_headroom']-capacity['free_bytes']),
                  process_control=False, gpu_use=False, continuation_authorized_by_this_observation=False)
    with out.open('x', encoding='utf8') as f:
        json.dump(result, f, indent=2)
    request_path = ROOT/'queue-20261030/execution-request.json'
    request = core.read(request_path)
    request['resource_capacity_observation'] = core.bound(out)
    if request.get('resource_hold_v24'):
        request['resource_hold_v24'].update(latest_capacity=core.bound(out),
            shortfall_bytes=result['shortfall_bytes'], budget_now_fits=capacity['fits'])
    temp = request_path.with_name('execution-request.'+label+'.capacity-v24.tmp')
    temp.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding='utf8')
    os.replace(temp, request_path)
    print(json.dumps(dict(path=str(out), capacity=capacity, shortfall_bytes=result['shortfall_bytes'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    observe(parser.parse_args().label)
