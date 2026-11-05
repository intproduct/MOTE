"""Formal-only resource adapter, copied as production_resource_preflight_v3.py.

Numerical diagnostic workers remain byte-identical. The legacy check() name is
retained only as an interface; returned provenance explicitly names this policy.
"""
import os
from pathlib import Path
import sys

ROOT=Path('E:/AI/MOTE-quantum')
V12=ROOT/'runtime/formal-supervision-v12-release-20261031'
V12_HASH='54d23c18597285fc9d35f8bce3c5dde7e34c1bf978da6933b832ace0b3e5f1f2'
sys.path.insert(0,str(V12))
import formal_runtime_v12 as core


def flag(words,name):
    core.require(words.count(name)==1 and words.index(name)+1<len(words),'Missing/ambiguous diagnostic '+name)
    return words[words.index(name)+1]


def bind_context(rows,configuration,words):
    path=Path(configuration)
    core.require(path.resolve().parent==core.MATRIX.parent.resolve(),'Unrecognized formal configuration directory')
    item=next((x for x in core.read(core.MATRIX)['configurations'] if Path(x['path']).resolve()==path.resolve()),None)
    core.require(item is not None,'Configuration absent from frozen matrix');core.bound(path,item['sha256'])
    config=core.read(path);row=next((r for r in rows if r['name']==config['name']),None)
    core.require(row==config,'Configuration changed')
    output=Path(flag(words,'--output')).resolve()
    core.require(output.parent==(core.DIAGNOSTICS/row['name']).resolve(),'Diagnostic output is not this formal run')
    core.require(flag(words,'--gpu')==row['gpu_uuid'],'Formal diagnostic device differs')
    checkpoint=Path(flag(words,'--checkpoint')).resolve()
    core.require(checkpoint.parent==(Path(row['output'])/'checkpoints').resolve(),'Checkpoint belongs to another run')
    core.require(checkpoint.name in {f'checkpoint-{t:010d}' for t in (3276800,16384000,32768000)},'Not a protected formal checkpoint')
    return row,output,checkpoint


def check():
    core.bound(V12/'MANIFEST.json',V12_HASH)
    for name,h in core.read(V12/'MANIFEST.json')['files'].items():core.bound(V12/name,h)
    rows,_,approval,acceptance=core.validate_acceptance()
    bundle=Path(__file__).parent/'MANIFEST.json'
    core.bound(bundle,approval['formal_diagnostic_worker_manifest_sha256'])
    for name,h in core.read(bundle)['files'].items():core.bound(bundle.parent/name,h)
    row,output,checkpoint=bind_context(rows,os.environ.get('QWEN_DIAGNOSTIC_FORMAL_CONFIG',''),sys.argv)
    budget=core.capacity_snapshot(rows,approval)
    return dict(policy='formal-v14-global-reservation',policy_sha256=core.sha(__file__),
                acceptance=acceptance,formal_configuration=row['name'],output=str(output),checkpoint=str(checkpoint),
                global_capacity=budget,acceptance_launch_fits=budget['fits'],
                processing_actual_plus_reserved_bytes=budget['processing_actual_plus_reserved_bytes'],
                free_bytes=budget['free_bytes'],
                note='Legacy interface name; replaces obsolete completed acceptance allowances with the bound all30-run global reservation. Worker-local extra processing guard remains conservative.')
