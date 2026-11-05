"""One-time immutable numerical-worker copies and tested orchestration release."""
import json
from pathlib import Path
import shutil
import time
import xml.etree.ElementTree as ET
import diagnostic_plan_v14 as p

c=p.core;R=c.ROOT/'runtime';B=R/'formal-diagnostics-v14-release-20261031'

def write(path,value):
    with path.open('x',encoding='utf8') as f:json.dump(value,f,indent=2);f.write('\n')

def main():
    old=p.dependencies()
    suite=ET.parse(R/'diagnostics-v14-cpu-r2.xml').getroot().find('testsuite')
    assert int(suite.attrib['tests'])==50 and all(int(suite.attrib[k])==0 for k in ('errors','failures','skipped'))
    assert not B.exists() and not p.WORKERS.exists()
    files={};provenance={}
    for kind in p.KINDS:
        item=old['diagnostic_bundles'][kind];manifest=Path(item['manifest']);value=c.read(manifest)
        for name,h in value['files'].items():
            if not name.endswith('.py') or name.startswith(('test_','run_')) or name=='production_resource_preflight_v3.py':continue
            if name in files:assert files[name]==h
            files[name]=h;provenance.setdefault(name,[]).append(dict(source=str(manifest.parent/name),sha256=h,manifest=c.bound(manifest)))
    p.WORKERS.mkdir()
    for name,refs in provenance.items():
        source=Path(refs[0]['source']);c.bound(source,files[name]);shutil.copyfile(source,p.WORKERS/name);c.bound(p.WORKERS/name,files[name])
    adapter=R/'formal_diagnostic_budget_v14.py'
    shutil.copyfile(adapter,p.WORKERS/'production_resource_preflight_v3.py')
    files['production_resource_preflight_v3.py']=c.sha(adapter)
    write(p.WORKERS/'MANIFEST.json',dict(schema='qwen-formal-diagnostic-workers-v14',created_epoch=time.time(),files=files,
        numerical_worker_sources_unchanged=True,numerical_provenance=provenance,formal_supervision_manifest_sha256=p.V12_HASH,
        resource_adapter_source=c.bound(adapter),scope='Only resource accounting interface changes; all numerical workers and numerical dependencies are byte-identical to independently GPU-audited originals. Final acceptance is mandatory.'))
    B.mkdir()
    names=['diagnostic_plan_v14.py','run_diagnostics_v14.py','formal_diagnostic_budget_v14.py','test_diagnostics_v14.py','freeze_diagnostics_v14.py','diagnostics-v14-cpu-r2.xml']
    for name in names:shutil.copyfile(R/name,B/name)
    write(B/'MANIFEST.json',dict(schema='qwen-diagnostic-controller-v14',created_epoch=time.time(),files={name:c.sha(B/name) for name in names},
        worker_manifest_sha256=c.sha(p.WORKERS/'MANIFEST.json'),cpu_tests=50,formal_supervision_manifest_sha256=p.V12_HASH,
        original_auditor_manifest_sha256=p.OLD_HASH,initial_missing_jobs=15,formal_jobs=270,adopted_initial_cells=60,
        mode_policy='Initial mode performs only missing FP16 local diagnostics before acceptance. Formal mode requires final acceptance and complete 60 initial references, then 270 checkpoint-bound cells. No automatic retry or process termination.',
        formal_approved=False))
    print(json.dumps(dict(controller=c.bound(B/'MANIFEST.json'),workers=c.bound(p.WORKERS/'MANIFEST.json'),numerical_files_unchanged=len(provenance))))

if __name__=='__main__':main()
