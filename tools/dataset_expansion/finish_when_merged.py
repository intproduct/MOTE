"""Bounded continuation of the current local release job."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
summary=ROOT/'Datasets/processed/shared_sft_v02_20261016/audit/merge_summary.json'
started=time.monotonic()
while not summary.exists():
 if time.monotonic()-started>7200:raise TimeoutError('Central merge did not finish within the continuation budget')
 time.sleep(10)
report=json.loads(summary.read_text(encoding='utf-8'))
assert report['input_records']>900000 and not report['errors']
for script,args in [('validate_release.py',['--materialize-views']),('finalize_release.py',[]),('package_training.py',[])]:
 print(json.dumps({'starting':script}),flush=True)
 subprocess.run([sys.executable,'-u',str(ROOT/'tools/dataset_expansion'/script),*args],cwd=ROOT,check=True)
print(json.dumps({'release_pipeline_complete':True}),flush=True)
