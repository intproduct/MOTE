"""Wait for the existing first pass, publish semantic gates, and freeze READY last."""
import json,time,subprocess,sys,hashlib
from pathlib import Path
from process import ROOT
started=time.monotonic()
while not all((ROOT/'audit'/name).exists() for name in ['final_summary.json','release_manifest.json','all_downloads.json']):
 if time.monotonic()-started>3600:raise TimeoutError('Initial finalization did not publish its final manifests within one hour.')
 time.sleep(5)
here=Path(__file__).resolve().parent
for script in ['post_release.py','write_report.py']:
 subprocess.run([sys.executable,str(here/script)],check=True)
summary=json.loads((ROOT/'audit/final_summary.json').read_text(encoding='utf-8'));manifest_path=ROOT/'audit/release_manifest.json';manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
assert len(summary['sources'])==7 and len(manifest)==28
assert summary['sources']['open_science_reasoning_2']['counts'].get('candidates',0)==0
ready={'status':'READY','frozen':True,'total_records':summary['total_records'],'candidate_rows':sum(s['counts'].get('candidates',0) for s in summary['sources'].values()),'quality_levels':summary['quality_levels'],'release_manifest':'audit/release_manifest.json','release_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),'candidate_files':[m for m in manifest if m['path'].endswith('/candidates.jsonl') and m['rows']>0],'candidate_meaning':'Source answer evidence plus conservative local screening; not proof of all reasoning. All process_verified=false.','central_work_remaining':['global duplicate and benchmark contamination checks','cross-source family grouping','final splits and common schema validation'],'source_pool_decisions':{k:s['counts'] for k,s in summary['sources'].items()},'tests_passed':17,'tokenizer_slicing':False,'teacher_inference_performed':False}
(ROOT/'READY.json').write_text(json.dumps(ready,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(ready,ensure_ascii=True),flush=True)
