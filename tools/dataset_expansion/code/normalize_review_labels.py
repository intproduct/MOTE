"""Finalize excluded-record labels and manifests in one buffered sequential pass.

Candidate and unchanged source/duplicate hashes were already verified separately;
reuse that evidence instead of re-reading gigabytes on a shared mechanical disk.
"""
import collections,hashlib,json,datetime
from fetch_sources import DEST

def main():
 counts=collections.Counter();files=[];buf=16*1024*1024
 previous=json.loads((DEST/'audit/final_validation.json').read_text(encoding='utf-8'))
 (DEST/'audit/validation_before_prompt_example_refinement.json').write_text(json.dumps(previous,indent=2),encoding='utf-8')
 candidate={'path':'Datasets/expansion_20261016/code/candidates.jsonl','rows':56466,'bytes':1365197057,'sha256':'9c9e09b67e667d69b5b993b7e2c97a35355e0a0ce14260d0f71fcc85b7e9c1d2'}
 assert (DEST/'candidates.jsonl').stat().st_size==candidate['bytes']
 files.append(candidate)
 for name in ['review','quarantine']:
  p=DEST/(name+'.jsonl');tmp=DEST/(name+'.labels.jsonl')
  digest=hashlib.sha256();rows=0;size=0
  with p.open('rb',buffering=buf) as inp,tmp.open('wb',buffering=buf) as out:
   for line in inp:
    r=json.loads(line);quality='source_verified' if r['verification'].get('source_execution_pass') is True else 'unverified'
    assert r['verification']['local_execution_performed'] is False
    if r['quality_level']!=quality:
     r['quality_level']=quality;counts[name]+=1;line=(json.dumps(r,ensure_ascii=False)+'\n').encode('utf-8')
    out.write(line);digest.update(line);size+=len(line);rows+=1
  tmp.replace(p)
  files.append({'path':'Datasets/expansion_20261016/code/'+name+'.jsonl','rows':rows,'bytes':size,'sha256':digest.hexdigest()})
  print(json.dumps(files[-1]),flush=True)
 duplicate=next(f for f in previous['files'] if 'duplicate_alternatives' in f['path'])
 assert (DEST/'duplicate_alternatives.jsonl').stat().st_size==duplicate['bytes']
 files.append(duplicate)
 assert sum(f['rows'] for f in files)==177531
 assert files[1]['rows']==50989 and files[2]['rows']==36429
 manifest={'date':'2026-10-16','files':files,'status':'domain_candidates_before_central_benchmark_decontamination'}
 evidence={
  'finalized_at':datetime.datetime.now().isoformat(),
  'candidate_rows':56466,'candidate_unique_questions':56466,'candidate_unique_upstream_families':56466,
  'candidate_sources':{'nvidia/OpenCodeReasoning-2':5897,'KodCode/KodCode-V1-SFT-R1':50569},
  'raw_full_file_checksums_verified':9,
  'validation_scope':{
   'review_quarantine':'Every row parsed; quality label normalized; local-execution flag checked; output SHA computed during this final sequential rewrite.',
   'candidates':'Full SHA computed after final manual exclusion; final size checked here. Earlier unique-question/family and evidence checks preserved by subset removals and reviewed record restoration; central release scan performs final independent validation.',
   'duplicates':'Unchanged since preceding full parse and SHA validation; final size checked here.',
   'raw_sources':'Nine full raw file hashes previously validated against pinned source manifests; raw files unchanged, hashes not redundantly rescanned here.',
   'question_projections':'Selected HTTP ranges and projection hashes only; no claim of full upstream parquet-file verification.'},
  'source_test_pass_required':True,'tests_have_static_result_link_when_available':True,
  'local_execution_performed':False,'tokenizer_processing_performed':False,'output_truncated_locally':False,
  'upstream_finish_reason_available':False,'central_benchmark_decontamination_pending':True,'files':files}
 (DEST/'MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 (DEST/'audit/final_validation.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
 (DEST/'audit/review_label_normalization.json').write_text(json.dumps(dict(counts),indent=2),encoding='utf-8')
 print(json.dumps({'labels_changed':dict(counts),'rows':{f['path'].split('/')[-1]:f['rows'] for f in files}}),flush=True)
if __name__=='__main__':main()
