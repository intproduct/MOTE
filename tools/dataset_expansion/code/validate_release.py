"""Full streaming integrity checks and manifests; no dataset program execution."""
import collections,hashlib,json
import pyarrow.parquet as pq
from fetch_sources import ROOT,DEST,RAW

def main():
 ids=set();questions=set();families=set();counts=collections.Counter();files=[];sample_ids=[];checked_raw=0
 for name in ['candidates','review','quarantine','duplicate_alternatives']:
  p=DEST/(name+'.jsonl');h=hashlib.sha256();rows=0
  with p.open('rb') as f:
   for line in f:
    h.update(line);r=json.loads(line);rows+=1
    assert r['domain']=='code' and r['source_revision'] and r['provenance']
    if name=='candidates':
     assert r['id'] not in ids and r['problem_id'] not in questions and r['family_id'] not in families
     ids.add(r['id']);questions.add(r['problem_id']);families.add(r['family_id']);assert not r['review_flags'] and r['verification']['source_execution_pass'] is True and r['verification']['local_execution_performed'] is False
     assert r['input'] and r['input']!='-' and r['original_split']=='train'
     assert r['verification']['solution_sha256']==hashlib.sha256(r['solution'].encode()).hexdigest()
     assert r['output'][slice(*r['output_sections']['final'])]==r['final']
     assert r['output'].count('<think>')==1 and r['output'].count('</think>')==1
     if r['source']=='KodCode/KodCode-V1-SFT-R1':assert r['verification']['local_test_static_evidence']['eligible'] is True
     counts[r['source']]+=1
     if len(sample_ids)<12:sample_ids.append(r['id'])
  files.append({'path':str(p.relative_to(ROOT)).replace('\\','/'),'rows':rows,'bytes':p.stat().st_size,'sha256':h.hexdigest()})
 # Independently hash complete source files against the pinned HF LFS metadata.
 for manifest_name in ['source_manifest.json','kodcode_source_manifest.json','apps_source_manifest.json']:
  data=json.loads((DEST/'audit'/manifest_name).read_text(encoding='utf-8'))
  for entry in data.get('files',[data]):
   p=ROOT/entry['local_path'];digest=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
   assert digest==entry['sha256']==entry['lfs']['oid'];checked_raw+=1
 summary=json.loads((DEST/'audit/processing_summary.json').read_text(encoding='utf-8'))
 assert sum(x['rows'] for x in files)==summary['counts']['raw_rows']
 for x in files:assert x['rows']==summary['counts'][x['path'].rsplit('/',1)[-1][:-6]]
 report={'candidate_rows':len(ids),'candidate_unique_questions':len(questions),'candidate_unique_upstream_families':len(families),'candidate_sources':dict(counts),'raw_full_file_checksums_verified':checked_raw,'all_rows_parsed':True,'candidate_ids_unique':True,'candidate_questions_unique':True,'source_test_pass_required':True,'tests_have_static_result_link_when_available':True,'local_execution_performed':False,'tokenizer_processing_performed':False,'output_truncated_locally':False,'files':files}
 (DEST/'audit/final_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 (DEST/'MANIFEST.json').write_text(json.dumps({'date':'2026-10-16','files':files,'status':'domain_candidates_before_central_benchmark_decontamination'},indent=2),encoding='utf-8')
 print(json.dumps(report),flush=True)
if __name__=='__main__':main()
