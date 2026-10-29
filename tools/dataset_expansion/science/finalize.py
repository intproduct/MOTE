"""Apply final source-level decisions, content-bound audits and full-record validation."""
import json,collections,hashlib,heapq,time
from pathlib import Path
from process import ROOT,sha,flags_for,classify
findings={r['id']:r for r in json.loads((ROOT/'audit/manual_findings.json').read_text(encoding='utf-8'))}
DEST=ROOT/'release';DEST.mkdir(exist_ok=True)
KINDS=['candidates','review','quarantine','tool_trajectories_review']
summaries={};manifest=[];manual_applied=[];samples=collections.defaultdict(list);all_ids=set()
for folder in sorted((ROOT/'normalized').iterdir()):
 if not folder.is_dir():continue
 summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'));out=DEST/folder.name;out.mkdir(exist_ok=True);streams={k:(out/(k+'.jsonl')).open('w',encoding='utf-8',newline='\n') for k in KINDS};counts=collections.Counter();domains=collections.defaultdict(collections.Counter);qualities=collections.Counter();flags=collections.Counter();n=0;last=time.monotonic()
 for old in KINDS:
  with (folder/(old+'.jsonl')).open(encoding='utf-8') as src:
   for line in src:
    r=json.loads(line);n+=1;assert r['id'] not in all_ids;all_ids.add(r['id']);assert r['source_split']=='train';assert r['input'] and r['output'] or old=='quarantine'
    extra=flags_for(r['input'],r['output'])
    if folder.name in ['sciq_train','openstax_physics_train']:extra=[x for x in extra if x!='very_short_solution_needs_review']
    r['review_flags']=sorted(set(r['review_flags']+extra))
    if r['id'] in findings:
     f=findings[r['id']];assert sha(r['input'])==f['input_sha256'];assert sha(r['output'])==f['output_sha256'];r['verification']['manual_audit']={k:f[k] for k in ['verdict','reason','input_sha256','output_sha256']};manual_applied.append(r['id'])
     if f['verdict'].startswith('incorrect'):
      r['verification']['reference_match_before_manual_audit']=r['verification']['answer_pass'];r['verification']['answer_pass']=False;r['review_flags'].append('known_incorrect_solution')
     elif f['verdict'].startswith('missing_context'):r['review_flags'].append('manually_confirmed_missing_context')
    if old=='tool_trajectories_review':bucket=old
    else:bucket=classify(r,old=='candidates' and folder.name not in ['textbook_reasoning','physicseval','nemotron_sft_science_v2','open_science_reasoning_2'])
    if old=='quarantine':bucket='quarantine';r['release_status']='quarantine'
    if bucket=='candidates':assert not r['review_flags'];assert r['verification']['answer_pass'] is True
    # Losslessness: all source output (or transparent foundation transformation) must remain intact.
    assert sha(r['output'])==r['provenance'][0].get('normalized_output_sha256',r['provenance'][0].get('raw_output_sha256'))
    for field in ['id','input','output','final','domain','source','source_revision','license','quality_level']:assert isinstance(r[field],str),(r['id'],field)
    assert r['solution'] is None and r['verification']['process_verified'] is False
    streams[bucket].write(json.dumps(r,ensure_ascii=False)+'\n');counts[bucket]+=1;domains[bucket][r['domain']]+=1;qualities[r['quality_level']]+=1;flags.update(r['review_flags'])
    if bucket=='candidates' or (folder.name=='nemotron_sft_science_v2' and bucket=='review'):
     key=(folder.name,r['domain']);item=(-int(r['problem_id'],16),r['id'],r)
     if len(samples[key])<10:heapq.heappush(samples[key],item)
     elif item>samples[key][0]:heapq.heapreplace(samples[key],item)
    if time.monotonic()-last>30:print(json.dumps({'source':folder.name,'processed':n}),flush=True);last=time.monotonic()
 for f in streams.values():f.close()
 summary['counts']=dict(counts);summary['domain_counts']={k:dict(v) for k,v in domains.items()};summary['review_flags']=dict(flags);summary['quality_levels']=dict(qualities);summary['records_after_source_exact_dedup']=n;summary['final_gates_applied']=True
 (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');summaries[folder.name]=summary
 for k in KINDS:
  p=out/(k+'.jsonl');h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
  manifest.append({'path':str(p.relative_to(ROOT)).replace('\\','/'),'rows':counts[k],'bytes':p.stat().st_size,'sha256':h.hexdigest()})
 print(json.dumps({'source_finished':folder.name,'counts':counts}),flush=True)
with (ROOT/'audit/final_stratified_samples.jsonl').open('w',encoding='utf-8') as f:
 for key,items in samples.items():
  for _,__,r in sorted(items,reverse=True):f.write(json.dumps(r,ensure_ascii=False)+'\n')
(ROOT/'audit/final_summary.json').write_text(json.dumps({'sources':summaries,'total_records':len(all_ids),'manual_audits_applied':manual_applied,'checks':{'unique_ids':True,'source_train_only':True,'full_output_preserved':True,'candidate_has_no_flags_and_pass':True,'no_process_verification_claim':True,'tokenizer_slicing_performed':False}},ensure_ascii=False,indent=2),encoding='utf-8')
(ROOT/'audit/release_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
(ROOT/'audit/all_downloads.json').write_text(json.dumps([json.loads(p.read_text(encoding='utf-8')) for p in sorted((ROOT/'audit').glob('*.download.json'))],ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'done':True,'total_records':len(all_ids),'files':len(manifest),'manual_audits_applied':len(manual_applied)}),flush=True)
