"""Stream sources into one schema, then enforce family splits and quality gates."""
import argparse,collections,hashlib,json,os,re,sqlite3,sys,time
from pathlib import Path
from normalize_records import normalize_record,primary_prompt,sha,norm,compact,supervision_signature
from evidence_checks import explicit_negative_evidence
from benchmark_matcher import BenchmarkMatcher,normalize as benchmark_normalize
from near_families import NearFamilies,strip_math_instruction
from math_content_gates import flags as math_content_flags

ROOT=Path(__file__).resolve().parents[2];EXP=ROOT/'Datasets/expansion_20261016';DEST=ROOT/'Datasets/processed/shared_sft_v02_20261016'

def quality_tier(r):
 q=r['quality_level'];v=r['verification'];domain=r['domain']
 manual=r['metadata'].get('manual_visible_content_review',{})
 if manual.get('decision')=='pass_visible' and manual.get('messages_sha256')==r['metadata'].get('source_messages_sha256'):return 'manually_screened'
 if v.get('method')=='publisher_human_demonstrations':return 'human_authored'
 if q in ['independently_recomputed','locally_recomputed','local_verified']:return 'locally_recomputed'
 if q in ['reference_matched','reference_checked','assistant_reviewed_final'] or v.get('independent_reference') and v.get('answer_pass') is True:return 'reference_checked'
 if q in ['human_authored','human_written','human_curated']:return 'human_authored'
 if q in ['source_verified','prior_execution_pass','source_execution_verified']:
  return 'publisher_constraints_checked' if domain in ['instruction','instruction_following'] else 'publisher_verified'
 if q in ['source_curated','structural_only']:return 'curated_unverified'
 return 'unverified'

def family_keys(r):
 m=r['metadata'];keys=['q:'+r['problem_id'],'context:'+m['context_sha256']]
 family=m.get('source_question_family_id') or m.get('group_key') or m.get('family_key') or m.get('source_family_id')
 if family:keys.append('family:'+r['source']+':'+str(family))
 legacy=m.get('dedup_cluster') or m.get('parent_id')
 if legacy:keys.append('legacy:'+r['source']+':'+str(legacy))
 # Near-identical numeric variants must not cross evaluation splits.
 prompt=benchmark_normalize(strip_math_instruction(primary_prompt(r)));keys.append('prompt_seed:'+sha(prompt));skeleton=re.sub(r'[-+]?\d+(?:\.\d+)?','<N>',prompt)
 if len(prompt)>=80 and skeleton!=prompt:keys.append('template:'+sha(skeleton))
 # Exact problem text across prompt wrappers is recovered when adapters supply it.
 original=m.get('original_question')
 if isinstance(original,str) and len(original)>=40 and r['domain'] not in ['reading','science_grounded']:keys.append('seed:'+sha(benchmark_normalize(original)))
 if r['domain'] in ['instruction','instruction_following','general_chat','writing']:
  first=next((x['content'] for x in r['messages'] if x.get('role')=='user' and isinstance(x.get('content'),str)),'');first=benchmark_normalize(first)
  if len(first)>=80:keys.append('conversation_start:'+r['source']+':'+sha(first))
 return keys

class Groups:
 def __init__(self):self.parent=[];self.keys={}
 def add(self,keys):
  i=len(self.parent);self.parent.append(i)
  for key in keys:
   if key in self.keys:self.union(i,self.keys[key])
   else:self.keys[key]=i
  return i
 def find(self,i):
  while self.parent[i]!=i:self.parent[i]=self.parent[self.parent[i]];i=self.parent[i]
  return i
 def union(self,a,b):
  a,b=self.find(a),self.find(b)
  if a!=b:self.parent[max(a,b)]=min(a,b)

def unique_record_id(record,seen_ids,path,line):
 """Disambiguate reused publisher IDs without dropping either source record."""
 original=record['id'];candidate=original
 if candidate in seen_ids:
  suffix=sha(path+':'+str(line));candidate=original+'--'+suffix
  assert candidate not in seen_ids,('duplicate_ingest_location',path,line)
  record['metadata']['original_record_id']=original
  record['metadata']['id_disambiguation']={'method':'original_id_plus_sha256_source_path_and_line','source_file':path,'line_1based':line}
  record['id']=candidate
 seen_ids.add(candidate)

def recover_committed_state(db,spool,config,groups,near,seen_content,seen_ids,collision_file,near_file):
 """Recover only committed rows; DB dispositions always override spool status.

 No normalization, source verification or benchmark matching is repeated. The
 uncommitted spool suffix is archived before truncation. Derived logs rebuild
 from the committed prefix so stale uncommitted indices cannot survive.
 """
 end=db.execute('SELECT COALESCE(MAX(offset+length),0) FROM records').fetchone()[0]
 assert spool.exists() and spool.stat().st_size>=end,'committed_spool_bytes_missing'
 info={'committed_rows':0,'committed_spool_bytes':end,'archived_tail':None}
 if spool.stat().st_size>end:
  tail=spool.with_name(spool.name+'.uncommitted-'+str(time.time_ns()))
  with spool.open('rb',buffering=16*1024*1024) as src,tail.open('xb',buffering=16*1024*1024) as out:
   src.seek(end)
   while True:
    block=src.read(16*1024*1024)
    if not block:break
    out.write(block)
   out.flush();os.fsync(out.fileno())
  with spool.open('r+b') as f:f.truncate(end)
  info['archived_tail']=str(tail)
 positions={x['path']:i for i,x in enumerate(config['inputs'])};assert len(positions)==len(config['inputs']),'repeated_input_paths_not_resumable'
 committed=collections.Counter();counts=collections.defaultdict(collections.Counter);manual_applied=[];expected_offset=0;last_position=-1
 with spool.open('rb',buffering=32*1024*1024) as sf:
  for idx,rid,offset,length,status,duplicate_of in db.execute('SELECT idx,id,offset,length,status,duplicate_of FROM records ORDER BY idx'):
   assert idx==len(groups.parent) and offset==expected_offset,('noncontiguous_committed_index',idx,offset,expected_offset)
   data=sf.read(length);assert len(data)==length and data.endswith(b'\n'),('incomplete_committed_spool_record',idx)
   r=json.loads(data);assert r['id']==rid,('spool_index_id_mismatch',idx)
   location=r['metadata']['ingest_location'];path=location['file'];number=location['line_1based']
   assert path in positions and positions[path]>=last_position,('resume_config_order_mismatch',path)
   assert number==committed[path]+1,('noncontiguous_source_prefix',path,number)
   assert location['pool']==config['inputs'][positions[path]]['pool'],('resume_pool_changed',path)
   last_position=positions[path];committed[path]=number;counts[path][status]+=1
   assert rid not in seen_ids,('committed_duplicate_id',rid);seen_ids.add(rid)
   assert groups.add(family_keys(r))==idx
   if location['pool']=='candidates' and r['domain'] not in ['reading','logic','science_grounded']:
    for other in near.add(idx,benchmark_normalize(strip_math_instruction(primary_prompt(r)))):
     groups.union(idx,other);near_file.write(json.dumps({'record_index':idx,'same_family_as_index':other,'method':'simhash_hamming_3_and_token_text_ratio_96','responses_deleted':False})+'\n')
   key=supervision_signature(r);old=seen_content.get(key)
   if old is None or (old[2]=='duplicate' and status!='duplicate') or (status=='eligible' and old[2]!='eligible'):seen_content[key]=(idx,rid,status)
   matches=r['metadata'].get('benchmark_matches')
   if matches:collision_file.write(json.dumps({'id':rid,'matches':matches},ensure_ascii=False)+'\n')
   if r['verification'].get('local_visible_content_review'):manual_applied.append(rid)
   expected_offset+=length;info['committed_rows']+=1
   if (idx+1)%10000==0:print(json.dumps({'phase':'resume_rebuild','rows':idx+1,'spool_bytes':expected_offset}),flush=True)
 assert expected_offset==end,'committed_spool_end_mismatch'
 info['committed_input_lines']=dict(committed)
 return committed,counts,manual_applied,info

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--limit-per-file',type=int,default=0);parser.add_argument('--dest',type=Path,default=DEST);parser.add_argument('--no-overlays',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args();dest=args.dest;dest.mkdir(parents=True,exist_ok=True);(dest/'audit').mkdir(exist_ok=True);(dest/'jsonl').mkdir(exist_ok=True)
 config=json.loads(args.config.read_text(encoding='utf-8'));paths=[EXP/'benchmarks/questions.jsonl']+sorted((EXP/'benchmarks/projections').glob('livecodebench_*.jsonl'));matcher=BenchmarkMatcher(paths)
 overlays={};overlay=EXP/'math/audit/symbolic_recheck.jsonl'
 if overlay.exists() and not args.no_overlays:
  assert overlay.with_suffix('.summary.json').exists(),'math_symbolic_audit_not_complete'
  with overlay.open(encoding='utf-8') as f:
   for line in f:
    if line.strip():o=json.loads(line);overlays[o['id']]=o
 manual_path=EXP/'math/audit/independent_manual_findings.json';manual={x['id']:x for x in json.loads(manual_path.read_text(encoding='utf-8'))['findings']} if manual_path.exists() else {};manual_applied=[]
 postgate_path=EXP/'math/audit/postgate_findings.json'
 if postgate_path.exists():manual.update({x['id']:x for x in json.loads(postgate_path.read_text(encoding='utf-8'))['findings']})
 openr1_path=EXP/'math/audit/openr1_postgate_findings.json'
 if openr1_path.exists():manual.update({x['id']:x for x in json.loads(openr1_path.read_text(encoding='utf-8'))['findings']})
 legacy_math_path=EXP/'math/audit/legacy_math_findings.json'
 if legacy_math_path.exists():manual.update({x['id']:x for x in json.loads(legacy_math_path.read_text(encoding='utf-8'))['findings']})
 policy_path=EXP/'source_admission_policy.json';source_policies=json.loads(policy_path.read_text(encoding='utf-8'))['sources'] if policy_path.exists() else {}
 source_origin=json.loads((EXP/'math/audit/local_shard_origin_check.json').read_text(encoding='utf-8'))
 if (dest/'MANIFEST.json').exists():raise FileExistsError('Published release exists; choose a new --dest to preserve it.')
 if not args.resume and (dest/'audit/index.sqlite').exists():raise FileExistsError('An unfinished index exists; use --resume or choose a new --dest.')
 if args.resume:assert (dest/'audit/index.sqlite').exists(),'resume_database_missing'
 db=sqlite3.connect(dest/'audit/index.sqlite');db.execute('PRAGMA journal_mode=WAL')
 if not args.resume:
  db.execute('DROP TABLE IF EXISTS records');db.execute('CREATE TABLE records (idx INTEGER PRIMARY KEY, id TEXT UNIQUE, offset INTEGER, length INTEGER, status TEXT, tier TEXT, original_split TEXT, benchmark INTEGER, problem_hash TEXT, content_hash TEXT, domain TEXT, source TEXT, group_root INTEGER, final_split TEXT, duplicate_of TEXT)');db.commit()
 groups=Groups();near=NearFamilies();seen_content={};seen_ids=set();counts=collections.Counter();summaries=[];spool=dest/'audit/normalized_spool.jsonl';collision_file=(dest/'audit/benchmark_matches.jsonl').open('w',encoding='utf-8');near_file=(dest/'audit/near_family_links.jsonl').open('w',encoding='utf-8');errors=[];start=time.monotonic();committed_lines=collections.Counter();committed_counts=collections.defaultdict(collections.Counter);resume_info=None
 if args.resume:
  committed_lines,committed_counts,restored_manual,resume_info=recover_committed_state(db,spool,config,groups,near,seen_content,seen_ids,collision_file,near_file);manual_applied.extend(restored_manual)
  (dest/'audit/resume_info.json').write_text(json.dumps(resume_info,ensure_ascii=False,indent=2),encoding='utf-8')
 with spool.open('ab' if args.resume else 'wb',buffering=32*1024*1024) as sf:
  for item in config['inputs']:
   if item.get('require_ready'):
    marker=ROOT/item['require_ready'];wait_start=time.monotonic();last_notice=0
    while True:
     ready=False
     if marker.exists():
      try:ready=json.loads(marker.read_text(encoding='utf-8')).get('status')=='READY'
      except json.JSONDecodeError:pass
     if ready:break
     if time.monotonic()-wait_start>1800:raise TimeoutError('Source did not freeze within 30 minutes: '+str(marker))
     if time.monotonic()-last_notice>60:print(json.dumps({'phase':'await_source_freeze','marker':str(marker)}),flush=True);last_notice=time.monotonic()
     time.sleep(5)
   if item.get('manifest'):
    entries=json.loads((ROOT/item['manifest']).read_text(encoding='utf-8'));entries=entries.get('files',[]) if isinstance(entries,dict) else entries
    matches=[x for x in entries if str(Path(item.get('manifest_base',''))/x['path']).replace('\\','/')==item['path']]
    assert len(matches)==1,('input_missing_from_frozen_manifest',item['path']);item={**item,'sha256':matches[0]['sha256'],'rows':matches[0]['rows']}
   path=ROOT/item['path'];before=path.stat();digest=hashlib.sha256();file_counts=committed_counts[item['path']].copy();pool=item['pool'];rows_read=0
   with path.open('rb',buffering=16*1024*1024) as f:
    for number,line in enumerate(f,1):
     if args.limit_per_file and number>args.limit_per_file:break
     digest.update(line);rows_read=number
     if number<=committed_lines[item['path']]:continue
     try:
      raw=json.loads(line);r=normalize_record(raw,item['path'],number,pool)
      unique_record_id(r,seen_ids,item['path'],number)
      if r['source']=='local/OpenMathReasoning_schema_shard' and source_origin['matched']:
       r['metadata']['previous_source_label']=r['source'];r['source']=source_origin['repo'];r['source_revision']=source_origin['revision'];r['source_split']='cot';r['license']='cc-by-4.0';r['metadata']['official_origin_confirmed_by_file_sha256']=True
      r['review_flags'].extend(math_content_flags(r))
      r['quality_tier']=quality_tier(r)
      status='eligible' if pool=='candidates' else 'quarantine' if pool=='quarantine' else 'duplicate' if pool=='duplicates' else 'review_required'
      if status=='eligible' and r['quality_tier'] in ['curated_unverified','unverified']:status='review_required';r['review_flags'].append('insufficient_correctness_or_curation_evidence')
      if r['review_flags'] and status=='eligible':status='quarantine'
      audit=overlays.get(r['id'])
      if audit:
       assert audit['output_sha256']==sha(raw['output']),('math_overlay_hash_mismatch',r['id'])
       r['verification']['local_symbolic_recheck']=audit
       if audit.get('match') is True:r['quality_tier']='reference_checked'
       elif audit.get('match') is False and status=='eligible':status='review_required';r['review_flags'].append('symbolic_recheck_no_match_requires_review')
      if r['id'] in manual:
       review=manual[r['id']];assert sha(raw['input'])==review['input_sha256'] and sha(raw['output'])==review['output_sha256'],('manual_review_content_mismatch',r['id']);r['verification']['local_visible_content_review']=review;manual_applied.append(r['id'])
       if review['action']=='quarantine':status='quarantine';r['review_flags'].append('manual_content_defect:'+review.get('classification','hash_bound_manual_content_review'))
       elif review['action']=='review' and status=='eligible':status='review_required';r['review_flags'].append('manual_content_requires_review:'+review.get('classification','hash_bound_manual_content_review'))
       elif review['action']=='clear_symbolic_mismatch_only':
        r['review_flags']=[x for x in r['review_flags'] if x!='symbolic_recheck_no_match_requires_review'];r['quality_tier']='reference_checked'
        if pool=='candidates' and status=='review_required' and not r['review_flags']:status='eligible'
      if status=='eligible' and r['source'] in source_policies:status=source_policies[r['source']]['disposition']
      if status=='eligible':
       negative=explicit_negative_evidence(r)
       if negative:status='review_required';r['review_flags'].extend('explicit_negative_evidence:'+field for field in negative)
      matches=matcher.match(primary_prompt(r))
      if matches:
       r['review_flags'].append('possible_benchmark_overlap');r['metadata']['benchmark_matches']=matches;status='quarantine';collision_file.write(json.dumps({'id':r['id'],'matches':matches},ensure_ascii=False)+'\n')
      # Do not admit a published evaluation split simply because a source adapter named its output candidates.
      source_split=str(r['source_split'] or '').lower()
      if status=='eligible' and source_split in ['test','validation','dev','evaluation'] and not item.get('legacy',False):status='quarantine';r['review_flags'].append('original_evaluation_split')
      r['release_status']=status;r['review_flags']=sorted(set(r['review_flags']));idx=groups.add(family_keys(r));content_key=r['record_content_sha256'];dedup_key=supervision_signature(r);duplicate_of=None
      if pool=='candidates' and r['domain'] not in ['reading','logic','science_grounded']:
       for other in near.add(idx,benchmark_normalize(strip_math_instruction(primary_prompt(r)))):
        groups.union(idx,other);near_file.write(json.dumps({'record_index':idx,'same_family_as_index':other,'method':'simhash_hamming_3_and_token_text_ratio_96','responses_deleted':False})+'\n')
      if dedup_key in seen_content:
       # Keep an evidence-rich copy; duplicates retain their full source records in a separate file.
       previous=seen_content[dedup_key]
       if status=='eligible' and previous[2]!='eligible':db.execute('UPDATE records SET status=?, duplicate_of=? WHERE idx=?',('duplicate',r['id'],previous[0]));seen_content[dedup_key]=(idx,r['id'],status)
       else:duplicate_of=previous[1];status='duplicate';r['release_status']=status
      else:seen_content[dedup_key]=(idx,r['id'],status)
      data=(json.dumps(r,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode('utf-8');offset=sf.tell();sf.write(data)
      original=r['original_split'];db.execute('INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',(idx,r['id'],offset,len(data),status,r['quality_tier'],original,bool(matches),r['problem_id'],content_key,r['domain'],r['source'],None,None,duplicate_of));counts[status]+=1;file_counts[status]+=1
     except Exception as e:
      errors.append({'file':item['path'],'line':number,'error':type(e).__name__+': '+str(e)})
      # Fail closed instead of silently dropping malformed input or provenance mismatches.
      raise
     if number%10000==0:sf.flush();os.fsync(sf.fileno());db.commit();print(json.dumps({'phase':'normalize','file':item['path'],'rows':number,'all_records':len(groups.parent),'elapsed_seconds':round(time.monotonic()-start)}),flush=True)
   after=path.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),('input_changed_while_reading',str(path))
   assert rows_read>=committed_lines[item['path']],('source_shorter_than_committed_prefix',item['path'])
   if not args.limit_per_file and item.get('sha256'):assert digest.hexdigest()==item['sha256'],('input_sha_mismatch',item['path'])
   if not args.limit_per_file and 'rows' in item:assert sum(file_counts.values())==item['rows'],('input_row_count_mismatch',item['path'])
   summaries.append({'path':item['path'],'pool':pool,'sha256':digest.hexdigest(),'bytes':before.st_size,'rows':sum(file_counts.values()),'counts':dict(file_counts),'limited':bool(args.limit_per_file)})
   sf.flush();os.fsync(sf.fileno());db.commit();print(json.dumps({'phase':'source_complete','file':item['path'],'counts':dict(file_counts)}),flush=True)
 collision_file.close();near_file.close()
 # Resolve connected components once all cross-source links are known.
 group_info={}
 for idx,original,benchmark,pid in db.execute('SELECT idx,original_split,benchmark,problem_hash FROM records'):
  root=groups.find(idx);info=group_info.setdefault(root,{'question_min':pid,'reserved':set(),'benchmark':False});info['question_min']=min(info['question_min'],pid);info['benchmark']|=bool(benchmark)
  if original in ['dev','validation','diagnostic_dev','test','smoke','smoke_only']:info['reserved'].add(original)
 splits={}
 for root,info in group_info.items():
  family='family-'+info['question_min'][:32]
  if info['benchmark']:split='excluded'
  elif 'test' in info['reserved']:split='test'
  elif info['reserved'] & {'dev','validation','diagnostic_dev'}:split='validation'
  elif info['reserved']:split='smoke'
  else:
   bucket=int(sha(family)[:8],16)%100;split='test' if bucket<5 else 'validation' if bucket<10 else 'train'
  splits[root]=(family,split)
 updates=[(groups.find(i),splits[groups.find(i)][1],i) for i in range(len(groups.parent))];db.executemany('UPDATE records SET group_root=?,final_split=? WHERE idx=?',updates);db.commit()
 streams={};output_counts=collections.Counter();domain_counts=collections.defaultdict(collections.Counter);tier_counts=collections.defaultdict(collections.Counter);source_counts=collections.defaultdict(collections.Counter);license_counts=collections.defaultdict(collections.Counter);flag_counts=collections.defaultdict(collections.Counter);actual_input_counts=collections.defaultdict(collections.Counter)
 for name in ['train','validation','test','smoke','review_required','quarantine','duplicates']:streams[name]=(dest/'jsonl'/(name+'.jsonl')).open('wb',buffering=8*1024*1024)
 with spool.open('rb',buffering=32*1024*1024) as f:
  for idx,offset,length,status,root,split,duplicate_of in db.execute('SELECT idx,offset,length,status,group_root,final_split,duplicate_of FROM records ORDER BY idx'):
   f.seek(offset);r=json.loads(f.read(length));r['family_id']=splits[root][0];r['split']=split;r['release_status']=status
   if split=='excluded' and status!='duplicate':status='quarantine';r['release_status']=status;r['review_flags']=sorted(set(r['review_flags']+['benchmark_family_excluded']))
   target=split if status=='eligible' else 'duplicates' if status=='duplicate' else status
   actual_input_counts[r['metadata']['ingest_location']['file']][status]+=1
   if duplicate_of:r['metadata']['duplicate_of']=duplicate_of
   streams[target].write((json.dumps(r,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode('utf-8'));output_counts[target]+=1;domain_counts[target][r['domain']]+=1;tier_counts[target][r['quality_tier']]+=1;source_counts[target][r['source']]+=1;license_counts[target][compact(r['license'])]+=1;flag_counts[target].update(r['review_flags'])
   if (idx+1)%20000==0:print(json.dumps({'phase':'export','rows':idx+1,'total':len(groups.parent)}),flush=True)
 for f in streams.values():f.close()
 db.execute("UPDATE records SET status='quarantine' WHERE final_split='excluded' AND status!='duplicate'");db.commit()
 for item in summaries:item['counts']=dict(actual_input_counts[item['path']]);item['rows']=sum(item['counts'].values())
 db.close();report={'schema_version':'mote-sft-jsonl-2.0','input_records':len(groups.parent),'families':len(group_info),'benchmark_index_rows':len(matcher.rows),'math_symbolic_overlays':len(overlays),'files':dict(output_counts),'domain_counts':{k:dict(v) for k,v in domain_counts.items()},'quality_tiers':{k:dict(v) for k,v in tier_counts.items()},'source_counts':{k:dict(v) for k,v in source_counts.items()},'license_counts':{k:dict(v) for k,v in license_counts.items()},'input_files':summaries,'errors':errors,'elapsed_seconds':round(time.monotonic()-start,2),'no_tokenizer_processing':True,'no_response_truncation':True,'source_quality_is_not_process_correctness':True}
 report['lexical_near_family_grouping']={'indexed_records':len(near.entries),'links':near.links,'overfull_bucket_lookups_skipped':near.overfull_buckets,'semantic_duplicate_guarantee':False}
 report['flag_counts']={k:dict(v) for k,v in flag_counts.items()}
 report['manual_math_reviews_applied']=manual_applied
 report['resume']=resume_info;report['id_disambiguation']='Keep first ID; subsequent reused IDs get a stable source-path/line SHA256 suffix; original ID retained in metadata.'
 (dest/'audit/merge_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({'complete':True,'counts':dict(output_counts),'elapsed_seconds':report['elapsed_seconds']}),flush=True)

if __name__=='__main__':main()
