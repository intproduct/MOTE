"""Validate every exported row and preservation of source message contents."""
import argparse,collections,hashlib,heapq,json,re,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'Datasets/expansion_20261016/cache/python_deps'))
from jsonschema import Draft202012Validator
from normalize_records import sha,compact,supervision_signature
from evidence_checks import explicit_negative_evidence

def check_preservation(r):
 conversion=r['metadata']['message_conversion'];assert conversion['truncated'] is False
 original=[dict(m) for m in r['messages']]
 for action in reversed(conversion['actions']):
  if action['action']=='fuse_reasoning_content_into_explicit_text_section':
   m=original[action['message']];text=m['content'];n=action['reasoning_characters'];assert text.startswith('<think>\n');assert text[8+n:8+n+11]=='\n</think>\n\n'
   m['reasoning_content']=text[8:8+n];m['content']=text[8+n+11:];assert len(m['content'])==action['original_content_characters']
  elif action['action']=='attach_top_level_auxiliary_reasoning':
   m=original[action['message']];assert m['reasoning_content']==r['auxiliary_reasoning']
   if action['previous_reasoning_present']:m['reasoning_content']=action['previous_reasoning_content']
   else:m.pop('reasoning_content')
 for field in conversion['source_fields']:
  m=original[field['index']];assert sha(compact(m.get('content')))==field['content_sha256'];assert sha(compact(m.get('reasoning_content')))==field['reasoning_sha256']
 assert sha(compact(original))==conversion['source_messages_sha256']
 assert sha(compact(r['messages']))==r['record_content_sha256']
 if 'source_auxiliary_reasoning_sha256' in conversion:
  assert 'auxiliary_reasoning' in r
  assert sha(compact(r['auxiliary_reasoning']))==conversion['source_auxiliary_reasoning_sha256']
 # Optional for compatibility with the already-running old-format export.
 if 'supervision_sha256' in r:assert r['supervision_sha256']==supervision_signature(r)

def main():
 p=argparse.ArgumentParser();p.add_argument('--dest',type=Path,default=ROOT/'Datasets/processed/shared_sft_v02_20261016');p.add_argument('--materialize-views',action='store_true');a=p.parse_args();dest=a.dest
 schema=json.loads((ROOT/'tools/dataset_expansion/schema_v2.json').read_text(encoding='utf-8'));validator=Draft202012Validator(schema);summary=json.loads((dest/'audit/merge_summary.json').read_text(encoding='utf-8'));counts=collections.Counter();domains=collections.defaultdict(collections.Counter);lengths=collections.defaultdict(list);ids=set();families={};eligible_content=set();manifest=[];failures=[];start=time.monotonic()
 views={};view_manifest={};pilot_heaps=collections.defaultdict(list);pilot_family_sets=collections.defaultdict(set);pilot_quotas={'math':500,'code':400,'logic':250,'reading':250,'instruction':250,'physics':150,'science_other':200}
 distinct_problems=collections.defaultdict(set);distinct_families=collections.defaultdict(set)
 conversation_lengths=collections.defaultdict(list);supervised_lengths=collections.defaultdict(list)
 policy_path=ROOT/'Datasets/expansion_20261016/source_admission_policy.json';source_policies=json.loads(policy_path.read_text(encoding='utf-8'))['sources'] if policy_path.exists() else {}
 def pilot_group(domain):
  if domain in ['math','code','logic','reading','physics']:return domain
  if domain in ['instruction','instruction_following','writing','general_chat','language_scienceqa','language']:return 'instruction'
  return 'science_other'
 for path in sorted((dest/'jsonl').glob('*.jsonl')):
  name=path.stem;digest=hashlib.sha256();rows=0
  with path.open('rb',buffering=16*1024*1024) as f:
   for number,line in enumerate(f,1):
    digest.update(line);rows+=1
    try:
     r=json.loads(line);validator.validate(r);assert r['id'] not in ids, 'duplicate_record_id';ids.add(r['id']);check_preservation(r)
     assert len(r['assistant_loss_mask'])==len(r['messages']);assert not r['metadata']['tokenizer_processing_performed'];assert r['verification']['process_verified'] is False
     if r['response_mode']=='prompt_completion':
      assert len(r['messages'])==2;assert r['input']==r['messages'][0]['content'];assert r['output']==r['messages'][-1]['content']
     else:assert r['input'] is None
     if r['release_status']=='eligible':
      assert r['source'] not in source_policies,'source_level_hold_bypassed'
      negative=explicit_negative_evidence(r);assert not negative,('explicit_negative_evidence',negative)
      assert name==r['split'] and name in ['train','validation','test','smoke'];assert not r['review_flags'];assert r['quality_tier'] not in ['curated_unverified','unverified'];assert r['messages'][-1]['role']=='assistant';assert any(r['assistant_loss_mask']);assert all(not active or r['messages'][i]['role']=='assistant' for i,active in enumerate(r['assistant_loss_mask']));assert all(isinstance(m['content'],str) for m in r['messages'])
      training_key=supervision_signature(r);assert training_key not in eligible_content,'duplicate_eligible_supervision';eligible_content.add(training_key);old=families.setdefault(r['family_id'],name);assert old==name,'family_crosses_splits';assert not r['metadata'].get('benchmark_matches')
      domains[name][r['domain']]+=1;lengths[r['domain']].append(len(r['output']));counts['eligible']+=1
      conversation_lengths[r['domain']].append(sum(len(m['content']) for m in r['messages']))
      supervised_lengths[r['domain']].append(sum(len(m['content']) for m,active in zip(r['messages'],r['assistant_loss_mask']) if active))
      distinct_problems[(name,r['domain'])].add(r['problem_id']);distinct_families[(name,r['domain'])].add(r['family_id'])
      if a.materialize_views:
       safe_domain=re.sub(r'[^a-zA-Z0-9_-]','_',r['domain']);key=(safe_domain,name)
       if key not in views:
        vp=dest/'by_domain'/safe_domain/(name+'.jsonl');vp.parent.mkdir(parents=True,exist_ok=True);views[key]=vp.open('wb',buffering=8*1024*1024);view_manifest[key]={'path':str(vp.relative_to(dest)).replace('\\','/'),'rows':0,'bytes':0,'digest':hashlib.sha256()}
       views[key].write(line);vm=view_manifest[key];vm['rows']+=1;vm['bytes']+=len(line);vm['digest'].update(line)
       if name=='train':
        group=pilot_group(r['domain']);heap=pilot_heaps[group];seen=pilot_family_sets[group];score=int(sha('inspection-v2:'+r['id']),16);quota=pilot_quotas[group]
        if r['family_id'] not in seen:
         entry=(-score,r['id'],r['family_id'],line)
         if len(heap)<quota:heapq.heappush(heap,entry);seen.add(r['family_id'])
         elif score < -heap[0][0]:removed=heapq.heapreplace(heap,entry);seen.remove(removed[2]);seen.add(r['family_id'])
     elif name=='duplicates':assert r['release_status']=='duplicate'
     else:assert name==r['release_status']
     counts[name]+=1
    except Exception as e:
     failures.append({'file':str(path.relative_to(dest)),'line':number,'error':type(e).__name__+': '+str(e)[:500]})
     if len(failures)>=20:break
    if number%20000==0:print(json.dumps({'phase':'validate','file':name,'rows':number,'failures':len(failures),'elapsed_seconds':round(time.monotonic()-start)}),flush=True)
  manifest.append({'path':str(path.relative_to(dest)).replace('\\','/'),'rows':rows,'bytes':path.stat().st_size,'sha256':digest.hexdigest()})
  print(json.dumps({'validated_file':name,'rows':rows,'failures':len(failures)}),flush=True)
  if failures:break
 for f in views.values():f.close()
 if a.materialize_views and not failures:
  vms=[]
  for key,vm in sorted(view_manifest.items()):vm['sha256']=vm.pop('digest').hexdigest();vm['domain'],vm['split']=key;vms.append(vm)
  (dest/'audit/domain_views_manifest.json').write_text(json.dumps(vms,indent=2),encoding='utf-8');(dest/'inspection').mkdir(exist_ok=True);pilot_path=dest/'inspection/stratified_train_sample.jsonl';h=hashlib.sha256();pilot_counts={}
  with pilot_path.open('wb',buffering=8*1024*1024) as f:
   for group,heap in sorted(pilot_heaps.items()):
    pilot_counts[group]=len(heap)
    for _,rid,family,line in sorted(heap,reverse=True):f.write(line);h.update(line)
  (dest/'inspection/manifest.json').write_text(json.dumps({'path':pilot_path.relative_to(dest).as_posix().replace('\\','/'),'rows':sum(pilot_counts.values()),'sha256':h.hexdigest(),'groups':pilot_counts,'requested_quotas':pilot_quotas,'selection':'Deterministic bounded hash sample; at most one selected record per family within each display group. Entire training messages preserved. Not a statistically representative error-rate estimate or prescribed training mixture.','subset_of':'jsonl/train.jsonl','do_not_count_as_new_data':True},indent=2),encoding='utf-8')
 if not failures:
  for name,n in summary['files'].items():
   if counts[name]!=n:failures.append({'error':'row_count_differs_from_merge','file':name,'expected':n,'actual':counts[name]})
 def distribution(values):
  values.sort();return {'rows':len(values),'characters_min':values[0],'characters_p50':values[int((len(values)-1)*.5)],'characters_p95':values[int((len(values)-1)*.95)],'characters_p99':values[int((len(values)-1)*.99)],'characters_max':values[-1],'unit':'Unicode characters; not model tokens'}
 report={'passed':not failures,'rows':len(ids),'counts':dict(counts),'eligible_unique_families':len(families),'source_message_preservation_checked_every_row':not failures,'output_files':manifest,'domain_counts':{k:dict(v) for k,v in domains.items()},'eligible_response_character_lengths':{k:distribution(v) for k,v in lengths.items()},'failures':failures,'elapsed_seconds':round(time.monotonic()-start,2),'limits':['Schema and content-preservation validation does not establish semantic correctness.','Family and benchmark matching are conservative lexical checks, not an exhaustive semantic contamination audit.','Response lengths are characters; no tokenizer, truncation, packing or model template has been applied.']}
 report['distinct_questions_and_families']=[{'split':split,'domain':domain,'questions':len(values),'families':len(distinct_families[(split,domain)])} for (split,domain),values in sorted(distinct_problems.items())]
 report['eligible_conversation_character_lengths']={k:distribution(v) for k,v in conversation_lengths.items()}
 report['eligible_supervised_content_character_lengths']={k:distribution(v) for k,v in supervised_lengths.items()}
 report['length_scope']='Unicode message-content characters only; conversation includes context, supervised content follows assistant_loss_mask. Auxiliary reasoning in separate fields and future model template/special-token overhead are not included. No token estimate.'
 (dest/'audit/validation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({k:report[k] for k in ['passed','rows','counts','failures','elapsed_seconds']}),flush=True)
 if failures:raise SystemExit(1)
if __name__=='__main__':main()
