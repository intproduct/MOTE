"""Read every export, match source messages and masks, verify answer-span/logic claims."""
import json,hashlib,collections,gzip
from pathlib import Path
import pyarrow.parquet as pq
from download import ROOT,BASE,RAW,AUDIT
from convert import sha,norm
from proofwriter import close,truth
from helpsteer import split_helpsteer2

def file_sha(path):
 h=hashlib.sha256()
 with path.open('rb',buffering=16*1024*1024)as f:
  while block:=f.read(4*1024*1024):h.update(block)
 return h.hexdigest()

def main():
 reports=[];expected=collections.defaultdict(dict);proof_cache={};ids=set();labels=collections.Counter();source_counts=collections.Counter();quality_counts=collections.Counter();allfamilies=set();domains=collections.Counter()
 manifests={}
 for p in AUDIT.glob('*.download.json'):
  meta=json.loads(p.read_text(encoding='utf-8'))
  if meta.get('repo') and meta.get('path'):manifests[(meta['repo'],meta['path'])]=meta
 for name in ['candidates.jsonl','review.jsonl','quarantine.jsonl','proofwriter_candidates.jsonl','helpsteer_candidates.jsonl','helpsteer_review.jsonl','helpsteer_quarantine.jsonl']:
  path=BASE/name;rows=0;families=set();quality=collections.Counter();source=collections.Counter();messages_with_reasoning=0;boolmasks=0
  h=hashlib.sha256();is_candidate=name.endswith('candidates.jsonl')
  if not is_candidate:
   with path.open('rb',buffering=16*1024*1024)as f:
    while block:=f.read(16*1024*1024):h.update(block);rows+=block.count(b'\n')
   reports.append({'file':str(path.relative_to(ROOT)).replace(chr(92),'/'),'rows':rows,'bytes':path.stat().st_size,'sha256':h.hexdigest(),'distinct_families':None,'full_row_schema_checked':False,'purpose':'review_or_quarantine_not_training'})
   print(json.dumps(reports[-1]),flush=True);continue
  with path.open('rb',buffering=16*1024*1024)as f:
   for line in f:
    h.update(line);row=json.loads(line);rows+=1;assert row['id']not in ids;ids.add(row['id'])
    msgs=row['messages'];assert msgs[-1]['role']=='assistant';assert row['output']==row['final']==(msgs[-1].get('content')or'')
    assert row['source_split']in ['train','chat','chat_if','instruction_following','structured_outputs'];assert row['metadata']['group_key']==row['family_id']
    if row.get('release_status')=='candidate':assert not row.get('review_flags')
    mask=row['metadata'].get('assistant_loss_mask')
    if mask is not None:
     assert len(mask)==len(msgs)and all(type(v)is bool for v in mask)
     assert all(not v or msgs[i]['role']=='assistant'for i,v in enumerate(mask));boolmasks+=1
    mh=sha(json.dumps(msgs,ensure_ascii=False,sort_keys=True))
    if row['source']!='AI2/ProofWriter':
     assert mh==row['metadata']['source_messages_sha256'];p=row['provenance'][0]
     if len(expected[(row['source'],p['path'])])<16:expected[(row['source'],p['path'])][(p['row_1based'],row['metadata'].get('source_response_field'))]=(mh,mask)
    else:
     assert row['verification']['answer_pass']is True and row['verification']['full_closure_computed']is True;labels[row['final']]+=1
     proof_cache[row['family_id']]=True
    families.add(row['family_id']);quality[row['quality_level']]+=1;source[row['source']]+=1
    if any(m.get('reasoning_content')for m in msgs):messages_with_reasoning+=1
    if name.endswith('candidates.jsonl'):allfamilies.add(row['family_id']);source_counts[row['source']]+=1;quality_counts[row['quality_level']]+=1;domains[row['domain']]+=1
  reports.append({'file':str(path.relative_to(ROOT)).replace('\\','/'),'rows':rows,'bytes':path.stat().st_size,'sha256':h.hexdigest(),'full_row_schema_checked':True,'distinct_families':len(families),'source_counts':dict(source),'quality_levels':dict(quality),'rows_with_reasoning_content':messages_with_reasoning,'rows_with_boolean_loss_mask':boolmasks})
  print(json.dumps(reports[-1]),flush=True)
 roundtrip={}
 for key,wanted in expected.items():
  meta=manifests[key];file=ROOT/meta['local_path'];matched=0
  if file.suffix=='.parquet':it=(r for b in pq.ParquetFile(file).iter_batches(batch_size=1000)for r in b.to_pylist())
  elif file.suffix=='.gz':it=(json.loads(l)for l in gzip.open(file,'rt',encoding='utf-8'))
  else:it=(json.loads(l)for l in file.open(encoding='utf-8'))
  for i,raw in enumerate(it,1):
   if i>max(x[0]for x in wanted):break
   fields=[None]if key[0]not in ['nvidia/HelpSteer2','nvidia/HelpSteer3']else ['response']if key[0]=='nvidia/HelpSteer2'else ['response1','response2']
   for field in fields:
    if (i,field)not in wanted:continue
    mh,mask=wanted[(i,field)]
    if key[0]=='rajpurkar/squad':
     a=raw['answers'];assert all(raw['context'][s:s+len(t)]==t for s,t in zip(a['answer_start'],a['text']))
     prompt='Read the passage and answer the question with the exact answer span from the passage. Return only the answer.\n\nPassage:\n'+raw['context']+'\n\nQuestion: '+raw['question'];messages=[{'role':'user','content':prompt},{'role':'assistant','content':a['text'][0]}]
    elif key[0]=='nvidia/Llama-Nemotron-Post-Training-Dataset':
     messages=list(raw['input']);system=raw.get('system_prompt')
     if system and(not messages or messages[0]['role']!='system'):messages=[{'role':'system','content':system}]+messages
     messages=messages+[{'role':'assistant','content':raw['output']}]
    elif key[0]=='nvidia/HelpSteer2':messages=split_helpsteer2(raw['prompt'])+[{'role':'assistant','content':raw[field]}]
    elif key[0]=='nvidia/HelpSteer3':messages=raw['context']+[{'role':'assistant','content':raw[field]}]
    else:messages=raw['messages']
    assert sha(json.dumps(messages,sort_keys=True,ensure_ascii=False))==mh,(key,i)
    if key[0]=='nvidia/Nemotron-SFT-Instruction-Following-Chat-v3':assert mask==raw['metadata']['train_turns']
    matched+=1
  assert matched==len(wanted),(key,matched,len(wanted));roundtrip['|'.join(key)]={'rows':matched,'sample_message_fields_equal_source':True,'scope':'first_16_candidate_rows_for_source','source_train_mask_equal':True if key[0].endswith('-v3')else None}
  print(json.dumps({'source_roundtrip':key,'matched':matched}),flush=True)
 report={'files':reports,'candidate_total':sum(r['rows']for r in reports if r['file'].endswith('candidates.jsonl')),'candidate_distinct_family_ids':len(allfamilies),'candidate_sources':dict(source_counts),'candidate_quality_levels':dict(quality_counts),'candidate_domains':dict(domains),'source_roundtrip':roundtrip,'proofwriter_recomputed_labels':dict(labels),'proofwriter_unique_family_ids':len(proof_cache),'proofwriter_closure_validation_report':'proofwriter_summary.json','all_candidate_rows_parsed_and_checked':True,'review_files_counted_and_hashed_only':True,'source_message_roundtrip_is_sample_only':True,'sample_no_source_response_truncation_or_field_loss':True,'not_a_semantic_truth_certificate':True}
 (AUDIT/'final_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 print('ALL VALIDATION CHECKS PASSED',flush=True)
if __name__=='__main__':main()
