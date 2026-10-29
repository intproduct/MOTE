"""Export model-independent input/output JSONL, with full traces and audit tiers."""
import collections,json,hashlib
from pathlib import Path
from common import *

def convert(r):
 reasoning=r['reasoning'];final=r['final'];direct=r['domain']=='instruction'
 if reasoning and not direct:
  start=len('<think>\n');body='<think>\n'+reasoning+'\n</think>\n\n';end=len(body)
  output=body+final;sections={'reasoning':[start,start+len(reasoning)],'final':[end,len(output)]}
 else:output=final;sections={'reasoning':None,'final':[0,len(final)]}
 solution=None
 if r['domain']=='code':
  selected=select_code(final)
  if selected['candidate_count']==1:solution=selected['selected']
  # Callable-specific extraction is stored below by the export main loop.
 result={'id':r['id'],'problem_id':r['question_hash'],'family_id':r['family_id'],'input':r['prompt'],'output':output,'final':final,
  'solution':solution,'auxiliary_reasoning':reasoning if direct and reasoning else None,'output_sections':sections,
  'response_mode':'direct' if direct else 'reasoning' if reasoning else 'solution',
  'domain':r['domain'],'source':r['source'],'dataset':'MoTE-local-curated-v0.1','difficulty':r['metadata'].get('difficulty'),'source_revision':r['metadata'].get('source_revision'),
  'source_split':r['metadata'].get('source_split'),'license':r['metadata'].get('license'),
  'split':r['split'],'original_split':r['original_split'],'expected_answer':r['reference_answer'],
  'quality_level':r['quality_level'],'release_status':r['release_status'],'verification':r['verification'],
  'review_flags':r['flags'],'provenance':r['provenance'],'metadata':r['metadata'],'schema_version':'mote-sft-jsonl-1.0'}
 if 'original_reference_answer' in r:result['original_expected_answer']=r['original_reference_answer']
 return result

def main():
 out=DEST/'jsonl';out.mkdir(exist_ok=True)
 names=['train','validation','test','smoke','review_required','quarantine']
 streams={name:(out/(name+'.jsonl')).open('w',encoding='utf-8',newline='\n') for name in names};counts={name:collections.Counter() for name in names};total=collections.Counter();examples={};pilot=[]
 pilot_ids={json.loads(line)['id'] for line in (DEST/'pilot/answer_checked_candidates.jsonl').open(encoding='utf-8')}
 jobs={j['id']:j for j in (json.loads(line) for line in (DEST/'code_review/jobs.jsonl').open(encoding='utf-8'))}
 id_seen=set();families={};violations=[];code_evidence=[]
 for line in (DEST/'canonical/records.jsonl').open(encoding='utf-8'):
  r=json.loads(line);s=convert(r)
  if s['id'] in jobs:s['solution']=jobs[s['id']]['code']
  assert s['id'] not in id_seen;id_seen.add(s['id'])
  if r['release_status']=='answer_checked':
   target='train' if r['split']=='train' else 'test' if r['split']=='test' else 'smoke' if r['split']=='smoke_only' else 'validation'
   assert r['split']!='benchmark_excluded' and not r['flags'] and r['verification']['answer_pass'] is True
   if s['family_id'] in families:assert families[s['family_id']]==target,('family_crosses_release_splits',s['id'])
   families[s['family_id']]=target
  else:target=r['release_status']
  sec=s['output_sections'];assert s['output'][slice(*sec['final'])]==r['final']
  if sec['reasoning']:assert s['output'][slice(*sec['reasoning'])]==r['reasoning']
  elif r['reasoning']:assert s['auxiliary_reasoning']==r['reasoning']
  if s['solution'] and s['verification']['method']=='hash_bound_prior_execution':assert sha(s['solution'])==s['verification']['selected_code_sha256']
  streams[target].write(json.dumps(s,ensure_ascii=False)+'\n');counts[target][s['domain']]+=1;total[target]+=1
  if s['id'] in pilot_ids:
   assert target=='train';pilot.append(s)
  if target=='train' and len(s['output'])<3500 and s['domain'] not in examples:examples[s['domain']]=s
  if s['domain']=='code':code_evidence.append({'id':s['id'],'status':s['verification'].get('execution_status'),'method':s['verification']['method'],'selected_code_sha256':s['verification'].get('selected_code_sha256')})
 for f in streams.values():f.close()
 write_jsonl(DEST/'pilot/train.jsonl',pilot)
 write_jsonl(DEST/'examples/sample_records.jsonl',examples.values())
 write_jsonl(DEST/'audit/code_evidence.jsonl',code_evidence)
 report={'schema_version':'mote-sft-jsonl-1.0','canonical_records':len(id_seen),'files':{k:{'rows':total[k],'domain_counts':dict(v),'bytes':(out/(k+'.jsonl')).stat().st_size} for k,v in counts.items()},'pilot_rows':len(pilot),'pilot_domains':dict(collections.Counter(r['domain'] for r in pilot)),
  'checks':{'unique_ids':True,'answer_checked_requires_pass':True,'no_flags_in_answer_checked':True,'release_family_disjoint':True,'reasoning_and_final_lossless':True,'code_hash_bound_when_prior_pass':True},
  'tokenizer_processing_performed':False,'responses_truncated':False,'code_execution_rerun_locally':False}
 write_json(DEST/'audit/export_validation.json',report)
 files=[]
 for folder in [DEST/'jsonl',DEST/'pilot',DEST/'examples']:
  for p in sorted(folder.glob('*.jsonl')):
   if p.name in ['answer_checked_candidates.jsonl','review_candidates.jsonl','delivery_messages.jsonl']:continue
   h=hashlib.sha256()
   with p.open('rb') as f:
    for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
   files.append({'path':str(p.relative_to(DEST)).replace('\\','/'),'bytes':p.stat().st_size,'sha256':h.hexdigest()})
 write_json(DEST/'MANIFEST.json',{'version':'v0.1','schema_version':'mote-sft-jsonl-1.0','date':'2026-10-16','files':files,'raw_sources_manifest':'audit/source_inventory.jsonl'})
 print(json.dumps(report,ensure_ascii=True))

if __name__=='__main__':main()
