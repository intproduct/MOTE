"""Model-independent, lossless science ingestion. No model inference or downloaded code execution."""
import os,re,json,hashlib,unicodedata,collections,argparse
from decimal import Decimal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]/'Datasets/expansion_20261016/science'
for k,v in {'HF_HOME':'cache/hf','HF_DATASETS_CACHE':'cache/datasets','HUGGINGFACE_HUB_CACHE':'cache/hub','TEMP':'tmp','TMP':'tmp','PIP_CACHE_DIR':'cache/pip'}.items():
 p=ROOT/v;p.mkdir(parents=True,exist_ok=True);os.environ[k]=str(p)
import pyarrow.parquet as pq
def sha(s):return hashlib.sha256(s.encode('utf-8')).hexdigest()
def norm(s):return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',s)).strip().casefold()
def boxes(s):
 result=[]
 for m in re.finditer(r'\\boxed\s*\{',s):
  level=1;i=m.end()
  while i<len(s) and level:
   if s[i]=='{':level+=1
   elif s[i]=='}':level-=1
   i+=1
  if level==0:result.append(s[m.end():i-1])
 return result
def norm_answer(s):
 s=s.strip().replace('\\left','').replace('\\right','').replace('\\dfrac','\\frac').replace('\\tfrac','\\frac')
 s=re.sub(r'\\(?:text|mathrm|mathbf|operatorname)\{([^{}]*)\}',r'\1',s)
 s=s.replace('\\(','').replace('\\)','').replace('\\[','').replace('\\]','').replace('$','')
 return re.sub(r'\s+','',s).casefold().rstrip('.')
def answer_match(extracted,expected):
 if not extracted or not expected:return None
 a,b=norm_answer(extracted),norm_answer(expected)
 if a==b:return True
 if re.fullmatch(r'[a-j]',a) and re.fullmatch(r'[a-j]',b):return False
 if re.fullmatch(r'[-+]?\d+(?:\.\d+)?',a) and re.fullmatch(r'[-+]?\d+(?:\.\d+)?',b):return Decimal(a)==Decimal(b)
 return None
def flags_for(q,a):
 flags=[]
 if not q.strip():flags.append('empty_input')
 if not a.strip():flags.append('empty_output')
 if '\ufffd' in q+a:flags.append('unicode_replacement_character')
 if re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f]',q+a):flags.append('unexpected_control_character')
 if a.count('<think>')!=a.count('</think>'):flags.append('unclosed_reasoning')
 if re.search(r'(?:as shown|shown|illustrated|indicated)\s+(?:in\s+)?(?:the\s+)?(?:figure|diagram|picture)|(?:figure|fig\.)\s*\d|refer to (?:the )?(?:figure|diagram)|in the (?:figure|diagram) below',q,re.I):flags.append('possible_missing_visual_context')
 if re.search(r'(?:theorem|proposition|lemma|equation|exercise|problem)\s+\d+(?:\.\d+)+',q,re.I):flags.append('possible_external_numbered_context')
 if re.search(r'!\[[^\]]*\]\(|<img\b',q,re.I):flags.append('image_required')
 paragraphs=[norm(x) for x in re.split(r'\n\s*\n',a) if len(x)>100]
 if any(v>=3 for v in collections.Counter(paragraphs).values()):flags.append('repeated_paragraphs')
 sentences=collections.Counter(norm(x) for x in re.split(r'(?<=[.!?])\s+|\n\s*\n',a) if len(x)>40)
 repeated_chars=sum(len(s)*(n-1) for s,n in sentences.items() if n>=3)
 if any(n>=5 for n in sentences.values()) and repeated_chars>0.05*len(a):flags.append('reasoning_repetition_warning')
 if a.strip().endswith(('...', '…')):flags.append('possible_incomplete_ending')
 if len(a.strip())<15:flags.append('very_short_solution_needs_review')
 if re.search(r'(?:reference|original|expected|provided|given) (?:document|answer|solution)',a,re.I):flags.append('answer_mentions_hidden_reference_or_expected_solution')
 if re.search(r'(?:curve|graph|figure|table|data|information|diagram).{0,70}(?:not provided|not given|missing)|(?:not provided|not given|missing).{0,70}(?:curve|graph|figure|table|diagram)',a,re.I):flags.append('answer_admits_missing_context')
 if re.search(r'assumes? a typical|suppose the (?:curve|graph) indicates|for illustration',a,re.I):flags.append('possible_invented_context_or_value')
 markers=len(re.findall(r'\b(?:I (?:think|guess|believe)|perhaps|maybe|let me (?:think|see|check)|let\x27s (?:see|check|think)|but wait)\b',a,re.I))
 if markers>=20 and markers>len(a)/1000:flags.append('sustained_answer_deliberation_needs_review')
 return flags

class Writer:
 def __init__(self,name):
  self.name=name;self.base=ROOT/'normalized'/name;self.base.mkdir(parents=True,exist_ok=True)
  self.stream={k:(self.base/(k+'.jsonl')).open('w',encoding='utf-8',newline='\n') for k in ['candidates','review','quarantine','tool_trajectories_review']}
  self.counts=collections.Counter();self.domains=collections.defaultdict(collections.Counter);self.flags=collections.Counter();self.seen=set();self.examples={};self.total=0;self.skipped_math=0
 def emit(self,r,bucket):
  self.total+=1;key=(r['problem_id'],sha(r['output']))
  if key in self.seen:self.counts['exact_duplicates']+=1;return
  self.seen.add(key);self.stream[bucket].write(json.dumps(r,ensure_ascii=False)+'\n');self.counts[bucket]+=1;self.domains[bucket][r['domain']]+=1;self.flags.update(r['review_flags'])
  if (bucket,r['domain']) not in self.examples:self.examples[bucket,r['domain']]=r
 def close(self,notes):
  for f in self.stream.values():f.close()
  report={'source':self.name,'source_rows_processed':self.total,'math_rows_deferred_to_math_workstream':self.skipped_math,'counts':self.counts,'domain_counts':dict(self.domains),'review_flags':self.flags,'notes':notes,'tokenizer_processing':False,'truncated_records':0,'model_inference_performed':False,'downloaded_code_executed':False}
  (self.base/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
  with (self.base/'examples.jsonl').open('w',encoding='utf-8') as f:
   for r in self.examples.values():f.write(json.dumps(r,ensure_ascii=False)+'\n')
  print(json.dumps(report,ensure_ascii=True),flush=True)

def base_record(meta,index,q,out,final,domain,expected=None):
 h=sha(norm(q));rid=meta['repo'].replace('/','_')+'-'+sha(str(index)+q)[:20]
 final_start=out.rfind(final) if final else -1
 return {'id':rid,'problem_id':h,'family_id':h,'input':q,'output':out,'final':final,'solution':None,'domain':domain,'source':meta['repo'],'source_revision':meta['revision'],'source_split':'train','license':meta['license'],'split':'train','expected_answer':expected,'quality_level':'structural_only','release_status':'review_required','verification':{'answer_pass':None,'process_verified':False,'independent_reference':False,'method':'source_present_only'},'review_flags':flags_for(q,out),'output_sections':{'final':[final_start,final_start+len(final)] if final_start>=0 else None},'response_mode':'reasoning' if '<think>' in out else 'solution','provenance':[{'dataset':meta['repo'],'revision':meta['revision'],'source_file':meta['path'],'source_file_sha256':meta['sha256'],'source_split':'train','row_index':index,'raw_output_sha256':sha(out)}],'metadata':{},'schema_version':'mote-sft-jsonl-2.0'}
def classify(r,candidate=False):
 hard={'empty_input','empty_output','unicode_replacement_character','unexpected_control_character','unclosed_reasoning','repeated_paragraphs','answer_reference_mismatch','known_incorrect_solution'}
 if hard.intersection(r['review_flags']):r['release_status']='quarantine';return 'quarantine'
 if r['review_flags'] or not candidate:r['release_status']='review_required';return 'review'
 r['release_status']='candidate';return 'candidates'
def read_meta(name):return json.loads((ROOT/'audit'/(name+'.download.json')).read_text(encoding='utf-8'))
def textbook():
 name='textbook_reasoning.parquet';meta=read_meta(name);w=Writer('textbook_reasoning');ix=0
 for batch in pq.ParquetFile(ROOT/'raw'/name).iter_batches(batch_size=512):
  for row in batch.to_pylist():
   ix+=1
   if row['subject']=='math':w.skipped_math+=1;continue
   q,out,expected=row['question'],row['answer'],row['reference_answer'];domain={'cs':'computer_science'}.get(row['subject'],row['subject'])
   r=base_record(meta,ix-1,q,out,out,domain,expected);b=boxes(out);match=answer_match(b[-1] if b else None,expected)
   r['verification'].update(answer_pass=match,method='normalized_last_box_reference_match' if match is not None else 'publisher_llm_curation',extracted_answer=b[-1] if b else None,reference_origin='Llama3.3-70B-Instruct summary of curated answer; not an independent human gold',publisher_process=['textbook-grounded extraction','DeepSeek-V3 context refinement','Llama3.3 defective-pair filtering','embedding plus LLM decontamination'],publisher_evidence='https://arxiv.org/html/2507.16812v2',publisher_process_rerun_locally=False)
   r['quality_level']='reference_matched' if match is True else 'source_curated'
   r['metadata']={'subject':row['subject'],'teacher_model':'DeepSeek-V3','full_source_answer_preserved':True,'scope':'publisher-curated science SFT, locally structurally screened'}
   if not expected.strip():r['review_flags'].append('missing_reference_answer')
   if match is False:r['review_flags'].append('answer_reference_mismatch')
   r['metadata']['source_level_release_decision']='review_only_after_hash_stratified_audit_found_wrong_solutions_and_missing_context'
   r['review_flags'].append('source_level_science_review_required')
   w.emit(r,classify(r,False))
 w.close(['Full single source shard scanned; every non-math subject processed, not a head sample.','Entire source downgraded to research/review: stratified sample found definite physics/mathematics errors and missing context.','Publisher curation and answer/reference agreement do not verify the whole reasoning chain.','CC-BY-NC-SA-4.0 kept per row.'])
def physicseval():
 name='physicseval_train.json';meta=read_meta(name);w=Writer('physicseval');rows=json.loads((ROOT/'raw'/name).read_text(encoding='utf-8'))
 for i,row in enumerate(rows):
  q=row['problem'];out=row['elaborated_solution_steps'];final='\n'.join(row['final_answers_in_brief']);r=base_record(meta,i,q,out,final,'physics',None)
  r['metadata']={k:v for k,v in row.items() if k not in ['problem','elaborated_solution_steps','final_answers_in_brief']};r['metadata']['provided_brief_answers']=row['final_answers_in_brief'];r['metadata']['source_output_is_complete']=True
  r['verification'].update(method='publisher_elaboration_only',publisher_evidence='https://arxiv.org/html/2508.00079v1',reference_origin='same source solution summarized by publisher; no independent verifier evidence')
  r['review_flags'].append('independent_physics_review_required')
  # A concrete audited error: this reaction violates charge and baryon number.
  if row['Problem_ID']=='8d1fa3f5-e5ce-4e7f-ada5-7186955707d5':
   r['review_flags'].append('known_incorrect_solution');r['verification']['answer_pass']=False;r['verification']['manual_audit']='p+p has Q=+2 and B=2; n+anti-n+p+anti-p has Q=0 and B=0. Source says possible, which is false.'
  w.emit(r,classify(r,False))
 w.close(['Only official train downloaded. All entries require independent physics review; publisher elaboration is not correctness verification.','Original question and elaborated solution preserved. Brief final is a separate source field and may not be a substring of output.','Known charge/baryon conservation error explicitly quarantined.'])
def osr2():
 name='osr2_0000.parquet';meta=read_meta(name);w=Writer('open_science_reasoning_2');ix=0
 for batch in pq.ParquetFile(ROOT/'raw'/name).iter_batches(batch_size=256):
  for row in batch.to_pylist():
   q,out,expected=row['input'],row['output'],str(row['expected_answer']);final=out.split('</think>',1)[-1].strip() if '</think>' in out else out
   r=base_record(meta,ix,q,out,final,'science_mixed',expected);ix+=1;b=boxes(final);match=answer_match(b[-1] if b else None,expected)
   r['verification'].update(answer_pass=match,method='normalized_last_box_reference_match',extracted_answer=b[-1] if b else None,reference_origin='publisher synthetic expected_answer; no independent oracle',publisher_evidence='https://huggingface.co/datasets/nvidia/OpenScienceReasoning-2')
   r['quality_level']='reference_matched' if match is True else 'structural_only';r['metadata']={'teacher_model':'DeepSeek-R1-0528','subject':'not_provided','sampling':'complete HF converted partial-train shard 0000, not global random sample','source_main_revision':'174b02c9cdf231f220765b2a1d5ece4550921894','full_source_output_preserved':True}
   if match is False:r['review_flags'].append('answer_reference_mismatch')
   if match is None:r['review_flags'].append('answer_not_confidently_extractable_or_matchable')
   if not final:r['review_flags'].append('empty_final')
   r['review_flags'].append('source_level_semantic_review_required')
   r['metadata']['source_level_release_decision']='review_only_after_fixed_seed_semantic_audit_found_material_errors'
   w.emit(r,classify(r,False))
 w.close(['Pinned HF conversion revision contains only a partial train view; shard 0000 fully processed.','No original subject label; science_mixed can include non-STEM domains and is not counted as physics.','Exact last-box reference agreement checks final label only, not scientific validity or whole CoT.'])
def nemotron():
 name='nemotron_v2_syn_mcq.jsonl';meta=read_meta(name);w=Writer('nemotron_sft_science_v2')
 for i,line in enumerate((ROOT/'raw'/name).open(encoding='utf-8')):
  row=json.loads(line);messages=row['messages'];users=[m.get('content') or '' for m in messages if m['role']=='user'];assistants=[m for m in messages if m['role']=='assistant'];last=assistants[-1] if assistants else {};final=last.get('content') or '';reasoning=last.get('reasoning_content') or '';out=('<think>\n'+reasoning+'\n</think>\n\n' if reasoning else '')+final
  r=base_record(meta,i,'\n\n'.join(users),out,final,(row.get('metadata',{}).get('topic') or 'science_mixed').lower());r['metadata']=row.get('metadata',{});r['metadata']['source_uuid']=row['uuid'];r['metadata']['used_in']=row.get('used_in');r['messages']=messages;r['tools']=row.get('tools');r['verification'].update(method='publisher_sft_release_no_independent_gold',publisher_evidence='https://huggingface.co/datasets/nvidia/Nemotron-SFT-Science-v2');r['review_flags'].append('independent_science_answer_review_required')
  tool=any(m['role']=='tool' or m.get('tool_calls') for m in messages)
  if tool:r['review_flags'].append('tool_trajectory_requires_messages_aware_training');bucket='tool_trajectories_review';r['release_status']='review_required'
  else:bucket=classify(r,False)
  w.emit(r,bucket)
 w.close(['Entire syn_mcq subset scanned, with true publisher topic retained.','All original messages and tools are preserved; tool-assisted examples excluded from plain SFT candidates.','No expected_answer or per-record correctness result was published, so examples are review material despite official publisher.'])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('source',choices=['textbook','physicseval','osr2','nemotron']);args=p.parse_args();globals()[args.source]()
