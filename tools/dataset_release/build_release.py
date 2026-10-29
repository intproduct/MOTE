"""Versioned local data curation. Raw files remain read-only. No model calls."""
import argparse,collections,hashlib,html,json,re,sys,time
from pathlib import Path
import pyarrow.parquet as pq
from common import *
from verifiers import verify_distilled,structural_flags

def metadata(r):
 result={k:r.get(k) for k in ['parent_id','source_id','source_url','source_revision','source_split','license','teacher_revision','quantization','generation_config','template_family','structural_signature','dedup_cluster','difficulty','skills','finish_reason','generation_seed','completed_at']}
 result['teacher_model_reported']=(r.get('metrics_before') or {}).get('engine',{}).get('model')
 return result

def main():
 started=time.time();DEST.mkdir(parents=True,exist_ok=True);rows=[];sources=[];errors=[];jobs=[];repairs=[];duplicates=[]
 def record_source(p):
  b=p.read_bytes();h=hashlib.sha256(b).hexdigest();sources.append({'path':str(p.relative_to(ROOT)),'sha256':h,'bytes':len(b)});return b,h
 def emit(r):
  r['curation_version']=VERSION;r.setdefault('flags',[]);r.setdefault('metadata',{});r.setdefault('reasoning','');r.setdefault('original_split',None)
  r['prompt']=clean(r['prompt']);r['final']=clean(r['final']);r['reasoning']=clean(r['reasoning'])
  r['question_hash']=sha(question_key(r['prompt']));r['answer_sha256']=sha(r['final']);r['reasoning_sha256']=sha(r['reasoning'])
  r['character_lengths']={'prompt':len(r['prompt']),'reasoning':len(r['reasoning']),'final':len(r['final'])}
  r['format_stage']='model_independent_text';rows.append(r)
 for p in sorted((ROOT/'Datasets/mixture-fromqwen3p8/results').glob('*.json')):
  b,h=record_source(p)
  try:r=json.loads(b)
  except ValueError as e:errors.append({'path':str(p),'error':str(e)});continue
  v=verify_distilled(r);flags=structural_flags(r['prompt'],r.get('raw_reasoning',''),r.get('final',''),r.get('finish_reason'))
  if v['answer_pass'] is False:flags.append('reference_or_oracle_disagreement')
  if v['reference_validated'] is False and str(v['method']).startswith('recomputed'):flags.append('reference_does_not_match_recomputed_oracle')
  level='prior_execution_pass' if v['method']=='hash_bound_prior_execution' else 'independently_recomputed' if v['answer_pass'] is True and v['reference_validated'] else 'reference_matched' if v['answer_pass'] is True else 'unverified'
  if r['id']=='math-01a5efe142c7314e':
   flags.append('source_reference_inconsistent_with_problem');v['manual_note']='A year ending 00 is divisible by 100, so its remainder mod 1000 cannot be 125 or 875. Source gold 4 is inconsistent; preserve teacher and source for review, do not auto-correct the dataset.'
  if r['domain']=='code':
   selection=select_code(r.get('final',''),(r.get('hidden_tests') or {}).get('fn_name'))
   if selection['selected']:
    jobs.append({'id':r['id'],'code':selection['selected'],'code_sha256':sha(selection['selected']),'hidden_tests':r.get('hidden_tests'),'original_split':r.get('split'),'old_execution':r.get('verification',{}).get('execution'),
     'status':'prior_run_pass_recheck_optional' if v['method']=='hash_bound_prior_execution' else 'pending_isolated_execution'})
    if sha(selection['selected'])!=r.get('verification',{}).get('code_sha256'):repairs.append({'id':r['id'],'action':'selected_complete_python_by_fence_and_callable','old_sha256':r.get('verification',{}).get('code_sha256'),'new_sha256':sha(selection['selected']),'changes_answer_text':False,'execution_confirmed':False})
   else:flags.append('code_selection_ambiguous_or_missing')
  emit({'id':r['id'],'origin':'distillation','source':r['source'],'domain':r['domain'],'prompt':r['prompt'],'reasoning':r.get('raw_reasoning',''),'final':r.get('final',''),'reference_answer':r.get('gold'),
   'original_split':r.get('split'),'quality_level':level,'verification':v,'old_verification':{k:r.get('verification',{}).get(k) for k in ['answer_pass','format_pass','verifier_version','failure_reason']},
   'metadata':metadata(r),'provenance':[{'path':str(p.relative_to(ROOT)),'sha256':h}], 'flags':flags,'existing_qwen_lengths':r.get('lengths')})
 # Physics: parse raw independently, retaining all exact origins and alternatives.
 seenfiles={};phys_count=0
 for p in sorted((ROOT/'Datasets/phy-selfuse').iterdir()):
  if p.suffix not in ['.json','.jsonl','.csv','.md']:continue
  b,h=record_source(p)
  if h in seenfiles:duplicates.append({'type':'identical_file','path':str(p.relative_to(ROOT)),'same_as':seenfiles[h]});continue
  seenfiles[h]=str(p.relative_to(ROOT));text=b.decode('utf-8-sig');items=[]
  if p.suffix=='.json':items=list(enumerate(json.loads(text),1))
  elif p.suffix=='.csv':items=list(enumerate(csv.DictReader(io.StringIO(text)),1))
  elif p.suffix=='.jsonl':
   for i,line in enumerate(text.splitlines(),1):
    if not line.strip():continue
    try:
     obj,actions=repair_json_text(line);items.append((i,obj))
     if actions:repairs.append({'path':str(p.relative_to(ROOT)),'line_1based':i,'actions':actions,'action':'repair_invalid_JSON_escape','raw_source_preserved':True})
    except ValueError as e:errors.append({'path':str(p.relative_to(ROOT)),'line_1based':i,'raw':line,'error':str(e)})
  else:
   recovered,bad=markdown_json_objects(text);items=[(i,{k:clean_recovered_markdown_value(v) if isinstance(v,str) else v for k,v in obj.items()}) for i,(obj,_) in enumerate(recovered,1)]
   errors.extend({'path':str(p.relative_to(ROOT)),'stage':'markdown_json_recovery',**e} for e in bad)
   repairs.append({'path':str(p.relative_to(ROOT)),'action':'recover_JSON_objects_from_Markdown','records_recovered':len(items),'object_repairs':[meta for _,meta in recovered if meta['repairs']]})
  for i,r in items:
   q=r.get('question',r.get('Input',r.get('problem','')));a=r.get('answer',r.get('Output',''));solution=r.get('solution')
   if solution:a=clean(solution)+'\n\n最终答案：'+clean(a)
   if not isinstance(q,str) or not isinstance(a,str) or None in r:errors.append({'path':str(p.relative_to(ROOT)),'record':i,'error':'unexpected_schema','row':r});continue
   q,qfix=restore_damaged_latex(q);a,afix=restore_damaged_latex(a)
   if qfix or afix:repairs.append({'path':str(p.relative_to(ROOT)),'record_1based':i,'action':'restore_recognizable_LaTeX_control_escapes','prompt_repairs':qfix,'answer_repairs':afix})
   flags=structural_flags(q,'',a)
   if re.search('如图|图中|下图|图示|见图|如右图|如左图|上图|所示|本段|这段|本书|本页|上述|上文',q):flags.append('possible_missing_context')
   if re.search('作者|出版社|出版时间|ISBN|书名|主编|责任编辑',q):flags.append('possible_book_metadata')
   v={'method':'structure_only','answer_pass':None,'process_verified':False}
   # Internal agreement is useful but is not independent proof of physics correctness.
   key=re.findall(r'####\s*【答案】\s*([A-D]+)',a);conclusion=re.findall(r'故选[：:]\s*([A-D]+)',a)
   if key and conclusion:v.update({'method':'reference_internal_choice_consistency','answer_pass':key[-1]==conclusion[-1],'reference_answer':key[-1]})
   if v['answer_pass'] is False:flags.append('reference_internal_contradiction')
   emit({'id':'physics-'+sha(str(p.relative_to(ROOT))+':'+str(i))[:20],'origin':'physics_local','source':'local/phy-selfuse','domain':'physics','prompt':q,'final':a,'reference_answer':key[-1] if key else None,
    'quality_level':'structural_only','verification':v,'metadata':{'local_id':r.get('id'),'latex_cleanup':p.suffix=='.md'},'provenance':[{'path':str(p.relative_to(ROOT)),'sha256':h,'record_1based':i,'record_type':'physical_line' if p.suffix=='.jsonl' else 'parsed_record','page_reference':r.get('file_name')}],'flags':flags})
   phys_count+=1
 # Math: preserve every complete reasoning trace, not just a prefix or suffix.
 p=ROOT/'Datasets/high-equal-examples/cot-00011-of-00144.parquet';_,h=record_source(p);idx=0
 for batch in pq.ParquetFile(p).iter_batches(batch_size=512):
  for r in batch.to_pylist():
   s=clean(r.get('generated_solution'));reasoning='';final=s;flags=[]
   if s.startswith('"') and s.endswith('"'):
    try:
     t=json.loads(s)
     if isinstance(t,str):s=t;repairs.append({'id':f'omr-local-{idx:05d}','action':'decode_nested_JSON_string'})
    except ValueError:pass
   if '<think>' in s and '</think>' in s:
    prefix,rest=s.split('<think>',1);reasoning,final=rest.split('</think>',1)
    if prefix.strip():flags.append('nonempty_text_before_think')
   else:flags.append('reasoning_boundary_missing')
   flags+=structural_flags(r['problem'],reasoning,final)
   expected=clean(r.get('expected_answer'));extracted=boxed(final);passed=answer_equal(extracted,expected) if extracted is not None and expected else None
   if passed is False:flags.append('reference_answer_disagreement')
   if not expected:flags.append('missing_reference_answer')
   if r['problem_type']!='has_answer_extracted':flags.append('reference_is_consensus_or_transformed_proof')
   emit({'id':f'omr-local-{idx:05d}','origin':'math_parquet','source':'local/OpenMathReasoning_schema_shard','domain':'math','prompt':r['problem'],'reasoning':reasoning,'final':final,'reference_answer':expected,
    'quality_level':'reference_matched' if passed is True and r['problem_type']=='has_answer_extracted' else 'unverified',
    'verification':{'method':'balanced_box_exact_or_rational','answer_pass':passed,'extracted_answer':extracted,'process_verified':False},
    'metadata':{k:r.get(k) for k in ['problem_type','problem_source','generation_model','inference_mode','pass_rate_72b_tir']},
    'provenance':[{'path':str(p.relative_to(ROOT)),'sha256':h,'row_0based':idx}],'flags':flags});idx+=1
 print(json.dumps({'phase':'parsed','records':len(rows),'physics_records':phys_count}),flush=True)
 manual_path=Path(__file__).with_name('manual_reviews.json');manual={x['id']:x for x in json.loads(manual_path.read_text(encoding='utf-8'))} if manual_path.exists() else {}
 manual_applied=[]
 for r in rows:
  review=manual.get(r['id'])
  if not review:continue
  assert review['input_sha256']==sha(r['prompt']) and review['final_sha256']==sha(r['final']),('review_content_changed',r['id'])
  r['verification']['manual_review']=review;r['verification']['answer_pass']=review['verdict']!='fail';r['quality_level']='assistant_reviewed_final' if review['verdict']!='fail' else 'unverified'
  if review['verdict']=='fail':r['flags'].append('manual_content_issue')
  if review['verdict']=='correct_reference':
   r['original_reference_answer']=r['reference_answer'];r['reference_answer']=review['corrected_reference'];r['flags']=[x for x in r['flags'] if x not in ['reference_or_oracle_disagreement','source_reference_inconsistent_with_problem']]
   repairs.append({'id':r['id'],'action':'correct_reference_after_explicit_review','original':review['original_reference'],'corrected':review['corrected_reference'],'note':review['note']})
  manual_applied.append(review)
 # Deduplicate identical Q + final + reasoning. Retain alternate trajectories without multiplying questions.
 unique={};merged=[]
 for r in rows:
  key=(r['question_hash'],r['answer_sha256'],r['reasoning_sha256'])
  if key in unique:
   prior=unique[key];prior['provenance']+=r['provenance'];prior.setdefault('duplicate_ids',[]).append(r['id'])
   prior.setdefault('merged_original_splits',[]).append(r['original_split']);duplicates.append({'type':'identical_record','id':r['id'],'kept_id':prior['id']})
  else:unique[key]=r;merged.append(r)
 rows=merged
 # Connected components preserve exact questions, numeric-template variants and legacy article/problem IDs.
 parent=list(range(len(rows)))
 def find(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 def union(a,b):
  a,b=find(a),find(b)
  if a!=b:parent[max(a,b)]=min(a,b)
 keys={}
 for i,r in enumerate(rows):
  q=question_key(r['prompt']);skeleton=re.sub(r'[-+]?\d+(?:\.\d+)?','<N>',q)
  rkeys=['q:'+q]
  if len(q)>=40 and '<N>' in skeleton:rkeys.append('template:'+skeleton)
  legacy=r['metadata'].get('dedup_cluster') or r['metadata'].get('parent_id')
  if legacy:rkeys.append('legacy:'+r['source']+':'+str(legacy))
  for key in rkeys:
   if key in keys:union(i,keys[key])
   else:keys[key]=i
 groups=collections.defaultdict(list)
 for i in range(len(rows)):groups[find(i)].append(i)
 bench_path=DEST/'audit/benchmark_questions.jsonl';benchmarks=[json.loads(line) for line in bench_path.read_text(encoding='utf-8').splitlines()] if bench_path.exists() else []
 exact_bench={};template_bench={}
 for b in benchmarks:
  q=question_key(b['prompt']);exact_bench.setdefault(q,[]).append({k:b[k] for k in ['benchmark','source_file','row']})
  sk=re.sub(r'[-+]?\d+(?:\.\d+)?','<N>',q)
  if len(q)>=40 and '<N>' in sk:template_bench.setdefault(sk,[]).append({k:b[k] for k in ['benchmark','source_file','row']})
 collisions=[];split_moves=[]
 for indices in groups.values():
  fid='family-'+min(rows[i]['question_hash'] for i in indices)[:24]
  reserved={s for i in indices for s in [rows[i]['original_split']]+rows[i].get('merged_original_splits',[]) if s and s!='train'}
  has_train=any(rows[i]['original_split']=='train' or 'train' in rows[i].get('merged_original_splits',[]) for i in indices)
  group_matches=[]
  for i in indices:
   q=question_key(rows[i]['prompt']);sk=re.sub(r'[-+]?\d+(?:\.\d+)?','<N>',q);matches=exact_bench.get(q) or template_bench.get(sk)
   if matches:group_matches.append({'id':rows[i]['id'],'match_type':'exact' if q in exact_bench else 'numeric_template','benchmarks':matches})
  if group_matches:split='benchmark_excluded';collisions+=group_matches
  elif reserved:split='smoke_only' if reserved=={'smoke_only'} else 'diagnostic_dev'
  elif has_train:split='train'
  else:
   bucket=int(sha(fid)[:8],16)%100;split='test' if bucket<5 else 'dev' if bucket<10 else 'train'
  for i in indices:
   r=rows[i];r['family_id']=fid;r['split']=split
   if r['original_split'] and r['original_split']!=split:split_moves.append({'id':r['id'],'from':r['original_split'],'to':split,'reason':'family_holdout_or_benchmark_exclusion'})
   if group_matches:r['flags'].append('benchmark_family_overlap')
 # Multiple answers are reviewed, never resolved by majority vote here.
 qgroups=collections.defaultdict(list)
 for r in rows:qgroups[r['question_hash']].append(r)
 answer_variants=[]
 for qhash,rs in qgroups.items():
  if len({r['answer_sha256'] for r in rs})>1:
   answer_variants.append({'question_hash':qhash,'ids':[r['id'] for r in rs],'count':len(rs),'note':'different text is not proof of contradiction'})
   for r in rs:
    if r['origin']=='physics_local':r['flags'].append('multiple_reference_texts_require_review')
 for r in rows:
  r['flags']=sorted(set(r['flags']))
  r['release_status']='quarantine' if r['flags'] else 'answer_checked' if r['quality_level'] in ['independently_recomputed','reference_matched','prior_execution_pass','assistant_reviewed_final'] else 'review_required'
 # One canonical archive; small indices avoid duplicating the entire long-CoT corpus.
 write_jsonl(DEST/'canonical/records.jsonl',rows)
 indexkeys=['id','family_id','origin','source','domain','split','original_split','quality_level','release_status','flags','character_lengths','existing_qwen_lengths','format_stage']
 write_jsonl(DEST/'manifests/all_records.jsonl',({k:r.get(k) for k in indexkeys} for r in rows))
 write_jsonl(DEST/'manifests/quarantine.jsonl',({k:r.get(k) for k in indexkeys} for r in rows if r['release_status']=='quarantine'))
 write_jsonl(DEST/'code_review/jobs.jsonl',jobs)
 for name,value in [('source_inventory',sources),('ingest_errors',errors),('repairs',repairs),('duplicates',duplicates),('benchmark_collisions',collisions),('split_changes',split_moves),('answer_variants',answer_variants)]:write_jsonl(DEST/f'audit/{name}.jsonl',value)
 write_jsonl(DEST/'audit/manual_reviews_applied.jsonl',manual_applied)
 # A deterministic initial selection; all selected reasoning is retained in full.
 quotas={'math':500,'physics':400,'code':300,'logic':200,'instruction':300,'reading':200,'stem':200,'writing':100}
 selected=[];review_selected=[]
 for domain,quota in quotas.items():
  pool=[r for r in rows if r['domain']==domain and r['split']=='train' and r['release_status']=='answer_checked']
  # Complete shorter delivery answers first; not truncation of traces.
  pool.sort(key=lambda r:(r['quality_level']!='independently_recomputed',len(r['reasoning'])+len(r['final']),sha(r['id'])))
  used=set()
  for r in pool:
   if r['family_id'] in used:continue
   used.add(r['family_id']);selected.append(r)
   if len(used)>=quota:break
  review=[r for r in rows if r['domain']==domain and r['split']=='train' and r['release_status']=='review_required']
  review.sort(key=lambda r:(len(r['final']),sha(r['id'])))
  for r in review[:quota]:review_selected.append(r)
 write_jsonl(DEST/'pilot/answer_checked_candidates.jsonl',selected)
 write_jsonl(DEST/'pilot/review_candidates.jsonl',review_selected)
 write_jsonl(DEST/'pilot/delivery_messages.jsonl',({'id':r['id'],'family_id':r['family_id'],'domain':r['domain'],'quality_level':r['quality_level'],'process_verified':False,'view':'delivery_only','messages':[{'role':'user','content':r['prompt']},{'role':'assistant','content':r['final']}]} for r in selected))
 summary={'version':VERSION,'source_files':len(sources),'source_records_parsed_before_record_dedup':len(merged)+sum(x['type']=='identical_record' for x in duplicates),'canonical_records':len(rows),'unique_questions':len(qgroups),'families':len(groups),'counts':{},'pilot_answer_checked':len(selected),'pilot_domains':dict(collections.Counter(r['domain'] for r in selected)),
 'pilot_review_required':len(review_selected),'benchmarks_indexed':len(benchmarks),'direct_benchmark_matches':len(collisions),'split_changes':len(split_moves),'ingest_errors':len(errors),'repair_actions':len(repairs),'code_jobs_pending':len(jobs),'elapsed_seconds':round(time.time()-started,2),'semantic_process_verified_records':0,'target_student_tokenized_records':0,'limitations':['Reference matching is not proof of the entire reasoning trace.','Code execution is pending because the local Docker engine failed to start.','Benchmark exclusion covers GSM8K/MATH exact and numeric-template matches, not all semantic paraphrases or all benchmarks.','Raw traces retained; no text clipping or synthetic EOS insertion.']}
 summary['code_job_statuses']=dict(collections.Counter(j['status'] for j in jobs))
 summary['code_jobs_pending']=sum(j['status']=='pending_isolated_execution' for j in jobs)
 summary.pop('target_student_tokenized_records',None)
 summary['tokenizer_processing_performed']=False
 summary['limitations'][1]='Selected code with matching hashes and complete historical test logs retains prior-run evidence; other code is pending execution. No code was rerun locally because the Docker engine failed to start.'
 for key in ['origin','domain','split','quality_level','release_status']:summary['counts'][key]=dict(collections.Counter(r[key] for r in rows))
 summary['flags']=dict(collections.Counter(flag for r in rows for flag in r['flags']))
 write_json(DEST/'audit/summary.json',summary);print(json.dumps(summary,ensure_ascii=True),flush=True)

if __name__=='__main__':main()
