"""Curate pinned math shards; publisher validation and local checks stay distinct."""
import collections,hashlib,json,re,sys
from pathlib import Path
import pyarrow.parquet as pq
from math_sources import BASE,ROOT
sys.path.insert(0,str(ROOT/'tools/dataset_release'))
from common import sha,boxed,answer_equal
from verifiers import structural_flags

def split_trace(text):
 if text.startswith('<think>') and text.count('<think>')==1 and text.count('</think>')==1:
  trace,final=text[len('<think>'):].split('</think>',1)
  return trace,final,[] if final.strip() else ['empty_final']
 return '',text,['reasoning_boundary_missing']

def emit_row(repo,rev,path,row_index,row,download):
 flags=[];publisher=False;metadata={};expected=None;local=None
 if repo=='open-r1/OpenR1-Math-220k':
  choices=[];completeness=row.get('is_reasoning_complete') or [];verdicts=row.get('correctness_math_verify') or [];finishes=row.get('finish_reasons') or []
  for i,g in enumerate(row.get('generations') or []):
   if i<len(verdicts) and verdicts[i] is True and i<len(completeness) and completeness[i] is True and (i>=len(finishes) or finishes[i] in [None,'stop']):choices.append((sha(g),i,g))
  if not choices:return None,'no_complete_math_verify_pass'
  _,selected,output=min(choices);reasoning,final,extra=split_trace(output);flags+=extra;prompt=row['problem'];expected=row['answer'];publisher=True
  metadata={'source_problem_id':row.get('uuid'),'problem_type':row.get('problem_type'),'question_type':row.get('question_type'),'problem_source':row.get('source'),'generation_index':selected,'available_generations':len(row.get('generations',[])),'selection':'one complete Math-Verify-passed trace per source row, minimum SHA-256, not shortest','publisher_math_verify':True,'publisher_reasoning_complete':True,'finish_reason':finishes[selected] if selected<len(finishes) else None,'generation_model':'DeepSeek-R1'}
  license='apache-2.0';proof='per_record_correctness_math_verify_and_reasoning_complete'
 else:
  if row.get('subset','').lower()!='cot':return None,'tool_integrated_subset_excluded'
  messages=row.get('messages') or []
  if len(messages)!=2 or [m.get('role') for m in messages]!=['user','assistant'] or row.get('tools') or any(m.get('tool_calls') for m in messages):return None,'tool_or_multiturn_requires_separate_pipeline'
  if not isinstance(messages[0].get('content'),str) or not isinstance(messages[1].get('content'),str):return None,'non_text_or_missing_message'
  prompt=messages[0]['content'];final=messages[1]['content'];reasoning=messages[1].get('reasoning_content') or '';expected=row.get('expected_answer');publisher=True
  if not reasoning.strip():flags.append('empty_reasoning')
  output='<think>\n'+reasoning+'\n</think>\n\n'+final if reasoning else final
  metadata={k:row.get(k) for k in ['uuid','source','url','user_url','username','dataset','subset','used_in']};metadata['source_problem_id']=row.get('uuid');metadata['generation_model']='DeepSeek-V4-Pro';metadata['reasoning_conversion']='reasoning_content placed in explicit think section; source preserved in pinned parquet'
  license=row.get('license');proof='publisher_card_all_final_answers_match_reference'
 flags+=structural_flags(prompt,reasoning,final)
 if '<think>' in reasoning or '</think>' in reasoning:flags.append('nested_reasoning_marker')
 if re.search(r'!\[[^\]]*\]\(|<img\b|as shown in (?:the )?(?:figure|diagram)|(?:figure|diagram) (?:below|above)',prompt,re.I):flags.append('possible_missing_visual_context')
 if re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f]',reasoning):flags.append('reasoning_control_character')
 if re.search(r'<\|(?:im_start|im_end|eot_id|start_header_id)\|>',output):flags.append('model_control_token')
 extracted=boxed(final);local=answer_equal(extracted,str(expected)) if extracted is not None and expected is not None else None
 if local is False:flags.append('definite_numeric_reference_disagreement')
 if not expected:flags.append('missing_reference_answer')
 if len(final.strip())<3:flags.append('suspiciously_empty_delivery')
 rid='math-exp-'+sha(repo+':'+path.name+':'+str(row_index))[:24]
 return {'id':rid,'input':prompt,'output':output,'final':final,'solution':None,'messages':[{'role':'user','content':prompt},{'role':'assistant','content':output}],'domain':'math','source':repo,'source_revision':rev,'source_split':'train','license':license,'expected_answer':expected,
  'quality_level':'reference_matched' if local is True else 'source_verified','release_status':'quarantine' if flags else 'candidate',
  'verification':{'publisher_pass':publisher,'publisher_evidence':proof,'local_answer_pass':local,'local_method':'balanced_box_exact_or_rational','local_extracted_answer':extracted,'process_verified':False,'independent_reference_recomputed':False},
  'review_flags':sorted(set(flags)),'metadata':metadata,'provenance':[{'path':str(path.relative_to(ROOT)).replace('\\','/'),'sha256':download['sha256'],'row_0based':row_index,'source_url':download['download_url']}],
  'content_integrity':{'prompt_sha256':sha(prompt),'output_sha256':sha(output),'reasoning_sha256':sha(reasoning),'final_sha256':sha(final),'characters':{'prompt':len(prompt),'output':len(output),'reasoning':len(reasoning),'final':len(final)},'truncated':False}},None

def main():
 streams={k:(BASE/(k+'.jsonl')).open('w',encoding='utf-8') for k in ['candidates','quarantine']};counts=collections.Counter();reasons=collections.Counter();sources=collections.defaultdict(collections.Counter);seen=set();audit=[]
 for folder in sorted((BASE/'raw').iterdir()):
  for path in sorted(folder.glob('*.parquet')):
   dpath=path.with_suffix('.download.json')
   if not dpath.exists():continue
   download=json.loads(dpath.read_text(encoding='utf-8'));repo=download['repo'];rev=download['revision'];idx=0
   for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
    for row in batch.to_pylist():
     counts['read']+=1;sources[repo]['read']+=1;r,reason=emit_row(repo,rev,path,idx,row,download);idx+=1
     if r is None:reasons[reason]+=1;continue
     qkey=re.sub(r'\s+',' ',r['input']).strip().casefold();key=sha(qkey)
     if key in seen:reasons['duplicate_question_in_math_expansion']+=1;continue
     # A quarantined trajectory does not consume the accepted question slot.
     target='quarantine' if r['review_flags'] else 'candidates'
     if target=='candidates':seen.add(key)
     streams[target].write(json.dumps(r,ensure_ascii=False)+'\n');counts[target]+=1;sources[repo][target]+=1;reasons.update(r['review_flags']);counts['local_answer_matched']+=int(r['verification']['local_answer_pass'] is True and target=='candidates')
   audit.append({'path':str(path.relative_to(ROOT)),'rows':idx,'sha256':download['sha256']});print(json.dumps({'processed':path.name,'repo':repo,'rows':idx,'counts':dict(counts)}),flush=True)
 for f in streams.values():f.close()
 report={'counts':dict(counts),'by_source':{k:dict(v) for k,v in sources.items()},'exclusions_and_flags':dict(reasons),'input_files':audit,'selection_scope':'Fixed published shards; not a representative sample of the entire upstream corpus. One accepted trajectory per normalized question.','process_verified':False,'tokenizer_processing':False}
 (BASE/'audit/processing_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
