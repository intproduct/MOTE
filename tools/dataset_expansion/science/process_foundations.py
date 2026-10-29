import json,re,collections,random
from process import ROOT,Writer,base_record,classify,read_meta,norm,sha,flags_for
import pyarrow.parquet as pq

def sciq():
 meta=read_meta('sciq_train.parquet');w=Writer('sciq_train')
 for i,row in enumerate(pq.read_table(ROOT/'raw/sciq_train.parquet').to_pylist()):
  question=row['question'];support=row['support'];answer=row['correct_answer'];q='Answer the science question using the passage. Give only the answer.\n\nPassage:\n'+support+'\n\nQuestion: '+question
  r=base_record(meta,i,q,answer,answer,'science_grounded',answer)
  r['problem_id']=sha(norm(question));r['family_id']=r['problem_id'];r['response_mode']='direct';r['quality_level']='source_verified'
  r['review_flags']=[f for f in r['review_flags'] if f!='very_short_solution_needs_review'];r['metadata']={'original_question':question,'support':support,'distractors':[row['distractor1'],row['distractor2'],row['distractor3']],'subject':'mixed_science_not_separately_labeled','transformation':'Support supplied as input context; output is exact crowdsourced correct_answer, not generated reasoning.'}
  exists=bool(support.strip()) and bool(re.search(r'(?<!\w)'+re.escape(norm(answer))+r'(?!\w)',norm(support)))
  r['verification'].update(method='crowdsourced_gold_with_literal_support_check',answer_pass=True if exists else None,reference_origin='crowdsourced science QA correct_answer',independent_reference=True,support_contains_answer=exists,process_verified=False,publisher_evidence='https://arxiv.org/abs/1707.06209',local_semantic_entailment_verified=False)
  if not support.strip():r['review_flags'].append('missing_support')
  elif not exists:r['review_flags'].append('answer_not_literal_in_support')
  if re.search(r'include all but|all of the following|which of the following|\b(?:figure|diagram|table) (?:below|above)|serves no purpose in humans|has no function',support,re.I):r['review_flags'].append('support_incomplete_external_or_known_oversimplification')
  if support and (support[0].islower() or len(support.split())<15):r['review_flags'].append('support_fragment_needs_semantic_review')
  w.emit(r,classify(r,exists))
 w.close(['Official train only.','Strict subset requires the human correct_answer to occur literally in the provided support, plus structural screening. Literal occurrence is evidence, not a full semantic correctness proof.','No artificial CoT: short answers are intentional and preserve the source answer exactly.','NC license retained.'])

def scienceqa():
 meta=read_meta('scienceqa_problems.json');w=Writer('scienceqa_text_train');d=json.loads((ROOT/'raw/scienceqa_problems.json').read_text(encoding='utf-8'));splits=json.loads((ROOT/'raw/scienceqa_pid_splits.json').read_text(encoding='utf-8'));train=splits['train'];assert not set(train)&set(splits['val']);assert not set(train)&set(splits['test']);excluded=[]
 for i,pid in enumerate(train):
  row=d[pid];assert row['split']=='train'
  if row.get('image'):
   excluded.append({'problem_id':pid,'reason':'source_image_present','source_split':'train'});continue
  options=row['choices'];idx=row['answer'];assert isinstance(idx,int) and 0<=idx<len(options)
  question=row['question'];hint=row.get('hint') or '';lecture=row.get('lecture') or '';solution=row.get('solution') or '';q='Explain the answer, then end with Final answer: (letter) answer text.\n\n'+('Context:\n'+hint+'\n\n' if hint else '')+'Question: '+question+'\n\n'+'\n'.join(f'({chr(65+j)}) {x}' for j,x in enumerate(options));final=f'Final answer: ({chr(65+idx)}) {options[idx]}'
  out='\n\n'.join(x for x in [lecture,solution,final] if x);topic=row.get('topic') or 'science_mixed';domain={'physics':'physics','chemistry':'chemistry','biology':'biology','earth-science':'earth_science','engineering':'engineering'}.get(topic, 'language' if row['subject']=='language science' else 'social_science' if row['subject']=='social science' else 'science_foundation')
  r=base_record(meta,i,q,out,final,domain,options[idx]);r['quality_level']='source_verified';r['problem_id']=sha(norm(question+'\n'+hint+'\n'+json.dumps(options)));r['family_id']=r['problem_id'];r['metadata']={k:v for k,v in row.items() if k not in ['question','lecture','solution']};r['metadata'].update(original_question=question,source_problem_id=pid,source_lecture=lecture,source_solution=solution,transformation='Original lecture and solution concatenated intact, then deterministic labeled source-gold final answer appended.',license_note='Official LICENSE-DATA is CC-BY-NC-SA-4.0; third-party HF card omitted NC.')
  r['verification'].update(method='publisher_explanation_and_gold_index_export',answer_pass=True,independent_reference=True,reference_origin='official publisher answer index and educational explanation',local_semantic_entailment_verified=False,publisher_evidence='https://arxiv.org/abs/2209.09513',source_train_membership_verified=True,image_free_verified=True)
  r['provenance'][0]['normalized_output_sha256']=r['provenance'][0]['raw_output_sha256']
  r['provenance'][0]['raw_output_hash_scope']='legacy name: constructed export output; see source_lecture_sha256 and source_solution_sha256 for original fields'
  r['provenance'][0].update(source_lecture_sha256=sha(lecture),source_solution_sha256=sha(solution))
  if not solution.strip():r['review_flags'].append('missing_source_solution')
  if re.search(r'\b(?:first|second|third|fourth|fifth|sixth) (?:answer )?(?:choice|option)',solution,re.I):r['review_flags'].append('source_explanation_ordinal_may_disagree_with_shuffled_options')
  # Image-null alone does not make a question self-contained.
  if re.search(r'look at (?:the )?(?:picture|diagram|map|graph)|(?:shown|highlighted|pictured)|this (?:diagram|picture|map)',question+' '+hint+' '+lecture+' '+solution,re.I):r['review_flags'].append('possible_unprovided_visual_dependency')
  w.emit(r,classify(r,True))
 (w.base/'excluded_image_records.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in excluded),encoding='utf-8')
 w.close([f'Official train has {len(train)} IDs; {len(excluded)} image-bearing examples excluded before conversion.','No validation/test example is exported. Source raw problems JSON contains original all-split data, but only verified train IDs are processed.','Output retains complete provided lecture and solution; final answer is a transparent deterministic rendering of the original answer index.','Official data license CC-BY-NC-SA-4.0 overrides third-party HF card label.'])
if __name__=='__main__':sciq();scienceqa()
