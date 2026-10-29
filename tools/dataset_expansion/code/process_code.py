"""Model-neutral code SFT curation. Syntax parsing only; never execute dataset code."""
import ast,collections,hashlib,json,re,unicodedata
from pathlib import Path
import pyarrow.parquet as pq
from fetch_sources import ROOT,DEST,RAW,REV,REPO
from fetch_questions import SOURCES

KOD_REPO='KodCode/KodCode-V1-SFT-R1'
KOD_REV='26c8a11c800d71b6c4bafe12a92c9090ef0b6214'

def sha(s):return hashlib.sha256(s.encode('utf-8')).hexdigest()
def norm(s):return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',s)).strip().casefold()
def jwrite(f,r):f.write(json.dumps(r,ensure_ascii=False)+'\n')
def syntax(code):
 try:ast.parse(code);return True
 except (SyntaxError,ValueError,RecursionError):return False

def inspect_response(text,solution):
 flags=[];final='';reasoning='';sections={}
 if text.count('<think>')!=1 or text.count('</think>')!=1 or not text.lstrip().startswith('<think>'):
  flags.append('invalid_reasoning_boundary')
 else:
  a=text.index('<think>')+7;b=text.index('</think>')
  if b<a:flags.append('invalid_reasoning_boundary')
  else:
   reasoning=text[a:b];final=text[b+8:];sections={'reasoning':[a,b],'final':[b+8,len(text)]}
   if not reasoning.strip():flags.append('empty_reasoning')
   if not final.strip():flags.append('missing_final')
 if not solution.strip():flags.append('empty_solution')
 elif not syntax(solution):flags.append('solution_python_syntax_error')
 blocks=re.findall(r'^\s*```(?:python|py)?[ \t]*\n(.*?)^\s*```[ \t]*$',final,re.M|re.S)
 matching=[b for b in blocks if b.strip()==solution.strip()]
 if len(matching)!=1:flags.append('final_code_not_exactly_bound_to_verified_solution')
 if len(blocks)>1:flags.append('multiple_final_code_blocks')
 if sum(1 for line in final.splitlines() if re.match(r'^\s*```',line))%2:flags.append('unclosed_final_fence')
 if re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f\ufffd]',text):flags.append('unexpected_control_or_replacement_character')
 paragraphs=[p.strip() for p in re.split(r'\n\s*\n',reasoning) if len(p.strip())>=160]
 if any(n>=3 for n in collections.Counter(paragraphs).values()):flags.append('repeated_long_reasoning_paragraph')
 return flags,final,sections

def load_questions():
 maps={}
 for key in SOURCES:
  rows=[]
  for p in sorted((RAW/'questions').glob(key.replace('/','_')+'_train-*.jsonl')):
   for line in p.open(encoding='utf-8'):
    r=json.loads(line)
    if key=='code_contests':q=r['description']
    elif key=='taco':q=r['question']
    else:
     q=r['description'] or ''
     if r.get('input_format'):q+='\n\nInput\n\n'+r['input_format']
     if r.get('output_format'):q+='\n\nOutput\n\n'+r['output_format']
     if r.get('examples'):
      q+='\n\nExamples'
      for x in r['examples']:
       if 'input' in x:q+='\n\nInput\n\n'+x['input']
       if 'output' in x:q+='\n\nOutput\n\n'+x['output']
     if r.get('note'):q+='\n\nNote\n\n'+r['note']
    rows.append(q)
  maps[(key,'train')]=rows
 apps=RAW/'questions/apps_train-00000-of-00001.jsonl'
 if apps.exists():maps[('apps','train')]=[json.loads(line)['question'] for line in apps.open(encoding='utf-8')]
 return maps

def base_record(row,repo,rev,p,row_index,q,out,solution):
 flags,final,sections=inspect_response(out,solution)
 if not q or q=='-':flags.append('question_not_restored')
 if re.search(r'!\[[^\]]*\]\(|\b\w+\.(?:png|jpg|jpeg|svg)\b',q,re.I):flags.append('question_has_external_image_reference')
 pid=sha(norm(q))
 source_family=re.sub(r'_(?:I|C|OJ)$','',str(row['question_id'])) if repo==KOD_REPO else str(row['question_id'])
 return {'id':('ocr2-' if repo==REPO else 'kodcode-r1-')+str(row.get('id') or row['question_id']),
  'problem_id':pid,'family_id':sha(repo+'::'+source_family),'input':q,'output':out,'final':final,'solution':solution,'auxiliary_reasoning':None,
  'output_sections':sections,'response_mode':'reasoning','domain':'code','source':repo,'source_revision':rev,'source_split':'python' if repo==REPO else 'train',
  'license':['cc-by-4.0',row['license']] if repo==REPO else 'cc-by-nc-4.0','split':'train','quality_level':'source_verified','release_status':'candidate',
  'verification':{'answer_pass':None,'process_verified':False,'local_execution_performed':False,'local_python_ast_parse':syntax(solution) if solution else False,'solution_sha256':sha(solution),'code_matches_final':not any('code' in f or 'fence' in f for f in flags),'source_execution_pass':None},
  'review_flags':flags,'provenance':[{'repository':repo,'revision':rev,'raw_file':str(p.relative_to(ROOT)).replace('\\','/'),'row_index':row_index,'upstream_id':row.get('id'),'upstream_question_id':row['question_id'],'upstream_split':row.get('split','train')}],
  'metadata':{'teacher':'DeepSeek-R1','source_question_family_id':repo+'::'+source_family,'characters':{'input':len(q),'output':len(out),'final':len(final)},'truncated_by_local_pipeline':False,'tokenizer_processing_performed':False,'finish_reason_available':False,'truncation_check':'closed single think section, nonempty final, closed code fence, exact final/code binding; upstream finish_reason not supplied','original_question_id':row['question_id']}}

def main():
 questions=load_questions();counts=collections.Counter();reasons=collections.Counter();platforms=collections.Counter();sources=collections.Counter();licenses=collections.Counter();seen=set();lens=[];samples=[]
 streams={k:(DEST/(k+'.jsonl')).open('w',encoding='utf-8',newline='\n') for k in ['candidates','review','quarantine','duplicate_alternatives']}
 files=[(p,REPO,REV) for p in sorted(RAW.glob('*.parquet'))]+[(p,KOD_REPO,KOD_REV) for p in sorted((RAW/'kodcode').glob('*.parquet'))]
 for p,repo,rev in files:
  offset=0
  for batch in pq.ParquetFile(p).iter_batches(batch_size=128):
   for row in batch.to_pylist():
    counts['raw_rows']+=1;idx=offset;offset+=1
    if repo==REPO:
     ds=row['dataset'];sp=row['split'];srcq=questions.get((ds,sp),[]);i=int(row['index']);q=srcq[i] if 0<=i<len(srcq) else ''
     r=base_record(row,repo,rev,p,idx,q,row['r1_generation'],row['solution'])
     pass_rate=float(row['pass_rate']);v=r['verification'];v.update({'method':'upstream_execution_pass_rate_and_local_structure','source_execution_pass':pass_rate==1.0,'source_pass_rate':pass_rate,'source_judgement':row['judgement'],'unit_tests_present_locally':False})
     r['metadata'].update({'question_source':ds,'question_source_split':sp,'question_source_index':i,'question_source_revision':'21e74ddf8de1a21436da12e3e653065c5213e9d1' if ds=='apps' else SOURCES.get(ds,[None,None])[1],'platform':row['source'],'difficulty':row['difficulty'],'question_restoration':'official dataset/split/index mapping at pinned revision; no model-generated question reconstruction'})
     r['original_split']=sp
     if pass_rate!=1:r['review_flags'].append('upstream_execution_not_full_pass' if pass_rate>=0 else 'upstream_execution_unavailable')
     if row['judgement']!='right':r['review_flags'].append('upstream_critic_disagrees')
     if sp!='train':r['review_flags'].append('original_nontraining_split_excluded');r['split']='benchmark_excluded'
     # Chronological snapshot mapping cannot be proven for post-release Codeforces revisions.
     if ds=='open-r1/codeforces':r['review_flags'].append('question_source_revision_alignment_requires_review')
    else:
     conv=row['conversations'];humans=[x['value'] for x in conv if x['from']=='human'];assistants=[x['value'] for x in conv if x['from']=='gpt'];q=humans[0] if len(humans)==1 else row['question'];out=assistants[0] if len(assistants)==1 else ''
     r=base_record(row,repo,rev,p,idx,q,out,row['r1_solution'])
     seq=row.get('r1_pass_sequence') or [];sourcepass=str(row.get('r1_correctness')).lower()=='true' and any(x==1 for x in seq)
     r['verification'].update({'method':'upstream_paired_unit_test_rejection_sampling_and_local_structure','source_execution_pass':sourcepass,'source_r1_correctness':row.get('r1_correctness'),'source_pass_sequence':seq,'source_pass_trial_num':row.get('r1_pass_trial_num'),'unit_tests_present_locally':bool(row.get('test')),'unit_tests_sha256':sha(row.get('test') or '')})
     r['metadata'].update({'subset':row['subset'],'style':row['style'],'difficulty':row['gpt_difficulty'],'test_info':row.get('test_info'),'original_metadata':row.get('metadata'),'usage_group':'noncommercial_research'})
     r['unit_tests']=row.get('test');r['original_split']='train'
     if not sourcepass:r['review_flags'].append('upstream_r1_not_passed')
     if len(humans)!=1 or len(assistants)!=1:r['review_flags'].append('unsupported_conversation_structure')
     if not r['unit_tests']:r['review_flags'].append('missing_unit_tests')
     elif row['style']!='online_judge' and not syntax(r['unit_tests']):r['review_flags'].append('unit_tests_python_syntax_error')
    flags=r['review_flags'];reasons.update(flags)
    hard={'upstream_execution_not_full_pass','upstream_r1_not_passed','original_nontraining_split_excluded','solution_python_syntax_error','empty_solution','invalid_reasoning_boundary','missing_final','unclosed_final_fence','unexpected_control_or_replacement_character','repeated_long_reasoning_paragraph'}
    target='quarantine' if hard.intersection(flags) else 'review' if flags else 'candidates'
    if target=='candidates':
     if r['problem_id'] in seen:target='duplicate_alternatives';r['metadata']['dedup_reason']='one complete verified response per normalized question'
     else:
      seen.add(r['problem_id']);platforms[r['metadata'].get('platform',r['metadata'].get('subset','unknown'))]+=1;sources[repo]+=1;licenses[str(r['license'])]+=1;lens.append(len(r['output']))
      if len(samples)<20:samples.append({'id':r['id'],'input_start':r['input'][:500],'final_end':r['final'][-600:]})
    r['quality_level']='source_verified' if r['verification']['source_execution_pass'] is True else 'unverified'
    r['release_status']=target;counts[target]+=1;jwrite(streams[target],r)
  print(json.dumps({'processed':p.name,'rows':offset,'candidates_so_far':counts['candidates']}),flush=True)
 for f in streams.values():f.close()
 lens.sort();summary={'counts':dict(counts),'candidate_unique_questions':len(seen),'candidate_sources':dict(sources),'candidate_platforms_or_subsets':dict(platforms),'candidate_license_groups':dict(licenses),'filter_reasons':dict(reasons),'output_character_percentiles':{str(p):lens[min(len(lens)-1,int((len(lens)-1)*p))] if lens else None for p in [.1,.5,.9,.99]},'question_inventory':{str(k):len(v) for k,v in questions.items()},'local_execution_performed':False,'all_candidate_outputs_structurally_complete_and_untruncated_locally':True,'all_candidates_source_execution_pass':True,'quality_caveat':'Upstream tests can be incomplete and are not proofs of question correctness or full reasoning correctness. No student tokenizer processing. Benchmark decontamination handled by final integration.'}
 (DEST/'audit/processing_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
 (DEST/'audit/spotcheck_samples.json').write_text(json.dumps(samples,indent=2,ensure_ascii=False),encoding='utf-8')
 print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
