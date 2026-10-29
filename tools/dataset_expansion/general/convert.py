"""Conservative source-aware curation. No tokenization, truncation, or trace rewriting."""
import sys,json,re,hashlib,collections,unicodedata,random
from pathlib import Path
from download import ROOT,BASE,RAW,AUDIT
sys.path.insert(0,str(BASE/'vendor'))
import pyarrow.parquet as pq
import yaml
import jsonschema
from lxml import etree

def sha(x):return hashlib.sha256(x.encode('utf-8')).hexdigest()
def norm(x):return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',x)).strip().casefold()
def dump(path,x):path.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def strict_json(s):
 def pairs(p):
  d={}
  for k,v in p:
   if k in d:raise ValueError('duplicate_json_key')
   d[k]=v
  return d
 return json.loads(s,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite_json')))
def simple_constraints(labels,answer,prompt=''):
 """Only unambiguous constraints with no natural-language parameter guessing."""
 results={};p=prompt.casefold()
 # Publisher taxonomy labels sometimes name constraints missing from the actual prompt.
 # Never enforce those labels without an explicit corresponding user request.
 if 'case:in english and lowercase' in labels and re.search(r'(?:entire|whole|all|only|in english and).{0,60}(?:lowercase|lower case)',p):results['lowercase']=answer==answer.lower()
 global_upper=re.search(r'(?:entire|whole) (?:response|output|text|answer).{0,60}(?:uppercase\b|capital letters\b)|(?:response|output|text|answer).{0,40}(?:entirely|all).{0,25}(?:uppercase\b|capital letters\b)|in english and (?:in )?(?:all )?(?:capital letters\b|uppercase\b)',p)
 if 'in english and capital' in labels and global_upper:results['uppercase']=answer==answer.upper()
 if 'punctuation:use no comma' in labels and re.search(r'\bno commas?\b|without (?:using )?(?:any )?commas?\b|do not use (?:any )?commas?\b|avoid (?:using )?commas?\b',p):results['no_comma']=',' not in answer
 json_whole=re.search(r'(?:entire|whole|only|exclusively).{0,40}\bjson\b|(?:response|answer|reply).{0,35}\b(?:in|as|using) (?:a )?json\b|ensure (?:the )?(?:format|response) is (?:in )?json\b',p)
 if 'format:use json format' in labels and json_whole:
  s=answer.strip();m=re.fullmatch(r'```(?:json)?\s*\n([\s\S]*?)\n```',s,re.I)
  if m:s=m.group(1)
  try:strict_json(s);results['json_parse']=True
  except Exception:results['json_parse']=False
 return results
def semantic_review_flags(messages,domain,kind):
 if domain not in ['instruction','general_chat','writing']:return []
 user='\n'.join(m.get('content') or '' for m in messages if m.get('role')=='user').casefold()
 flags=[]
 if re.search(r'\b(?:attached|uploaded) (?:image|file|report|document)|\b(?:this|the|above) (?:image|screenshot|diagram)\b',user):flags.append('external_visual_or_attachment_context_unverified')
 if re.search(r'\b(?:latest|recent|ongoing|current|today|yesterday)\b.{0,65}\b(?:report|album|news|conflict|games|prices|statistics|polls|events)\b',user):flags.append('time_sensitive_external_context_unverified')
 if re.search(r'\b(?:synthe(?:sis|tic routes?)|reagents?|retrosynth\w*|drug dos(?:e|age)|clinical treatment|anorectal|surgical procedures?)\b',user):flags.append('expert_factual_content_not_independently_verified')
 if kind=='instruction' and re.search(r'\b(?:biograph\w*|early life|career beginnings)\b',user):flags.append('named_entity_biography_not_independently_verified')
 return flags
def structural(messages):
 flags=[]
 if not messages or messages[-1].get('role')!='assistant':flags.append('missing_terminal_assistant');return flags
 if not any(m.get('role')=='user' for m in messages):flags.append('missing_user')
 for m in messages:
  role=m.get('role');text=m.get('content')
  if role not in ['system','user','assistant']:flags.append('tool_or_unsupported_role')
  if not isinstance(text,str):flags.append('missing_or_nontext_content');continue
  if role!='system' and not text.strip():flags.append('empty_message_content')
  for field in ['content','reasoning_content']:
   s=m.get(field)
   if s is None:continue
   if not isinstance(s,str):flags.append('nontext_reasoning');continue
   if '\ufffd' in s:flags.append('replacement_character')
   if re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f]',s):flags.append('control_character')
   if role=='assistant':
    paras=[norm(p) for p in re.split(r'\n\s*\n',s) if len(p)>80]
    if any(n>=3 for n in collections.Counter(paras).values()):flags.append('repeated_long_paragraph')
    if s.count('<think>')!=s.count('</think>'):flags.append('unbalanced_think_tags')
  if m.get('tool_calls'):flags.append('tool_environment_required')
 return sorted(set(flags))

def format_check(prompt,answer):
 """Validate only syntax/schema, never claim factual or semantic correctness."""
 result={'method':'structured_output_parser','answer_pass':None,'format_pass':None,'semantic_content_verified':False,'reasoning_verified':False}
 a=answer.strip()
 try:
  if a.startswith('<'):
   if '<!DOCTYPE' in a or '<!ENTITY' in a:raise ValueError('external_entities_not_allowed')
   parser=etree.XMLParser(resolve_entities=False,no_network=True)
   node=etree.fromstring(a.encode(),parser)
   schemas=re.findall(r'<(?:xs|xsd):schema\b[\s\S]*?</(?:xs|xsd):schema>',prompt)
   result['format']='xml';result['parse_pass']=True
   if len(schemas)==1:
    if re.search(r'<(?:xs|xsd):(?:import|include|redefine)\b',schemas[0]):result['schema_status']='external_schema_disabled'
    else:
     try:
      schema=etree.XMLSchema(etree.fromstring(schemas[0].encode(),parser));schema.assertValid(node);result['format_pass']=True;result['method']='local_xml_xsd_validation'
     except etree.XMLSchemaParseError as e:result['schema_status']='source_schema_invalid';result['error']=str(e)[:400]
  else:
   if a.startswith(('{','[')):
    value=strict_json(a);result['format']='json';result['parse_pass']=True
   elif re.search(r'\byaml\b',prompt,re.I):
    value=yaml.safe_load(a);result['format']='yaml';result['parse_pass']=True
   else:result['format']='unrecognized';return result
   decoder=json.JSONDecoder();schemas=[]
   for match in re.finditer(r'\{',prompt):
    try:o,end=decoder.raw_decode(prompt[match.start():])
    except ValueError:continue
    if isinstance(o,dict):
     if isinstance(o.get('schema'),dict) and ('properties' in o['schema'] or '$schema' in o['schema']):schemas.append(o['schema'])
     elif ('properties' in o and 'type' in o) or '$schema' in o:schemas.append(o)
   # Nested object schemas are not separate root candidates; prefer containing root schema.
   schemas=list({json.dumps(s,sort_keys=True):s for s in schemas}.values())
   if schemas:
    schema=max(schemas,key=lambda s:len(json.dumps(s)))
    serialized=json.dumps(schema)
    if re.search(r'"\$ref"\s*:\s*"(?!#)',serialized):result['schema_status']='external_schema_disabled'
    else:
     cls=jsonschema.validators.validator_for(schema);cls.check_schema(schema);cls(schema,format_checker=jsonschema.FormatChecker()).validate(value)
     result['format_pass']=True;result['method']='local_json_schema_validation'
   else:result['schema_status']='schema_not_found'
 except (jsonschema.ValidationError,etree.DocumentInvalid) as e:result['format_pass']=False;result['error']=str(e)[:700]
 except Exception as e:result['parse_pass']=False;result['error']=str(e)[:700]
 return result

def base_record(raw,meta,index,messages,domain,quality,scope):
 final=messages[-1].get('content') or ''
 users=[m.get('content','') for m in messages[:-1] if m.get('role')=='user']
 inp=users[-1] if users else ''
 context=messages[:-1]
 # Include all roles/content/reasoning in context identity; retain raw messages verbatim.
 ph=sha(json.dumps(context,ensure_ascii=False,sort_keys=True))
 sid=str(raw.get('uuid') or raw.get('id') or raw.get('prompt_id') or index)
 constraints=raw.get('constraints')
 rr={'id':'general-'+sha(meta['repo']+'|'+meta['path']+'|'+sid)[:24],
  'problem_id':ph,'family_id':'general-context-'+ph,'input':inp,'output':final,'final':final,'solution':None,
  'messages':messages,'auxiliary_reasoning':messages[-1].get('reasoning_content'),
  'domain':domain,'source':meta['repo'],'source_revision':meta['revision'],'source_split':meta['source_split'],
  'license':raw.get('license') or meta.get('license'),'split':'train','quality_level':quality,'release_status':'candidate',
  'verification':{'method':scope,'answer_pass':None,'reasoning_verified':False,'source_quality_claim':scope,'local_semantic_correctness_verified':False},
  'review_flags':[],
  'provenance':[{'repo':meta['repo'],'revision':meta['revision'],'path':meta['path'],'source_record_id':sid,'row_1based':index,'source_split':meta['source_split'],'download_sha256':meta['sha256'],'selection':meta['selection']}],
  'metadata':{'requires_messages':len(messages)!=2 or any(m.get('role')=='system' for m in messages),
   'source_metadata':{k:v for k,v in raw.items() if k not in ['messages','input','output','context','answers']},
   'source_messages_sha256':sha(json.dumps(messages,ensure_ascii=False,sort_keys=True)),
   'training_target':'last_assistant_message','full_messages_preserved':True,'truncated':False,
   'reasoning_available':any(bool(m.get('reasoning_content')) for m in messages),'output_contains_reasoning':False,
   'all_source_constraints_locally_verified':False,'constraints':constraints,'local_evidence_scope':scope}}
 mask=raw.get('metadata',{}).get('train_turns')
 if mask is not None:rr['metadata']['source_train_turns']=mask
 return rr

def main():
 streams={x:(BASE/(x+'.jsonl')).open('w',encoding='utf-8',newline='\n') for x in ['candidates','review','quarantine']}
 counters=collections.defaultdict(collections.Counter);seen=set();duplicates=[];examples=collections.defaultdict(list);audit_reservoir={};rng=random.Random(20261016)
 tulu_labels={norm(r['prompt']):r['constraints'] for r in pq.read_table(RAW/'tulu_if_train.parquet').select(['prompt','constraints']).to_pylist()}
 reviews={r['id']:r for r in json.loads((AUDIT/'manual_content_reviews.json').read_text(encoding='utf-8'))} if (AUDIT/'manual_content_reviews.json').exists() else {}
 jobs=[('llama_nemotron_chat.jsonl','chat'),('nemotron_chat_if.prefix50000.jsonl','chat_if'),('nemotron_if_v3.prefix30000.jsonl','instruction'),('nemotron_structured.jsonl','structured'),('tulu_if_train.parquet','tulu'),('squad_train.parquet','reading'),('no_robots_train.parquet','human')]
 for filename,kind in jobs:
  file=RAW/filename;mf=AUDIT/(filename+'.download.json')
  if not(file.exists() and mf.exists()):raise FileNotFoundError(file)
  meta=json.loads(mf.read_text());card=json.loads((AUDIT/(meta['repo'].split('/')[-1]+'.metadata.json')).read_text(encoding='utf-8'))['cardData'];meta['license']=card.get('license')
  if kind=='chat_if':meta['license']='odc-by-1.0'
  it=(r for batch in pq.ParquetFile(file).iter_batches(batch_size=1000) for r in batch.to_pylist()) if file.suffix=='.parquet' else (json.loads(l) for l in file.open(encoding='utf-8'))
  for index,raw in enumerate(it,1):
   if kind=='reading':
    context=raw['context'];q=raw['question'];answers=raw['answers']['text'];starts=raw['answers']['answer_start'];final=answers[0] if answers else ''
    valid=bool(answers) and len(answers)==len(starts) and all(context[s:s+len(a)]==a for a,s in zip(answers,starts))
    prompt='Read the passage and answer the question with the exact answer span from the passage. Return only the answer.\n\nPassage:\n'+context+'\n\nQuestion: '+q
    messages=[{'role':'user','content':prompt},{'role':'assistant','content':final}]
    row=base_record(raw,meta,index,messages,'reading','reference_matched','human_reference_answer_span_checked')
    row['verification'].update(answer_pass=valid,reference_span_pass=valid,semantic_content_verified=False)
    row['expected_answer']=answers;row['metadata'].update(article_title=raw['title'],context_sha256=sha(context),answer_starts=starts,original_question=q)
    row['family_id']='squad-article-'+sha(norm(raw['title']))
    target='candidates' if valid else 'quarantine'
    if not valid:row['review_flags'].append('reference_span_mismatch')
   else:
    if kind=='chat':
     messages=list(raw.get('input') or []);system=raw.get('system_prompt')
     if system and (not messages or messages[0].get('role')!='system'):messages=[{'role':'system','content':system}]+messages
     messages=messages+[{'role':'assistant','content':raw.get('output')}]
    else:messages=raw.get('messages') or []
    if not messages:messages=[{'role':'user','content':''},{'role':'assistant','content':''}]
    domain='instruction' if kind in ['instruction','tulu','structured'] or raw.get('capability_target')=='instruction_following' else 'general_chat'
    if kind=='human':domain={'Generation':'writing','Brainstorm':'writing','Rewrite':'writing','Summarize':'reading','Closed QA':'reading','Extract':'reading','Classify':'instruction','Coding':'code'}.get(raw.get('category'),'general_chat')
    source_verified=kind=='chat_if' and raw.get('capability_target')=='instruction_following'
    quality='source_verified' if source_verified else 'source_curated'
    scope='publisher_instruction_verifiers_and_model_judge' if source_verified else 'publisher_human_demonstrations' if kind=='human' else 'publisher_sft_curation_not_locally_reverified'
    row=base_record(raw,meta,index,messages,domain,quality,scope)
    target='candidates'
    labels=raw.get('constraints')
    if labels is None:
     first_user=next((m.get('content') for m in messages if m.get('role')=='user'),None)
     if isinstance(first_user,str):labels=tulu_labels.get(norm(first_user))
    if labels:
     checks=simple_constraints(labels,row['final'],row['input']);row['verification']['local_constraint_checks']=checks
     row['verification']['all_instruction_constraints_verified']=False
     if any(v is False for v in checks.values()):target='quarantine';row['review_flags'].append('local_instruction_constraint_failed')
    if kind=='chat' or (kind=='chat_if' and not source_verified):
     if target!='quarantine':target='review'
     row['review_flags'].append('general_chat_requires_semantic_review')
    if kind=='tulu':
     if target!='quarantine':target='review'
     row['review_flags'].append('incomplete_instruction_constraint_verification_and_audit_failures')
    if kind=='human' and raw.get('category')=='Chat':
     if target!='quarantine':target='review'
     row['review_flags'].append('persona_roleplay_outside_current_training_mix')
    if kind=='human' and domain=='code':target='review';row['review_flags'].append('code_execution_not_verified')
    if kind=='structured':
     row['verification']=format_check('\n'.join(m.get('content') or '' for m in messages[:-1]),row['final'])
     row['metadata']['task_subtype']='structured_output'
     # Schema compliance does not validate invented document facts; keep all such rows outside trusted mixture.
     target='review';row['review_flags'].append('schema_format_only_content_grounding_unverified')
     if row['verification'].get('format_pass') is False or row['verification'].get('parse_pass') is False:target='quarantine';row['review_flags'].append('structured_format_validation_failed')
    if raw.get('tools'):target='quarantine';row['review_flags'].append('tool_environment_required')
    extra=semantic_review_flags(messages,domain,kind)
    if extra:
     row['review_flags'].extend(extra)
     if target!='quarantine':target='review'
   flags=structural(row['messages']);row['review_flags']=sorted(set(row['review_flags']+flags))
   if flags:target='quarantine'
   if kind=='human':row['metadata']['noncommercial_license']=True
   if row['source']=='nvidia/Nemotron-SFT-Instruction-Following-Chat-v3':
    mask=row['metadata']['source_train_turns']
    assert len(mask)==len(row['messages']) and all(type(v)==bool for v in mask)
    assert all(not v or row['messages'][i]['role']=='assistant' for i,v in enumerate(mask))
    row['metadata']['assistant_loss_mask']=mask;row['metadata']['training_target']='source_boolean_message_loss_mask'
   row['metadata']['group_key']=row['family_id']
   if row['id'] in reviews:
    review=reviews[row['id']];assert review['messages_sha256']==row['metadata']['source_messages_sha256']
    row['metadata']['manual_content_review']=review
    if review['decision']=='review':
     row['review_flags'].append('manual_content_review_requires_followup')
     if target!='quarantine':target='review'
   row['release_status']={'candidates':'candidate','review':'review_required','quarantine':'quarantine'}[target]
   exact=sha(json.dumps(row['messages'],sort_keys=True,ensure_ascii=False))
   if exact in seen:duplicates.append({'id':row['id'],'source':row['source'],'row':index,'reason':'exact_messages_duplicate'});continue
   seen.add(exact)
   streams[target].write(json.dumps(row,ensure_ascii=False)+'\n');counters[filename][target]+=1;counters[filename]['domain:'+row['domain']]+=1
   for flag in row['review_flags']:counters[filename]['flag:'+flag]+=1
   if row['verification'].get('format_pass'):counters[filename]['local_schema_pass']+=1
   group=filename+'|'+domain if kind!='reading' else filename+'|reading'
   entry={'id':row['id'],'source':row['source'],'source_file':filename,'source_row':index,'domain':row['domain'],'release_status':row['release_status'],'messages_sha256':exact,'messages':row['messages'],'verification':row['verification'],'review_flags':row['review_flags']}
   bucket=audit_reservoir.setdefault(group,{'seen':0,'items':[]});bucket['seen']+=1
   if len(bucket['items'])<12:bucket['items'].append(entry)
   else:
    k=rng.randrange(bucket['seen'])
    if k<12:bucket['items'][k]=entry
  print(json.dumps({'source':filename,'counts':counters[filename]}),flush=True)
 for f in streams.values():f.close()
 with (AUDIT/'exact_duplicates.jsonl').open('w',encoding='utf-8') as f:
  for r in duplicates:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 with (AUDIT/'fixed_seed_content_sample.jsonl').open('w',encoding='utf-8') as f:
  for k,b in sorted(audit_reservoir.items()):
   for r in b['items']:r['sample_group']=k;f.write(json.dumps(r,ensure_ascii=False)+'\n')
 report={'sources':dict(counters),'exact_duplicates_removed':len(duplicates),'rows':len(seen),'tokenization_performed':False,'response_truncation_performed':False,'dataset_test_splits_downloaded':False,'content_sample_seed':20261016,'semantic_correctness_of_synthetic_reasoning_certified':False}
 dump(AUDIT/'conversion_summary.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
