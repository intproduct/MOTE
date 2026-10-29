"""Shared model-independent JSONL schema, preserving full conversation context."""
import copy,hashlib,json,re,unicodedata

VERSION='mote-sft-jsonl-2.0'
def sha(s):return hashlib.sha256(s.encode('utf-8')).hexdigest()
def compact(obj):return json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def norm(s):return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',s)).strip().casefold()
def supervision_signature(record):
 """Training identity; the existing content hash remains messages-only."""
 return sha(compact({'messages_sha256':record['record_content_sha256'],'assistant_loss_mask':record['assistant_loss_mask']}))

def normalize_record(r,origin_file,origin_line,origin_pool):
 r=copy.deepcopy(r);flags=list(r.get('review_flags') or r.get('flags') or []);metadata=r.get('metadata') or {}
 if not isinstance(metadata,dict):metadata={'source_metadata_original':metadata}
 source=r.get('source') or 'unknown';domain=r.get('domain') or 'unspecified';messages=r.get('messages');changes=[]
 if not messages:
  messages=[{'role':'user','content':r.get('input',r.get('prompt',''))},{'role':'assistant','content':r.get('output',r.get('final',''))}]
 original_message_hash=sha(compact(messages));original_contents=[]
 for i,m in enumerate(messages):
  if not isinstance(m,dict):raise ValueError('Message must be an object')
  original_contents.append({'index':i,'content_sha256':sha(compact(m.get('content'))),'reasoning_sha256':sha(compact(m.get('reasoning_content')))})
 auxiliary=r.get('auxiliary_reasoning')
 if auxiliary:
  if not isinstance(auxiliary,str):flags.append('non_text_auxiliary_reasoning')
  elif not messages or messages[-1].get('role')!='assistant':flags.append('unattached_auxiliary_reasoning')
  else:
   last_message=messages[-1];existing=last_message.get('reasoning_content')
   if existing not in (None,''):
    if existing!=auxiliary:flags.append('conflicting_auxiliary_reasoning')
   elif isinstance(last_message.get('content'),str) and ('<think>' in last_message['content'] or '</think>' in last_message['content']):
    flags.append('ambiguous_dual_reasoning_representation')
   else:
    changes.append({'message':len(messages)-1,'action':'attach_top_level_auxiliary_reasoning','previous_reasoning_present':'reasoning_content' in last_message,'previous_reasoning_content':existing})
    last_message['reasoning_content']=auxiliary
 for i,m in enumerate(messages):
  if not isinstance(m,dict):raise ValueError('Message must be an object')
  content=m.get('content');reasoning=m.get('reasoning_content')
  if m.get('role') not in ['system','developer','user','assistant','tool']:flags.append('unsupported_message_role')
  if not isinstance(content,str):flags.append('non_text_or_null_message_content')
  if m.get('role')=='tool' or m.get('tool_calls'):flags.append('tool_environment_required')
  if reasoning is not None and not isinstance(reasoning,str):flags.append('non_text_reasoning_content')
  if m.get('role')=='assistant' and isinstance(reasoning,str) and reasoning:
   if domain in ['instruction','instruction_following','structured_output','structured_outputs']:
    changes.append({'message':i,'action':'retain_reasoning_as_auxiliary_field_for_exact_instruction_output'})
   elif isinstance(content,str):
    if '<think>' in content or '</think>' in content:flags.append('ambiguous_dual_reasoning_representation')
    else:
     m['content']='<think>\n'+reasoning+'\n</think>\n\n'+content;m.pop('reasoning_content',None)
     changes.append({'message':i,'action':'fuse_reasoning_content_into_explicit_text_section','reasoning_characters':len(reasoning),'original_content_characters':len(content)})
 if not messages or messages[-1].get('role')!='assistant':flags.append('conversation_does_not_end_with_assistant')
 if not any(m.get('role')=='user' and isinstance(m.get('content'),str) and m['content'].strip() for m in messages):flags.append('missing_user_context')
 last=messages[-1].get('content') if messages else None
 if not isinstance(last,str) or not last.strip():flags.append('empty_assistant_response')
 users=[m['content'] for m in messages if m.get('role')=='user' and isinstance(m.get('content'),str)]
 single=len(messages)==2 and [m.get('role') for m in messages]==['user','assistant'] and all(isinstance(m.get('content'),str) for m in messages)
 context=compact(messages[:-1]);problem=metadata.get('original_question') or metadata.get('source_question_text') or r.get('input') or '\n\n'.join(users)
 if not isinstance(problem,str):problem=compact(problem)
 if domain in ['reading','instruction','instruction_following','science_grounded']:
  problem=compact([m for m in messages if m.get('role') in ['system','developer','user']])
 # For formal logic, the same query over a different theory is a different problem.
 if domain=='logic' and metadata.get('source_theory_text'):problem=metadata['source_theory_text']+'\n\n'+problem
 keys={'id','problem_id','family_id','input','prompt','output','final','solution','messages','auxiliary_reasoning','domain','source','source_revision','source_split','license','split','original_split','expected_answer','quality_level','release_status','verification','review_flags','flags','provenance','metadata','schema_version','output_sections','response_mode','dataset','difficulty','original_expected_answer'}
 extra={k:v for k,v in r.items() if k not in keys}
 if extra:metadata['source_record_extra_fields']=extra
 metadata['ingest_location']={'file':origin_file,'line_1based':origin_line,'pool':origin_pool};metadata['source_family_id']=r.get('family_id');metadata['source_problem_id_original']=r.get('problem_id');metadata['requires_messages']=not single;metadata['context_sha256']=sha(context)
 metadata['message_conversion']={'source_messages_sha256':original_message_hash,'source_fields':original_contents,'actions':changes,'truncated':False}
 if 'auxiliary_reasoning' in r:metadata['message_conversion']['source_auxiliary_reasoning_sha256']=sha(compact(auxiliary))
 metadata['tokenizer_processing_performed']=False
 verification=r.get('verification') or {}
 if verification.get('process_verified'):verification['source_process_verified_claim']=verification['process_verified']
 verification['process_verified']=False
 result={'schema_version':VERSION,'id':r.get('id') or 'row-'+sha(origin_file+':'+str(origin_line))[:24],'problem_id':sha(norm(problem)),'family_id':None,
  'input':messages[0]['content'] if single else None,'output':last if isinstance(last,str) else None,'messages':messages,'final':r.get('final') if isinstance(r.get('final'),str) else last if isinstance(last,str) else '',
  'solution':r.get('solution'),'domain':domain,'source':source,'source_revision':r.get('source_revision'),'source_split':r.get('source_split'),'license':r.get('license'),'split':None,'original_split':r.get('original_split') or r.get('split'),
  'expected_answer':r.get('expected_answer'),'quality_level':r.get('quality_level','unverified'),'quality_tier':None,'release_status':None,'verification':verification,'review_flags':sorted(set(flags)),'provenance':r.get('provenance') or [{'derived_file':origin_file,'line_1based':origin_line}],
  'metadata':metadata,'response_mode':'prompt_completion' if single else 'conversation','assistant_loss_mask':None}
 if 'original_expected_answer' in r:result['original_expected_answer']=r['original_expected_answer']
 if 'auxiliary_reasoning' in r:result['auxiliary_reasoning']=auxiliary
 # Default to the last assistant only; explicit source target masks must be supplied by their adapters.
 mask=metadata['assistant_loss_mask'] if 'assistant_loss_mask' in metadata else r.get('assistant_loss_mask')
 if mask is not None:
  if not isinstance(mask,list) or len(mask)!=len(messages) or any(type(v) is not bool for v in mask):flags.append('invalid_assistant_loss_mask')
  else:result['assistant_loss_mask']=mask
 if result['assistant_loss_mask'] is None:result['assistant_loss_mask']=[i==len(messages)-1 and m.get('role')=='assistant' for i,m in enumerate(messages)]
 if any(active and messages[i].get('role')!='assistant' for i,active in enumerate(result['assistant_loss_mask'])):flags.append('non_assistant_supervision')
 result['review_flags']=sorted(set(flags));result['record_content_sha256']=sha(compact(messages));result['supervision_sha256']=supervision_signature(result);return result

def primary_prompt(record):
 """Only user messages are checked for benchmark question overlap, not generated answers."""
 return '\n\n'.join(m['content'] for m in record['messages'] if m.get('role')=='user' and isinstance(m.get('content'),str))
