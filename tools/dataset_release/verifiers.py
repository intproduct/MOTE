import ast,json,re
from common import *

def verify_distilled(r):
 typ=r['verifier_type'];f=clean(r.get('final'));g=r.get('gold');result={'method':None,'answer_pass':None,'reference_validated':False,'process_verified':False}
 if typ in ['strict_json','exact_text']:
  expected=instruction_oracle(r)
  if expected is None:result['method']='unsupported_instruction_template';return result
  result['method']='recomputed_filter_sort_from_prompt';result['reference_validated']=json_equal(expected,g)
  try:value=json.loads(f) if isinstance(expected,dict) else f
  except ValueError:value=None
  result['answer_pass']=json_equal(value,expected)
 elif typ=='state_replay':
  expected=inventory_oracle(r)
  if expected is None:result['method']='unsupported_inventory_event';return result
  result['method']='recomputed_inventory_from_prompt';result['reference_validated']=json_equal(expected,g)
  candidates=[v for v in objects(f) if isinstance(v,dict) and set(v)=={'stock','total_units','inventory_value'}]
  result['answer_pass']=len(candidates)==1 and json_equal(candidates[0],expected)
 elif typ=='gym_oracle':
  expected=gym_oracle(r);a=tagged(f)
  if expected is None:result['method']='gym_reference_only';result['answer_pass']=literal_equal(a,g) if a is not None else False
  else:
   if isinstance(expected,list):
    # Pooling tasks explicitly require output rounded to two decimal places.
    result['method']='recomputed_pool_matrix_2dp'
    def matrix(x):return [[Decimal(v) for v in line.split()] for line in x.strip().splitlines()]
    expected=[[Decimal(f'{x:.2f}') for x in row] for row in expected]
    try:result['reference_validated']=matrix(g)==expected;result['answer_pass']=matrix(a)==expected if a is not None else False
    except (ValueError,InvalidOperation):result['answer_pass']=False
    return result
   result['method']='recomputed_gym_'+r.get('gym_name','unknown');result['reference_validated']=bool(literal_equal(expected,g))
   result['answer_pass']=literal_equal(a,expected) if a is not None else False
 elif typ in ['math_symbolic','math_numeric']:
  a=boxed(f);result['method']='balanced_box_exact_or_rational';result['extracted_answer']=a
  result['answer_pass']=answer_equal(a,str(g)) if a is not None and g is not None else False
 elif typ=='choice_evidence':
  a=tagged(f);result['method']='choice_reference_match';result['answer_pass']=a==g if a is not None else False
 elif typ=='span_evidence':
  a=tagged(f);result['method']='exact_span_reference_match'
  answers=g if isinstance(g,list) else [g]
  passage=re.search(r'<passage>(.*?)</passage>',r['prompt'],re.S)
  result['answer_pass']=bool(a is not None and any(norm(a)==norm(x) for x in answers))
  result['answer_in_passage']=bool(a is not None and passage and norm(a) in norm(passage[1]))
 elif typ=='sandbox_tests':
  selection=select_code(f,(r.get('hidden_tests') or {}).get('fn_name'));result['method']='code_static_selection_only'
  result.update({k:v for k,v in selection.items() if k!='selected'});result['selected_code_sha256']=sha(selection['selected']) if selection['selected'] else None
  result['execution_status']='pending_isolated_runner'
  old=r.get('verification',{});execution=old.get('execution') or {};tests=execution.get('tests') or [];expected=len((r.get('hidden_tests') or {}).get('inputs',[]))
  if selection['selected'] and result['selected_code_sha256']==old.get('code_sha256') and old.get('answer_pass') is True and execution.get('passed') is True and expected>0 and execution.get('tests_run')==execution.get('tests_expected')==len(tests)==expected and all(t.get('passed') is True and t.get('returncode')==0 for t in tests):
   result.update({'answer_pass':True,'method':'hash_bound_prior_execution','execution_status':'prior_run_pass_not_rerun_locally','tests_passed':expected,'controller_sha256':execution.get('controller_sha256')})
 else:result['method']='semantic_review_required'
 return result

def structural_flags(prompt,reasoning,final,finish=None):
 flags=[]
 if not clean(prompt):flags.append('empty_prompt')
 if not clean(final):flags.append('empty_final')
 if finish is not None and finish!='stop':flags.append('generation_not_completed')
 if '\ufffd' in prompt+final:flags.append('unicode_replacement_character')
 if re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f]',prompt+final):flags.append('unexpected_control_character')
 if code_blocks(final)[1]:flags.append('unclosed_code_fence')
 if re.search(r'<\|(?:im_start|im_end|eot_id|start_header_id)\|>',final):flags.append('teacher_control_token_in_final')
 # Consecutive exact paragraphs only, not normal recurring algebra or quoted text.
 paragraphs=[x.strip() for x in re.split(r'\n\s*\n',reasoning+'\n\n'+final) if len(x.strip())>=80]
 if any(paragraphs[i]==paragraphs[i-1]==paragraphs[i-2] for i in range(2,len(paragraphs))):flags.append('consecutive_repeated_paragraphs')
 return flags
