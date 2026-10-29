import ast,copy,json,re,tempfile,unittest
from pathlib import Path
from benchmark_matcher import BenchmarkMatcher
from normalize_records import normalize_record,sha,compact,supervision_signature
from evidence_checks import explicit_negative_evidence
from near_families import NearFamilies,strip_math_instruction
from validate_release import check_preservation
from math_content_gates import flags as math_flags,unbalanced_tex_braces

class ManualFindingsContractTests(unittest.TestCase):
 def audit_files(self):
  base=Path(__file__).resolve().parents[2]/'Datasets/expansion_20261016/math/audit'
  names=['independent_manual_findings.json','postgate_findings.json','openr1_postgate_findings.json','legacy_math_findings.json']
  paths=[base/name for name in names]
  if not all(path.exists() for path in paths):self.skipTest('Frozen manual audit fixtures are unavailable')
  return paths
 def test_all_manual_findings_match_merge_contract(self):
  seen=set()
  for path in self.audit_files():
   document=json.loads(path.read_text(encoding='utf-8'));self.assertIsInstance(document,dict);self.assertIsInstance(document['findings'],list)
   for row in document['findings']:
    with self.subTest(file=path.name,id=row.get('id')):
     self.assertIsInstance(row,dict)
     for key in ['id','input_sha256','output_sha256','action']:
      self.assertIsInstance(row[key],str);self.assertTrue(row[key])
     for key in ['input_sha256','output_sha256']:self.assertRegex(row[key],r'^[0-9a-f]{64}$')
     self.assertIn(row['action'],['quarantine','review','clear_symbolic_mismatch_only','no_change'])
     if 'classification' in row:self.assertIsInstance(row['classification'],str)
     self.assertNotIn(row['id'],seen);seen.add(row['id'])
 def test_actual_legacy_shape_runs_merge_manual_branch(self):
  # Execute the real branch, without starting main or touching release files.
  # Synthetic bound text isolates the field-shape contract from source scanning.
  source=Path(__file__).with_name('merge_release.py').read_text(encoding='utf-8')
  tree=ast.parse(source)
  branches=[node for node in ast.walk(tree) if isinstance(node,ast.If) and ast.unparse(node.test)=="r['id'] in manual"]
  self.assertEqual(len(branches),1)
  program=compile(ast.fix_missing_locations(ast.Module(body=[branches[0]],type_ignores=[])),'manual_review_branch','exec')
  legacy=json.loads(self.audit_files()[-1].read_text(encoding='utf-8'))['findings']
  covered=set()
  for original in legacy:
   with self.subTest(id=original['id'],action=original['action']):
    self.assertNotIn('classification',original)
    review=copy.deepcopy(original);raw={'input':'Bound fixture question','output':'Bound fixture complete response'}
    review['input_sha256']=sha(raw['input']);review['output_sha256']=sha(raw['output'])
    record={'id':review['id'],'verification':{},'review_flags':[],'quality_tier':'reference_checked'}
    state={'r':record,'raw':raw,'manual':{review['id']:review},'manual_applied':[],'sha':sha,'status':'eligible','pool':'candidates'}
    exec(program,state)
    expected={'quarantine':'quarantine','review':'review_required','no_change':'eligible'}[review['action']]
    self.assertEqual(state['status'],expected);self.assertEqual(state['manual_applied'],[review['id']])
    self.assertEqual(record['verification']['local_visible_content_review'],review)
    if review['action']!='no_change':self.assertTrue(record['review_flags'][0].endswith(':hash_bound_manual_content_review'))
    covered.add(review['action'])
  self.assertEqual(covered,{'quarantine','review','no_change'})

class MathContentTests(unittest.TestCase):
 def test_final_truncation_not_hidden_by_complete_reasoning(self):
  r={'source':'open-r1/OpenR1-Math-220k','final':'The area is \\frac{1','messages':[{'role':'user','content':'Find the area.'}]}
  self.assertIn('unbalanced_final_latex_braces',math_flags(r));self.assertIn('missing_explicit_math_final_delivery',math_flags(r))
 def test_explicit_gold_in_question_isolated(self):
  r={'source':'open-r1/OpenR1-Math-220k','final':'\\boxed{4}','messages':[{'role':'user','content':'Find the area.\nAnswer: $4$.'}]}
  self.assertIn('source_prompt_contains_answer_or_solution_section',math_flags(r))
  r['messages'][0]['content']='Find the area.\nAnswer: ______';self.assertEqual(math_flags(r),[])
 def test_escaped_braces_are_not_missing_tex_groups(self):
  self.assertFalse(unbalanced_tex_braces(r'\\boxed{\\{1,2\\}}'))

class FamilyTests(unittest.TestCase):
 def test_equivalent_wrappers_group_but_different_questions_do_not(self):
  n=NearFamilies();q='A farmer collects exactly forty five apples from the old tree and gives twelve apples to a neighbour. Explain how many apples remain in the basket and justify your answer.'
  self.assertEqual(n.add(0,q),[])
  self.assertEqual(n.add(1,q.upper()),[0])
  self.assertEqual(n.add(2,'Explain the difference between stable equilibria and unstable equilibria in a dynamical system with nonlinear feedback, including examples of each case and their physical meaning.'),[])
  self.assertEqual(strip_math_instruction(q+' Please reason step by step, and put your final answer within \\boxed{}.'),q)

class NormalizerTests(unittest.TestCase):
 def row(self,**kwargs):
  r={'id':'x','input':'Question','output':'Answer','final':'Answer','domain':'math','source':'test','metadata':{},'verification':{}}
  r.update(kwargs);return r
 def run_row(self,r):return normalize_record(r,'file',1,'candidate')
 def test_full_math_reasoning(self):
  r=self.run_row(self.row(messages=[{'role':'user','content':'Q'},{'role':'assistant','content':'Final','reasoning_content':'Full trace'}]))
  self.assertEqual(r['output'],'<think>\nFull trace\n</think>\n\nFinal');self.assertEqual(r['input'],'Q');self.assertNotIn('reasoning_content',r['messages'][-1])
  check_preservation(r)
  r['messages'][-1]['content']=r['messages'][-1]['content'][:-1]
  with self.assertRaises(AssertionError):check_preservation(r)
 def test_strict_json_reasoning_is_auxiliary(self):
  r=self.run_row(self.row(domain='instruction',messages=[{'role':'user','content':'Only JSON'},{'role':'assistant','content':'{"x":1}','reasoning_content':'Think'}]))
  self.assertEqual(r['output'],'{"x":1}');self.assertEqual(r['messages'][-1]['reasoning_content'],'Think')
  check_preservation(r)
 def test_multiturn_never_flattens_to_last_user(self):
  messages=[{'role':'user','content':'First'},{'role':'assistant','content':'Old'},{'role':'user','content':'Follow-up'},{'role':'assistant','content':'New'}]
  r=self.run_row(self.row(messages=messages));self.assertIsNone(r['input']);self.assertEqual(r['messages'],messages);self.assertEqual(r['assistant_loss_mask'],[False,False,False,True])
 def test_source_supervision_mask_preserved(self):
  r=self.run_row(self.row(metadata={'assistant_loss_mask':[False,True]},messages=[{'role':'user','content':'Q'},{'role':'assistant','content':'A'}]));self.assertEqual(r['assistant_loss_mask'],[False,True])
 def test_wrong_supervision_quarantinable(self):
  r=self.run_row(self.row(metadata={'assistant_loss_mask':[True,True]}));self.assertIn('non_assistant_supervision',r['review_flags'])
 def test_same_reading_question_different_context_not_same_problem(self):
  a=self.run_row(self.row(domain='reading',input='Context A. What year?'));b=self.run_row(self.row(domain='reading',input='Context B. What year?'));self.assertNotEqual(a['problem_id'],b['problem_id'])
 def test_no_source_process_claim_promoted(self):
  r=self.run_row(self.row(verification={'process_verified':True}));self.assertFalse(r['verification']['process_verified']);self.assertTrue(r['verification']['source_process_verified_claim'])
 def test_top_level_trace_with_existing_messages_roundtrips(self):
  original=[{'role':'user','content':'Q'},{'role':'assistant','content':'Final'}]
  r=self.run_row(self.row(messages=original,auxiliary_reasoning='Previously lost full trace'))
  self.assertEqual(r['output'],'<think>\nPreviously lost full trace\n</think>\n\nFinal')
  self.assertEqual(r['auxiliary_reasoning'],'Previously lost full trace')
  self.assertEqual(r['metadata']['message_conversion']['source_messages_sha256'],sha(compact(original)))
  self.assertEqual(original[-1],{'role':'assistant','content':'Final'});check_preservation(r)
  r['auxiliary_reasoning']='tampered'
  with self.assertRaises(AssertionError):check_preservation(r)
 def test_top_level_trace_keeps_strict_json_output_exact(self):
  r=self.run_row(self.row(domain='instruction',messages=[{'role':'user','content':'Only JSON'},{'role':'assistant','content':'{"a":1}'}],auxiliary_reasoning='Trace'))
  self.assertEqual(r['output'],'{"a":1}');self.assertEqual(r['messages'][-1]['reasoning_content'],'Trace');check_preservation(r)
 def test_conflicting_traces_are_both_preserved_and_flagged(self):
  r=self.run_row(self.row(messages=[{'role':'user','content':'Q'},{'role':'assistant','content':'Final','reasoning_content':'Message trace'}],auxiliary_reasoning='Different top-level trace'))
  self.assertIn('conflicting_auxiliary_reasoning',r['review_flags']);self.assertIn('Message trace',r['output']);self.assertEqual(r['auxiliary_reasoning'],'Different top-level trace');check_preservation(r)
 def test_empty_and_null_source_reasoning_roundtrip(self):
  for value in [None,'']:
   with self.subTest(value=value):
    r=self.run_row(self.row(messages=[{'role':'user','content':'Q'},{'role':'assistant','content':'Final','reasoning_content':value}],auxiliary_reasoning='Top-level trace'));check_preservation(r)
 def test_trace_without_messages_still_roundtrips(self):
  r=self.run_row(self.row(auxiliary_reasoning='Fallback trace'));self.assertIn('Fallback trace',r['output']);check_preservation(r)
 def test_old_release_without_optional_signature_remains_valid(self):
  # This is exactly the pre-change conversion contract: only message fields,
  # a messages-only content hash and reversible per-message fusion actions.
  r=self.run_row(self.row(messages=[{'role':'user','content':'Q'},{'role':'assistant','content':'Final','reasoning_content':'Trace'}]))
  r.pop('supervision_sha256');check_preservation(r)
 def test_different_supervision_is_not_a_duplicate_training_example(self):
  messages=[{'role':'user','content':'Q1'},{'role':'assistant','content':'A1'},{'role':'user','content':'Q2'},{'role':'assistant','content':'A2'}]
  first=self.run_row(self.row(messages=messages,assistant_loss_mask=[False,True,False,False]))
  last=self.run_row(self.row(messages=messages,assistant_loss_mask=[False,False,False,True]))
  self.assertEqual(first['record_content_sha256'],last['record_content_sha256'])
  self.assertNotEqual(supervision_signature(first),supervision_signature(last));check_preservation(first);check_preservation(last)
  same=self.run_row(self.row(messages=messages,assistant_loss_mask=[False,False,False,True]))
  self.assertEqual(supervision_signature(last),supervision_signature(same))
  last['assistant_loss_mask']=[False,True,False,False]
  with self.assertRaises(AssertionError):check_preservation(last)

class NegativeEvidenceTests(unittest.TestCase):
 def record(self,verification):
  return normalize_record({'id':'evidence-test','input':'Q','output':'A','final':'A','source':'test','domain':'code','quality_level':'source_verified','verification':verification},'fixture',1,'candidates')
 def override(self,r,field):
  return {'field':field,'decision':'superseded','method':'independent_execution_recheck','evidence':'audit/reexecution.json: current solution and tests passed','messages_sha256':sha(compact(r['messages'])),'output_sha256':sha(r['output'])}
 def test_label_does_not_override_failed_execution_or_binding(self):
  r=self.record({'source_execution_pass':False,'code_matches_final':False})
  self.assertEqual(set(explicit_negative_evidence(r)),{'source_execution_pass','code_matches_final'})
 def test_unknown_unperformed_and_symbolic_mismatch_are_not_failures(self):
  r=self.record({'process_verified':False,'local_execution_performed':False,'reference_validated':False,'independent_reference':False,'unit_tests_present_locally':False,'local_answer_pass':False,'local_symbolic_recheck':{'match':False},'source_pass_rate':-1,'method':'normalized_last_box_reference_match','answer_pass':False})
  self.assertEqual(explicit_negative_evidence(r),[])
 def test_known_execution_and_reference_methods_reject_explicit_failures(self):
  for method in ['hash_bound_prior_execution','human_reference_answer_span_checked','independent_finite_owa_horn_closure']:
   with self.subTest(method=method):self.assertIn('answer_pass',explicit_negative_evidence(self.record({'method':method,'answer_pass':False})))
  self.assertEqual(explicit_negative_evidence(self.record({'format_pass':False,'parse_pass':False,'reference_span_pass':False})),['format_pass','parse_pass','reference_span_pass'])
 def test_source_trials_are_not_individual_final_failures(self):
  r=self.record({'method':'upstream_paired_unit_test_rejection_sampling_and_local_structure','source_execution_pass':True,'source_r1_correctness':'True','source_pass_sequence':[0,1,0]})
  self.assertEqual(explicit_negative_evidence(r),[])
  r['verification']['source_r1_correctness']='False';self.assertIn('source_r1_correctness',explicit_negative_evidence(r))
 def test_pass_rate_unknown_differs_from_failed(self):
  r=self.record({'method':'upstream_execution_pass_rate_and_local_structure','source_pass_rate':-1});self.assertEqual(explicit_negative_evidence(r),[])
  r['verification']['source_pass_rate']=0.75;self.assertEqual(explicit_negative_evidence(r),['source_pass_rate'])
 def test_bound_override_clears_only_named_failure(self):
  r=self.record({'source_execution_pass':False,'code_matches_final':False});r['verification']['negative_evidence_overrides']=[self.override(r,'source_execution_pass')]
  self.assertEqual(explicit_negative_evidence(r),['code_matches_final'])
  r['verification']['negative_evidence_overrides'].append(self.override(r,'code_matches_final'));self.assertEqual(explicit_negative_evidence(r),[])
 def test_stale_or_unsubstantiated_override_cannot_clear_failure(self):
  for change in ['input','output','evidence','method']:
   with self.subTest(change=change):
    r=self.record({'source_execution_pass':False});o=self.override(r,'source_execution_pass');r['verification']['negative_evidence_overrides']=[o]
    if change=='input':r['messages'][0]['content']='Different question'
    elif change=='output':r['output']='Different answer'
    elif change=='evidence':o['evidence']=''
    else:o['method']='source_claim_only'
    self.assertEqual(explicit_negative_evidence(r),['source_execution_pass'])
 def test_symbolic_only_manual_clear_does_not_clear_execution_failure(self):
  r=self.record({'source_execution_pass':False,'local_symbolic_recheck':{'match':False},'local_visible_content_review':{'action':'clear_symbolic_mismatch_only'}})
  self.assertEqual(explicit_negative_evidence(r),['source_execution_pass'])

class BenchmarkTests(unittest.TestCase):
 def test_matching(self):
  with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]/'Datasets/expansion_20261016/tmp') as d:
   p=Path(d)/'index.jsonl';q='A farmer collects exactly 45 apples from the red tree and then gives 12 apples to a neighbour. How many apples are left?'
   p.write_text(json.dumps({'prompt':q,'benchmark':'test','source_file':'x','row':0})+'\n',encoding='utf-8');m=BenchmarkMatcher([p])
   self.assertEqual(m.match(q)[0]['match_type'],'normalized_exact')
   self.assertTrue(m.match('Solve carefully. '+q+' Show all work.'))
   self.assertEqual(m.match(q.replace('45','50'))[0]['match_type'],'numeric_template')
   self.assertFalse(m.match('A farmer has an entirely different problem about market prices and there are no shared numerical premises.'))
   self.assertFalse(m.match('How many apples are left?'))

if __name__=='__main__':unittest.main()
