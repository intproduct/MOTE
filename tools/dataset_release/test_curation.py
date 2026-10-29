import json,unittest
from common import *
from verifiers import verify_distilled,structural_flags

class CurationTests(unittest.TestCase):
 def test_control_latex_repair_is_targeted(self):
  fixed,changes=restore_damaged_latex('\x0crac{1}{2} + \x08eta + \theta + \x08oxed{2}')
  self.assertEqual(fixed,r'\frac{1}{2} + \beta + \theta + \boxed{2}');self.assertTrue(changes)
  self.assertEqual(restore_damaged_latex('a\tordinary\ntext')[0],'a\tordinary\ntext')
  self.assertEqual(restore_damaged_latex('\x07lpha_i + \x0barepsilon_{ijk}')[0],r'\alpha_i + \varepsilon_{ijk}')
 def test_nested_box(self):self.assertEqual(boxed(r'work \boxed{\frac{1}{2}}'),r'\frac{1}{2}')
 def test_no_partial_box(self):self.assertIsNone(boxed(r'\boxed{\frac{1}{2}'))
 def test_rational(self):self.assertTrue(answer_equal(r'\(\frac{1}{2}\)','0.5'))
 def test_symbolic_unknown_is_not_false(self):self.assertIsNone(answer_equal(r'\sqrt{2}','2^(1/2)'))
 def test_numeric_disagreement(self):self.assertFalse(answer_equal('0','4'))
 def test_bool_is_not_int(self):self.assertFalse(json_equal({'x':True},{'x':1}))
 def test_code_fences_exclude_prose(self):
  s='```text\nx = y\n```\nwhere `x` is explanatory.\n```python\ndef f(x):\n    return x+1\n```'
  c=select_code(s,'f');self.assertEqual(c['selected'],'def f(x):\n    return x+1\n')
 def test_code_example_not_solution(self):
  s='```python\n"example"\n```\n```python\ndef f(x):\n return x\n```'
  self.assertEqual(select_code(s,'f')['candidate_count'],1)
 def test_ambiguous_code_not_silently_selected(self):
  s='```python\ndef f(x):\n return x\n```\n```python\ndef f(x):\n return x+1\n```'
  self.assertIsNone(select_code(s,'f')['selected'])
 def test_unclosed_code(self):self.assertIn('unclosed_code_fence',structural_flags('q','','```python\nx=1'))
 def test_finished_flag(self):self.assertIn('generation_not_completed',structural_flags('q','r','a','length'))
 def test_multiple_answer_tags(self):self.assertIsNone(tagged('<answer>1</answer><answer>2</answer>'))
 def test_json_escape_repair(self):
  value,repairs=repair_json_text(r'{"answer":"\sqrt{2}"}');self.assertEqual(value['answer'],r'\sqrt{2}');self.assertEqual(len(repairs),1)
 def test_markdown_multiple_arrays(self):
  s=r'\[ {"id":"a\_1","problem":"q","solution":"s","answer":"a"} \]'+ '\n'+r'\[ {"id":"b","problem":"q2","solution":"s2","answer":"a2"} \]'
  good,bad=markdown_json_objects(s);self.assertEqual(len(good),2);self.assertFalse(bad)
 def test_latex_nu_not_corrupted(self):self.assertEqual(clean_recovered_markdown_value(r'\nu + a\nnext'),r'\nu + a\nnext')
 def test_inventory_from_prompt_not_gold(self):
  p='Track this inventory in order. Initial stock: {"A": 5, "B": 2}. Unit prices: {"A": 3, "B": 7}.\n1. Transfer up to 8 units from A to B, limited by the stock in A.\n2. Remove up to 3 units from B, stopping at zero.'
  self.assertEqual(inventory_oracle({'prompt':p}),{'stock':{'A':0,'B':4},'total_units':4,'inventory_value':28})
 def test_sort_ties_and_filter(self):
  p='Use these records: [{"id":"B","group":"blue","score":4,"units":2},{"id":"A","group":"blue","score":4,"units":3},{"id":"C","group":"amber","score":9,"units":1}]\nFilter: score >= 4 AND group is not amber. Sort by score descending, breaking ties by id ascending. Keep the first 2 records after filtering and sorting. Return only valid JSON'
  self.assertEqual(instruction_oracle({'prompt':p})['ids'],['A','B'])
 def test_prior_execution_requires_hash_and_all_tests(self):
  code='def f(x):\n return x\n';r={'verifier_type':'sandbox_tests','final':'```python\n'+code+'```','hidden_tests':{'fn_name':'f','inputs':[[1]],'outputs':[[1]]},'verification':{'code_sha256':sha(code),'answer_pass':True,'execution':{'passed':True,'tests_run':1,'tests_expected':1,'tests':[{'passed':True,'returncode':0}]}}}
  self.assertTrue(verify_distilled(r)['answer_pass']);r['verification']['code_sha256']='wrong';self.assertIsNone(verify_distilled(r)['answer_pass'])

if __name__=='__main__':unittest.main()
