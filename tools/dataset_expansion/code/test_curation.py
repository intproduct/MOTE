import unittest
from process_code import inspect_response
from test_evidence import analyse
from example_evidence import analyse as analyse_examples

class CurationTests(unittest.TestCase):
 def test_lossless_sections(self):
  out='<think>\nReason in full.\n</think>\n\n```python\nx = 1\n```'
  flags,final,sections=inspect_response(out,'x = 1')
  self.assertEqual(flags,[]);self.assertEqual(out[slice(*sections['final'])],final)
 def test_nonmatching_code_is_not_verified(self):
  flags,_,_=inspect_response('<think>r</think>\n```python\nx = 2\n```','x = 1')
  self.assertIn('final_code_not_exactly_bound_to_verified_solution',flags)
 def test_missing_final(self):
  flags,_,_=inspect_response('<think>r</think>','x = 1');self.assertIn('missing_final',flags)
 def test_unclosed_trace(self):
  flags,_,_=inspect_response('<think>still reasoning','x = 1');self.assertIn('invalid_reasoning_boundary',flags)
 def test_repeated_long_paragraph(self):
  body='Repeated explanation. '*20
  flags,_,_=inspect_response('<think>'+('\n\n'.join([body]*3))+'</think>\n```python\nx = 1\n```','x = 1')
  self.assertIn('repeated_long_reasoning_paragraph',flags)
 def test_multiple_final_programs(self):
  flags,_,_=inspect_response('<think>r</think>\n```python\nx = 1\n```\n```python\nx = 2\n```','x = 1')
  self.assertIn('multiple_final_code_blocks',flags)
 def test_empty_assertions_rejected(self):
  result=analyse('from solution import f\ndef test_f():\n f(1)\n assert True\n assert 1\n assert 2 == 2','def f(x): return x')
  self.assertFalse(result['eligible']);self.assertEqual(result['linked_value_checks'],0)
 def test_linked_result_checks_pass(self):
  result=analyse('from solution import f\ndef test_f():\n a=f(1)\n assert a==1\n assert f(2)==2\n assert f(3)==3','def f(x): return x')
  self.assertTrue(result['eligible']);self.assertEqual(result['linked_value_checks'],3)
 def test_unrelated_assertions_rejected(self):
  result=analyse('from solution import f\ndef test_f():\n result=f(1)\n x=1\n assert x==1\n assert x==1\n assert x==1','def f(x): return x')
  self.assertFalse(result['eligible'])
 def test_tautological_result_comparison_rejected(self):
  result=analyse('from solution import f\ndef test_f():\n a=f(1)\n assert a==a\n assert a==a\n assert a==a','def f(x): return x')
  self.assertFalse(result['eligible'])
 def test_unittest_instance_result_does_not_taint_self(self):
  result=analyse('from solution import f\nclass T:\n def test_f(self):\n  self.result=f(1)\n  self.assertEqual(1,1)\n  self.assertEqual(2,2)\n  self.assertEqual(3,3)','def f(x): return x')
  self.assertFalse(result['eligible'])
 def test_parameterized_checks(self):
  result=analyse('from solution import f\nimport pytest\n@pytest.mark.parametrize("x", [1,2,3])\ndef test_f(x):\n assert f(x)==x','def f(x): return x')
  self.assertTrue(result['eligible'])
 def test_overwritten_result_is_not_a_check(self):
  result=analyse('from solution import f\ndef test_f():\n a=f(1)\n a=1\n assert a==1\n assert a==1\n assert a==1','def f(x): return x')
  self.assertFalse(result['eligible'])
 def test_alias_of_result_is_not_expected_output(self):
  result=analyse('from solution import f\ndef test_f():\n a=f(1)\n b=a\n assert a==b\n assert a==b\n assert a==b','def f(x): return x')
  self.assertFalse(result['eligible'])
 def test_assert_before_solution_call_is_not_linked(self):
  result=analyse('from solution import f\ndef test_f():\n a=1\n assert a==1\n assert a==1\n assert a==1\n a=f(1)','def f(x): return x')
  self.assertFalse(result['eligible'])
 def test_invalid_inline_doctest(self):
  r=analyse_examples('>>> f(1) 2','assert f(1)==2','def f(x): return x+1')
  self.assertIn('prompt_doctest_python_syntax_error',r['flags'])
 def test_doctest_conflicting_with_literal_test(self):
  r=analyse_examples('>>> f([1,2,3])\n120\n','def test_f():\n a=[1,2,3]\n result=f(a)\n assert result==6','def f(x): return 6')
  self.assertIn('prompt_example_conflicts_with_unit_test_oracle',r['flags'])
 def test_doctest_matching_test(self):
  r=analyse_examples('>>> f([1,2,3])\n6\n','def test_f():\n assert f([1,2,3])==6','def f(x): return 6')
  self.assertEqual(r['flags'],[]);self.assertEqual(r['matched_test_oracles'],1)
 def test_unresolved_test_expression_is_not_a_conflict(self):
  r=analyse_examples('>>> f(1)\n[0,0,0]\n','def test_f():\n assert f(1)==[0]*3','def f(x): return [0,0,0]')
  self.assertEqual(r['flags'],[]);self.assertEqual(r['matched_test_oracles'],0)

if __name__=='__main__':unittest.main()
