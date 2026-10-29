import unittest
from proofwriter import parse,atom,negate,close,truth
class ProofWriterTests(unittest.TestCase):
 def test_unknown_is_not_false(self):
  q=('a','is','blue','+');self.assertEqual(truth(q,set()),'Unknown');self.assertEqual(truth(q,{negate(q)}),'False')
 def test_explicit_negative_true(self):
  q=('a','is','blue','-');self.assertEqual(truth(q,{q}),'True')
 def test_contradiction_rejected(self):
  q=('a','is','blue','+')
  with self.assertRaises(ValueError):truth(q,{q,negate(q)})
 def test_variable_join_and_multistep(self):
  facts={('a','likes','b','+'),('b','is','blue','+'),('c','is','blue','+')}
  rules=[((('something','likes','someone','+'),('someone','is','blue','+')),('something','is','happy','+')),((('something','is','happy','+'),),('something','is','sad','-'))]
  got,depth=close(facts,rules);self.assertIn(('a','is','happy','+'),got);self.assertIn(('a','is','sad','-'),got);self.assertNotIn(('c','is','happy','+'),got);self.assertEqual(depth[('a','is','sad','-')],2)
 def test_cycle_without_seed(self):
  rules=[((('something','is','blue','+'),),('something','is','green','+')),((('something','is','green','+'),),('something','is','blue','+'))]
  got,_=close({('a','is','red','+')},rules);self.assertEqual(len(got),1)
 def test_no_negation_as_failure(self):
  rules=[((('something','is','blue','-'),),('something','is','happy','+'))]
  got,_=close({('a','is','red','+')},rules);self.assertNotIn(('a','is','happy','+'),got)
 def test_constant_head_grounding(self):
  rules=[((('something','is','blue','+'),),('mouse','likes','something','+'))]
  got,_=close({('a','is','blue','+')},rules);self.assertIn(('mouse','likes','a','+'),got)
 def test_reject_ignored_code(self):
  with self.assertRaises(ValueError):parse('("a" "is" "red" "+") print(1)')
 def test_negative_alias(self):self.assertEqual(atom(parse('("a" "is" "red" "~")')),('a','is','red','-'))
if __name__=='__main__':unittest.main()
