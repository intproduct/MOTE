import unittest
from process import boxes,answer_match,flags_for
class TestScienceCuration(unittest.TestCase):
 def test_balanced_box(self):self.assertEqual(boxes(r'a \boxed{\frac{1}{2}} b'),[r'\frac{1}{2}'])
 def test_unclosed_box(self):self.assertEqual(boxes(r'\boxed{\frac{1}{2}'),[])
 def test_simple_numeric_exact(self):self.assertTrue(answer_match('2.00','2'))
 def test_large_numeric_not_float_collapsed(self):self.assertFalse(answer_match('100000000000000001','100000000000000002'))
 def test_unsupported_symbolic_not_rejected(self):self.assertIsNone(answer_match('x+x','2x'))
 def test_same_publisher_answer_not_independence(self):self.assertTrue(answer_match(r'\text{A}','A'))
 def test_wrong_mcq(self):self.assertFalse(answer_match('A','B'))
 def test_expected_answer_leakage(self):self.assertIn('answer_mentions_hidden_reference_or_expected_solution',flags_for('Find force.','The reference document states 185.7 so use that answer.'))
 def test_missing_curve(self):self.assertIn('answer_admits_missing_context',flags_for('Find saturation.','The curve is not provided, so suppose it indicates forty percent.'))
 def test_visual_question(self):self.assertIn('possible_missing_visual_context',flags_for('As shown in the figure, find x.','The answer is forty.'))
 def test_full_reasoning_kept_no_length_flag(self):self.assertNotIn('length',flags_for('Solve.','A'*100000))
if __name__=='__main__':unittest.main()
