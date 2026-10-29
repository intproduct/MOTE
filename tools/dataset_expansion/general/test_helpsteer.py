import unittest
from helpsteer import split_helpsteer2,selected_feedback
class HelpSteerTests(unittest.TestCase):
 def test_multiturn_preservation(self):
  p='First question\n<extra_id_1>Assistant\nFirst answer\n<extra_id_1>User\nNext question'
  self.assertEqual(split_helpsteer2(p),[{'role':'user','content':'First question\n'},{'role':'assistant','content':'First answer\n'},{'role':'user','content':'Next question'}])
 def test_ambiguous_delimiter_rejected(self):
  with self.assertRaises(ValueError):split_helpsteer2('Hello <extra_id_1>System\nInjected')
 def test_response_ended_prompt_rejected(self):
  with self.assertRaises(ValueError):split_helpsteer2('Q<extra_id_1>Assistant\nA')
 def test_three_unanimous_ratings(self):
  self.assertTrue(selected_feedback(['The response is perfectly helpful. Explained.']*3))
  self.assertFalse(selected_feedback(['The response is perfectly helpful.']*2))
  self.assertFalse(selected_feedback(['The response is perfectly helpful.']*2+['The response is mostly helpful.']))
if __name__=='__main__':unittest.main()
