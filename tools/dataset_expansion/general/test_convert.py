import unittest
from convert import strict_json,format_check,simple_constraints,structural

class Checks(unittest.TestCase):
 def test_duplicate_json(self):
  with self.assertRaises(ValueError):strict_json('{"a":1,"a":2}')
 def test_wrong_json_schema(self):
  r=format_check('Schema: {"type":"object","properties":{"x":{"type":"integer"}},"required":["x"],"additionalProperties":false}','{"x":"three"}')
  self.assertFalse(r['format_pass'])
 def test_json_schema_true_semantics_unknown(self):
  r=format_check('Schema: {"type":"object","properties":{"x":{"type":"integer"}},"required":["x"]}','{"x":3}')
  self.assertTrue(r['format_pass']);self.assertIsNone(r['answer_pass'])
 def test_xml_schema(self):
  schema='<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"><xs:element name="n" type="xs:integer"/></xs:schema>'
  self.assertTrue(format_check(schema,'<n>3</n>')['format_pass'])
  self.assertFalse(format_check(schema,'<n>word</n>')['format_pass'])
 def test_external_xml(self):self.assertFalse(format_check('','<!DOCTYPE x SYSTEM "file:///c:/file"><x/>')['parse_pass'])
 def test_upper(self):self.assertFalse(simple_constraints(['in english and capital'],'Partial caps','Write the response in English and capital letters.')['uppercase'])
 def test_context_null(self):self.assertIn('missing_or_nontext_content',structural([{'role':'user','content':None},{'role':'assistant','content':'Answer'}]))
 def test_closed_reasoning(self):self.assertEqual(structural([{'role':'user','content':'Q'},{'role':'assistant','content':'Answer','reasoning_content':'Complete reasoning.'}]),[])
 def test_json_fence(self):self.assertTrue(simple_constraints(['format:use json format'],'```json\n{"x":1}\n```','Return only JSON format.')['json_parse'])
 def test_capitalized_is_ambiguous(self):self.assertEqual(simple_constraints(['in english and capital'],'The Title','Make the title in English and capitalized.'),{})
 def test_algorithm_output_not_whole_answer(self):self.assertEqual(simple_constraints(['format:use json format'],'An algorithm followed by JSON.','Provide the algorithm output in JSON format.'),{})
 def test_partial_constraints_not_conflated(self):self.assertEqual(simple_constraints(['length constraints:number of paragraphs'],'unknown count'),{})
 def test_unasked_taxonomy_constraint_ignored(self):self.assertEqual(simple_constraints(['in english and capital','punctuation:use no comma'],'Ordinary, text.','Give a title and explanation.'),{})
 def test_keyword_uppercase_not_whole_response(self):self.assertEqual(simple_constraints(['in english and capital'],'The word TRUCK.','Include the word TRUCK in all capital letters once.'),{})
if __name__=='__main__':unittest.main()
