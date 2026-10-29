import unittest
from lxml import html
from recover_openstax import render

class OpenStaxConversionTests(unittest.TestCase):
 def test_power(self):
  self.assertEqual(render(html.fromstring('<p>Voltage <math><mn>9</mn><mo>×</mo><msup><mn>10</mn><mn>3</mn></msup></math> V</p>')),r'Voltage $9\times {10}^{3}$ V')
 def test_fraction(self):
  self.assertIn(r'\frac{a}{{b}^{2}}',render(html.fromstring('<p><math><mfrac><mi>a</mi><msup><mi>b</mi><mn>2</mn></msup></mfrac></math></p>')))
 def test_semantics_no_duplicate_annotation(self):
  self.assertEqual(render(html.fromstring('<p><math><semantics><mi>x</mi><annotation encoding="application/x-tex">x</annotation></semantics></math></p>')),'$x$')
 def test_copy_namespace_attributes(self):
  self.assertEqual(render(html.fromstring('<p>A <a epub:type="noteref">word</a>.</p>')),'A word.')
 def test_superscript_subscript(self):
  self.assertIn(r'{x}_{i}^{2}',render(html.fromstring('<p><math><msubsup><mi>x</mi><mi>i</mi><mn>2</mn></msubsup></math></p>')))
 def test_nuclear_prescript(self):
  self.assertEqual(render(html.fromstring('<p><math><mmultiscripts><mi>Pu</mi><mprescripts/><none/><mn>239</mn></mmultiscripts></math></p>')),'${}^{239}{Pu}$')
if __name__=='__main__':unittest.main()
