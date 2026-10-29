"""Conservative integrity/answer-leak gates, not a general mathematical judge."""
import re
REPOS={'nvidia/Nemotron-SFT-Math-v4','open-r1/OpenR1-Math-220k','nvidia/OpenMathReasoning','local/OpenMathReasoning_schema_shard'}

def unbalanced_tex_braces(text):
 depth=0
 for i,c in enumerate(text):
  # An odd run of backslashes escapes a literal brace.
  j=i-1;slashes=0
  while j>=0 and text[j]=='\\':slashes+=1;j-=1
  if slashes%2:continue
  if c=='{':depth+=1
  elif c=='}':
   depth-=1
   if depth<0:return True
 return depth!=0

def flags(record):
 if record['source'] not in REPOS:return []
 final=record.get('final') or '';prompt='\n\n'.join(m['content'] for m in record['messages'] if m.get('role')=='user' and isinstance(m.get('content'),str));found=[]
 if unbalanced_tex_braces(final):found.append('unbalanced_final_latex_braces')
 if not re.search(r'\\(?:boxed|fbox)\s*\{',final):found.append('missing_explicit_math_final_delivery')
 for m in re.finditer(r'(?im)^\s*(?:answer|solution|答案|解答)\s*[:：]\s*([^\n]+)',prompt):
  tail=m.group(1).strip()
  if tail and re.search(r'[0-9A-Za-z\u3400-\u9fff]',tail) and not re.fullmatch(r'(?i)(?:here|below|your answer|write here|fill in)[. _-]*',tail):found.append('source_prompt_contains_answer_or_solution_section');break
 if re.search(r'(?i)(?:problem|question|statement).{0,65}(?:contains? a typo|is (?:ill[- ]posed|under[- ]?specified|ambiguous|inconsistent)|is not (?:well[- ]posed|uniquely determined))|(?:intended|corrected) (?:equation|question|problem)|(?:standard interpretation|problem as stated likely expects)|(?:expected|provided|reference) answer.{0,70}(?:does not match|is incorrect|is wrong)|(?:force|match) (?:the )?(?:given|reference|expected) answer',final):found.append('final_relies_on_reinterpreting_or_repairing_problem')
 return sorted(set(found))
