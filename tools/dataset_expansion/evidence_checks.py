"""Known explicit negative checks, not generic false-valued metadata.

Unknown/missing evidence, process_verified=False, unperformed local execution,
and symbolic answer non-equivalence are deliberately NOT classified as failures.
Overrides are per-field, content-bound local reviews; never infer one merely
from a higher quality label or a symbolic matcher passing another expression.
"""
from normalize_records import sha,compact

EXPLICIT_FALSE_FIELDS=('source_execution_pass','code_matches_final','local_python_ast_parse',
 'format_pass','parse_pass','reference_span_pass')
ANSWER_FAILURE_METHODS={
 'hash_bound_prior_execution','independent_finite_owa_horn_closure',
 'human_reference_answer_span_checked','official_textbook_answer_key_link',
 'publisher_explanation_and_gold_index_export','crowdsourced_gold_with_literal_support_check',
}
OVERRIDE_METHODS={'independent_execution_recheck','independent_recomputation','manual_content_recheck'}

def explicit_negative_evidence(record):
 """Return unresolved failure field paths for an otherwise eligible record.

 Optional verification.negative_evidence_overrides items must contain:
 field, decision='superseded', method (one of OVERRIDE_METHODS), evidence
 (nonempty audit reference/reason), messages_sha256 and output_sha256.
 A matching override only clears its named field, never another failure.
 This describes supplied local audit evidence; it performs no execution itself.
 """
 v=record.get('verification') or {};failures=[]
 for field in EXPLICIT_FALSE_FIELDS:
  if v.get(field) is False:failures.append(field)
 if v.get('answer_pass') is False and v.get('method') in ANSWER_FAILURE_METHODS:failures.append('answer_pass')
 if v.get('method')=='independent_finite_owa_horn_closure':
  for field in ('published_gold_match','published_closure_match'):
   if v.get(field) is False:failures.append(field)
 if v.get('method')=='upstream_execution_pass_rate_and_local_structure':
  rate=v.get('source_pass_rate')
  if type(rate) in (int,float) and 0<=rate<1:failures.append('source_pass_rate')
 if v.get('method')=='upstream_paired_unit_test_rejection_sampling_and_local_structure':
  if v.get('source_r1_correctness') is False or v.get('source_r1_correctness')=='False':failures.append('source_r1_correctness')
 if not failures:return []
 overrides=v.get('negative_evidence_overrides') or []
 if not isinstance(overrides,list):return failures
 # Use actual complete messages, including context; do not trust a stale hash.
 messages_hash=sha(compact(record['messages']));output_hash=sha(record.get('output') or '')
 cleared={o['field'] for o in overrides if isinstance(o,dict) and
  o.get('decision')=='superseded' and o.get('method') in OVERRIDE_METHODS and
  isinstance(o.get('evidence'),str) and o['evidence'].strip() and
  o.get('messages_sha256')==messages_hash and o.get('output_sha256')==output_hash and
  o.get('field') in failures}
 return [field for field in failures if field not in cleared]
