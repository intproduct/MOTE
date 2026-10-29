"""Export official OpenStax question and answer HTML recovered without inference."""
import json,re
from process import ROOT,Writer,base_record,classify,read_meta,sha

def process():
 meta=read_meta('physcorp_a.parquet');w=Writer('openstax_physics_train')
 for i,line in enumerate((ROOT/'raw/openstax_recovered.jsonl').open(encoding='utf-8')):
  row=json.loads(line);q=row['input'];a=row['output'];md=row['metadata'];r=base_record(meta,i,q,a,a,'physics',a)
  r['id']='openstax_official-'+sha(row['source_row']['source'])[:20]
  r['quality_level']='source_verified';r['response_mode']='direct';r['review_flags']=[x for x in r['review_flags'] if x!='very_short_solution_needs_review']
  r['metadata']={'subject':'physics','topic':md['topic'],'book':md['book'],'chapter':md['chapter'],'problem_number':md['problem_number'],'source_problem_id':row['source_row']['source'],'original_text_license':'CC-BY-4.0','collection_license':'CC-BY-NC-4.0','selection':'All 1429 OpenStax rows from pinned PhysCorp-A train; all other 5003 rows excluded.','transformation':'Official question and answer-key HTML converted to text/LaTex with MathML exponents retained; no generated explanation.','source_secondary_plain_question':json.loads(row['source_row']['messages'])[0]['content'],'source_secondary_plain_answer':row['source_row']['solution']}
  r['verification'].update(method='official_textbook_answer_key_link',answer_pass=True,independent_reference=True,reference_origin='OpenStax official linked answer key, not a teacher summary',publisher_evidence=row['answer_url'],local_semantic_entailment_verified=False,process_verified=False)
  r['provenance'][0]['raw_output_hash_scope']='converted official answer-key text, not the secondary corpus plain answer'
  r['provenance'].extend([{'source_url':row[k]['url'],'source_file':row[k]['local_path'],'source_file_sha256':row[k]['sha256'],'retrieval_date':'2026-10-16','dom_id':row[idk],'license':'CC-BY-4.0'} for k,idk in [('question_page','question_dom_id'),('answer_page','answer_dom_id')]])
  if row['has_visual'] or row['answer_has_visual']:r['review_flags'].append('official_problem_or_answer_contains_image')
  if re.search(r'\b(?:figure|table|graph|diagram)\b|shown below|shown above|previous (?:problem|exercise)|above (?:problem|exercise)|the (?:preceding|following) (?:problem|exercise)',q,re.I):r['review_flags'].append('textbook_external_context_or_visual_dependency')
  if re.search(r'\b(?:sketch|draw|plot)\b',q,re.I):r['review_flags'].append('requires_graphical_answer')
  if re.search(r'flint glass|crown glass|density of|specific heat of|latent heat of|heat capacity of|refractive index of',q,re.I):r['review_flags'].append('material_constant_may_require_book_table')
  if re.search(r'heat|melt|boil|thermal|temperature',q,re.I) and re.search(r'aluminum|copper|iron|steel|water|ice|lead|silver|mercury',q,re.I):r['review_flags'].append('thermal_material_constants_need_explicit_check')
  if re.search(r'\\text\{[^{}]*(?:x|\\times)\s*10[^{}]*\}\}\^',q+a):r['review_flags'].append('source_mathml_exponent_base_may_be_malformed')
  if re.search(r'\b(?:explain|why|discuss|justify|show that|verify|does your|does the answer)\b',q,re.I):r['review_flags'].append('answer_key_may_omit_requested_explanation')
  if any(x in md['topic'] for x in ['Nuclear','Radioactivity','Medical']):r['review_flags'].append('nuclear_constants_or_source_notation_needs_review')
  if row['source_row']['source']=='openstax_univ_phys_vol3_ch10_prob_59':r['review_flags'].append('known_incorrect_solution');r['verification']['answer_pass']=False;r['verification']['manual_audit']='Official answer-key MathML writes helium nuclei as H instead of He in reactions (ii)/(iii), despite the correct charge numbers. Preserve source notation error in quarantine, do not silently fix.'
  w.emit(r,classify(r,True))
 w.close(['All 1429 selected OpenStax train records recovered from 156 official question/answer pages with SHA256.','Source reference is official answer-key text, not an independently re-solved scientific proof.','MathML formatting recovery fixes secondary extraction exponent loss. Questions/answers needing visual context and nuclear/table constants remain review.','No SFT is manufactured from Physics StackExchange or UGPhysics subsets.','Original OpenStax text license CC-BY-4.0; conservative selected collection license CC-BY-NC-4.0 retained.'])
if __name__=='__main__':process()
