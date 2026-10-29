import json,heapq,math
from process import ROOT,sha
notes={
'allenai_sciq-2a5863306f7bfc4011c2':('missing_support','Support says only that irrigation has problems; it does not define irrigation as the question requires. Literal answer occurrence is insufficient.'),
'allenai_sciq-32d570fb09bd7fa64903':('known_oversimplification_in_support','The support says the human appendix serves no purpose. The answer word is unrelated to this known oversimplification, but retaining the full support teaches it; move to review.'),
'allenai_sciq-8ac37adb6c001c7d696f':('support_biological_sequence_needs_review','The support states daughter cells complete the first meiotic division before puberty then the second after puberty. Human oocytes arrest in prophase I prenatally and complete meiosis I upon later maturation; the simplified passage is misleading.'),
'lupantech_ScienceQA-2e2c2e12ec387df59240':('source_option_order_contradiction','Explanation calls the vague Paul/Aaron sentence second, but source choices place it first; correct gold index alone does not fix explanation.'),
'lupantech_ScienceQA-b6bc2b0403041a80ef18':('source_option_order_contradiction','Explanation calls vague them sentence first but source choices place it second.'),
'lupantech_ScienceQA-4bdb728c2658359e0247':('missing_dynamic_assumptions','Starting two different masses at the same speed does not by itself establish a force comparison. Equal acceleration/starting time and friction assumptions are missing, and lecture incorrectly associates force directly with speed.'),
'openstax_official-acf91c9df7274aa9ab23':('ambiguous_count_of_minima','Single-slit minima occur on both sides. The answer interprets50/1000 as order or one-sided counts, but prompt says number of minima without specifying. Ambiguous counting convention; review rather than silently choosing.'),
'openstax_official-2584cff960706d503dec':('missing_reference_frame','0.991c follows if0.800c is the relative ship speed: (.999-.8)/(1-.999*.8). If both ships move at0.800c in a third frame the answer differs. Prompt does not specify frame, so retain for review.')
}
records={};samples=[]
for source,domain in [('sciq_train',None),('scienceqa_text_train','physics'),('openstax_physics_train',None)]:
 folder=ROOT/'normalized'/source
 rows=[json.loads(x) for x in (folder/'candidates.jsonl').open(encoding='utf-8')]
 chosen=heapq.nsmallest(10,(r for r in rows if domain is None or r['domain']==domain),key=lambda r:r['problem_id'])
 for r in chosen:
  samples.append({'id':r['id'],'input_sha256':sha(r['input']),'output_sha256':sha(r['output']),'verdict':'review_required' if r['id'] in notes else 'no_clear_error_in_answer_not_full_certification','scope':'Manual reading of full exported input/output; short numerical checks where feasible.','input':r['input'],'output':r['output'],'provenance':r['provenance']})
 for bucket in ['candidates','review','quarantine']:
  for line in (folder/(bucket+'.jsonl')).open(encoding='utf-8'):
   r=json.loads(line)
   if r['id'] in notes:
    code,reason=notes[r['id']];records[r['id']]={'id':r['id'],'input_sha256':sha(r['input']),'output_sha256':sha(r['output']),'flag':code,'reason':reason,'input':r['input'],'output':r['output'],'provenance':r['provenance']}
assert set(records)==set(notes)
records['allenai_sciq-8ac37adb6c001c7d696f']['primary_references']=['https://openstax.org/books/biology-2e/pages/43-3-human-reproductive-anatomy-and-gametogenesis']
records['allenai_sciq-32d570fb09bd7fa64903']['primary_references']=['https://openstax.org/books/concepts-biology/pages/17-3-adaptive-immunity']
report={'sampling':'Ten lowest normalized-question hashes from updated candidate pool of each source (ScienceQA physics only); diagnostic, not global correctness estimate.','sample_count':len(samples),'known_findings':list(records.values()),'samples':samples,'independent_calculation_examples':{'plate_m_per_second':.04/(365.25*24*3600),'plate_km_per_Myr':.04*1e6/1000,'bags_lb':2*13.5+10.2,'force_newton_before_significant_digits':55*.0255,'wave_12MHz_m':299792458/12e6,'poiseuille_radius_ratio':(.1/1.2)**.25,'radar_500MHz_m':299792458/500e6,'cheetah_acceleration_mps2':30/7},'process_verified_for_entire_source':False}
(ROOT/'audit/foundation_semantic_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({'samples':len(samples),'findings':len(records)}))
