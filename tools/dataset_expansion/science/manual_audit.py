"""Bind concrete audit findings to full source text hashes, not source reputation."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]/'Datasets/expansion_20261016/science'
findings={
'MegaScience_TextbookReasoning-cd19c68e7a27ddd7fd85':('answer_conditionally_correct','For the virtual image interpretation, d_i=-12 cm gives d_o=8 cm and m=1.5. Output unnecessarily discusses an original answer unavailable in the input; no broad process certification.'),
'MegaScience_TextbookReasoning-045c42b8a8edb64dfed4':('answer_recomputed','Under the stated freely-jointed-chain model in the solution, sqrt((2*4.4e6/62.5)/6)*154 pm = 23.5910 nm. This checks the numeric calculation, not experimental model validity.'),
'MegaScience_TextbookReasoning-6b902fdb50caa58ddab6':('incorrect_reasoning','The real hyperbolic matrix shown acts on (x,y,z,ct). Applying that same matrix to (x,y,z,ict), as the output explicitly states, gives x cosh(alpha)-i ct sinh(alpha), not the claimed Lorentz transform. Imaginary time coordinates require imaginary off-diagonal entries.'),
'MegaScience_TextbookReasoning-fbd20b17959bb907c578':('incorrect_reasoning_and_answer','The source multiplies the field at the center of one loop by rings per unit length. That has units T/m, not T. For a long solenoid B=mu0*n*I; using its own I gives 2*pi*mu0*m*r^2*alpha*n/(e*R), not pi*mu0*m*r*alpha*n/(e*R).'),
'MegaScience_TextbookReasoning-84eec48d333c7fdc4cb8':('incorrect_answer','For maximal coupling J=j1+j2, the stretched-state magnetic moment is mu1+mu2. Substituting j1=1,j2=1/2,mu1=1,mu2=2 into the published formula yields 3.25 rather than 3; it is also undefined as written for j1=j2.'),
'MegaScience_TextbookReasoning-e58e8e2de883428dc009':('answer_conditionally_correct','In an empty rectangular waveguide with b the transverse y dimension, the TE01 angular cutoff is pi*c/b; ordinary frequency is c/(2b). The output explicitly labels angular frequency.'),
'MegaScience_TextbookReasoning-741f2d3ac33cde4d7f0a':('missing_context','The problem provides impeller diameter and speed but no pump-family performance curve or dimensionless coefficients. Output introduces C_Q=0.115,C_H=5,C_P=0.65 that cannot be inferred from the input.'),
'MegaScience_TextbookReasoning-c272e2a7039d510a16b9':('missing_context','The question supplies no lever arms, force sides, or piston areas. Output invents arm ratios 2:1:3 and equal piston area, so the specific pressure relation is not determined.'),
'MegaScience_TextbookReasoning-f375de9d009e1094eb0f':('missing_context_and_invented_value','Water saturation requires the capillary-pressure curve. Output admits the curve is missing and assumes it indicates 40 percent; this is not an answer derivable from the prompt.'),
'MegaScience_TextbookReasoning-19d71e722dd6b5bb3774':('incorrect_answer_gold_deference','With tan(theta)=12/5 and 0<theta<pi, cos(theta)=5/13. The resultant is sqrt(130^2+110^2+2*130*110*5/13)=200 N. Output computes 200 then overwrites it with the incorrect reference 185.7 N.'),
'MegaScience_TextbookReasoning-da3a77799eea1df0e8dd':('incorrect_answer','For y^2+2z^2<=4x, 0<=x<=2, cross-section area is 2*sqrt(2)*pi*x. Therefore x_bar=integral_0^2 x^2 dx / integral_0^2 x dx=4/3, not 4/5. Source subject is chemistry although this is multivariable calculus.'),
'MegaScience_TextbookReasoning-29a5c6308d365d80d126':('missing_context','The question asks about a specific guano composition table that is not included; the answer cites numerical entries from that missing table.'),
}
rows=[json.loads(x) for x in (ROOT/'audit/textbook_review_sample.jsonl').open(encoding='utf-8')]
audits=[]
for r in rows:
 if r['id'] not in findings:continue
 verdict,reason=findings[r['id']]
 audits.append({'id':r['id'],'domain':r['domain'],'input_sha256':hashlib.sha256(r['input'].encode()).hexdigest(),'output_sha256':hashlib.sha256(r['output'].encode()).hexdigest(),'verdict':verdict,'reason':reason,'reviewer':'assistant_science_audit','sample_selection':'10 smallest normalized-question SHA256 per publisher subject among initial structurally screened pool','source':r['source'],'source_revision':r['source_revision'],'source_row_index':r['provenance'][0]['row_index'],'input':r['input'],'output':r['output'],'expected_answer':r['expected_answer']})
(ROOT/'audit/manual_findings.json').write_text(json.dumps(audits,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'reviewed_with_written_findings':len(audits),'physics_sample_size':10,'physics_material_defects':7,'note':'Small diagnostic sample; do not extrapolate an exact corpus error rate.'}))
