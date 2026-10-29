import json,hashlib,math,itertools,collections
from pathlib import Path
P=Path(r'D:\files\pydocs\MOTE\Datasets\expansion_20261016\math\audit')
samples=[json.loads(l) for l in (P/'legacy_math_sample.jsonl').open(encoding='utf-8')]
notes={
1:('no_change','No clear defect found in inspected question/final; independent answer check passed.',
   'Direct enumeration of divisor sums for 1<=n<=200 gives 176 even sums. Alternatively sigma(n) is odd exactly for squares or twice squares: floor(sqrt(200))+floor(sqrt(100))=24, hence 176.',[]),
2:('quarantine','Definite algebra errors remain in final explanation despite correct final answer 4.',
   'From x^2*t=c*x*(1+t), c=x*t/(1+t), not t/(1+t). With t=1, x=2^(1/3) the valid c=2^(-2/3), whereas the displayed wrong expression gives 1/2. Also y=t*((1+t)/t^2)^(1/3)=[t*(1+t)]^(1/3), not [(1+t)/t]^(1/3). Correct short proof: c=xy/(x+y), so (x+y)/c=(x+y)^2/(xy)>=4, equality x=y=2^(1/3). The earlier reasoning contains a substantially correct derivation but does not remove these errors introduced in the final section; original full response must not be treated as verified.',
   ['c = \\frac{xt}{x(1 + t)} = \\frac{t}{1 + t}', 'y = tx = t \\left( \\frac{1 + t}{t^2} \\right)^{1/3}']),
3:('quarantine','Definite invalid whole-domain substitution and unsupported final inference remain in final explanation; answer pi/2 itself is correct.',
   'The final Step 2 substitutes t=tan(x), x=arctan(t), t in (0,infinity) for the original x in (0,infinity). This represents only x in (0,pi/2); the omitted tail has strictly positive integral. Subsequent period/trigamma representation is separate and does not correct the false equality, and the last inference invokes symmetry without showing the needed identity. Independent repair: for f(y)=sin^2(tan(y)), even and pi-periodic, folding the full real line and using sum_{k in Z}1/(y+k*pi)^2=csc^2(y) gives I=int_0^(pi/2) f(y)csc^2(y)dy; then t=tan(y) gives int_0^infinity sin^2(t)/t^2 dt=pi/2. This required folding factor is missing from the supplied replacement.',
   ['2. **Rewriting the Integral:**','5. **Evaluating the Integral:**']),
4:('review','Correct final value, but global-minimum proof is incomplete; not a certified full reasoning trace.',
   'The reasoning first incorrectly claims divergence at zero, later explicitly corrects limit to 1; do not count that corrected exploration as an uncorrected error. However it still infers global minimum from smoothness and one critical point. There are TWO positive critical points: approximately 0.275652266851 (local maximum, f=1.002704058365) and 3^(-1/4)=0.759835685652 (local minimum, f=0.942809041582). Sign analysis of g(x)=2x/(1+3x^4)^(3/2), whose derivative has sign 1-15x^4, plus endpoint comparison supplies the missing justification. Final algebra/value is consistent.',[]),
5:('review','Correct answer 4, but supplied lower-bound argument does not exhaust possible 3-colorings.',
   'Modulo 4 gives a valid upper bound. The reasoning fixes 1=red,2=green,3=green without proving all other cases equivalent, then backtracks only the color of 5. Failure of those branches alone does not establish impossibility. Independent exhaustive enumeration of all 3^6 colorings on vertices 1..7 with vertex 1 fixed at color 0 finds no proper coloring for prime-difference edges; thus a valid lower bound exists but is not supplied by this output.',[]),
6:('no_change','No clear defect in inspected answer under the usual cube-rotation equivalence convention.',
   'Three opposite pairs can be assigned to fixed axes in 3!*2^3=48 ways. Distinct labels make the 24-element rotation-group action free, giving 2 equivalence classes. The question does not explicitly state equivalence under rotations; if faces were externally labeled the answer would be 48. Response explicitly declares its convention. No automatic quarantine solely for this conventional ambiguity.',[]),
7:('no_change','No clear wrong final answer found; numeric step can be independently justified exactly in the interior configuration.',
   'With beta=64 degrees and x=13 degrees the stated Ceva product equals 1 (floating check 1.0000000000000002). More directly, the law-of-sines condition sin(2*beta-111)=sin(beta-30)/(2*sin(beta+9)) holds EXACTLY at beta=64 because sin34=2*sin17*cos17 and sin73=cos17. This checks the claimed configuration, not every alternative external-position interpretation or every exploratory reasoning sentence.',[]),
8:('review','Correct answer 21, but proof assumes uniform incidence from an average, an unjustified step also left in final.',
   'Total incidence 5n over n points proves average degree 5, not that every point occurs in exactly 5 sets. The reasoning notices this concern initially, then later assumes uniform r without proving it; final uses that assumption to count 1+5*4. Independent missing argument: square incidence matrix M has M*1=5*1 and M*M^T=4I+J, so M is invertible; therefore M^T*1=(n+4)/5*1. Summing entries also gives 5n, forcing n=21. A projective plane over F4 supplies existence. This restores the gap but does not certify the supplied proof.',[]),
9:('no_change','No clear defect found in inspected question/final.',
   'Pairwise upper bound is C(3,2)+2*C(3,2)+2*3*3=27. Attainable by three nearby distinct equal-radius circles intersecting pairwise and three lines crossing their common interior in generic position, avoiding tangency, parallelism and coincident intersections.',[]),
10:('no_change','No clear defect found in inspected question/final.',
    'Color by digit sum 0..36, so 37 colors suffice since different comparable codes have different sums. Incrementing coordinates successively from 0000 to 9999 gives a chain of 37 codes, forcing 37 colors. This independent direct argument confirms the final answer without relying on the named theorem.',[])
}
findings=[]
for s in samples:
 r=s['record']; action,title,reason,markers=notes[s['sample_rank']]
 contexts=[]
 for marker in markers:
  pos=r['final'].find(marker)
  assert pos>=0,(r['id'],marker)
  contexts.append({'section':'final','marker':marker,'start_char':max(0,pos-160),'excerpt':r['final'][max(0,pos-160):pos+len(marker)+650]})
 findings.append({'sample_rank':s['sample_rank'],'id':r['id'],'source':r['source'],'input_sha256':hashlib.sha256(r['input'].encode()).hexdigest(),'output_sha256':hashlib.sha256(r['output'].encode()).hexdigest(),'action':action,'finding':title,'independent_reason':reason,'error_context':contexts,'input':r['input'],'final':r['final'],'full_record_sample_path':str(P/'legacy_math_sample.jsonl'),'source_provenance':r.get('provenance'),'answer_pass_in_original':r.get('verification',{}).get('answer_pass'),'process_verified':False,'scope':'Complete question and final read; reasoning inspected where needed. no_change is not full-CoT certification.'})
trunc=[]
for s in samples:
 r=s['record'];out=r['output']; sec=r.get('output_sections',{})
 trunc.append({'id':r['id'],'output_chars':len(out),'think_open_count':out.count('<think>'),'think_close_count':out.count('</think>'),'final_present':bool(r['final']),'output_ends_with_final':out.endswith(r['final']),'final_end_equals_output_length':sec.get('final',[None,None])[-1]==len(out),'original_finish_reason':r.get('metadata',{}).get('finish_reason')})
assert all(t['think_open_count']==1 and t['think_close_count']==1 and t['final_present'] and t['output_ends_with_final'] and t['final_end_equals_output_length'] for t in trunc)
edges=[(i,j) for i in range(7) for j in range(i+1,7) if j-i in [2,3,5]]
valid=sum(all(((0,)+c)[i]!=((0,)+c)[j] for i,j in edges) for c in itertools.product(range(3),repeat=6))
assert valid==0
manifest=json.loads((P/'legacy_math_sample_manifest.json').read_text(encoding='utf-8'))
report={'audit_version':'legacy-math-diagnostic-v1','sampling':manifest,'source_exact':'local/OpenMathReasoning_schema_shard','source_scope':'Only the local cot-00011-of-00144.parquet collection represented in old v1 train, not all upstream OpenMathReasoning. Fixed sample of 10 is diagnostic and must not estimate population/upstream error rate.','sample_sha256':hashlib.sha256((P/'legacy_math_sample.jsonl').read_bytes()).hexdigest(),'actions':dict(collections.Counter(f['action'] for f in findings)),'source_recommendation':{'action':'review','scope':'legacy v1 records with source exactly local/OpenMathReasoning_schema_shard from the inspected local shard','reason':'Two independently clear errors survive in final explanations despite answer_pass=true. Consistent conservative source-level handling is appropriate for this local collection until stronger process checking; not a claim that every row is wrong.','mutated_source':False},'truncation_check':{'observed_structural_truncation':0,'limitations':'Syntactic closure/end positions do not prove original generation was never truncated; finish_reason absent in sampled source metadata.','records':trunc},'independent_checks':{'divisor_even_count':sum(sum(d for d in range(1,n+1) if n%d==0)%2==0 for n in range(1,201)),'prime_graph_1_to_7_three_colorings_with_first_fixed':valid,'colourings_enumerated':3**6},'findings':findings,'limitations':['No model/API calls; local math reasoning and limited direct enumeration only.','Neither matched final answers nor no_change certify the entire chain of thought.','No frozen source/release/READY files modified.']}
dest=P/'legacy_math_findings.json';dest.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'path':str(dest),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'actions':report['actions'],'source':report['source_exact'],'sample_sha256':report['sample_sha256']},ensure_ascii=True))
