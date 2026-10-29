import json,collections,heapq
from process import ROOT
out=ROOT/'verification_queue';out.mkdir(exist_ok=True)
groups=collections.defaultdict(list)
path=ROOT/'release/nemotron_sft_science_v2/review.jsonl'
for line in path.open(encoding='utf-8'):
 r=json.loads(line)
 if r['domain']!='physics' or set(r['review_flags'])!={'independent_science_answer_review_required'}:continue
 key=r['metadata'].get('subtopic','unspecified').strip().lower()
 if key in ['general chemistry','inorganic chemistry']:continue
 if key in ['undefined','unspecified']:key='other'
 score=int(r['problem_id'],16);item=(-score,r['id'],r)
 if len(groups[key])<2000:heapq.heappush(groups[key],item)
 elif item>groups[key][0]:heapq.heapreplace(groups[key],item)
groups={k:[item[2] for item in sorted(items,reverse=True)] for k,items in groups.items()};selected=[];offset=0
while len(selected)<2000:
 added=0
 for k in sorted(groups):
  if offset<len(groups[k]) and len(selected)<2000:selected.append(groups[k][offset]);added+=1
 if not added:break
 offset+=1
count=collections.Counter()
with (out/'physics_2000_blind_teacher.jsonl').open('w',encoding='utf-8') as prompts,(out/'physics_2000_source_evidence.jsonl').open('w',encoding='utf-8') as evidence:
 for r in selected:
  count[r['metadata'].get('subtopic','unspecified').strip().lower()]+=1
  prompt={'id':r['id'],'domain':r['domain'],'subtopic':r['metadata'].get('subtopic'),'input':r['input'],'messages':[{'role':'system','content':'Solve independently from the problem statement. Check whether the stated information is sufficient and whether exactly one offered answer is valid. Show a clear, complete solution, checking signs, dimensions, assumptions and numerical calculations when relevant. If the problem is ambiguous or no option is correct, say so instead of forcing an option. Follow the requested final-answer format when a valid answer exists.'},{'role':'user','content':r['input']}],'expected_answer':None,'source_answer_hidden':True,'token_limit_policy':'Do not truncate a completed solution to fit a fixed short response budget. Log finish_reason and full returned reasoning/final separately.','quality_status':'pending_independent_review'}
  prompts.write(json.dumps(prompt,ensure_ascii=False)+'\n');evidence.write(json.dumps(r,ensure_ascii=False)+'\n')
sampling_counts=collections.Counter()
for key,value in count.items():sampling_counts['other' if key in ['undefined','unspecified'] else key]+=value
report={'rows':len(selected),'subtopics':dict(count),'sampling_groups':dict(sampling_counts),'selection':'Round-robin subtopic quota, ascending normalized-question SHA256 within each subtopic, no tools/structural flags. Capitalization normalized and undefined merged into other for quotas; original subject labels retained in records.','source_evidence_file':'physics_2000_source_evidence.jsonl','teacher_input_file':'physics_2000_blind_teacher.jsonl','teacher_model_invoked':False,'next_gate':'Compare blinded solution to the source; disagreement requires a third check, agreement alone is not process proof. Do not show source_evidence to the blind teacher.'};(out/'README.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report))
