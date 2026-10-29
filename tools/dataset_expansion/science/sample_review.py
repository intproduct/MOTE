import json,heapq,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]/'Datasets/expansion_20261016/science'
samples=collections.defaultdict(list)
for line in (ROOT/'normalized/textbook_reasoning/candidates.jsonl').open(encoding='utf-8'):
 r=json.loads(line);domain=r['domain'];key=int(r['problem_id'],16)
 item=(-key,r['id'],r)
 if len(samples[domain])<10:heapq.heappush(samples[domain],item)
 elif item>samples[domain][0]:heapq.heapreplace(samples[domain],item)
with (ROOT/'audit/textbook_review_sample.jsonl').open('w',encoding='utf-8') as f:
 for domain,items in samples.items():
  for _,__,r in sorted(items,reverse=True):
   f.write(json.dumps(r,ensure_ascii=False)+'\n')
   print(json.dumps({'id':r['id'],'domain':domain,'question':r['input'],'expected_answer':r['expected_answer'],'output':r['output'] if domain=='physics' else r['output'][:250],'matching':r['verification']['answer_pass']},ensure_ascii=True))
