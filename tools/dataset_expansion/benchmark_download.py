"""Build a question-only evaluation exclusion index; never training data."""
import concurrent.futures,hashlib,json,re
from pathlib import Path
import requests,pyarrow.parquet as pq
from benchmark_inventory import ROOT,BASE

SELECT={
 'openai/openai_humaneval':['openai_humaneval/test-00000-of-00001.parquet'],
 'google-research-datasets/mbpp':['full/test-00000-of-00001.parquet','full/validation-00000-of-00001.parquet','full/prompt-00000-of-00001.parquet'],
 'cais/mmlu':['all/test-00000-of-00001.parquet','all/validation-00000-of-00001.parquet','all/dev-00000-of-00001.parquet'],
 'TIGER-Lab/MMLU-Pro':['data/test-00000-of-00001.parquet','data/validation-00000-of-00001.parquet'],
 'google/IFEval':['ifeval_input_data.jsonl'],
 'allenai/ai2_arc':['ARC-Challenge/test-00000-of-00001.parquet','ARC-Challenge/validation-00000-of-00001.parquet','ARC-Easy/test-00000-of-00001.parquet','ARC-Easy/validation-00000-of-00001.parquet'],
 'Rowan/hellaswag':['data/validation-00000-of-00001.parquet','data/test-00000-of-00001.parquet'],
 'HuggingFaceH4/aime_2024':['data/train-00000-of-00001.parquet'],
 'math-ai/aime25':['test.jsonl'],
 'bigcode/bigcodebench':['data/v0.1.4-00000-of-00001.parquet']}

def fetch(item):
 repo,revision,meta=item;url=f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{meta['path']}";p=BASE/'raw'/(repo.replace('/','__')+'__'+meta['path'].replace('/','__'));p.parent.mkdir(exist_ok=True)
 if not p.exists():
  r=requests.get(url,timeout=120);r.raise_for_status();p.write_bytes(r.content)
 digest=hashlib.sha256(p.read_bytes()).hexdigest();assert p.stat().st_size==meta['size'];expected=meta.get('lfs',{}).get('oid');assert not expected or digest==expected
 records=pq.read_table(p).to_pylist() if p.suffix=='.parquet' else [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
 result=[]
 for idx,r in enumerate(records):
  prompt=r.get('question') or r.get('problem') or r.get('prompt') or r.get('text') or r.get('ctx') or r.get('instruct_prompt')
  if repo=='bigcode/bigcodebench':prompt=r.get('instruct_prompt') or r.get('complete_prompt')
  if not isinstance(prompt,str) or not prompt.strip():raise ValueError((repo,idx,list(r)))
  result.append({'benchmark':repo,'revision':revision,'source_file':meta['path'],'row':idx,'prompt':prompt,'source_id':r.get('task_id',r.get('question_id',r.get('id',r.get('key'))))})
 return result,{'repo':repo,'revision':revision,'path':str(p.relative_to(ROOT)),'source_file':meta['path'],'sha256':digest,'rows':len(records),'url':url}

if __name__=='__main__':
 tasks=[]
 for repo,paths in SELECT.items():
  inv=json.loads((BASE/(repo.replace('/','__')+'.json')).read_text(encoding='utf-8'));by={x['path']:x for x in inv['files']}
  tasks.extend((repo,inv['revision'],by[path]) for path in paths)
 sources=[];rows=[];errors=[]
 old=ROOT/'Datasets/processed/local_sft_v01_20261016/audit/benchmark_questions.jsonl'
 rows.extend(json.loads(x) for x in old.open(encoding='utf-8'));sources.append({'inherited_index':str(old.relative_to(ROOT)),'sha256':hashlib.sha256(old.read_bytes()).hexdigest(),'rows':len(rows)})
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  futures={pool.submit(fetch,t):t for t in tasks}
  for future in concurrent.futures.as_completed(futures):
   try:
    rr,ss=future.result();rows.extend(rr);sources.append(ss);print(json.dumps({'file':ss['source_file'],'repo':ss['repo'],'rows':len(rr)}),flush=True)
   except Exception as e:errors.append({'repo':futures[future][0],'path':futures[future][2]['path'],'error':str(e)})
 rows.sort(key=lambda r:(r['benchmark'],r['source_file'],r['row']))
 with (BASE/'questions.jsonl').open('w',encoding='utf-8') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 report={'rows':len(rows),'files':sources,'errors':errors,'scope':'Public benchmark questions for exclusion only. Exact/near-text exclusion is not a guarantee against semantic contamination.'};(BASE/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({'rows':len(rows),'errors':errors}))
