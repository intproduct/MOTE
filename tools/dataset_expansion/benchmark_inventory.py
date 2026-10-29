"""Inventory public evaluation sets solely to exclude matching training data."""
import concurrent.futures,json
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'Datasets/expansion_20261016/benchmarks';BASE.mkdir(exist_ok=True)
REPOS=['openai/openai_humaneval','google-research-datasets/mbpp','cais/mmlu','TIGER-Lab/MMLU-Pro','google/IFEval','allenai/ai2_arc','Rowan/hellaswag','HuggingFaceH4/aime_2024','math-ai/aime25']
def fetch(repo):
 try:
  r=requests.get('https://huggingface.co/api/datasets/'+repo,timeout=35);r.raise_for_status();v=r.json();revision=v['sha'];files=[];url=f'https://huggingface.co/api/datasets/{repo}/tree/{revision}?recursive=true&expand=false&limit=1000'
  while url:
   r=requests.get(url,timeout=35);r.raise_for_status();files.extend(r.json());url=r.links.get('next',{}).get('url')
  value={'repo':repo,'revision':revision,'license':(v.get('cardData') or {}).get('license'),'files':files};(BASE/(repo.replace('/','__')+'.json')).write_text(json.dumps(value,indent=2),encoding='utf-8')
  data=[{'path':x['path'],'bytes':x.get('size')} for x in files if x.get('path','').endswith(('.parquet','.json','.jsonl'))]
  return {'repo':repo,'files':data[:18],'file_count':len(data)}
 except Exception as e:return {'repo':repo,'error':str(e)}
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  for r in pool.map(fetch,REPOS):print(json.dumps(r),flush=True)
