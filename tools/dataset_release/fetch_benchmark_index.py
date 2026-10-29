"""Download only public test shards, pinned to repository revisions, for exclusion."""
import hashlib,json
from pathlib import Path
import requests,pyarrow.parquet as pq
from common import DEST,write_json,write_jsonl

def main():
 cache=DEST/'audit/benchmark_cache';cache.mkdir(parents=True,exist_ok=True)
 rows=[];sources=[];errors=[]
 for repo,prefix in [('openai/gsm8k','main/'),('EleutherAI/hendrycks_math','')]:
  try:
   response=requests.get('https://huggingface.co/api/datasets/'+repo,timeout=20);response.raise_for_status();meta=response.json();revision=meta['sha']
   paths=[s['rfilename'] for s in meta['siblings'] if s['rfilename'].startswith(prefix) and '/test-' in s['rfilename'] and s['rfilename'].endswith('.parquet')]
   for path in paths:
    url=f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}'
    target=cache/(repo.replace('/','__')+'__'+path.replace('/','__'))
    if not target.exists():
     response=requests.get(url,timeout=60);response.raise_for_status();target.write_bytes(response.content)
    file_sha=hashlib.sha256(target.read_bytes()).hexdigest();data=pq.read_table(target).to_pylist()
    for i,r in enumerate(data):rows.append({'benchmark':repo,'revision':revision,'source_file':path,'row':i,'prompt':r.get('question',r.get('problem'))})
    sources.append({'repo':repo,'revision':revision,'file':path,'url':url,'sha256':file_sha,'rows':len(data)})
  except Exception as e:errors.append({'repo':repo,'error':str(e)})
 write_jsonl(DEST/'audit/benchmark_questions.jsonl',rows)
 write_json(DEST/'audit/benchmark_sources.json',{'sources':sources,'errors':errors,'questions':len(rows),'scope':'GSM8K main test and MATH test only; no blanket decontamination guarantee.'})
 print(json.dumps({'questions':len(rows),'sources':len(sources),'errors':errors}))
if __name__=='__main__':main()
