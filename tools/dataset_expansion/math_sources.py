"""Pin mathematical dataset metadata; all downloaded assets live under Datasets."""
import concurrent.futures,hashlib,json,os,time
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'Datasets/expansion_20261016/math'
REPOS=['nvidia/Nemotron-SFT-Math-v4','open-r1/OpenR1-Math-220k','nvidia/OpenMathReasoning']

def inventory(repo):
 target=BASE/'audit'/repo.replace('/','__');target.mkdir(parents=True,exist_ok=True)
 s=requests.Session();r=s.get('https://huggingface.co/api/datasets/'+repo,timeout=90);r.raise_for_status();info=r.json();revision=info['sha']
 (target/'repository.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
 card=s.get(f'https://huggingface.co/datasets/{repo}/resolve/{revision}/README.md',timeout=90);card.raise_for_status();(target/'README.md').write_bytes(card.content)
 files=[];url=f'https://huggingface.co/api/datasets/{repo}/tree/{revision}?recursive=true&expand=false&limit=1000'
 while url:
  r=s.get(url,timeout=90);r.raise_for_status();files.extend(r.json());url=r.links.get('next',{}).get('url')
 manifest={'repo':repo,'revision':revision,'readme_sha256':hashlib.sha256(card.content).hexdigest(),'license':(info.get('cardData') or {}).get('license'),'files':files}
 (target/'inventory.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
 data=[x for x in files if x.get('path','').endswith(('.parquet','.jsonl','.jsonl.gz'))]
 return {'repo':repo,'revision':revision,'license':manifest['license'],'files':len(data),'bytes':sum(x.get('size',0) for x in data),'examples':data[:5]}

if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  for r in pool.map(inventory,REPOS):print(json.dumps(r),flush=True)
