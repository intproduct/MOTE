import os,json,concurrent.futures
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]/'Datasets/expansion_20261016/science'
for k,v in {'HF_HOME':'cache/hf','HF_DATASETS_CACHE':'cache/datasets','HUGGINGFACE_HUB_CACHE':'cache/hub','TEMP':'tmp','TMP':'tmp','PIP_CACHE_DIR':'cache/pip'}.items():
 p=ROOT/v;p.mkdir(parents=True,exist_ok=True);os.environ[k]=str(p)
import requests
REPOS=['nvidia/OpenScienceReasoning-2','nvidia/OpenScience','MegaScience/TextbookReasoning','MegaScience/MegaScience','IUTVanguard/PhysicsEval']
def go(repo):
 d=requests.get('https://huggingface.co/api/datasets/'+repo,timeout=60).json()
 out={'repo':repo,'sha':d.get('sha'),'card':d.get('cardData'),'files':d.get('siblings')}
 (ROOT/'audit'/ (repo.replace('/','__')+'.json')).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
 if d.get('sha'):
  card=requests.get(f'https://huggingface.co/datasets/{repo}/raw/{d["sha"]}/README.md',timeout=60)
  (ROOT/'audit'/(repo.replace('/','__')+'_README.md')).write_text(card.text,encoding='utf-8')
 return out
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
 for x in pool.map(go,REPOS):print(json.dumps(x,ensure_ascii=True),flush=True)
