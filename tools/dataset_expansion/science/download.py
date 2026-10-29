import os,json,hashlib,time,concurrent.futures
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]/'Datasets/expansion_20261016/science'
for k,v in {'HF_HOME':'cache/hf','HF_DATASETS_CACHE':'cache/datasets','HUGGINGFACE_HUB_CACHE':'cache/hub','TEMP':'tmp','TMP':'tmp','PIP_CACHE_DIR':'cache/pip'}.items():
 p=ROOT/v;p.mkdir(parents=True,exist_ok=True);os.environ[k]=str(p)
import requests
SOURCES=[
 ('nvidia/OpenScienceReasoning-2','c684c79c75600eb28175ba444c24890b760a9e5b','default/partial-train/0000.parquet','osr2_0000.parquet','cc-by-4.0'),
 ('MegaScience/TextbookReasoning','ca7ecbec76d01bff2e99f3dc17735b02f87d4e96','data/train-00000-of-00001.parquet','textbook_reasoning.parquet','cc-by-nc-sa-4.0'),
 ('IUTVanguard/PhysicsEval','de38cf20db6c4c4c6476497a3de3d8f39efe13ad','train.json','physicseval_train.json','cc-by-sa-4.0')]
def go(s):
 repo,rev,path,name,license=s;url=f'https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}';dest=ROOT/'raw'/name
 meta={'repo':repo,'revision':rev,'path':path,'url':url,'license':license,'source_split':'train','local_path':str(dest.relative_to(ROOT))}
 if not dest.exists():
  for attempt in range(4):
   try:
    with requests.get(url,stream=True,timeout=(30,120)) as r:
     r.raise_for_status();meta['content_length']=r.headers.get('Content-Length');meta['etag']=r.headers.get('ETag')
     with dest.with_suffix(dest.suffix+'.part').open('wb') as f:
      n=0;last=time.monotonic()
      for chunk in r.iter_content(4*1024*1024):
       f.write(chunk);n+=len(chunk)
       if time.monotonic()-last>30:print(json.dumps({'downloading':name,'bytes':n}),flush=True);last=time.monotonic()
    dest.with_suffix(dest.suffix+'.part').replace(dest);break
   except Exception as e:
    print(json.dumps({'retry':name,'attempt':attempt,'error':str(e)}),flush=True)
    if attempt==3:raise
    time.sleep(2**attempt)
 h=hashlib.sha256()
 with dest.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 meta.update(bytes=dest.stat().st_size,sha256=h.hexdigest(),download_completed=True)
 (ROOT/'audit'/(name+'.download.json')).write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(meta),flush=True);return meta
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:items=list(pool.map(go,SOURCES))
 (ROOT/'audit/download_manifest.json').write_text(json.dumps(items,ensure_ascii=False,indent=2),encoding='utf-8')
