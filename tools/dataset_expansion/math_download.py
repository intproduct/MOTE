"""Bounded, resumable, hash-checked downloads without a C-drive HF cache."""
import concurrent.futures,hashlib,json,time
from pathlib import Path
import requests
from math_sources import BASE

TASKS={'nvidia/Nemotron-SFT-Math-v4':['data/train-00000-of-00012.parquet','data/train-00005-of-00012.parquet'],
       'open-r1/OpenR1-Math-220k':['data/train-00000-of-00010.parquet','data/train-00005-of-00010.parquet']}

def download(item):
 repo,revision,meta=item;folder=BASE/'raw'/repo.replace('/','__');folder.mkdir(parents=True,exist_ok=True);dest=folder/Path(meta['path']).name;part=dest.with_suffix(dest.suffix+'.part');expected=meta.get('lfs',{}).get('oid')
 if not dest.exists():
  url=f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{meta['path']}?download=true"
  for attempt in range(5):
   try:
    offset=part.stat().st_size if part.exists() else 0
    with requests.get(url,headers={'Range':f'bytes={offset}-'} if offset else {},stream=True,timeout=(30,120)) as r:
     r.raise_for_status()
     if offset and r.status_code!=206:offset=0
     if offset:assert r.headers.get('Content-Range','').startswith(f'bytes {offset}-')
     with part.open('ab' if offset else 'wb') as f:
      for block in r.iter_content(chunk_size=1024*1024):
       if block:f.write(block)
    assert part.stat().st_size==meta['size'],(part.stat().st_size,meta['size'])
    part.replace(dest);break
   except Exception as e:
    print(json.dumps({'event':'retry','repo':repo,'file':meta['path'],'attempt':attempt,'error':str(e)}),flush=True)
    if attempt==4:raise
    time.sleep(min(10,2**attempt))
 h=hashlib.sha256()
 with dest.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 digest=h.hexdigest();assert dest.stat().st_size==meta['size'];assert not expected or digest==expected
 result={'repo':repo,'revision':revision,'source_path':meta['path'],'local_path':str(dest),'bytes':dest.stat().st_size,'sha256':digest,'upstream_lfs_sha256_match':bool(expected),'download_url':f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{meta['path']}"}
 (dest.with_suffix('.download.json')).write_text(json.dumps(result,indent=2),encoding='utf-8');return result

if __name__=='__main__':
 items=[]
 for repo,paths in TASKS.items():
  inv=json.loads((BASE/'audit'/repo.replace('/','__')/'inventory.json').read_text(encoding='utf-8'));by={x['path']:x for x in inv['files']}
  for path in paths:items.append((repo,inv['revision'],by[path]))
 # Alternate sources so first usable files arrive from both sources early.
 items.sort(key=lambda x:(x[2]['path'].split('-')[1],x[0]))
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  for f in concurrent.futures.as_completed([pool.submit(download,x) for x in items]):print(json.dumps(f.result()),flush=True)
