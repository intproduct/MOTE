"""Pinned, bounded downloads; all artifacts stay under D:/.../Datasets."""
import concurrent.futures, hashlib, json, os, time
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parents[3]
DEST=ROOT/'Datasets/expansion_20261016/code'
RAW=DEST/'raw'
for p in [RAW,DEST/'audit',DEST/'tmp']:p.mkdir(parents=True,exist_ok=True)
os.environ['TMP']=os.environ['TEMP']=str(DEST/'tmp')
REPO='nvidia/OpenCodeReasoning-2'
REV='eadf535931451525f3e5621d0f960c240bc62fd9'

def get(url):
 r=requests.get(url,timeout=90);r.raise_for_status();return r

def download(info):
 p=RAW/Path(info['path']).name
 repo=info.get('_repository',REPO);rev=info.get('_revision',REV)
 if info.get('_prefix'):p=RAW/info['_prefix']/Path(info['path']).name;p.parent.mkdir(parents=True,exist_ok=True)
 url=f'https://huggingface.co/datasets/{repo}/resolve/{rev}/{info["path"]}'
 if p.exists() and p.stat().st_size==info['size']:
  h=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
  if h==info['lfs']['oid']:return {**info,'url':url,'local_path':str(p.relative_to(ROOT)),'sha256':h}
 for attempt in range(4):
  try:
   part=p.with_suffix('.part');size=part.stat().st_size if part.exists() else 0
   # Bounded range requests avoid long full-object CDN stalls and can resume.
   with part.open('ab') as f:
    while size<info['size']:
     end=min(size+8*1024*1024,info['size'])-1
     r=requests.get(url,headers={'Range':f'bytes={size}-{end}'},timeout=(30,90));r.raise_for_status()
     assert r.status_code==206 and len(r.content)==end-size+1 and r.headers.get('Content-Range')==f'bytes {size}-{end}/{info["size"]}',(r.status_code,len(r.content),size,end,r.headers.get('Content-Range'))
     f.write(r.content);f.flush();size=end+1
     if size%(64*1024*1024)==0:print(json.dumps({'download_progress':p.name,'bytes':size}),flush=True)
   h=hashlib.file_digest(part.open('rb'),'sha256')
   assert size==info['size'] and h.hexdigest()==info['lfs']['oid'],(size,info['size'])
   p.with_suffix('.part').replace(p)
   result={**info,'url':url,'local_path':str(p.relative_to(ROOT)),'sha256':h.hexdigest()}
   print(json.dumps({'download_complete':p.name,'bytes':size}),flush=True)
   return result
  except Exception:
   if attempt==3:raise
   time.sleep(2)

def main():
 readme=get(f'https://huggingface.co/datasets/{REPO}/raw/{REV}/README.md').text
 (RAW/'README_OpenCodeReasoning-2.md').write_text(readme,encoding='utf-8')
 listing=get(f'https://huggingface.co/api/datasets/{REPO}/tree/{REV}/train/python?limit=100').json()
 (DEST/'audit/upstream_python_listing.json').write_text(json.dumps(listing,indent=2),encoding='utf-8')
 selected=listing[:4]
 with concurrent.futures.ThreadPoolExecutor(2) as pool:files=list(pool.map(download,selected))
 manifest={'repository':REPO,'revision':REV,'dataset_card_url':f'https://huggingface.co/datasets/{REPO}/blob/{REV}/README.md','license':'cc-by-4.0','files':files,'download_policy':'first four Python shards, complete records; no generated code executed locally','verification_notes':'pass_rate is upstream execution fraction, -1 means unavailable; judgement is QwQ judgement, not execution','decontamination_claim':'card excludes CodeContests/open-r1-codeforces test; local benchmark overlap checks pending','question_restoration':'question field is placeholder; original upstream question must be restored using dataset/split/index'}
 (DEST/'audit/source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 print(json.dumps(manifest),flush=True)

if __name__=='__main__':main()
