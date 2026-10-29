"""Pinned, resumable downloads; preserve complete source lines and provenance."""
import os,json,time,hashlib,concurrent.futures
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'Datasets/expansion_20261016/general'
RAW=BASE/'raw';AUDIT=BASE/'audit'
for d in [RAW,AUDIT,BASE/'tmp']:d.mkdir(parents=True,exist_ok=True)
for k in ['TMP','TEMP','TMPDIR','HF_HOME','HF_DATASETS_CACHE','HUGGINGFACE_HUB_CACHE']:
 os.environ[k]=str(BASE/'tmp')
JOBS=[
 ('nvidia/Nemotron-Instruction-Following-Chat-v1','data/structured_outputs.jsonl','nemotron_structured.jsonl',None),
 ('nvidia/Llama-Nemotron-Post-Training-Dataset','SFT/chat/chat.jsonl','llama_nemotron_chat.jsonl',None),
 ('nvidia/Nemotron-Instruction-Following-Chat-v1','data/chat_if.jsonl','nemotron_chat_if.prefix50000.jsonl',50000),
 ('nvidia/Nemotron-SFT-Instruction-Following-Chat-v3','data/instruction_following.jsonl','nemotron_if_v3.prefix30000.jsonl',30000),
]
def fetch(job):
 repo,path,name,limit=job
 meta=json.loads((AUDIT/(repo.split('/')[-1]+'.metadata.json')).read_text(encoding='utf-8'))
 file=next(x for x in meta['siblings'] if x['rfilename']==path);rev=meta['sha']
 url=f'https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}?download=true'
 dest=RAW/name;done=AUDIT/(name+'.download.json');part=dest.with_suffix(dest.suffix+'.part')
 if done.exists() and dest.exists():return json.loads(done.read_text())
 for attempt in range(8):
  try:
   # Restart partial JSONL at the last complete newline, then HTTP Range resume.
   start=part.stat().st_size if part.exists() else 0
   if start:
    with part.open('rb+') as f:
     f.seek(max(0,start-1048576));tail=f.read();pos=tail.rfind(b'\n');start=start-len(tail)+pos+1 if pos>=0 else 0;f.truncate(start)
   rows=0
   if start and limit:
    with part.open('rb') as f:rows=sum(1 for _ in f)
   if limit and rows>=limit:break
   headers={'Range':f'bytes={start}-'} if start else {}
   with requests.get(url,headers=headers,stream=True,timeout=(40,120)) as response:
    response.raise_for_status()
    if start and response.status_code!=206:start=0;rows=0
    with part.open('ab' if start else 'wb') as f:
     if limit:
      for line in response.iter_lines(chunk_size=1024*1024):
       if not line:continue
       json.loads(line);f.write(line+b'\n');rows+=1
       if rows>=limit:break
     else:
      for chunk in response.iter_content(1024*1024):f.write(chunk)
   break
  except Exception as e:
   print(json.dumps({'job':name,'attempt':attempt+1,'error':str(e)}),flush=True)
   if attempt==7:raise
   time.sleep(min(20,attempt*3+2))
 h=hashlib.sha256()
 with part.open('rb') as f:
  rows=0
  for line in f:
   h.update(line);json.loads(line);rows+=1
 digest=h.hexdigest()
 if not limit:
  assert part.stat().st_size==file['size'],(name,part.stat().st_size,file['size'])
  assert digest==file['lfs']['sha256'],name
 part.replace(dest)
 report={'repo':repo,'revision':rev,'path':path,'url':url,'local_path':str(dest.relative_to(ROOT)).replace('\\','/'),'source_bytes':file['size'],'source_lfs_sha256':file.get('lfs',{}).get('sha256'),'bytes':dest.stat().st_size,'sha256':digest,'rows':rows,'selection':'full_file' if not limit else f'first_{limit}_complete_nonempty_lines','full_file_hash_verified':not bool(limit),'source_split':path.split('/')[-1].replace('.jsonl',''),'download_completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
 done.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True);return report
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
  for report in ex.map(fetch,JOBS):pass
