"""Stream official benchmark JSONL; retain only questions, not gigabytes of tests."""
import concurrent.futures,hashlib,json,time
from pathlib import Path
import requests
from benchmark_inventory import ROOT,BASE

def project(item):
 repo,revision,meta=item;folder=BASE/'projections';folder.mkdir(exist_ok=True);dest=folder/('livecodebench_'+meta['path']);sidecar=dest.with_suffix('.manifest.json')
 if dest.exists() and sidecar.exists():return json.loads(sidecar.read_text(encoding='utf-8'))
 url=f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{meta['path']}";partial=dest.with_suffix('.partial')
 for attempt in range(3):
  try:
   h=hashlib.sha256();size=0;buf=b'';rows=0
   with requests.get(url,stream=True,timeout=(30,120)) as response,partial.open('w',encoding='utf-8') as f:
    response.raise_for_status()
    def process(line):
     nonlocal rows
     if not line.strip():return
     r=json.loads(line);prompt=r.get('question_content')
     if not isinstance(prompt,str) or not prompt.strip():raise ValueError(('question_content_missing',list(r)))
     f.write(json.dumps({'benchmark':repo,'revision':revision,'source_file':meta['path'],'row':rows,'prompt':prompt,'source_id':r.get('question_id'),'platform':r.get('platform'),'contest_date':r.get('contest_date')},ensure_ascii=False)+'\n');rows+=1
    for block in response.iter_content(1024*1024):
     h.update(block);size+=len(block);buf+=block
     lines=buf.split(b'\n');buf=lines.pop()
     for line in lines:process(line)
    if buf:process(buf)
   assert size==meta['size'];assert not meta.get('lfs') or h.hexdigest()==meta['lfs']['oid']
   partial.replace(dest);result={'repo':repo,'revision':revision,'source_file':meta['path'],'projection_path':str(dest.relative_to(ROOT)),'source_bytes_streamed':size,'source_sha256':h.hexdigest(),'upstream_lfs_sha256_match':bool(meta.get('lfs')),'projection_sha256':hashlib.file_digest(dest.open('rb'),'sha256').hexdigest(),'rows':rows,'full_source_retained':False,'purpose':'benchmark exclusion only'};sidecar.write_text(json.dumps(result,indent=2),encoding='utf-8');return result
  except Exception as e:
   print(json.dumps({'retry':attempt,'file':meta['path'],'error':str(e)}),flush=True)
   if attempt==2:raise
   time.sleep(3)

if __name__=='__main__':
 repo='livecodebench/code_generation_lite';inv=json.loads((BASE/(repo.replace('/','__')+'.json')).read_text(encoding='utf-8'));items=[(repo,inv['revision'],m) for m in inv['files'] if m['path'].endswith('.jsonl')];results=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  for f in concurrent.futures.as_completed([pool.submit(project,x) for x in items]):
   r=f.result();results.append(r);print(json.dumps(r),flush=True)
 (BASE/'livecodebench_manifest.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
