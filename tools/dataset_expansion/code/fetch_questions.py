"""Read only question columns from pinned parquet files via byte-range requests."""
import concurrent.futures, hashlib, io, json, os, time
from pathlib import Path
import requests
import pyarrow.parquet as pq
from fetch_sources import ROOT, DEST, get

SOURCES={
 'code_contests':('deepmind/code_contests','802411c3010cb00d1b05bad57ca77365a3c699d6','data',['description','name']),
 'taco':('BAAI/TACO','d593ed0a2becbbc952230bb89be09189bf1056dc','ALL',['question','url']),
 # Last complete pre-release revision; later versions may reorder indexed questions.
 'open-r1/codeforces':('open-r1/codeforces','8fd706e298b713125ac8df18cd5d847440fe3fb9','data',['description','input_format','output_format','examples','note','id']),
}

class RangeReader(io.RawIOBase):
 def __init__(self,url,size,cache):
  self.url=url;self.size=size;self.pos=0;self.cache=cache;self.cache.mkdir(parents=True,exist_ok=True);self.calls=[];self.session=requests.Session()
 def readable(self):return True
 def seekable(self):return True
 def tell(self):return self.pos
 def seek(self,offset,whence=0):
  self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
  return self.pos
 def read(self,n=-1):
  start=self.pos;end=self.size-1 if n<0 else min(self.size-1,start+n-1)
  if end<start:return b''
  p=self.cache/f'{start}-{end}.bin'
  if p.exists():b=p.read_bytes()
  else:
   for attempt in range(4):
    try:
     r=self.session.get(self.url,headers={'Range':f'bytes={start}-{end}'},timeout=(20,50));r.raise_for_status();b=r.content
     assert r.status_code==206 and len(b)==end-start+1 and r.headers.get('Content-Range')==f'bytes {start}-{end}/{self.size}',(r.status_code,len(b),start,end)
     p.write_bytes(b);break
    except Exception:
     if attempt==3:raise
     time.sleep(1)
  self.pos=end+1;self.calls.append({'start':start,'end':end,'sha256':hashlib.sha256(b).hexdigest()});return b

def project(task):
 key,repo,rev,columns,item=task;label=key.replace('/','_')+'_'+Path(item['path']).name
 out=DEST/'raw/questions'/label.replace('.parquet','.jsonl');out.parent.mkdir(parents=True,exist_ok=True)
 meta=out.with_suffix('.manifest.json')
 if out.exists() and meta.exists():return json.loads(meta.read_text())
 url=f'https://huggingface.co/datasets/{repo}/resolve/{rev}/{item["path"]}'
 rr=RangeReader(url,item['size'],DEST/'raw/range_cache'/label)
 pf=pq.ParquetFile(rr);available=pf.schema_arrow.names
 selected=[x for x in columns if x in available]
 table=pf.read(columns=selected,use_threads=False)
 with out.open('w',encoding='utf-8') as f:
  for row in table.to_pylist():f.write(json.dumps(row,ensure_ascii=False)+'\n')
 result={'source_key':key,'repository':repo,'revision':rev,'upstream_path':item['path'],'upstream_bytes':item['size'],'upstream_declared_lfs_sha256':item.get('lfs',{}).get('oid'),'columns':selected,'rows':table.num_rows,'projection_path':str(out.relative_to(ROOT)),'projection_sha256':hashlib.file_digest(out.open('rb'),'sha256').hexdigest(),'downloaded_ranges':rr.calls,'full_file_hash_verified':False}
 meta.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({'questions':label,'rows':table.num_rows}),flush=True);return result

def main():
 tasks=[]
 for key,(repo,rev,folder,columns) in SOURCES.items():
  listing=get(f'https://huggingface.co/api/datasets/{repo}/tree/{rev}/{folder}?limit=100').json()
  (DEST/'audit'/('question_listing_'+key.replace('/','_')+'.json')).write_text(json.dumps(listing,indent=2),encoding='utf-8')
  for item in listing:
   if item['path'].endswith('.parquet') and Path(item['path']).name.startswith('train-'):tasks.append((key,repo,rev,columns,item))
 with concurrent.futures.ThreadPoolExecutor(2) as pool:results=list(pool.map(project,tasks))
 (DEST/'audit/question_sources_manifest.json').write_text(json.dumps(results,indent=2),encoding='utf-8')

if __name__=='__main__':main()
