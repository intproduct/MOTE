import json,hashlib,concurrent.futures,requests
from download import RAW,AUDIT,ROOT
JOBS=[('nvidia/HelpSteer2','train.jsonl.gz','helpsteer2_train.jsonl.gz'),('nvidia/HelpSteer3','feedback/train.jsonl.gz','helpsteer3_feedback_train.jsonl.gz'),('nvidia/HelpSteer3','edit_quality/train.jsonl.gz','helpsteer3_edit_quality_train.jsonl.gz')]
def fetch(job):
 repo,path,name=job;meta=json.loads((AUDIT/(repo.split('/')[-1]+'.metadata.json')).read_text());rev=meta['sha'];info=next(x for x in meta['siblings'] if x['rfilename']==path)
 dest=RAW/name;reportfile=AUDIT/(name+'.download.json')
 if reportfile.exists():return json.loads(reportfile.read_text())
 url=f'https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}?download=true'
 with requests.get(url,stream=True,timeout=(40,180))as r:
  r.raise_for_status()
  with dest.with_suffix('.part').open('wb') as f:
   for block in r.iter_content(1048576):f.write(block)
 part=dest.with_suffix('.part');digest=hashlib.sha256(part.read_bytes()).hexdigest();assert digest==info['lfs']['sha256'];assert part.stat().st_size==info['size'];part.replace(dest)
 report={'repo':repo,'revision':rev,'path':path,'url':url,'local_path':str(dest.relative_to(ROOT)).replace(chr(92),'/'),'bytes':dest.stat().st_size,'sha256':digest,'full_file_hash_verified':True,'selection':'full_train_split','source_split':'train'}
 reportfile.write_text(json.dumps(report,indent=2),encoding='utf-8');return report
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:
  for result in pool.map(fetch,JOBS):print(json.dumps(result),flush=True)
