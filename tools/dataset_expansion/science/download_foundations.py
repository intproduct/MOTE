import json,hashlib
from pathlib import Path
from download import ROOT,go
import requests
repo='allenai/sciq';rev='2c94ad3e1aafab77146f384e23536f97a4849815'
d=requests.get(f'https://huggingface.co/api/datasets/{repo}/revision/{rev}',timeout=60);d.raise_for_status();(ROOT/'audit/allenai__sciq.json').write_text(d.text,encoding='utf-8')
c=requests.get(f'https://huggingface.co/datasets/{repo}/raw/{rev}/README.md',timeout=60);c.raise_for_status();(ROOT/'audit/allenai__sciq_README.md').write_text(c.text,encoding='utf-8')
go((repo,rev,'data/train-00000-of-00001.parquet','sciq_train.parquet','cc-by-nc-3.0'))
repo='lupantech/ScienceQA';d=requests.get(f'https://api.github.com/repos/{repo}/commits/main',timeout=60);d.raise_for_status();meta=d.json();rev=meta['sha'];(ROOT/'audit/ScienceQA_commit.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
for path,name in [('data/scienceqa/problems.json','scienceqa_problems.json'),('data/scienceqa/pid_splits.json','scienceqa_pid_splits.json'),('LICENSE-DATA','scienceqa_LICENSE-DATA.txt'),('LICENSE','scienceqa_LICENSE.txt'),('README.md','scienceqa_README.md')]:
 url=f'https://raw.githubusercontent.com/{repo}/{rev}/{path}';r=requests.get(url,timeout=120);r.raise_for_status();dest=ROOT/('raw' if path.endswith('.json') else 'audit')/name;dest.write_bytes(r.content)
 item={'repo':repo,'revision':rev,'path':path,'url':url,'license':'mit' if path=='LICENSE' else 'cc-by-nc-sa-4.0','source_split':'train selected by pid_splits.json; source problems file contains all original splits','local_path':str(dest.relative_to(ROOT)),'bytes':len(r.content),'sha256':hashlib.sha256(r.content).hexdigest(),'download_completed':True}
 (ROOT/'audit'/(name+'.download.json')).write_text(json.dumps(item,indent=2),encoding='utf-8');print(json.dumps(item),flush=True)
