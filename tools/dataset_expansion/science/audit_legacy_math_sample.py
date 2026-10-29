import json, hashlib, heapq
from pathlib import Path

ROOT=Path(r'D:\files\pydocs\MOTE')
SRC=ROOT/'Datasets/processed/local_sft_v01_20261016/jsonl/train.jsonl'
OUT=ROOT/'Datasets/expansion_20261016/math/audit'
OUT.mkdir(parents=True,exist_ok=True)
seed='legacy-math-diagnostic-20261016-v1'
h=hashlib.sha256(); selected=[]; count=0; total=0
with SRC.open('rb', buffering=16*1024*1024) as f:
    for line in f:
        h.update(line); total+=1
        r=json.loads(line)
        if r.get('domain')!='math': continue
        count+=1
        key=hashlib.sha256((seed+'\0'+r['id']).encode()).hexdigest()
        selected.append((key,r))
        selected=sorted(selected,key=lambda x:x[0])[:10]
with (OUT/'legacy_math_sample.jsonl').open('w',encoding='utf-8',newline='\n') as f:
    for rank,(key,r) in enumerate(selected,1):
        f.write(json.dumps({'sample_rank':rank,'selection_sha256':key,'record':r},ensure_ascii=False)+'\n')
meta={'source_path':str(SRC),'source_sha256':h.hexdigest(),'source_rows':total,'math_rows':count,'sample_count':len(selected),'sampling':'lowest 10 SHA256(seed + NUL + original id) among domain==math','seed':seed,'fields':'input/output/final use standard names; complete original records retained'}
(OUT/'legacy_math_sample_manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(meta))
for i,(_,r) in enumerate(selected,1):
    print(json.dumps({'rank':i,'id':r['id'],'input':r['input'],'final':r.get('final'),'expected_answer':r.get('expected_answer'),'verification':r.get('verification')},ensure_ascii=True))
