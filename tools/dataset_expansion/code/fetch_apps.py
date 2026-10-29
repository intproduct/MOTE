import json,time,hashlib
from fetch_sources import DEST,RAW,ROOT,get,download

REPO='codeparrot/apps';REV='21e74ddf8de1a21436da12e3e653065c5213e9d1'
def main():
 listing=get(f'https://huggingface.co/api/datasets/{REPO}/tree/{REV}?limit=100').json()
 info=next(x for x in listing if x['path']=='train.jsonl')
 while not (DEST/'audit/kodcode_source_manifest.json').exists():time.sleep(5)
 result=download({**info,'_repository':REPO,'_revision':REV,'_prefix':'apps'})
 questions=RAW/'questions/apps_train-00000-of-00001.jsonl'
 with questions.open('w',encoding='utf-8') as f:
  for i,line in enumerate((RAW/'apps/train.jsonl').open(encoding='utf-8')):
   r=json.loads(line);f.write(json.dumps({'question':r['question'],'problem_id':r.get('problem_id'),'row_index':i},ensure_ascii=False)+'\n')
 result.update({'projected_question_rows':i+1,'projection_sha256':hashlib.file_digest(questions.open('rb'),'sha256').hexdigest()})
 (DEST/'audit/apps_source_manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
