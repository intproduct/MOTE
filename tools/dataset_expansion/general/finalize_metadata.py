"""Add non-destructive context-family aids before checksums/READY are issued."""
import json
from download import BASE
from convert import norm,sha
def main():
 for filename in ['candidates.jsonl','review.jsonl','quarantine.jsonl','proofwriter_candidates.jsonl','helpsteer_candidates.jsonl','helpsteer_review.jsonl','helpsteer_quarantine.jsonl']:
  path=BASE/filename;tmp=path.with_suffix('.metadata.tmp');count=0
  with path.open(encoding='utf-8')as src,tmp.open('w',encoding='utf-8',newline='\n')as out:
   for line in src:
    row=json.loads(line);first=next((m.get('content')for m in row['messages']if m.get('role')=='user'),None)
    if isinstance(first,str):
     row['metadata']['first_user_sha256']=sha(norm(first));row['metadata']['first_user_normalized_characters']=len(norm(first))
    raw=row['metadata'].get('source_metadata',{})
    for field in ['conversation_id','conversation_uuid','chat_id','thread_id']:
     if field in raw:row['metadata']['source_conversation_id']={'field':field,'value':raw[field]};break
    out.write(json.dumps(row,ensure_ascii=False)+'\n');count+=1
  tmp.replace(path);print(json.dumps({'finalized':filename,'rows':count}),flush=True)
if __name__=='__main__':main()
