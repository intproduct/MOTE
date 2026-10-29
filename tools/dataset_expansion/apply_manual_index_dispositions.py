"""Content-bound manual downgrades before the ongoing central export starts."""
import argparse,hashlib,json,sqlite3,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'Datasets/processed/shared_sft_v02_20261016'
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--findings',type=Path,required=True);p.add_argument('--minimum-records',type=int,default=0);a=p.parse_args();findings=json.loads(a.findings.read_text(encoding='utf-8'))['findings'];db=sqlite3.connect(BASE/'audit/index.sqlite',timeout=120);start=time.monotonic()
 while db.execute('SELECT COALESCE(MAX(idx),-1)+1 FROM records').fetchone()[0]<a.minimum_records:
  if time.monotonic()-start>7200:raise TimeoutError('Required records were not normalized')
  time.sleep(10)
 assert db.execute('SELECT COUNT(*) FROM records WHERE final_split IS NOT NULL').fetchone()[0]==0,'Export has already started';applied=[]
 with (BASE/'audit/normalized_spool.jsonl').open('rb') as f:
  for item in findings:
   if item['action'] not in ['quarantine','review']:continue
   row=db.execute('SELECT offset,length,status FROM records WHERE id=?',(item['id'],)).fetchone();assert row,('row_not_yet_normalized',item['id']);offset,length,status=row;f.seek(offset);r=json.loads(f.read(length));assert digest(r['input'])==item['input_sha256'] and digest(r['output'])==item['output_sha256'],('manual_hash_mismatch',item['id'])
   target='quarantine' if item['action']=='quarantine' else 'review_required'
   if status=='duplicate' or status=='quarantine' and target=='review_required':continue
   db.execute('UPDATE records SET status=? WHERE id=?',(target,item['id']));applied.append({**item,'previous_release_status':status,'applied_release_status':target})
 db.commit();db.close();out=BASE/'audit'/('applied_'+a.findings.stem+'.json');out.write_text(json.dumps({'before_export':True,'source':str(a.findings),'applied':applied,'record_content_modified':False},ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({'applied':len(applied),'audit':str(out)}))
if __name__=='__main__':main()
