"""Apply a newly established source-level hold before the ongoing export starts.

The main merger has already normalized mathematics. Its authoritative SQLite
dispositions are read again at export. Original responses are never modified.
"""
import argparse,json,sqlite3,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'Datasets/processed/shared_sft_v02_20261016';POLICY=ROOT/'Datasets/expansion_20261016/source_admission_policy.json'
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--minimum-records',type=int,default=0);parser.add_argument('--audit-name',default='source_policy_application.json');args=parser.parse_args()
 start=time.monotonic();path=BASE/'audit/index.sqlite';policy=json.loads(POLICY.read_text(encoding='utf-8'))
 while True:
  db=sqlite3.connect(path,timeout=60)
  # Code is after both math input files. Seeing a committed code row proves
  # the complete mathematical sources have been normalized and committed.
  complete=(db.execute('SELECT COALESCE(MAX(idx),-1)+1 FROM records').fetchone()[0]>=args.minimum_records) if args.minimum_records else db.execute("SELECT 1 FROM records WHERE source='nvidia/OpenCodeReasoning-2' LIMIT 1").fetchone()
  if complete:
   assert db.execute('SELECT COUNT(*) FROM records WHERE final_split IS NOT NULL').fetchone()[0]==0,'Too late: export phase has already started'
   changes=[]
   for source,p in policy['sources'].items():
    cursor=db.execute("UPDATE records SET status=? WHERE source=? AND status='eligible'",(p['disposition'],source));changes.append({'source':source,'rows_moved_from_eligible':cursor.rowcount,'disposition':p['disposition'],'evidence':p['evidence']})
   db.commit();db.close();report={'applied_before_export':True,'response_content_modified':False,'changes':changes,'policy':str(POLICY.relative_to(ROOT)).replace('\\','/'),'elapsed_seconds':time.monotonic()-start};(BASE/'audit'/args.audit_name).write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True);return
  db.close()
  if time.monotonic()-start>7200:raise TimeoutError('Required source normalization did not finish in time')
  time.sleep(10)
if __name__=='__main__':main()
