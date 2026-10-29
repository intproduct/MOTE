"""Bounded subprocess audit of final-answer equivalence, never proof of reasoning."""
import argparse,collections,hashlib,json,multiprocessing as mp,sys,time,logging
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'Datasets/expansion_20261016/math';DEPS=ROOT/'Datasets/expansion_20261016/cache/math_verify_deps'

def unwrap(s):
 s=s.strip()
 for left,right in [('\\(','\\)'),('\\[','\\]'),('$$','$$'),('$','$')]:
  if s.startswith(left) and s.endswith(right):s=s[len(left):-len(right)].strip()
 return s

def worker(conn):
 sys.path.insert(0,str(DEPS));logging.disable(logging.CRITICAL)
 from math_verify import parse,verify,LatexExtractionConfig
 conn.send({'ready':True})
 while True:
  item=conn.recv()
  if item is None:break
  rid,gold,pred,binding=item
  result={'id':rid,'output_sha256':binding,'method':'math_verify_0.9.0_strict_answer_equivalence','process_verified':False}
  try:
   config=[LatexExtractionConfig()]
   g=parse('$'+unwrap(gold)+'$',extraction_config=config,fallback_mode='no_fallback',extraction_mode='first_match',parsing_timeout=None,raise_on_error=True)
   a=parse('$'+unwrap(pred)+'$',extraction_config=config,fallback_mode='no_fallback',extraction_mode='first_match',parsing_timeout=None,raise_on_error=True)
   result.update({'reference_parsed':bool(g),'prediction_parsed':bool(a),'reference_expression':str(g)[:500],'prediction_expression':str(a)[:500]})
   if len(g)!=1 or len(a)!=1:result.update(status='unparsed_or_ambiguous',match=None)
   else:result.update(status='compared',match=bool(verify(g,a,strict=True,timeout_seconds=None,raise_on_error=True)))
  except Exception as e:result.update(status='parser_or_comparison_error',match=None,error=type(e).__name__+': '+str(e)[:300])
  conn.send(result)

def main():
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);p.add_argument('--limit',type=int,default=0);p.add_argument('--seconds',type=float,default=5);p.add_argument('--output',default='symbolic_recheck.jsonl');a=p.parse_args()
 output=BASE/'audit'/a.output;done=set()
 if output.exists():done={json.loads(x)['id'] for x in output.open(encoding='utf-8')}
 jobs=[];skipped=collections.Counter()
 for line in (BASE/'candidates.jsonl').open(encoding='utf-8'):
  r=json.loads(line);v=r['verification'];pred=v.get('local_extracted_answer');gold=r.get('expected_answer')
  if r['id'] in done:continue
  if v.get('local_answer_pass') is True:skipped['already_exact_or_rational_match']+=1;continue
  if not isinstance(gold,str) or not pred:skipped['missing_extracted_answer']+=1;continue
  if max(len(gold),len(pred))>2000:skipped['expression_above_audit_size_budget']+=1;continue
  jobs.append((r['id'],gold,pred,hashlib.sha256(r['output'].encode()).hexdigest()))
  if a.limit and len(jobs)>=a.limit:break
 slots=[];cursor=0;stats=collections.Counter();start=time.monotonic();last=start
 def start_slot():
  parent,child=mp.Pipe();proc=mp.Process(target=worker,args=(child,));proc.start();child.close();return {'pipe':parent,'proc':proc,'ready':False,'job':None,'started':time.monotonic()}
 for _ in range(min(a.workers,len(jobs))):slots.append(start_slot())
 with output.open('a',encoding='utf-8') as f:
  while slots:
   for s in list(slots):
    result=None
    if s['pipe'].poll():
     try:msg=s['pipe'].recv()
     except EOFError:msg={'crashed':True}
     if msg.get('ready'):s['ready']=True
     elif msg.get('crashed'):
      if s['job']:result={'id':s['job'][0],'output_sha256':s['job'][3],'status':'worker_crash','match':None,'process_verified':False}
      s['proc'].join(timeout=1);s['pipe'].close();slots.remove(s);slots.append(start_slot())
     else:result=msg;s['job']=None
    if s not in slots:
     if result:f.write(json.dumps(result)+'\n');stats[result['status']]+=1
     continue
    if s['job'] and time.monotonic()-s['started']>a.seconds:
     result={'id':s['job'][0],'output_sha256':s['job'][3],'status':'timeout','match':None,'seconds':a.seconds,'process_verified':False};s['proc'].terminate();s['proc'].join(timeout=2);s['pipe'].close();slots.remove(s);slots.append(start_slot())
    if result:
     f.write(json.dumps(result)+'\n');stats[result['status']]+=1;stats['match_'+str(result['match'])]+=1
    if s in slots and s['ready'] and s['job'] is None:
     if cursor<len(jobs):s['job']=jobs[cursor];cursor+=1;s['started']=time.monotonic();s['pipe'].send(s['job'])
     else:s['pipe'].send(None);s['proc'].join(timeout=2);s['pipe'].close();slots.remove(s)
    if s in slots and not s['ready'] and time.monotonic()-s['started']>45:raise RuntimeError('Worker startup failed')
   if time.monotonic()-last>30:
    f.flush();print(json.dumps({'assigned':cursor,'jobs':len(jobs),'stats':dict(stats),'elapsed_seconds':round(time.monotonic()-start,1)}),flush=True);last=time.monotonic()
   time.sleep(.005)
 report={'new_jobs':len(jobs),'already_completed':len(done),'stats':dict(stats),'skipped':dict(skipped),'elapsed_seconds':round(time.monotonic()-start,2),'strict':True,'per_record_timeout_seconds':a.seconds,'meaning':'Mathematical equivalence of extracted final answer to supplied reference; not independent solution of question or verification of CoT.'};output.with_suffix('.summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))

if __name__=='__main__':main()
