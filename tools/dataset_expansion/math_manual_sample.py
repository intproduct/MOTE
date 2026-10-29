"""Small reproducible audit sample; diagnostic excerpts are never training exports."""
import collections,hashlib,heapq,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'Datasets/expansion_20261016/math'
def main():
 overlays={}
 with (BASE/'audit/symbolic_recheck.jsonl').open(encoding='utf-8') as f:
  for line in f:
   if line.strip():r=json.loads(line);overlays[r['id']]=r
 samples=collections.defaultdict(list)
 with (BASE/'candidates.jsonl').open(encoding='utf-8') as f:
  for line in f:
   r=json.loads(line);audit=overlays.get(r['id'],{});state='symbolic_false' if audit.get('match') is False else 'general';key=(r['source'],state);score=int(hashlib.sha256(('math-manual-v2:'+r['id']).encode()).hexdigest(),16);heap=samples[key]
   if len(heap)<6:heapq.heappush(heap,(-score,r,audit))
   elif score < -heap[0][0]:heapq.heapreplace(heap,(-score,r,audit))
 rows=[]
 for key,items in sorted(samples.items()):
  for _,r,audit in sorted(items,reverse=True):
   row={'id':r['id'],'source':key[0],'sample_group':key[1],'question':r['input'],'reference':r['expected_answer'],'prediction':r['verification'].get('local_extracted_answer'),'full_final':r['final'],'reasoning_tail_excerpt':r['output'][-3500:],'output_sha256':hashlib.sha256(r['output'].encode()).hexdigest(),'symbolic_audit':audit};rows.append(row)
 target=BASE/'audit/manual_sample.json';target.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({'rows':len(rows),'path':str(target)}))
if __name__=='__main__':main()
