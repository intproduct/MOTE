"""Do not count instruct/complete renderings of one KodCode problem as new problems."""
import collections,hashlib,json,re
from fetch_sources import DEST

def main():
 source=DEST/'candidates.jsonl';tmp=DEST/'candidates.families.jsonl';seen=set();moved=0;sources=collections.Counter();platforms=collections.Counter();licenses=collections.Counter();lengths=[]
 with source.open(encoding='utf-8') as inp,tmp.open('w',encoding='utf-8',newline='\n') as good,(DEST/'duplicate_alternatives.jsonl').open('a',encoding='utf-8',newline='\n') as dup:
  for line in inp:
   r=json.loads(line);qid=r['metadata']['original_question_id']
   family=re.sub(r'_(?:I|C|OJ)$','',qid) if r['source']=='KodCode/KodCode-V1-SFT-R1' else qid
   key=r['source']+'::'+family;r['family_id']=hashlib.sha256(key.encode()).hexdigest();r['metadata']['source_question_family_id']=key
   if key in seen:
    r['release_status']='duplicate_alternatives';r['metadata']['dedup_reason']='same upstream problem across instruct/complete rendering';dup.write(json.dumps(r,ensure_ascii=False)+'\n');moved+=1;continue
   seen.add(key);good.write(json.dumps(r,ensure_ascii=False)+'\n');sources[r['source']]+=1;platforms[r['metadata'].get('platform',r['metadata'].get('subset','unknown'))]+=1;licenses[str(r['license'])]+=1;lengths.append(len(r['output']))
 tmp.replace(source);p=DEST/'audit/processing_summary.json';s=json.loads(p.read_text(encoding='utf-8'));s['counts']['candidates']-=moved;s['counts']['duplicate_alternatives']+=moved;s['candidate_unique_questions']=len(lengths);s['candidate_unique_upstream_problem_families']=len(seen);s['candidate_sources']=dict(sources);s['candidate_platforms_or_subsets']=dict(platforms);s['candidate_license_groups']=dict(licenses);lengths.sort();s['output_character_percentiles']={str(p):lengths[int((len(lengths)-1)*p)] for p in [.1,.5,.9,.99]};s['family_dedup_additional_moved']=moved;p.write_text(json.dumps(s,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps(s),flush=True)
if __name__=='__main__':main()
