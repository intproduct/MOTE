import collections,hashlib,json
from fetch_sources import DEST

def main():
 reviews={x['id']:x for x in json.loads((DEST/'audit/manual_content_rejections.json').read_text(encoding='utf-8'))};found=[];source=DEST/'candidates.jsonl';tmp=DEST/'candidates.manual.jsonl';sources=collections.Counter();platforms=collections.Counter();licenses=collections.Counter();lengths=[]
 with source.open(encoding='utf-8') as inp,tmp.open('w',encoding='utf-8',newline='\n') as good,(DEST/'review.jsonl').open('a',encoding='utf-8',newline='\n') as bad:
  for line in inp:
   r=json.loads(line)
   if r['id'] in reviews:
    review=reviews[r['id']];assert hashlib.sha256(r['input'].encode()).hexdigest()==review['input_sha256'];r['review_flags'].append(review['reason']);r['verification']['manual_content_review']=review;r['release_status']='review';bad.write(json.dumps(r,ensure_ascii=False)+'\n');found.append(r['id']);continue
   good.write(line);sources[r['source']]+=1;platforms[r['metadata'].get('platform',r['metadata'].get('subset','unknown'))]+=1;licenses[str(r['license'])]+=1;lengths.append(len(r['output']))
 tmp.replace(source);p=DEST/'audit/processing_summary.json';s=json.loads(p.read_text(encoding='utf-8'));s['counts']['candidates']-=len(found);s['counts']['review']+=len(found)
 for rid in found:
  key=reviews[rid]['reason'];s['filter_reasons'][key]=s['filter_reasons'].get(key,0)+1
 s['candidate_unique_questions']=len(lengths);s['candidate_unique_upstream_problem_families']=len(lengths);s['candidate_sources']=dict(sources);s['candidate_platforms_or_subsets']=dict(platforms);s['candidate_license_groups']=dict(licenses);lengths.sort();s['output_character_percentiles']={str(p):lengths[int((len(lengths)-1)*p)] for p in [.1,.5,.9,.99]};s['manual_content_rejections_applied']=found;p.write_text(json.dumps(s,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps(s),flush=True)
if __name__=='__main__':main()
