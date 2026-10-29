"""Apply the test-evidence gate to already-created candidates, retaining rejects."""
import collections,hashlib,json
from fetch_sources import DEST
from test_evidence import analyse

def main():
 source=DEST/'candidates.jsonl';tmp=DEST/'candidates.refined.jsonl';counts=collections.Counter();domains=collections.Counter();platforms=collections.Counter();licenses=collections.Counter();lengths=[];rejected=0
 with source.open(encoding='utf-8') as fin,tmp.open('w',encoding='utf-8',newline='\n') as good,(DEST/'review.jsonl').open('a',encoding='utf-8',newline='\n') as bad:
  for line in fin:
   r=json.loads(line)
   if r['source']=='KodCode/KodCode-V1-SFT-R1':
    evidence=analyse(r['unit_tests'],r['solution'],r['metadata'].get('test_info'));r['verification']['local_test_static_evidence']=evidence
    counts['linked_checks_'+str(min(evidence['linked_value_checks'],10))]+=1
    if not evidence['eligible']:
     r['review_flags'].append('insufficient_statically_linked_test_checks');r['release_status']='review';bad.write(json.dumps(r,ensure_ascii=False)+'\n');rejected+=1;continue
   good.write(json.dumps(r,ensure_ascii=False)+'\n');domains[r['source']]+=1;platforms[r['metadata'].get('platform',r['metadata'].get('subset','unknown'))]+=1;licenses[str(r['license'])]+=1;lengths.append(len(r['output']))
 tmp.replace(source)
 p=DEST/'audit/processing_summary.json';s=json.loads(p.read_text(encoding='utf-8'));s['counts']['candidates']-=rejected;s['counts']['review']+=rejected;s['filter_reasons']['insufficient_statically_linked_test_checks']=s['filter_reasons'].get('insufficient_statically_linked_test_checks',0)+rejected
 s['candidate_unique_questions']=len(lengths);s['candidate_sources']=dict(domains);s['candidate_platforms_or_subsets']=dict(platforms);s['candidate_license_groups']=dict(licenses);lengths.sort();s['output_character_percentiles']={str(p):lengths[int((len(lengths)-1)*p)] for p in [.1,.5,.9,.99]}
 s['static_test_refinement']={'newly_held_for_review':rejected,'linked_check_histogram':dict(counts),'no_code_executed':True}
 p.write_text(json.dumps(s,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps(s),flush=True)
if __name__=='__main__':main()
