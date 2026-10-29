import collections,json
from fetch_sources import DEST
from example_evidence import analyse

def main():
 source=DEST/'candidates.jsonl';tmp=DEST/'candidates.examples.jsonl';reasons=collections.Counter();sources=collections.Counter();platforms=collections.Counter();licenses=collections.Counter();lengths=[];rejected=0;checked=0;matched=0
 with source.open(encoding='utf-8') as inp,tmp.open('w',encoding='utf-8',newline='\n') as good,(DEST/'review.jsonl').open('a',encoding='utf-8',newline='\n') as bad,(DEST/'audit/prompt_example_issues.jsonl').open('w',encoding='utf-8',newline='\n') as audit:
  for line in inp:
   r=json.loads(line)
   if r['source']=='KodCode/KodCode-V1-SFT-R1':
    evidence=analyse(r['input'],r['unit_tests'],r['solution'],r['metadata'].get('test_info'));r['verification']['local_prompt_example_evidence']=evidence;checked+=bool(evidence['doctest_examples']);matched+=evidence['matched_test_oracles']
    if evidence['flags']:
     r['review_flags'].extend(evidence['flags']);reasons.update(evidence['flags']);r['release_status']='review';bad.write(json.dumps(r,ensure_ascii=False)+'\n');audit.write(json.dumps({'id':r['id'],'evidence':evidence},ensure_ascii=False)+'\n');rejected+=1;continue
   good.write(json.dumps(r,ensure_ascii=False)+'\n');sources[r['source']]+=1;platforms[r['metadata'].get('platform',r['metadata'].get('subset','unknown'))]+=1;licenses[str(r['license'])]+=1;lengths.append(len(r['output']))
 tmp.replace(source);p=DEST/'audit/processing_summary.json';s=json.loads(p.read_text(encoding='utf-8'));s['counts']['candidates']-=rejected;s['counts']['review']+=rejected
 for key,n in reasons.items():s['filter_reasons'][key]=s['filter_reasons'].get(key,0)+n
 s['candidate_unique_questions']=len(lengths);s['candidate_unique_upstream_problem_families']=len(lengths);s['candidate_sources']=dict(sources);s['candidate_platforms_or_subsets']=dict(platforms);s['candidate_license_groups']=dict(licenses);lengths.sort();s['output_character_percentiles']={str(p):lengths[int((len(lengths)-1)*p)] for p in [.1,.5,.9,.99]};s['prompt_example_refinement']={'newly_held_for_review':rejected,'reasons':dict(reasons),'records_with_doctests_checked':checked,'example_oracles_compared':matched,'no_code_executed':True};p.write_text(json.dumps(s,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps(s),flush=True)
if __name__=='__main__':main()
