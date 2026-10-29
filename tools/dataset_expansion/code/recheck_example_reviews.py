"""Re-adjudicate example-only review items with the current static checker."""
import collections,json
from fetch_sources import DEST
from example_evidence import analyse

def main():
 issue_path=DEST/'audit/prompt_example_issues.jsonl';issues={json.loads(l)['id'] for l in issue_path.open(encoding='utf-8')};old_counts=collections.Counter();new_counts=collections.Counter();promoted=0;matched_delta=0
 source=DEST/'review.jsonl';tmp=DEST/'review.examples_rechecked.jsonl';audit_tmp=issue_path.with_suffix('.new.jsonl')
 with source.open(encoding='utf-8') as inp,tmp.open('w',encoding='utf-8',newline='\n') as bad,(DEST/'candidates.jsonl').open('a',encoding='utf-8',newline='\n') as good,audit_tmp.open('w',encoding='utf-8',newline='\n') as audit:
  for line in inp:
   r=json.loads(line)
   if r['id'] in issues:
    old=r['verification']['local_prompt_example_evidence'];old_counts.update(old['flags']);new=analyse(r['input'],r['unit_tests'],r['solution'],r['metadata'].get('test_info'));r['verification']['local_prompt_example_evidence']=new;new_counts.update(new['flags']);matched_delta+=new['matched_test_oracles']-old['matched_test_oracles']
    r['review_flags']=[f for f in r['review_flags'] if f not in old['flags']]+new['flags']
    if not r['review_flags']:r['release_status']='candidates';good.write(json.dumps(r,ensure_ascii=False)+'\n');promoted+=1;continue
    audit.write(json.dumps({'id':r['id'],'evidence':new},ensure_ascii=False)+'\n');bad.write(json.dumps(r,ensure_ascii=False)+'\n')
   else:bad.write(line)
 tmp.replace(source);audit_tmp.replace(issue_path)
 sources=collections.Counter();platforms=collections.Counter();licenses=collections.Counter();lengths=[]
 for line in (DEST/'candidates.jsonl').open(encoding='utf-8'):
  r=json.loads(line);sources[r['source']]+=1;platforms[r['metadata'].get('platform',r['metadata'].get('subset','unknown'))]+=1;licenses[str(r['license'])]+=1;lengths.append(len(r['output']))
 p=DEST/'audit/processing_summary.json';s=json.loads(p.read_text(encoding='utf-8'));s['counts']['candidates']+=promoted;s['counts']['review']-=promoted
 for key in old_counts.keys()|new_counts.keys():s['filter_reasons'][key]=s['filter_reasons'].get(key,0)-old_counts[key]+new_counts[key]
 s['candidate_unique_questions']=len(lengths);s['candidate_unique_upstream_problem_families']=len(lengths);s['candidate_sources']=dict(sources);s['candidate_platforms_or_subsets']=dict(platforms);s['candidate_license_groups']=dict(licenses);lengths.sort();s['output_character_percentiles']={str(p):lengths[int((len(lengths)-1)*p)] for p in [.1,.5,.9,.99]};s['prompt_example_refinement']['newly_held_for_review']-=promoted;s['prompt_example_refinement']['reasons']=dict(new_counts);s['prompt_example_refinement']['example_oracles_compared']+=matched_delta;s['prompt_example_refinement']['conservative_recheck_restored']=promoted;p.write_text(json.dumps(s,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps(s),flush=True)
if __name__=='__main__':main()
