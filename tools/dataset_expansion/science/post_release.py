"""Publish final semantic decisions after finalize.py completes; no rerunning regex on long CoT."""
import json,hashlib,collections,os,heapq
from process import ROOT,sha
K=['candidates','review','quarantine','tool_trajectories_review']
summary=json.loads((ROOT/'audit/final_summary.json').read_text(encoding='utf-8'))
osr_audit=json.loads((ROOT/'audit/osr2_semantic_findings.json').read_text(encoding='utf-8'))
osr={r['input_sha256']:r for r in osr_audit['findings']}
foundation=json.loads((ROOT/'audit/foundation_semantic_audit.json').read_text(encoding='utf-8'));known={r['id']:r for r in foundation['known_findings']}
manifest=json.loads((ROOT/'audit/release_manifest.json').read_text(encoding='utf-8'));changed=set();new_manifest=[];sample=collections.defaultdict(list)
for name in ['open_science_reasoning_2','sciq_train','scienceqa_text_train','openstax_physics_train']:
 folder=ROOT/'release'/name;folder.mkdir(exist_ok=True)
 src=ROOT/'normalized'/name if name=='openstax_physics_train' else folder
 source_summary=json.loads((src/'summary.json').read_text(encoding='utf-8'))
 streams={k:(folder/(k+'.finalizing.jsonl')).open('wb',buffering=16*1024*1024) for k in K};hashes={k:hashlib.sha256() for k in K};sizes=collections.Counter();counts=collections.Counter();domains=collections.defaultdict(collections.Counter);levels=collections.Counter();flags=collections.Counter();ids=set()
 for old in K:
  for line in (src/(old+'.jsonl')).open(encoding='utf-8',buffering=16*1024*1024):
   r=json.loads(line);assert r['id'] not in ids;ids.add(r['id']);bucket=old
   assert sha(r['output'])==r['provenance'][0].get('normalized_output_sha256',r['provenance'][0].get('raw_output_sha256'))
   if name=='open_science_reasoning_2':
    r['review_flags'].append('source_level_semantic_review_required');r['metadata']['source_level_release_decision']='review_only_after_fixed_seed_semantic_audit_found_material_errors';r['metadata']['conversion_to_main_revision_linkage_verified']=False
    if old=='candidates':bucket='review'
    f=osr.get(sha(r['input']))
    if f:
     assert sha(r['output'])==f['output_sha256'];f['normalized_record_id']=r['id'];f['provenance']=r['provenance'];r['verification']['manual_semantic_audit']={k:f[k] for k in ['sample_index','verdict','reason','primary_references','input_sha256','output_sha256']}
     if f['verdict']=='definite_material_error':bucket='quarantine';r['review_flags'].append('known_incorrect_problem_or_reasoning');r['verification']['source_reference_match_before_semantic_audit']=r['verification']['answer_pass'];r['verification']['answer_pass']=False
   if r['id'] in known:
    f=known[r['id']];assert sha(r['input'])==f['input_sha256'] and sha(r['output'])==f['output_sha256'];r['verification']['manual_semantic_audit']={k:f[k] for k in ['flag','reason','input_sha256','output_sha256']};r['review_flags'].append(f['flag'])
    if bucket=='candidates':bucket='review'
   if name=='scienceqa_text_train':r['provenance'][0].pop('raw_output_sha256',None);r['provenance'][0].pop('raw_output_hash_scope',None)
   r['review_flags']=sorted(set(r['review_flags']));r['release_status']={'candidates':'candidate','review':'review_required','quarantine':'quarantine','tool_trajectories_review':'review_required'}[bucket]
   if bucket=='candidates':assert not r['review_flags'] and r['verification']['answer_pass'] is True
   blob=(json.dumps(r,ensure_ascii=False)+'\n').encode('utf-8');streams[bucket].write(blob);hashes[bucket].update(blob);sizes[bucket]+=len(blob);counts[bucket]+=1;domains[bucket][r['domain']]+=1;levels[r['quality_level']]+=1;flags.update(r['review_flags'])
   if bucket=='candidates':
    item=(-int(r['problem_id'],16),r['id'],r);key=(name,r['domain'])
    if len(sample[key])<10:heapq.heappush(sample[key],item)
    elif item>sample[key][0]:heapq.heapreplace(sample[key],item)
 for f in streams.values():f.close()
 for k in K:
  p=folder/(k+'.jsonl');os.replace(folder/(k+'.finalizing.jsonl'),p);rel=str(p.relative_to(ROOT)).replace('\\','/');changed.add(rel);new_manifest.append({'path':rel,'rows':counts[k],'bytes':sizes[k],'sha256':hashes[k].hexdigest()})
 source_summary.update(counts=dict(counts),domain_counts={k:dict(v) for k,v in domains.items()},quality_levels=dict(levels),review_flags=dict(flags),records_after_source_exact_dedup=len(ids),final_gates_applied=True)
 if name=='open_science_reasoning_2':source_summary['notes'].append('Whole source review-only after20-record fixed-seed semantic audit:8 definite material defects,6 unresolved,6 not certified; not a global error-rate estimate.')
 (folder/'summary.json').write_text(json.dumps(source_summary,ensure_ascii=False,indent=2),encoding='utf-8');summary['sources'][name]=source_summary
 print(json.dumps({'published':name,'counts':dict(counts)}),flush=True)
(ROOT/'audit/osr2_semantic_findings.json').write_text(json.dumps(osr_audit,ensure_ascii=False,indent=2),encoding='utf-8')
manifest=[r for r in manifest if r['path'] not in changed]+new_manifest
qualities=collections.Counter()
for s in summary['sources'].values():qualities.update(s['quality_levels'])
summary['total_records']=sum(s['records_after_source_exact_dedup'] for s in summary['sources'].values());summary['quality_levels']=dict(qualities);summary['semantic_audits']={'osr2':'audit/osr2_semantic_findings.json','foundations':'audit/foundation_semantic_audit.json','textbook':'audit/manual_findings.json'}
summary['checks']['unique_ids_scope']='Initial six-source global scan plus per-source recheck and disjoint openstax_official prefix; central merger will recheck global IDs.'
(ROOT/'audit/final_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
(ROOT/'audit/release_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
with (ROOT/'audit/final_stratified_samples.jsonl').open('w',encoding='utf-8') as f:
 for items in sample.values():
  for _,__,r in sorted(items,reverse=True):f.write(json.dumps(r,ensure_ascii=False)+'\n')
(ROOT/'audit/all_downloads.json').write_text(json.dumps([json.loads(p.read_text(encoding='utf-8')) for p in sorted((ROOT/'audit').glob('*.download.json'))],ensure_ascii=False,indent=2),encoding='utf-8')
(ROOT/'audit/openstax_html_manifest.json').write_text(json.dumps([json.loads(p.read_text(encoding='utf-8')) for p in sorted((ROOT/'raw/openstax_html').glob('*.json'))],ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'ready':True,'records':summary['total_records'],'files':len(manifest),'quality_levels':dict(qualities)}))
