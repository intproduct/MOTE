"""OWA ProofWriter: independent finite Horn closure, never execute dataset code."""
import json,re,hashlib,zipfile,collections,itertools,random
from download import ROOT,BASE,RAW,AUDIT
VAR={'something','someone'}
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def parse(s):
 tokens=re.findall(r'"(?:\\.|[^"\\])*"|\(|\)|->',s);pos=0
 def consume():
  nonlocal pos
  if pos>=len(tokens):raise ValueError('unexpected_eof')
  token=tokens[pos];pos+=1
  if token=='(':
   out=[]
   while pos<len(tokens) and tokens[pos]!=')':out.append(consume())
   if pos==len(tokens):raise ValueError('unclosed_parenthesis')
   pos+=1;return out
  if token.startswith('"'):return json.loads(token)
  if token=='->':return token
  raise ValueError('unexpected_token')
 result=consume()
 if pos!=len(tokens):raise ValueError('trailing_tokens')
 # No ignored text, including invented code or unsupported operators.
 remainder=re.sub(r'"(?:\\.|[^"\\])*"|\(|\)|->|\s+','',s)
 if remainder:raise ValueError('unsupported_representation')
 return result
def atom(v):
 if not isinstance(v,list) or len(v)!=4 or not all(isinstance(x,str) for x in v):raise ValueError('atom_shape')
 if v[3] not in ['+','-','~']:raise ValueError('atom_polarity')
 return tuple(v[:3]+['-' if v[3]=='~' else v[3]])
def negate(a):return a[:3]+('-' if a[3]=='+' else '+',)
def formal(theory):
 facts={atom(parse(x['representation'])) for x in theory['triples'].values()};rules=[]
 for x in theory['rules'].values():
  v=parse(x['representation'])
  if len(v)!=3 or v[1]!='->' or not isinstance(v[0],list) or not v[0]:raise ValueError('rule_shape')
  rules.append((tuple(atom(a) for a in v[0]),atom(v[2])))
 return facts,rules
def family(theory):
 facts,rules=formal(theory)
 return sha(json.dumps({'facts':sorted(facts),'rules':sorted((sorted(pre),post)for pre,post in rules)},sort_keys=True))
def close(facts,rules):
 entities={a[0] for a in facts}
 for a in facts:
  if a[1]!='is':entities.add(a[2])
 for pre,post in rules:
  for a in (*pre,post):
   entities.add(a[0])
   if a[1]!='is':entities.add(a[2])
 entities-=VAR;ground=[]
 for pre,post in rules:
  variables=sorted({v for a in (*pre,post) for v in (a[0],a[2]) if v in VAR})
  for values in itertools.product(sorted(entities),repeat=len(variables)):
   sub=dict(zip(variables,values))
   def replace(a):return (sub.get(a[0],a[0]),a[1],sub.get(a[2],a[2]),a[3])
   ground.append((set(replace(a) for a in pre),replace(post)))
 known=set(facts);depth={a:0 for a in facts};step=0
 while True:
  step+=1;new={}
  for pre,post in ground:
   if post not in known and pre<=known:new[post]=min(new.get(post,10000),max(depth[a] for a in pre)+1)
  if not new:break
  known.update(new);depth.update(new)
  if step>1000:raise ValueError('closure_nontermination')
 return known,depth
def truth(query,closure):
 if query in closure and negate(query) in closure:raise ValueError('inconsistent_query')
 return 'True' if query in closure else 'False' if negate(query) in closure else 'Unknown'
def main():
 meta=json.loads((AUDIT/'proofwriter.download.json').read_text());stats=collections.Counter();errors=[];selected=[];used=set();heldout=set();evaluated=[];counts=collections.Counter();rng=random.Random(20261016)
 with zipfile.ZipFile(RAW/'proofwriter-dataset-V2020.12.3.zip') as z:
  # All official OWA development/test theories are exclusion-only, never SFT targets.
  for name in z.namelist():
   if '/OWA/' in name and name.endswith(('meta-dev.jsonl','meta-test.jsonl')):
    with z.open(name) as f:
     for line in f:
      r=json.loads(line)
      try:heldout.add(family(r))
      except Exception:stats['heldout_parse_failures']+=1
  for d in [0,1,2,3,5]:
   name=f'proofwriter-dataset-V2020.12.3/OWA/depth-{d}/meta-train.jsonl'
   with z.open(name) as f:theories=[(i,json.loads(l))for i,l in enumerate(f,1)]
   rng.shuffle(theories)
   for index,r in theories:
    if all(counts[(d,l)]>=2000 for l in ['True','False','Unknown']):break
    stats['theories_examined']+=1
    try:
     fh=family(r)
     if fh in heldout:stats['heldout_theory_excluded']+=1;continue
     if fh in used:stats['duplicate_theory_skipped']+=1;continue
     facts,rules=formal(r);closure,depth=close(facts,rules)
     if any(negate(a) in closure for a in closure):raise ValueError('inconsistent_theory')
     details={atom(parse(x['representation'])) for x in r.get('proofDetails',[])}
     if details and details!=closure:raise ValueError('independent_closure_differs_from_published_proof_details')
     qrows=[]
     for qid,q in r['questions'].items():
      query=atom(parse(q['representation']));computed=truth(query,closure);gold='True' if q['answer'] is True else 'False' if q['answer'] is False else q['answer']
      if computed!=gold:raise ValueError('gold_mismatch:'+qid+':'+str(gold)+':'+computed)
      qrows.append((qid,q,computed,query))
     stats['theories_closure_verified']+=1;stats['questions_gold_verified']+=len(qrows)
    except Exception as e:
     stats['theories_rejected']+=1;errors.append({'source_path':name,'row':index,'theory_id':r.get('id'),'error':str(e)});continue
    rng.shuffle(qrows);taken=set()
    for qid,q,label,query in qrows:
     if label in taken or counts[(d,label)]>=2000:continue
     taken.add(label);counts[(d,label)]+=1
     prompt='Use only the following facts and rules. Apply rules forward; do not use outside knowledge or assume that an unproved statement is false. Answer True if the statement follows, False if its explicit negation follows, and Unknown if neither follows. Return only True, False, or Unknown.\n\nFacts and rules:\n'+r['theory']+'\n\nStatement: '+q['question']
     ph=sha(prompt);mid='proofwriter-'+sha(name+'|'+r['id']+'|'+qid)[:24]
     messages=[{'role':'user','content':prompt},{'role':'assistant','content':label}]
     row={'id':mid,'problem_id':ph,'family_id':'proofwriter-theory-'+fh,'input':prompt,'output':label,'final':label,'solution':None,'auxiliary_reasoning':None,'messages':messages,'domain':'logic','source':'AI2/ProofWriter','source_revision':'V2020.12.3;sha256:'+meta['sha256'],'source_split':'train','license':None,'split':'train','quality_level':'independently_recomputed','release_status':'candidate','verification':{'method':'independent_finite_owa_horn_closure','answer_pass':True,'published_gold_match':True,'published_closure_match':True,'full_closure_computed':True,'closure_size':len(closure),'reasoning_verified':None,'generated_cot_present':False},'expected_answer':label,'review_flags':[],'provenance':[{'url':meta['url'],'version':'V2020.12.3','zip_sha256':meta['sha256'],'path':name,'row_1based':index,'source_record_id':r['id'],'source_question_id':qid,'source_split':'train'}],'metadata':{'group_key':'proofwriter-theory-'+fh,'requires_messages':False,'training_target':'last_assistant_message','source_theory_id':r['id'],'source_question_id':qid,'source_max_depth':d,'source_question_depth':q.get('QDep'),'proof_depth_recomputed':depth.get(query,depth.get(negate(query))),'theory_family':r['id'].split('-')[0],'formal_facts':sorted(facts),'formal_rules':[{'premises':list(pre),'conclusion':post}for pre,post in rules],'formal_query':query,'closure_sha256':sha(json.dumps(sorted(closure))),'source_theory_text':r['theory'],'source_question_text':q['question'],'license_status':'not_declared_in_official_release','not_distilled_cot':True,'response_truncated':False,'official_dev_test_theory_overlap':False}}
     selected.append(row)
    used.add(fh)
   print(json.dumps({'depth':d,'selected':{l:counts[(d,l)]for l in ['True','False','Unknown']},'stats':stats}),flush=True)
 out=BASE/'proofwriter_candidates.jsonl'
 with out.open('w',encoding='utf-8',newline='\n') as f:
  for row in selected:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 with (AUDIT/'proofwriter_rejected_theories.jsonl').open('w',encoding='utf-8') as f:
  for row in errors:f.write(json.dumps(row)+'\n')
 with (AUDIT/'proofwriter_content_sample.jsonl').open('w',encoding='utf-8') as f:
  for label in ['True','False','Unknown']:
   pool=[r for r in selected if r['final']==label]
   for row in rng.sample(pool,min(8,len(pool))):f.write(json.dumps(row,ensure_ascii=False)+'\n')
 report={'rows':len(selected),'labels':dict(collections.Counter(r['final']for r in selected)),'depth_label_counts':{str(k):v for k,v in counts.items()},'distinct_theory_families':len({r['family_id']for r in selected}),'heldout_theory_families_excluded':len(heldout),'statistics':stats,'license_status':'not_declared_in_official_release','no_distilled_reasoning_claim':True,'all_answers_independently_recomputed':True,'full_official_archive_downloaded_but_only_train_rows_exported':True,'selection_seed':20261016,'candidate_file':str(out.relative_to(ROOT)).replace('\\','/')}
 (AUDIT/'proofwriter_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
