"""Human-rating selected general SFT. Keep absolute rating evidence, not just preference wins."""
import gzip,json,re,collections,random
from download import BASE,RAW,AUDIT
from convert import base_record,structural,semantic_review_flags,sha,dump

def split_helpsteer2(prompt):
 parts=re.split(r'<extra_id_1>(Assistant|User)\n',prompt)
 messages=[{'role':'user','content':parts[0]}]
 for role,content in zip(parts[1::2],parts[2::2]):messages.append({'role':role.lower(),'content':content})
 if any('<extra_id_' in m['content'] for m in messages):raise ValueError('ambiguous_role_delimiter')
 if messages[-1]['role']!='user':raise ValueError('missing_final_user')
 if any(m['role']!=('user' if i%2==0 else 'assistant')for i,m in enumerate(messages)):raise ValueError('role_alternation')
 return messages

def selected_feedback(feedback):
 return len(feedback)>=3 and all(re.match(r'^The response is perfectly helpful\b',x.strip(),re.I)for x in feedback)

def domain_for(messages,original):
 p=messages[-2]['content'].lower()
 if re.search(r'\b(?:write|rewrite|compose|draft|brainstorm|poem|story|essay|email)\b',p):return 'writing'
 if re.search(r'\b(?:summariz|extract|classif|translate|translation)\w*\b',p):return 'reading'
 return 'general_chat'

def main():
 stats=collections.Counter();seen=set();allrows=[];excluded=[];rng=random.Random(20261016)
 for filename,kind in [('helpsteer2_train.jsonl.gz','rating'),('helpsteer3_feedback_train.jsonl.gz','feedback')]:
  meta=json.loads((AUDIT/(filename+'.download.json')).read_text());meta['license']='cc-by-4.0'
  with gzip.open(RAW/filename,'rt',encoding='utf-8')as f:
   for index,line in enumerate(f,1):
    raw=json.loads(line);stats[kind+':source_rows']+=1
    if kind=='rating':
     if not all(raw[k]==4 for k in ['helpfulness','correctness','coherence']):stats['rating:below_perfect_threshold']+=1;continue
     try:context=split_helpsteer2(raw['prompt'])
     except ValueError as e:excluded.append({'file':filename,'row':index,'reason':str(e)});continue
     responses=[(1,raw['response'],{'human_scores':{k:raw[k]for k in ['helpfulness','correctness','coherence','complexity','verbosity']}})]
    else:
     if raw['domain'] not in ['general','multilingual']:stats['feedback:outside_general_scope']+=1;continue
     context=raw['context'];responses=[]
     for n in [1,2]:
      if selected_feedback(raw['feedback'+str(n)]):responses.append((n,raw['response'+str(n)],{'human_feedback':raw['feedback'+str(n)]}))
      else:stats['feedback:below_unanimous_perfect_threshold']+=1
    for n,response,evidence in responses:
     messages=context+[{'role':'assistant','content':response}];mh=sha(json.dumps(messages,sort_keys=True,ensure_ascii=False))
     if mh in seen:stats['exact_duplicates']+=1;continue
     seen.add(mh);domain=domain_for(messages,raw.get('domain'))
     cleanraw={'id':str(index)+'-response'+str(n),'language':raw.get('language','english'),'source_domain':raw.get('domain','not_provided'),**evidence}
     row=base_record(cleanraw,meta,index,messages,domain,'source_verified','human_absolute_rating_selection_not_independent_fact_check')
     row['metadata'].update({'group_key':row['family_id'],'source_response_field':'response' if kind=='rating' else 'response'+str(n),'source_prompt_encoding':'documented_extra_id_1_role_delimiters' if kind=='rating' else 'structured_context','assistant_loss_mask':[False]*(len(messages)-1)+[True],'human_rating_selection':{'helpfulness':4,'correctness':4,'coherence':4}if kind=='rating'else{'minimum_annotators':3,'all_perfectly_helpful':True},'language':raw.get('language','english'),'domain_assignment':'source_general_multilingual_then_explicit_prompt_keyword_heuristic'})
     row['verification'].update({'human_rating_evidence':evidence,'human_rating_not_guarantee_of_factual_correctness':True})
     flags=structural(messages);soft=semantic_review_flags(messages,domain,'helpsteer')
     # Code/STEM have separate executed/recomputed streams. Human ratings alone are insufficient here.
     user='\n'.join(m['content']for m in messages if m['role']=='user')
     if kind=='rating' and re.search(r'```|\b(?:python|javascript|typescript|c\+\+|sql query|source code|write.{0,30}code)\b',user,re.I):soft.append('code_requires_execution_not_human_rating_only')
     if re.search(r'\b(?:diagnos\w*|prescri\w*|dosage|medical advice|legal advice|investment advice|lawsuit)\b',user,re.I):soft.append('specialist_advice_requires_independent_review')
     if re.search(r'\b(?:pretend.{0,50}(?:human|girlfriend|boyfriend)|roleplay|role play|act as my)\b',user,re.I):soft.append('persona_roleplay_outside_current_training_mix')
     row['review_flags']=flags+soft
     # Fixed-seed semantic audit found material errors even under unanimous top ratings.
     # Preserve the whole high-rating pool for review; only explicit hash-bound manual
     # acceptance can promote a row, and that is not a blanket factual certificate.
     soft.append('perfect_human_ratings_failed_source_semantic_spot_audit')
     row['review_flags']=flags+soft
     row['release_status']='quarantine'if flags else 'review'
     stats[kind+':'+row['release_status']]+=1;allrows.append(row)
 reviews_path=AUDIT/'helpsteer_manual_reviews.json'
 if reviews_path.exists():
  reviews={r['id']:r for r in json.loads(reviews_path.read_text(encoding='utf-8'))}
  for row in allrows:
   rev=reviews.get(row['id'])
   if rev:
    assert row['metadata']['source_messages_sha256']==rev['messages_sha256']
    row['metadata']['manual_visible_content_review']=rev
    if rev['decision']=='review':row['review_flags'].append('manual_content_review_requires_followup')
    elif rev['decision']=='pass_visible' and row['release_status']=='review' and row['review_flags']==['perfect_human_ratings_failed_source_semantic_spot_audit']:
     row['release_status']='candidate';row['quality_level']='source_curated';row['review_flags']=[]
 for status,outname in [('candidate','helpsteer_candidates'),('review','helpsteer_review'),('quarantine','helpsteer_quarantine')]:
  with (BASE/(outname+'.jsonl')).open('w',encoding='utf-8',newline='\n')as f:
   for row in allrows:
    if row['release_status']==status:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 groups=collections.defaultdict(list)
 for row in allrows:groups[(row['source'],row['domain'])].append(row)
 with (AUDIT/'helpsteer_content_sample.jsonl').open('w',encoding='utf-8')as f:
  for key,rows in sorted(groups.items()):
   for row in rng.sample(rows,min(5,len(rows))):f.write(json.dumps(row,ensure_ascii=False)+'\n')
 dump(AUDIT/'helpsteer_summary.json',{'stats_before_manual_review':dict(stats),'final_counts':dict(collections.Counter(r['release_status']for r in allrows)),'candidate_domains':dict(collections.Counter(r['domain']for r in allrows if r['release_status']=='candidate')),'candidate_languages':dict(collections.Counter(r['metadata']['language']for r in allrows if r['release_status']=='candidate')),'edit_quality_archived_not_exported':'Good means feedback adherence only, not independently assessed response correctness; no automatic SFT acceptance.','excluded_decode_rows':excluded,'manual_reviews_applied':reviews_path.exists(),'selection_seed':20261016})
 print((AUDIT/'helpsteer_summary.json').read_text(),flush=True)
if __name__=='__main__':main()
