"""Recover selected training questions and answer keys from official OpenStax HTML with MathML intact.
Run with bundled Python (lxml available). No executable source code is downloaded.
"""
import os,json,re,hashlib,concurrent.futures,time,copy
from pathlib import Path
from urllib.parse import urljoin
ROOT=Path(__file__).resolve().parents[3]/'Datasets/expansion_20261016/science'
for k in ['TEMP','TMP','PIP_CACHE_DIR','HF_HOME']:os.environ[k]=str(ROOT/'tmp')
from urllib.request import urlopen,Request
from lxml import html,etree
RAW=ROOT/'raw/openstax_html';RAW.mkdir(exist_ok=True)
def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def fetch(url):
 assert url.startswith('https://openstax.org/books/')
 path=RAW/(hashlib.sha256(url.encode()).hexdigest()+'.html');metapath=path.with_suffix('.json')
 if path.exists() and metapath.exists():return json.loads(metapath.read_text())
 for n in range(3):
  try:
   with urlopen(Request(url,headers={'User-Agent':'Mozilla/5.0 research dataset curation'}),timeout=90) as r:body=r.read()
   assert b'<html' in body.lower();path.write_bytes(body);meta={'url':url,'local_path':str(path.relative_to(ROOT)),'sha256':sha_bytes(body),'bytes':len(body),'retrieval_date':'2026-10-16','source':'OpenStax official book HTML'};metapath.write_text(json.dumps(meta,indent=2),encoding='utf-8');return meta
  except Exception:
   if n==2:raise
   time.sleep(n+1)
def mm(node):
 tag=etree.QName(node).localname;children=list(node)
 if tag=='semantics':return mm(children[0]) if children else ''
 if tag.startswith('annotation'):return ''
 if tag in ['none','mprescripts','mphantom']:return ''
 if tag=='mmultiscripts' and children:
  base=mm(children[0]);post=[];pre=[];target=post
  for child in children[1:]:
   if child.tag=='mprescripts':target=pre
   else:target.append(child)
  def scripts(items):
   result=''
   for i in range(0,len(items),2):
    low=mm(items[i]);high=mm(items[i+1]) if i+1<len(items) else ''
    result+=('_{'+low+'}' if low else '')+('^{'+high+'}' if high else '')
   return result
  return ('{}'+scripts(pre) if pre else '')+'{'+base+'}'+scripts(post)
 if tag in ['msup','msub'] and len(children)==2:return '{'+mm(children[0])+'}'+('^' if tag=='msup' else '_')+'{'+mm(children[1])+'}'
 if tag=='msubsup' and len(children)==3:return '{'+mm(children[0])+'}_{'+mm(children[1])+'}^{'+mm(children[2])+'}'
 if tag=='mfrac' and len(children)==2:return '\\frac{'+mm(children[0])+'}{'+mm(children[1])+'}'
 if tag=='msqrt':return '\\sqrt{'+''.join(mm(x) for x in children)+'}'
 if tag=='mroot' and len(children)==2:return '\\sqrt['+mm(children[1])+']{'+mm(children[0])+'}'
 if tag=='mspace':return ' '
 if tag in ['mover','munder'] and len(children)==2:return ('\\overset' if tag=='mover' else '\\underset')+'{'+mm(children[1])+'}{'+mm(children[0])+'}'
 if tag=='munderover' and len(children)==3:return '{'+mm(children[0])+'}_{'+mm(children[1])+'}^{'+mm(children[2])+'}'
 if tag=='mtable':return '\\begin{matrix}'+r' \\ '.join(mm(x) for x in children)+'\\end{matrix}'
 if tag=='mtr':return ' & '.join(mm(x) for x in children)
 if tag=='mfenced':return node.get('open','(')+','.join(mm(x) for x in children)+node.get('close',')')
 text=(node.text or '').strip();text=text.replace('\u00d7',r'\times ').replace('\u2212','-').replace('\u2013','-').replace('\u00b1',r'\pm ')
 if tag=='mtext' and re.search('[A-Za-z]',text):text='\\text{'+text.replace('{',r'\{').replace('}',r'\}')+'}'
 return text+''.join(mm(x)+(x.tail or '').strip() for x in children)
def render(node):
 node=copy.deepcopy(node)
 for math in list(node.iter('math')):
  replacement=etree.Element('span');replacement.text='$'+mm(math)+'$';replacement.tail=math.tail;math.getparent().replace(math,replacement)
 for unwanted in node.xpath('.//script|.//style|.//*[contains(@class,"os-number")]|.//*[contains(@class,"os-divider")]'):
  p=unwanted.getparent()
  if p is not None:p.remove(unwanted)
 for block in node.xpath('.//p|.//div|.//li|.//br'):
  block.tail=(block.tail or '')+'\n'
 text=''.join(node.itertext());text=re.sub(r'[ \t\r\f\v]+',' ',text);text=re.sub(r'\n\s*\n+','\n\n',text);return text.strip()
def parse_page(meta):return html.fromstring((ROOT/meta['local_path']).read_bytes(),parser=html.HTMLParser(encoding='utf-8'))
def process():
 selected=[json.loads(x) for x in (ROOT/'raw/openstax_selection.jsonl').open(encoding='utf-8')];urls=sorted({json.loads(r['metadata'])['source_url'] for r in selected})
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:pages=dict(zip(urls,pool.map(fetch,urls)))
 print(json.dumps({'question_pages_downloaded':len(pages)}),flush=True)
 trees={u:parse_page(m) for u,m in pages.items()};mapped=[];failures=[];answer_urls=set()
 for row in selected:
  md=json.loads(row['metadata']);url=md['source_url'];tree=trees[url];number=str(md['problem_number']);exs=tree.xpath('//*[@data-type="exercise"]');found=[]
  for e in exs:
   nums=e.xpath('.//*[contains(@class,"os-number")]')
   if nums and ''.join(nums[0].itertext()).strip()==number:found.append(e)
  if len(found)!=1:failures.append({'source':row['source'],'reason':'exercise_number_ambiguous_or_missing','matches':len(found)});continue
  e=found[0];parts=e.xpath('.//*[contains(@class,"os-problem-container")]');links=e.xpath('.//a[@data-page-fragment]')
  if len(parts)!=1 or not links:failures.append({'source':row['source'],'reason':'missing_problem_or_answer_link'});continue
  link=links[0];answer_url=urljoin(url,link.get('href','').split('#')[0]);answer_id=link.get('data-page-fragment');answer_urls.add(answer_url)
  mapped.append({'source_row':row,'metadata':md,'input':render(parts[0]),'has_visual':bool(parts[0].xpath('.//img|.//svg')),'question_page':pages[url],'question_dom_id':e.get('id'),'answer_url':answer_url,'answer_dom_id':answer_id})
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:answers=dict(zip(sorted(answer_urls),pool.map(fetch,sorted(answer_urls))))
 print(json.dumps({'answer_pages_downloaded':len(answers),'mapped_questions':len(mapped)}),flush=True)
 answer_trees={u:parse_page(m) for u,m in answers.items()};results=[]
 for r in mapped:
  nodes=answer_trees[r['answer_url']].xpath('//*[@id=$identifier]',identifier=r['answer_dom_id'])
  if len(nodes)!=1:failures.append({'source':r['source_row']['source'],'reason':'answer_dom_id_not_unique','matches':len(nodes)});continue
  parts=nodes[0].xpath('.//*[contains(@class,"os-solution-container")]');node=parts[0] if len(parts)==1 else nodes[0]
  r['output']=render(node);r['answer_has_visual']=bool(node.xpath('.//img|.//svg'));r['answer_page']=answers[r['answer_url']];results.append(r)
 out=ROOT/'raw/openstax_recovered.jsonl';out.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results),encoding='utf-8')
 report={'selected':len(selected),'recovered':len(results),'failures':failures,'html_pages':list(pages.values())+list(answers.values()),'mathml_conversion':'superscripts,subscripts,fractions,roots,limits and matrices retained as LaTex; no inference or missing context supplied'};(ROOT/'audit/openstax_recovery.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({'done':True,'recovered':len(results),'failures':len(failures)}),flush=True)
if __name__=='__main__':process()
