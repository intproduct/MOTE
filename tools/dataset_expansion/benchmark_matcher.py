"""Auditable exact/contained and conservative fuzzy benchmark text matching."""
import collections,json,re,sys,unicodedata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'Datasets/expansion_20261016/cache/python_deps'))
import ahocorasick
from rapidfuzz.fuzz import ratio

def normalize(text):
 s=unicodedata.normalize('NFKC',text).casefold()
 for token in ['\\(','\\)','\\[','\\]','$']:s=s.replace(token,' ')
 return re.sub(r'\s+',' ',s).strip()

def words(text):return re.findall(r'\w+',text.casefold(),re.UNICODE)

def read_rows(path):
 with Path(path).open(encoding='utf-8') as f:
  for line in f:yield json.loads(line)

class BenchmarkMatcher:
 def __init__(self,paths):
  self.rows=[];self.exact=collections.defaultdict(list);self.templates=collections.defaultdict(list);self.contained=ahocorasick.Automaton();self.anchors=ahocorasick.Automaton();anchor_map=collections.defaultdict(set)
  for path in paths:
   for r in read_rows(path):
    q=normalize(r['prompt']);idx=len(self.rows);self.rows.append((r,q));self.exact[q].append(idx)
    sk=re.sub(r'[-+]?\d+(?:\.\d+)?','<N>',q)
    if len(q)>=80 and sk!=q:self.templates[sk].append(idx)
    tok=words(q)
    if len(tok)>=16:
     starts={0,max(0,(len(tok)-13)//2),len(tok)-13}
     for start in starts:anchor_map[' '.join(tok[start:start+13])].add(idx)
  for q,indices in self.exact.items():
   if len(q)>=55:self.contained.add_word(q,indices)
  for anchor,indices in anchor_map.items():
   if len(indices)<=4 and len(anchor)>=55:self.anchors.add_word(anchor,tuple(indices))
  self.contained.make_automaton();self.anchors.make_automaton()
 def match(self,text):
  q=normalize(text);found={}
  for i in self.exact.get(q,[]):found[i]='normalized_exact'
  sk=re.sub(r'[-+]?\d+(?:\.\d+)?','<N>',q)
  for i in self.templates.get(sk,[]):found.setdefault(i,'numeric_template')
  for end,indices in self.contained.iter(q):
   for i in indices:
    bq=self.rows[i][1];start=end-len(bq)+1
    if (start==0 or not q[start-1].isalnum()) and (end==len(q)-1 or not q[end+1].isalnum()):found.setdefault(i,'contained_normalized_question')
  tokens=' '.join(words(q));candidates=set()
  for _,indices in self.anchors.iter(tokens):candidates.update(indices)
  for i in candidates:
   if i in found:continue
   b=' '.join(words(self.rows[i][1]))
   # A full benchmark question must remain mostly identical; a shared stock phrase is not enough.
   if b in tokens and len(b)>=70:found[i]='contained_word_normalized_question'
   elif min(len(tokens),len(b))/max(len(tokens),len(b),1)>=.7 and ratio(tokens,b)>=92:found[i]='high_similarity_with_rare_13word_anchor'
  return [{'benchmark':self.rows[i][0]['benchmark'],'source_file':self.rows[i][0]['source_file'],'row':self.rows[i][0]['row'],'source_id':self.rows[i][0].get('source_id'),'match_type':kind} for i,kind in sorted(found.items())]
