from __future__ import annotations
import ast,collections,csv,hashlib,io,json,math,re,unicodedata
from decimal import Decimal,InvalidOperation
from fractions import Fraction
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'Datasets/processed/local_sft_v01_20261016'
VERSION='local-curation-v01-20261016'

def sha(s):return hashlib.sha256(s.encode('utf-8')).hexdigest()
def write_json(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def write_jsonl(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',encoding='utf-8',newline='\n') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
def clean(s):
 if s is None:return ''
 return str(s).replace('\r\n','\n').replace('\r','\n').lstrip('\ufeff').strip()
def norm(s):return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',clean(s))).strip().casefold()
def question_key(s):
 s=re.sub(r'\n\nReason through the problem and put your final answer within \\boxed\{\}\.?\s*$','',s)
 return norm(s)
def boxed(s):
 found=[]
 for m in re.finditer(r'\\(?:boxed|fbox)\s*\{',s):
  start=m.end();depth=1;i=start
  while i<len(s) and depth:
   if s[i]=='{':depth+=1
   elif s[i]=='}':depth-=1
   i+=1
  if depth==0:found.append(s[start:i-1])
 return found[-1] if found else None
def tagged(s):
 a=re.findall(r'<answer>(.*?)</answer>',s,re.S|re.I)
 return a[-1].strip() if len(a)==1 else None
def objects(s):
 d=json.JSONDecoder();out=[];i=0
 while i<len(s):
  if s[i] not in '[{':i+=1;continue
  try:v,n=d.raw_decode(s[i:]);out.append(v);i+=n
  except ValueError:i+=1
 return out
def code_blocks(s):
 out=[];opened=None;buf=[]
 for line in s.splitlines():
  if opened is None:
   m=re.fullmatch(r'\s*(`{3,}|~{3,})\s*([^\s`]*)\s*',line)
   if m:opened=(m[1],m[2].lower());buf=[]
  elif re.fullmatch(r'\s*'+re.escape(opened[0][0])+r'{'+str(len(opened[0]))+r',}\s*',line):
   out.append({'language':opened[1],'code':'\n'.join(buf)});opened=None
  else:buf.append(line)
 return out,opened is not None
def select_code(s,fn_name=None):
 blocks,unclosed=code_blocks(s);candidates=[]
 for b in blocks:
  if b['language'] not in ['python','python3','py','']:continue
  try:t=ast.parse(b['code'])
  except SyntaxError:continue
  funcs={n.name for n in ast.walk(t) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
  executable=any(isinstance(n,(ast.FunctionDef,ast.ClassDef,ast.Import,ast.ImportFrom,ast.Assign,ast.For,ast.While,ast.If)) for n in t.body)
  if executable and (not fn_name or fn_name in funcs):candidates.append(b['code'])
 candidates=list(dict.fromkeys(candidates))
 return {'selected':candidates[0]+'\n' if len(candidates)==1 else None,'candidate_count':len(candidates),'unclosed_fence':unclosed,'all_blocks':len(blocks)}

def math_norm(s):
 s=clean(s).replace('$','').replace('−','-').replace('\\left','').replace('\\right','').replace('\\dfrac','\\frac').replace('\\tfrac','\\frac')
 for marker in ['\\(','\\)','\\[','\\]']:s=s.replace(marker,'')
 return re.sub(r'\s+','',s).rstrip('.')
def scalar(s):
 s=math_norm(s)
 m=re.fullmatch(r'\\frac\{([-+]?\d+)\}\{([-+]?\d+)\}',s)
 if m and int(m[2]):return Fraction(int(m[1]),int(m[2]))
 if re.fullmatch(r'[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?',s):s=s.replace(',','')
 if re.fullmatch(r'[-+]?\d+(?:\.\d+)?(?:/[+-]?\d+)?',s):
  try:return Fraction(s)
  except (ValueError,ZeroDivisionError):return None
 return None
def answer_equal(a,b):
 if not isinstance(a,str) or not isinstance(b,str):return a==b
 if math_norm(a)==math_norm(b):return True
 x,y=scalar(a),scalar(b)
 return x==y if x is not None and y is not None else None
def json_equal(a,b):
 # Do not conflate bool and int as Python equality does.
 if isinstance(a,bool) or isinstance(b,bool):return type(a) is type(b) and a==b
 if isinstance(a,(float,int)) and isinstance(b,(float,int)):
  return math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-9)
 if type(a) is not type(b):return False
 if isinstance(a,dict):return a.keys()==b.keys() and all(json_equal(a[k],b[k]) for k in a)
 if isinstance(a,list):return len(a)==len(b) and all(json_equal(x,y) for x,y in zip(a,b))
 return a==b

def instruction_oracle(r):
 p=r['prompt'];m=re.search(r'Use these records: (\[.*?\])\s*\nFilter: (.*?)\. Sort by (score|units|id) (ascending|descending), breaking ties by id ascending\. Keep the first (\d+) records',p,re.S)
 if not m:return None
 records=json.loads(m[1]);conds=m[2].split(' AND ');selected=[]
 for x in records:
  ok=True
  for c in conds:
   if z:=re.fullmatch(r'score >= (\d+)',c):ok &= x['score']>=int(z[1])
   elif z:=re.fullmatch(r'group is not (\w+)',c):ok &= x['group']!=z[1]
   elif c=='units is even':ok &= x['units']%2==0
   else:return None
  if ok:selected.append(x)
 selected.sort(key=lambda x:x['id'])
 selected.sort(key=lambda x:x[m[3]],reverse=m[4]=='descending');selected=selected[:int(m[5])]
 if 'Return only valid JSON' in p:return {'ids':[x['id'] for x in selected],'count':len(selected),'units_total':sum(x['units'] for x in selected),'weighted_score':sum(x['score']*x['units'] for x in selected)}
 if 'Return only a numbered list' in p:return '\n'.join(f'{i}. {x["id"]}: {x["score"]}' for i,x in enumerate(selected,1)) or 'EMPTY'
 if 'Return only CSV' in p:
  f=io.StringIO();w=csv.writer(f,lineterminator='\n');w.writerow(['id','group','score','units']);w.writerows([[x[k] for k in ['id','group','score','units']] for x in selected]);return f.getvalue().strip()
 return None

def inventory_oracle(r):
 p=r['prompt'];m=re.match(r'Track this inventory in order\. Initial stock: (\{.*?\})\. Unit prices: (\{.*?\})\.',p)
 if not m:return None
 state=json.loads(m[1]);prices=json.loads(m[2]);events=re.findall(r'^\d+\. (.+)$',p,re.M)
 if not events:return None
 for e in events:
  if m:=re.fullmatch(r'Add (\d+) units to (\w+)\.',e):state[m[2]]+=int(m[1])
  elif m:=re.fullmatch(r'Remove up to (\d+) units from (\w+), stopping at zero\.',e):state[m[2]]=max(0,state[m[2]]-int(m[1]))
  elif m:=re.fullmatch(r'Cap (\w+) at (\d+) units \(discard excess, do not add stock\)\.',e):state[m[1]]=min(state[m[1]],int(m[2]))
  elif m:=re.fullmatch(r'Swap the entire quantities in (\w+) and (\w+)\.',e):state[m[1]],state[m[2]]=state[m[2]],state[m[1]]
  elif m:=re.fullmatch(r'Transfer up to (\d+) units from (\w+) to (\w+), limited by the stock in \2\.',e):
   n=min(int(m[1]),state[m[2]]);state[m[2]]-=n;state[m[3]]+=n
  elif m:=re.fullmatch(r'If (\w+) has at least (\d+) units now, add (\d+) to (\w+); otherwise remove up to (\d+) from \4, stopping at zero\.',e):
   condition=state[m[1]]>=int(m[2]);state[m[4]]=state[m[4]]+int(m[3]) if condition else max(0,state[m[4]]-int(m[5]))
  else:return None
 return {'stock':state,'total_units':sum(state.values()),'inventory_value':sum(v*prices[k] for k,v in state.items())}

def gym_oracle(r):
 name=r.get('gym_name');m=r.get('gym_entry',{}).get('metadata',{});p=r['prompt']
 # Only independently implemented tasks; metadata solution is never used as oracle.
 if name=='number_sorting':
  z=re.search(r'Sort these numbers in (ascending|descending) order: ([^\n]+)',p)
  if not z:return None
  values=[v.strip() for v in z[2].split(',')];return str(sorted(values,key=Decimal,reverse=z[1]=='descending'))
 if name=='number_filtering':
  a=m['original_numbers'];x=Decimal(m['filter_value']);op=m['operation']
  if op not in ['keep_smaller','keep_larger','remove_smaller','remove_larger']:return None
  rules={'keep_smaller':lambda v:v<x,'keep_larger':lambda v:v>x,'remove_smaller':lambda v:v>=x,'remove_larger':lambda v:v<=x}
  return str([v for v in a if rules[op](Decimal(v))])
 if name=='binary_alternation':
  s=m['string'];n=len(s);a=s.count('0');b=n-a
  if abs(a-b)>1:return '-1'
  options=[]
  for first in '01':
   target=''.join(str((int(first)+i)%2) for i in range(n))
   if target.count('0')==a:options.append(sum(x!=y for x,y in zip(s,target))//2)
  return str(min(options))
 if name in ['binary_matrix','rotten_oranges']:
  a=m['matrix'];nr=len(a);nc=len(a[0]);d=[[-1]*nc for _ in range(nr)];q=collections.deque()
  for i in range(nr):
   for j in range(nc):
    if a[i][j]==(0 if name=='binary_matrix' else 2):d[i][j]=0;q.append((i,j))
  while q:
   i,j=q.popleft()
   for di,dj in [(1,0),(-1,0),(0,1),(0,-1)]:
    x,y=i+di,j+dj
    if 0<=x<nr and 0<=y<nc and d[x][y]<0 and (name=='binary_matrix' or a[x][y]==1):d[x][y]=d[i][j]+1;q.append((x,y))
  if name=='binary_matrix':return '\n'.join(' '.join(map(str,row)) for row in d)
  fresh=[d[i][j] for i in range(nr) for j in range(nc) if a[i][j]==1]
  return str(-1 if -1 in fresh else max(fresh,default=0))
 if name=='spiral_matrix':
  a=[row[:] for row in m['matrix']];values=[]
  while a:
   values+=a.pop(0)
   if a and a[0]:
    for row in a:values.append(row.pop())
   if a:values+=a.pop()[::-1]
   if a and a[0]:
    for row in a[::-1]:values.append(row.pop(0))
  return ' '.join(map(str,values))
 if name=='pool_matrix':
  a=m['matrix'];k=m['pool_size'];out=[]
  for i in range(0,len(a),k):
   row=[]
   for j in range(0,len(a[0]),k):
    vals=[a[x][y] for x in range(i,min(i+k,len(a))) for y in range(j,min(j+k,len(a[0])))];row.append(max(vals) if m['pool_type']=='max' else sum(vals)/len(vals))
   out.append(row)
  return out
 return None

def literal_equal(a,b):
 if norm(a)==norm(b):return True
 try:return json_equal(ast.literal_eval(a),ast.literal_eval(b))
 except (ValueError,SyntaxError,TypeError,MemoryError,RecursionError):return None

def repair_json_text(text):
 """Only repair unescaped Markdown underscore and known LaTeX command escapes."""
 actions=[]
 for _ in range(1000):
  try:return json.loads(text),actions
  except json.JSONDecodeError as e:
   if not e.msg.startswith('Invalid \\escape'):raise
   tail=text[e.pos:]
   if tail.startswith('\\_'):
    text=text[:e.pos]+text[e.pos+1:];actions.append({'kind':'markdown_underscore_escape','offset':e.pos})
   elif len(tail)>1:
    text=text[:e.pos]+'\\'+text[e.pos:];actions.append({'kind':'double_invalid_LaTeX_backslash','offset':e.pos})
   else:raise
 raise ValueError('too_many_JSON_repairs')

def markdown_json_objects(text):
 """Recover separate JSON objects from multiple Markdown-wrapped arrays."""
 import html
 text=html.unescape(text).replace('\u00a0',' ');found=[];bad=[]
 for match in re.finditer(r'\{\s*"id"\s*:',text):
  start=match.start();quoted=False;escaped=False;depth=0;end=None
  for pos in range(start,len(text)):
   c=text[pos]
   if quoted:
    if escaped:escaped=False
    elif c=='\\':escaped=True
    elif c=='"':quoted=False
   elif c=='"':quoted=True
   elif c=='{':depth+=1
   elif c=='}':
    depth-=1
    if depth==0:end=pos+1;break
  if end is None:bad.append({'offset':start,'error':'unclosed_object'});continue
  try:
   obj,actions=repair_json_text(text[start:end])
   if not isinstance(obj,dict) or not {'problem','solution','answer'}<=obj.keys():raise ValueError('unexpected_object_schema')
   found.append((obj,{'offset':start,'repairs':actions}))
  except (ValueError,TypeError) as e:bad.append({'offset':start,'error':str(e)})
 return found,bad

def clean_recovered_markdown_value(s):
 # Source-specific Markdown-overescaped LaTeX and literal newline cleanup.
 s=clean(s)
 s=re.sub(r'\\{2,}(?=[A-Za-z])',lambda m:'\\',s)
 return re.sub(r'(?<!\\)\\n(?![A-Za-z])','\n',s)

def restore_damaged_latex(s):
 """Repair only recognizable LaTeX commands whose backslash became a control byte."""
 s=str(s);changes=[]
 specs=[('\x07',r'(?:lpha|ngle|pprox|rrow)(?![A-Za-z])','a'),('\x0c',r'rac(?![A-Za-z])','f'),('\x08',r'(?:eta|ar|egin|ig|oldsymbol|oxed)(?![A-Za-z])','b'),('\x0b',r'(?:arepsilon|arphi|artheta|ec)(?![A-Za-z])','v'),('\t',r'(?:heta|imes|ext|ilde|au)(?![A-Za-z])','t'),('\r',r'(?:ight|ho|angle)(?![A-Za-z])','r')]
 for c,tail,letter in specs:
  pattern=re.escape(c)+'(?='+tail+')';s,n=re.subn(pattern,lambda m:'\\'+letter,s)
  if n:changes.append({'control_codepoint':ord(c),'replacement_prefix':'\\'+letter,'count':n})
 return s,changes
