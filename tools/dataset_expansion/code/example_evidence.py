"""Check prompt doctest syntax and compare literal examples with stored test oracles.

All comparison is static AST/literal inspection. No imported or generated code is run.
"""
import ast,copy,doctest,json,math
from test_evidence import scope_nodes

def canonical(value):
 if isinstance(value,tuple):return ['tuple',[canonical(v) for v in value]]
 if isinstance(value,list):return ['list',[canonical(v) for v in value]]
 if isinstance(value,dict):return ['dict',sorted([[canonical(k),canonical(v)] for k,v in value.items()],key=repr)]
 return [type(value).__name__,value]

def expand(node,bindings,depth=0):
 if depth>12:raise ValueError('binding depth')
 if isinstance(node,ast.Name) and node.id in bindings:return expand(bindings[node.id],bindings,depth+1)
 node=copy.deepcopy(node)
 for field,value in ast.iter_fields(node):
  if isinstance(value,ast.AST):setattr(node,field,expand(value,bindings,depth+1))
  elif isinstance(value,list):setattr(node,field,[expand(v,bindings,depth+1) if isinstance(v,ast.AST) else v for v in value])
 return node

def call_key(node,bindings,names):
 node=expand(node,bindings)
 if not isinstance(node,ast.Call):return None
 name=node.func.id if isinstance(node.func,ast.Name) else node.func.attr if isinstance(node.func,ast.Attribute) else None
 if name not in names or any(k.arg is None for k in node.keywords):return None
 try:
  args=[canonical(ast.literal_eval(a)) for a in node.args];kwargs=sorted((k.arg,canonical(ast.literal_eval(k.value))) for k in node.keywords)
  return json.dumps([name,args,kwargs],sort_keys=True,ensure_ascii=False)
 except (ValueError,TypeError,SyntaxError,RecursionError):return None

def numeric_equal(a,b):
 if type(a) is not type(b) and not isinstance(a,(int,float)):return False
 if isinstance(a,(int,float)) and not isinstance(a,bool) and isinstance(b,(int,float)) and not isinstance(b,bool):return math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-12)
 if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(numeric_equal(x,y) for x,y in zip(a,b))
 if isinstance(a,dict) and isinstance(b,dict):return a.keys()==b.keys() and all(numeric_equal(a[k],b[k]) for k in a)
 return type(a)==type(b) and a==b

def test_oracles(test,names):
 t=ast.parse(test);oracle={}
 for root in [t]+[n for n in ast.walk(t) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]:
  bindings={}
  for n in scope_nodes(root):
   if isinstance(n,ast.Assign):
    for target in n.targets:
     if isinstance(target,ast.Name):
      try:bindings[target.id]=expand(n.value,bindings)
      except ValueError:pass
   pair=None
   if isinstance(n,ast.Assert) and isinstance(n.test,ast.Compare) and len(n.test.ops)==1 and isinstance(n.test.ops[0],ast.Eq):pair=(n.test.left,n.test.comparators[0])
   if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in {'assertEqual','assertListEqual','assertTupleEqual','assertDictEqual','assertSequenceEqual','assertAlmostEqual'} and len(n.args)>=2:pair=(n.args[0],n.args[1])
   if pair:
    for left,right in [pair,pair[::-1]]:
     try:
      key=call_key(left,bindings,names)
      if key is not None:
       expected=ast.literal_eval(expand(right,bindings))
       oracle.setdefault(key,[]).append(expected)
     except (ValueError,TypeError,SyntaxError,RecursionError):pass
 return oracle

def analyse(prompt,test,solution,test_info=None):
 flags=[];conflicts=[];syntax_errors=[];matched=0
 if '>>>' not in prompt:return {'flags':[],'doctest_examples':0,'matched_test_oracles':0,'conflicts':[]}
 try:examples=doctest.DocTestParser().get_examples(prompt)
 except ValueError as e:return {'flags':['prompt_doctest_parse_error'],'doctest_examples':None,'matched_test_oracles':0,'conflicts':[],'parse_error':str(e)[:300]}
 names={n.name for n in ast.parse(solution).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
 names.update(x.get('function_name') for x in (test_info or []) if x.get('function_name'))
 oracle=test_oracles(test,names)
 for ex in examples:
  try:tree=ast.parse(ex.source)
  except SyntaxError as e:syntax_errors.append({'line':ex.lineno,'source':ex.source[:300],'error':str(e)});continue
  if len(tree.body)!=1 or not isinstance(tree.body[0],ast.Expr):continue
  expr=tree.body[0].value;expected=None;known=False;call=expr
  want='\n'.join(x for x in ex.want.splitlines() if x.strip() not in ['"""',"'''"]).strip()
  try:
   if isinstance(expr,ast.Compare) and len(expr.ops)==1 and isinstance(expr.ops[0],ast.Eq) and want in ['','True']:
    call=expr.left;expected=ast.literal_eval(expr.comparators[0]);known=True
   elif want:expected=ast.literal_eval(want);known=True
  except (ValueError,SyntaxError,TypeError):pass
  if not known:continue
  try:key=call_key(call,{},names)
  except (ValueError,RecursionError):continue
  if key in oracle and oracle[key]:
   matched+=1
   if not any(numeric_equal(expected,other) for other in oracle[key]):conflicts.append({'line':ex.lineno,'call':ex.source.strip()[:500],'prompt_expected':repr(expected)[:500],'test_expected':[repr(x)[:500] for x in oracle[key]]})
 if syntax_errors:flags.append('prompt_doctest_python_syntax_error')
 if conflicts:flags.append('prompt_example_conflicts_with_unit_test_oracle')
 return {'flags':flags,'doctest_examples':len(examples),'matched_test_oracles':matched,'conflicts':conflicts[:10],'syntax_errors':syntax_errors[:10],'local_execution_performed':False}
