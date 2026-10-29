"""Conservative static evidence that tests check outputs of the tested solution.

This is neither execution nor a proof of test coverage/correctness. Unsupported
fixture/helper flows are held for review rather than inferred to be verified.
"""
import ast

CHECK_METHODS={'assertEqual','assertNotEqual','assertAlmostEqual','assertNotAlmostEqual','assertTrue','assertFalse','assertIs','assertIsNot','assertIsNone','assertIsNotNone','assertIn','assertNotIn','assertGreater','assertGreaterEqual','assertLess','assertLessEqual','assertCountEqual','assertListEqual','assertTupleEqual','assertDictEqual','assertSetEqual','assertSequenceEqual','assert_array_equal','assert_array_almost_equal','assert_allclose','assert_frame_equal','assert_series_equal','assert_index_equal'}

def scope_nodes(root):
 result=[]
 def visit(node):
  result.append(node)
  for child in ast.iter_child_nodes(node):
   if isinstance(child,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef,ast.Lambda)):continue
   visit(child)
 visit(root);return result

def analyse(test,solution,test_info=None):
 try:t=ast.parse(test);sol=ast.parse(solution)
 except (SyntaxError,ValueError,RecursionError):return {'eligible':False,'reason':'test_or_solution_syntax_invalid','linked_value_checks':0}
 names={x.name for x in sol.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
 names.update(x.get('function_name') for x in (test_info or []) if x.get('function_name'))
 aliases=set(names);modules={'solution'}
 for n in ast.walk(t):
  if isinstance(n,ast.ImportFrom) and n.module=='solution':
   for a in n.names:
    if a.name in names:aliases.add(a.asname or a.name)
  if isinstance(n,ast.Import):
   for a in n.names:
    if a.name=='solution':modules.add(a.asname or a.name)
 def solvercall(n,taint):
  if not isinstance(n,ast.Call):return False
  if isinstance(n.func,ast.Name):return n.func.id in aliases
  return isinstance(n.func,ast.Attribute) and ((n.func.attr in names and isinstance(n.func.value,ast.Name) and n.func.value.id in modules) or ast.unparse(n.func.value) in taint)
 def linked(node,taint):return any((isinstance(n,(ast.Name,ast.Attribute)) and ast.unparse(n) in taint) or solvercall(n,taint) for n in ast.walk(node))
 def targets_of(target):
  if isinstance(target,(ast.Name,ast.Attribute)):return {ast.unparse(target)}
  if isinstance(target,(ast.Tuple,ast.List)):return set().union(*(targets_of(x) for x in target.elts))
  return set()
 def tautology(node):
  if isinstance(node,ast.Constant):return True
  if isinstance(node,ast.Compare) and len(node.comparators)==1:
   return ast.dump(node.left)==ast.dump(node.comparators[0])
  return False
 linked_checks=0;empty_checks=0;parameterized_cases=0;solver_calls=0;check_functions=0
 roots=[t]+[n for n in ast.walk(t) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
 global_taint=set()
 for root in roots:
  nodes=scope_nodes(root);taint=set(global_taint);origins={}
  if hasattr(root,'args'):taint.difference_update(x.arg for x in root.args.args)
  def origin(expr):
   key=ast.unparse(expr)
   return origins.get(key,key)
  local_checks=0
  for n in nodes:
   if isinstance(n,(ast.Assign,ast.AnnAssign,ast.NamedExpr)) and n.value is not None:
    target_nodes=n.targets if isinstance(n,ast.Assign) else [n.target]
    target_keys=set().union(*(targets_of(x) for x in target_nodes))
    depends=linked(n.value,taint);value_origin=origin(n.value)
    taint.difference_update(target_keys)
    if depends:taint.update(target_keys)
    for key in target_keys:origins[key]=value_origin
   if isinstance(n,(ast.For,ast.AsyncFor)):
    keys=targets_of(n.target);depends=linked(n.iter,taint);taint.difference_update(keys)
    if depends:taint.update(keys)
   if solvercall(n,taint):solver_calls+=1
   if isinstance(n,ast.Assert):
    same_alias=isinstance(n.test,ast.Compare) and len(n.test.comparators)==1 and origin(n.test.left)==origin(n.test.comparators[0])
    if tautology(n.test) or same_alias:empty_checks+=1
    elif linked(n.test,taint):linked_checks+=1;local_checks+=1
   if isinstance(n,ast.Call):
    name=n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id if isinstance(n.func,ast.Name) else ''
    if name in CHECK_METHODS:
     if not n.args or (len(n.args)>=2 and origin(n.args[0])==origin(n.args[1])):empty_checks+=1
     elif any(linked(arg,taint) for arg in n.args):linked_checks+=1;local_checks+=1
  if local_checks:
   check_functions+=root is not t
   for d in getattr(root,'decorator_list',[]):
    if isinstance(d,ast.Call) and isinstance(d.func,ast.Attribute) and d.func.attr=='parametrize' and len(d.args)>=2 and isinstance(d.args[1],(ast.List,ast.Tuple)):
     parameterized_cases+=len(d.args[1].elts)
  if root is t:global_taint=taint
  # Empty constant assertions are recorded even when other real assertions exist.
 enough=linked_checks>=3 or (linked_checks>=1 and parameterized_cases>=3)
 return {'eligible':enough,'reason':None if enough else 'insufficient_statically_linked_value_checks','linked_value_checks':linked_checks,'constant_or_self_comparison_checks':empty_checks,'test_functions_with_linked_checks':check_functions,'parameterized_cases_in_linked_tests':parameterized_cases,'recognized_solution_calls':solver_calls,'policy':'at least 3 statically linked result assertions, or >=3 explicit parameterized cases with linked assertions; unsupported flows review','proof_of_correctness':False}
