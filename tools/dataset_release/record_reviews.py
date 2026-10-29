"""Record the assistant's explicit, bounded review decisions with content hashes."""
from common import *
notes={
 'complex_analysis_001':'Expanded (x+iy)^2 and checked both real and imaginary components.',
 'complex_analysis_002':'Expanded 3(x+iy)-1 and checked real/imaginary components.',
 'complex_analysis_003':'Checked componentwise complex addition.',
 'complex_analysis_004':'Checked addition and multiplication using i^2=-1.',
 'complex_analysis_005':'Checked integer power: 3^3 exp(-6i).',
 'complex_analysis_006':'Checked conjugation in Cartesian and polar coordinates.',
 'complex_analysis_008':'Checked Euler identity.',
 'complex_analysis_009':'Checked conjugate of 7+8i.',
 'complex_analysis_011':'Checked subtraction gives -2-7i.',
 'complex_analysis_012':'Checked multiplication gives 41-37i.',
 'complex_analysis_013':'Checked division by multiplying by 1-i, giving -5i.',
 'complex_analysis_014':'Checked multiplication of polar forms gives 10 exp(-5i).'}
reviews=[]
p=next((ROOT/'Datasets/phy-selfuse').glob('*.md'));objects_,errors=markdown_json_objects(p.read_text(encoding='utf-8-sig'))
for i,(obj,_) in enumerate(objects_,1):
 if obj['id'] not in notes:continue
 obj={k:clean_recovered_markdown_value(v) for k,v in obj.items()}
 reviews.append({'id':'physics-'+sha(str(p.relative_to(ROOT))+':'+str(i))[:20],'input_sha256':sha(clean(obj['problem'])),'final_sha256':sha(clean(obj['solution'])+'\n\n最终答案：'+clean(obj['answer'])),'verdict':'pass','scope':'visible_solution_and_answer','reviewer':'assistant_semantic_review','note':notes[obj['id']]})
choices={'gaokao_phy.jsonl':{1:'Momentum is a vector; mass, speed and kinetic energy are scalars.',2:'On a horizontal moving belt, friction accelerates the object; acceleration duration v/(mu*g) increases with belt speed.'},
 '期中.jsonl':{2:'Under the stated free-fall approximation without air resistance, speed increases and gravitational acceleration stays approximately constant.'},
 '模拟.jsonl':{1:'The airborne diver is the free-fall case in the listed elementary-mechanics scenarios.',2:'With contact retained and downward acceleration, N=m(g-a)<mg, while action/reaction forces remain equal.'}}
for name,selected in choices.items():
 p=ROOT/'Datasets/phy-selfuse'/name
 for i,line in enumerate(p.read_text(encoding='utf-8-sig').splitlines(),1):
  if i not in selected:continue
  r,_=repair_json_text(line)
  reviews.append({'id':'physics-'+sha(str(p.relative_to(ROOT))+':'+str(i))[:20],'input_sha256':sha(clean(r['question'])),'final_sha256':sha(clean(r['answer'])),'verdict':'pass','scope':'visible_solution_and_answer','reviewer':'assistant_semantic_review','note':selected[i]})
p=ROOT/'Datasets/mixture-fromqwen3p8/results/math-01a5efe142c7314e.json';r=json.loads(p.read_text(encoding='utf-8'))
assert [y for y in range(1600,3501) if y%100==0 and y%1000 in (125,875)]==[]
assert boxed(r['final'])=='0'
reviews.append({'id':r['id'],'input_sha256':sha(clean(r['prompt'])),'final_sha256':sha(clean(r['final'])),'verdict':'correct_reference','corrected_reference':'0','original_reference':'4','scope':'visible_solution_and_answer','reviewer':'assistant_review_and_exhaustive_integer_check','note':'A multiple of 100 has a remainder modulo 1000 divisible by 100. Exhaustive integer enumeration over 1600..3500 confirms zero solutions. Teacher final is correct for the supplied question; source gold is corrected only in derived data.'})
write_json(Path(__file__).with_name('manual_reviews.json'),reviews)
print(json.dumps({'bounded_manual_reviews':len(reviews)}))
