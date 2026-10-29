import requests,json
from download import ROOT
u='https://openstax.org/books/college-physics-2e/pages/19-problems-exercises'
r=requests.get(u,timeout=60);(ROOT/'audit/openstax_probe.html').write_text(r.text,encoding='utf-8')
print(r.status_code,len(r.content));print(r.text[:300]);print('exercise',r.text.find('data-type="exercise"'))
