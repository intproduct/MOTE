"""Select only the pinned collection's OpenStax train rows; exclude other origins."""
import json
import pyarrow.parquet as pq
from process import ROOT
rows=pq.read_table(ROOT/'raw/physcorp_a.parquet').to_pylist()
selected=[r for r in rows if r['source'].startswith('openstax_')]
assert len(rows)==6432 and len(selected)==1429
(ROOT/'raw/openstax_selection.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected),encoding='utf-8')
print(json.dumps({'selected_openstax':len(selected),'other_sources_excluded':len(rows)-len(selected)}))
