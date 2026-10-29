import json,collections
from pathlib import Path
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[3]/'Datasets/expansion_20261016/science'
for name in ['textbook_reasoning.parquet','osr2_0000.parquet']:
 p=ROOT/'raw'/name
 if not p.exists():continue
 f=pq.ParquetFile(p);print(json.dumps({'name':name,'rows':f.metadata.num_rows,'schema':str(f.schema_arrow),'row_groups':f.num_row_groups}))
 row=next(f.iter_batches(batch_size=1)).to_pylist()[0]
 print(json.dumps({'sample':row},ensure_ascii=True)[:15000])
 if 'subject' in f.schema_arrow.names:
  c=collections.Counter()
  for b in f.iter_batches(columns=['subject']):c.update(b.column(0).to_pylist())
  print(json.dumps({'subject_counts':c}))
