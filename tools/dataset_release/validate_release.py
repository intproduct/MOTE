"""Validate JSONL exports, their schema, split isolation and source integrity."""
import argparse,collections,hashlib,json,sys
from pathlib import Path

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('release',type=Path);p.add_argument('--source-root',type=Path);p.add_argument('--deps',type=Path);a=p.parse_args()
 if a.deps:sys.path.insert(0,str(a.deps))
 from jsonschema import Draft202012Validator
 schema=json.loads((a.release/'schema.json').read_text(encoding='utf-8'));Draft202012Validator.check_schema(schema);validator=Draft202012Validator(schema)
 counts={};seen=set();families={};training=set();total=0
 for path in sorted((a.release/'jsonl').glob('*.jsonl')):
  count=0
  for line in path.open(encoding='utf-8'):
   r=json.loads(line);validator.validate(r);assert r['id'] not in seen;seen.add(r['id']);count+=1
   section=r['output_sections'];assert r['output'][slice(*section['final'])]==r['final']
   if path.stem in ['train','validation','test','smoke']:
    assert r['input'].strip() and r['output'].strip()
    assert r['release_status']=='answer_checked' and r['verification']['answer_pass'] is True and not r['review_flags']
    assert r['split']!='benchmark_excluded'
    assert families.setdefault(r['family_id'],path.stem)==path.stem
   if path.stem=='train':training.add(r['id'])
  counts[path.name]=count;total+=count
 pilot=0;pilot_families=set()
 for line in (a.release/'pilot/train.jsonl').open(encoding='utf-8'):
  r=json.loads(line);validator.validate(r);assert r['id'] in training and r['family_id'] not in pilot_families;pilot_families.add(r['family_id']);pilot+=1
 manifest=json.loads((a.release/'MANIFEST.json').read_text(encoding='utf-8'))
 for item in manifest['files']:assert digest(a.release/item['path'])==item['sha256'],item['path']
 source_count=0
 if a.source_root:
  for line in (a.release/'audit/source_inventory.jsonl').open(encoding='utf-8'):
   item=json.loads(line);assert digest(a.source_root/item['path'])==item['sha256'],item['path'];source_count+=1
 report={'passed':True,'schema':'Draft 2020-12','record_count':total,'files':counts,'pilot_records':pilot,'source_files_checked_unchanged':source_count,'manifest_files_checked':len(manifest['files']),'unique_ids':True,'released_family_split_isolation':True,'pilot_is_subset_of_train':True,'pilot_one_per_family':True}
 (a.release/'audit/validation_final.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report))

if __name__=='__main__':main()
