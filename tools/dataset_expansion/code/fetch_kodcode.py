import concurrent.futures,json,time
from fetch_sources import DEST,RAW,get,download

REPO='KodCode/KodCode-V1-SFT-R1'
REV='26c8a11c800d71b6c4bafe12a92c9090ef0b6214'

def main():
 (RAW/'README_KodCode-V1-SFT-R1.md').write_text(get(f'https://huggingface.co/datasets/{REPO}/raw/{REV}/README.md').text,encoding='utf-8')
 listing=get(f'https://huggingface.co/api/datasets/{REPO}/tree/{REV}/data?limit=100').json()
 (DEST/'audit/kodcode_listing.json').write_text(json.dumps(listing,indent=2),encoding='utf-8')
 train=[r for r in listing if r['path'].startswith('data/train-')]
 selected=[{**train[i],'_repository':REPO,'_revision':REV,'_prefix':'kodcode'} for i in [0,4,8,10]]
 # Share the network fairly: no overlap with the two OCR large-download workers.
 while not (DEST/'audit/source_manifest.json').exists():time.sleep(5)
 with concurrent.futures.ThreadPoolExecutor(2) as pool:files=list(pool.map(download,selected))
 manifest={'repository':REPO,'revision':REV,'license':'cc-by-nc-4.0','files':files,'verification_notes':'upstream paired-unit-test rejection sampling, not local execution; require r1_correctness true and r1_solution matching assistant response; retain unit tests','selection':'four spaced training shards 0,4,8,10; incorrect and use_with_caution splits excluded'}
 (DEST/'audit/kodcode_source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 print(json.dumps(manifest),flush=True)

if __name__=='__main__':main()
