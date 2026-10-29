"""Conservative lexical near-duplicate grouping, without deleting any response."""
import collections,hashlib,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'Datasets/expansion_20261016/cache/python_deps'))
import numpy as np
from rapidfuzz.fuzz import ratio

def strip_math_instruction(s):
 return re.sub(r'\s*(?:Reason through the problem and put your final answer within \\boxed\{\}\.?|Please reason step by step, and put your final answer within \\boxed\{\}\.?|Please reason step by step and put your final answer within \\boxed\{\}\.?)\s*$','',s,flags=re.I)

class NearFamilies:
 def __init__(self):self.bands=collections.defaultdict(list);self.entries=[];self.links=0;self.overfull_buckets=0;self.byte_bits=np.unpackbits(np.arange(256,dtype=np.uint8)[:,None],axis=1)
 def add(self,index,text):
  tokens=re.findall(r'\\[A-Za-z]+|\w+|[^\w\s]',text.casefold());normal=' '.join(tokens)
  if len(tokens)<20:return []
  # Deterministic bounded feature sample for indexing only; full text is used to confirm a link.
  step=max(1,(len(tokens)-1)//256);features={tokens[i]+' '+tokens[i+1] for i in range(0,len(tokens)-1,step)}
  hashes=b''.join(hashlib.blake2b(x.encode(),digest_size=8).digest() for x in features)
  values=np.frombuffer(hashes,dtype=np.uint8).reshape(-1,8);bits=self.byte_bits[values].reshape(-1,64).sum(axis=0)>=len(features)/2;fingerprint=int.from_bytes(np.packbits(bits).tobytes(),'little')
  candidates=set();keys=[(band,(fingerprint>>(band*16))&65535) for band in range(4)]
  for key in keys:
   bucket=self.bands[key]
   if len(bucket)>250:self.overfull_buckets+=1;continue
   candidates.update(bucket)
  matches=[]
  for j in candidates:
   old_index,old_hash,old_text=self.entries[j]
   if (fingerprint^old_hash).bit_count()<=3 and min(len(normal),len(old_text))/max(len(normal),len(old_text))>=.94 and ratio(normal,old_text)>=96:
    matches.append(old_index)
  position=len(self.entries);self.entries.append((index,fingerprint,normal))
  for key in keys:self.bands[key].append(position)
  self.links+=len(matches);return matches
