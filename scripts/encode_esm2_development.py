"""Create sequence-only ESM2 features without labels, padding or truncated residues."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import sys,json,time,hashlib,argparse
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModel
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from phgeofuse.io import read_manifest
from phgeofuse.cache import atomic_json
OUT=ROOT/'experiments/phgeofuse_redesign_20260914/esm2_masked';OUT.mkdir(exist_ok=True)
ap=argparse.ArgumentParser();ap.add_argument('--test',action='store_true');args=ap.parse_args();suffix='_test' if args.test else ''
torch.set_num_threads(4)
repo='facebook/esm2_t33_650M_UR50D'
tokenizer=AutoTokenizer.from_pretrained(repo,local_files_only=True)
model=AutoModel.from_pretrained(repo,local_files_only=True).cuda().eval().half()
records=[r for r in read_manifest(ROOT/'experiments/phgeofuse_phopt_full_20260913/inputs/manifest.csv') if r.split in (('test',) if args.test else ('train','validation'))]
source_hash=hashlib.sha256((ROOT/'experiments/phgeofuse_phopt_full_20260913/inputs/manifest.csv').read_bytes()).hexdigest()
atomic_json(OUT/f'provenance{suffix}.json',{'repo':repo,'model_config':model.config.to_dict(),'pooling':'all residue tokens; excludes padding/CLS/EOS; chunks <=1022 residues; population std','splits':['test'] if args.test else ['train','validation'],'manifest_sha256':source_hash,'labels_used':False})
cache=OUT/'pooled';cache.mkdir(exist_ok=True)
pending=[r for r in records if not (cache/f'{r.sequence_sha256}.npz').exists()]
pending.sort(key=lambda r:len(r.sequence))
for i,r in enumerate(pending):
 chunks=[]
 for start in range(0,len(r.sequence),1022):
  seq=r.sequence[start:start+1022]
  tokens=tokenizer(seq,return_tensors='pt',add_special_tokens=True)
  tokens={k:v.cuda() for k,v in tokens.items()}
  with torch.inference_mode(): h=model(**tokens).last_hidden_state[0,1:len(seq)+1].float().cpu()
  assert len(h)==len(seq);chunks.append(h)
 h=torch.cat(chunks);assert len(h)==len(r.sequence)
 np.savez(cache/f'{r.sequence_sha256}.npz',mean=h.mean(0).numpy(),std=h.std(0,unbiased=False).numpy())
 if i%20==0:
  atomic_json(OUT/f'status{suffix}.json',{'done':len(records)-len(pending)+i+1,'total':len(records),'protein':r.protein_id,'updated':time.time()});print('ENCODE',i+1,len(pending),r.protein_id,flush=True)
means=[];std=[]
for r in records:
 with np.load(cache/f'{r.sequence_sha256}.npz') as z:means.append(z['mean']);std.append(z['std'])
np.savez(OUT/f'features{suffix}.npz',mean=np.array(means),std=np.array(std),keys=np.array([r.split+'::'+r.protein_id for r in records]))
atomic_json(OUT/f'status{suffix}.json',{'status':'complete','total':len(records),'updated':time.time()})
print('FEATURES_COMPLETE',flush=True)

