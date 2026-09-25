"""Shardable, resumable frozen ESM1v residue cache for an audited pHenv manifest.

Only sample keys and sequences enter encoding. Retain all real residues in
float16 after float32 ESM inference, matching the existing PHOPT cache.
"""
import argparse
import csv
import fcntl
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import esm
from localph.phenv_data import sha_file, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records',type=Path,required=True)
    p.add_argument('--certificate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,default=Path.home()/'.cache/torch/hub/checkpoints/esm1v_t33_650M_UR90S_1.pt')
    p.add_argument('--shard-count',type=int,default=1)
    p.add_argument('--shard-index',type=int,default=0)
    a=p.parse_args()
    if not 0<=a.shard_index<a.shard_count:
        raise ValueError('invalid shard selection')
    cert=json.loads(a.certificate.read_text())
    if sha_file(a.records)!=cert['records_sha256']:
        raise ValueError('records differ from certificate')
    with a.records.open(newline='') as f:
        rows=[{'key':r['key'],'sequence':r['sequence']} for r in csv.DictReader(f)]
    if len(set(r['key'] for r in rows))!=len(rows):
        raise ValueError('duplicate sample keys')
    for r in rows:
        if not 32<=len(r['sequence'])<=1022 or set(r['sequence'])-set('ACDEFGHIKLMNPQRSTVWYX'):
            raise ValueError('unsupported sequence; no silent normalization/truncation')
    # Stable length ordering improves reproducibility and balances residue load by interleaving shards.
    rows=sorted(rows,key=lambda r:(len(r['sequence']),r['key']))[a.shard_index::a.shard_count]
    if not rows: raise ValueError('empty shard')
    offsets=np.r_[0,np.cumsum([len(r['sequence']) for r in rows])]
    protocol={'records_sha256':cert['records_sha256'],'certificate_sha256':sha_file(a.certificate),
      'source_code_sha256':sha_file(Path(__file__)),'checkpoint_sha256':sha_file(a.checkpoint),
      'model':'esm1v_t33_650M_UR90S_1','layer':33,'encoder_dtype':'float32','cache_dtype':'float16',
      'token_policy':'every real residue, no BOS/EOS, no padding, no truncation','batch_size':1,
      'shard_count':a.shard_count,'shard_index':a.shard_index,'keys':[r['key'] for r in rows],
      'residues':int(offsets[-1]),'torch':torch.__version__,'labels_consumed':False}
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        protocol_path=a.output/'protocol.json'
        if protocol_path.exists():
            if json.loads(protocol_path.read_text())!=protocol:
                raise ValueError('cache recipe/source changed; use new output')
        else:
            write_json(protocol_path,protocol)
        complete_path=a.output/'complete.json'
        if complete_path.exists():
            done=json.loads(complete_path.read_text())
            for name,digest in done['hashes'].items():
                if sha_file(a.output/name)!=digest: raise ValueError('completed shard changed')
            print('VERIFIED_EXISTING_ENVIRONMENT_CACHE',flush=True); return
        packed_path=a.output/'tokens.npy'; progress_path=a.output/'progress.json'
        written=0
        if progress_path.exists(): written=int(json.loads(progress_path.read_text())['written'])
        if not 0<=written<=len(rows): raise ValueError('invalid progress')
        if written and not packed_path.exists(): raise ValueError('missing partially written cache')
        packed=np.lib.format.open_memmap(packed_path,mode='r+' if packed_path.exists() else 'w+',
                                        dtype=np.float16,shape=(int(offsets[-1]),1280))
        if packed.shape!=(int(offsets[-1]),1280) or packed.dtype!=np.float16:
            raise ValueError('cache dimensions differ')
        index=a.output/'index.npz'
        if not index.exists():
            np.savez(index,keys=np.array(protocol['keys']),offsets=offsets)
        else:
            with np.load(index,allow_pickle=False) as z:
                if not np.array_equal(z['keys'],protocol['keys']) or not np.array_equal(z['offsets'],offsets):
                    raise ValueError('index differs')
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        model,alphabet=esm.pretrained.load_model_and_alphabet_local(str(a.checkpoint))
        model=model.float().cuda().eval().requires_grad_(False)
        convert=alphabet.get_batch_converter()
        start=time.monotonic(); start_index=written; emitted=start
        for i in range(written,len(rows)):
            r=rows[i]
            _,_,tokens=convert([(r['key'],r['sequence'])])
            with torch.inference_mode():
                h=model(tokens.cuda(),repr_layers=[33],return_contacts=False)['representations'][33][0,1:-1]
                h=h.half().cpu().numpy()
            if h.shape!=(len(r['sequence']),1280) or not np.isfinite(h).all():
                raise ValueError('nonfinite or incomplete residue encoding')
            packed[offsets[i]:offsets[i+1]]=h
            if i%100==99 or i+1==len(rows):
                packed.flush()
                write_json(progress_path,{'written':i+1,'last_key':r['key'],'protocol_sha256':sha_file(protocol_path)})
            now=time.monotonic()
            if now-emitted>=30 or i+1==len(rows):
                status={'state':'encoding','written':i+1,'total':len(rows),'resumed_from':start_index,
                        'session_seconds':now-start,'last_length':len(r['sequence'])}
                write_json(a.output/'status.json',status); print(json.dumps(status),flush=True); emitted=now
        packed.flush(); del packed
        complete={'state':'complete','rows':len(rows),'residues':int(offsets[-1]),'resumed_from':start_index,
                  'session_seconds':time.monotonic()-start,'labels_consumed':False,
                  'hashes':{n:sha_file(a.output/n) for n in ['tokens.npy','index.npz','protocol.json']}}
        write_json(complete_path,complete); write_json(a.output/'status.json',complete)
        print(json.dumps(complete),flush=True)


if __name__=='__main__': main()
