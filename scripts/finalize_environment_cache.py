"""Verify encoded shards and expose a virtual cache without copying token data."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from localph.phenv_data import sha_file,write_json
from localph.environment_training import load_cache


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True,help='Existing parent of shard_00000 etc.')
    p.add_argument('--records',type=Path,required=True)
    p.add_argument('--certificate',type=Path,required=True)
    p.add_argument('--shard-count',type=int,required=True)
    a=p.parse_args()
    if a.shard_count<1: raise ValueError('invalid shard count')
    if (a.output/'complete.json').exists(): raise FileExistsError('cache already finalized')
    cert=json.loads(a.certificate.read_text())
    if sha_file(a.records)!=cert['records_sha256']: raise ValueError('source records differ')
    with a.records.open(newline='') as f: expected={r['key'] for r in csv.DictReader(f)}
    all_keys=set(); shards=[]; shared=None
    for i in range(a.shard_count):
        path=a.output/f'shard_{i:05d}'
        packed,_=load_cache(path)
        protocol=json.loads((path/'protocol.json').read_text())
        if protocol['shard_count']!=a.shard_count or protocol['shard_index']!=i or protocol['certificate_sha256']!=sha_file(a.certificate):
            raise ValueError('shard provenance differs')
        current={k:v for k,v in protocol.items() if k not in ('shard_index','keys','residues')}
        if shared is not None and current!=shared: raise ValueError('shard encoding recipes differ')
        shared=current
        keys=set(packed['keys'])
        if keys&all_keys: raise ValueError('duplicated shard sample')
        all_keys|=keys
        shards.append({'directory':path.name,'complete_sha256':sha_file(path/'complete.json'),'rows':len(keys)})
    if all_keys!=expected: raise ValueError('shards do not cover manifest exactly')
    write_json(a.output/'protocol.json',shared)
    write_json(a.output/'complete.json',{'state':'complete','rows':len(all_keys),'shards':shards,
        'records_sha256':cert['records_sha256'],'protocol_sha256':sha_file(a.output/'protocol.json'),
        'source_code_sha256':sha_file(Path(__file__))})
    print(f'VERIFIED_SHARDED_CACHE {len(all_keys)} sequences',flush=True)


if __name__=='__main__': main()
