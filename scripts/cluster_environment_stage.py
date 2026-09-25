"""Validate and run one pHenv cluster stage inside an existing allocation.

This does not submit jobs, change resource directives, or infer cluster paths.
Use --dry-run to review the exact command before running an allocated stage.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from localph.phenv_data import sha_file


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['encode','finalize','pretrain'])
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--shard-count',type=int,required=True)
    p.add_argument('--shard-index',type=int)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if a.shard_count<1: p.error('shard count must be positive')
    cert=a.data/'complete.json'; verified=a.data/'verification.json'; records=a.data/'records.csv'
    for path in (cert,verified,records):
        if not path.is_file(): raise FileNotFoundError(f'required prepared dataset artifact: {path}')
    v=json.loads(verified.read_text()); c=json.loads(cert.read_text())
    if not v.get('verified') or v.get('complete_sha256')!=sha_file(cert) or c['records_sha256']!=sha_file(records):
        raise ValueError('prepared dataset lineage/verification differs')
    command=[sys.executable,'-B','-u']
    if a.stage=='encode':
        if a.shard_index is None or not 0<=a.shard_index<a.shard_count: p.error('valid --shard-index required for encode')
        if a.checkpoint is None or not a.checkpoint.is_file(): p.error('existing --checkpoint required for encode')
        command += [str(ROOT/'scripts/cache_environment_tokens.py'),'--records',str(records.resolve()),
          '--certificate',str(cert.resolve()),'--checkpoint',str(a.checkpoint.resolve()),
          '--output',str((a.cache/f'shard_{a.shard_index:05d}').resolve()),
          '--shard-count',str(a.shard_count),'--shard-index',str(a.shard_index)]
    elif a.stage=='finalize':
        command += [str(ROOT/'scripts/finalize_environment_cache.py'),'--records',str(records.resolve()),
          '--certificate',str(cert.resolve()),'--output',str(a.cache.resolve()),'--shard-count',str(a.shard_count)]
    else:
        if a.output is None: p.error('--output required for pretrain')
        if not (a.cache/'complete.json').is_file(): raise ValueError('finalize all shards before pretraining')
        command += [str(ROOT/'scripts/pretrain_environment_encoder.py'),'--data',str(a.data.resolve()),
          '--cache',str(a.cache.resolve()),'--output',str(a.output.resolve())]
    print(json.dumps({'stage':a.stage,'data_scope':c.get('scope'),'data_rows':v.get('rows'),
          'command':command,'display_command':shlex.join(command),'dry_run':a.dry_run,
          'submits_job':False,'changes_scheduler_resources':False},ensure_ascii=False),flush=True)
    if not a.dry_run: subprocess.run(command,cwd=ROOT,check=True)


if __name__=='__main__': main()
