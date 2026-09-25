"""Sequential pHenv encoding, auxiliary pretraining and fixed PHOPT comparison.

Use the process-group budget supervisor locally. This script never retries a
failed training stage, alters the default model, or starts a cluster job.
"""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from localph.phenv_data import sha_file,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    cert=json.loads((a.data/'complete.json').read_text())
    if cert['state']!='complete' or cert['phopt_labels_consumed'] or sha_file(a.data/'records.csv')!=cert['records_sha256']:
        raise ValueError('pilot data preparation is not certified')
    verification=json.loads((a.data/'verification.json').read_text())
    if not verification['verified'] or verification['complete_sha256']!=sha_file(a.data/'complete.json'):
        raise ValueError('independent pilot verification is missing or stale')
    a.output.mkdir(parents=True,exist_ok=False)
    with (a.output/'pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        python=[sys.executable,'-B','-u']
        stages=[
          ('encoding',python+['scripts/cache_environment_tokens.py','--records',str(a.data/'records.csv'),
              '--certificate',str(a.data/'complete.json'),'--output',str(a.output/'cache')]),
          ('environment_pretraining',python+['scripts/pretrain_environment_encoder.py','--data',str(a.data),
              '--cache',str(a.output/'cache'),'--output',str(a.output/'environment')]),
          ('phopt_comparison',python+['scripts/evaluate_environment_transfer.py','--environment',str(a.output/'environment'),
              '--phopt-cache','experiments/residue_field_phopt_20260917/full_features','--output',str(a.output/'comparison')])]
        source_files=[Path(__file__),ROOT/'localph/environment_transfer.py',ROOT/'localph/environment_training.py']
        source_files += [ROOT/command[3] for _,command in stages]
        source_hashes={str(f.relative_to(ROOT)):sha_file(f) for f in source_files}
        write_json(a.output/'protocol.json',{'data_complete_sha256':sha_file(a.data/'complete.json'),
            'stages':stages,'source_hashes':source_hashes,'default_model_replaced':False,
            'phopt_selection':'fixed recipes and epochs; all outer predictions reported'})
        start=time.monotonic()
        for stage,command in stages:
            if any(sha_file(ROOT/name)!=digest for name,digest in source_hashes.items()):
                raise ValueError('source changed during pipeline')
            write_json(a.output/'status.json',{'state':'running','stage':stage,'seconds':time.monotonic()-start})
            print(json.dumps({'event':'stage_start','stage':stage,'command':command}),flush=True)
            with (a.output/(stage+'.log')).open('w') as log:
                result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                write_json(a.output/'status.json',{'state':'failed','stage':stage,'returncode':result.returncode,'seconds':time.monotonic()-start})
                raise RuntimeError(f'{stage} failed; inspect its log; no automatic retry')
        write_json(a.output/'status.json',{'state':'complete','seconds':time.monotonic()-start,'default_model_replaced':False})
        print('ENVIRONMENT_TRANSFER_PILOT_COMPLETE',flush=True)


if __name__=='__main__': main()
