"""Launch one durable local DeltaRef runner; the runner owns the experiment lock."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    from phgeofuse.config import load_config,path
    from phgeofuse.cache import atomic_json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/delta_ref_phopt.yaml')
    parser.add_argument('--output')
    args=parser.parse_args()
    config=load_config(args.config)
    output=Path(args.output).resolve() if args.output else path(config,'paths.output')
    output.mkdir(parents=True,exist_ok=True)
    # Serialize dispatchers; the child then acquires its separate lifetime lock.
    with (output/'launcher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pidfile=output/'runner.pid.json'
        if pidfile.exists():
            old=json.loads(pidfile.read_text());proc=Path('/proc')/str(old['pid'])
            if (proc/'cmdline').exists() and 'phgeofuse.delta_ref' in (proc/'cmdline').read_text().replace('\0',' '):
                raise RuntimeError(f"runner already active: PID {old['pid']}")
        command=[sys.executable,'-B','-u','-m','phgeofuse.delta_ref','run','--config',config['_config_path'],'--output',str(output)]
        log=output/'runner.log'
        env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',
                 HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES='0',PYTHONUNBUFFERED='1')
        with log.open('a') as handle:
            process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,
                                     start_new_session=True,env=env)
        row={'pid':process.pid,'command':command,'log':str(log),'started':time.time(),
             'host':'local','gpu':'CUDA_VISIBLE_DEVICES=0','scheduler':'none'}
        atomic_json(pidfile,row)
        print(json.dumps(row,indent=2))


if __name__=='__main__':main()
