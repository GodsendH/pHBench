"""Finish a live corrected baseline, train v3, calibrate, and export full Dual."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from phgeofuse.cache import atomic_json,atomic_text,sha256_file


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--experiment',type=Path,required=True)
    ap.add_argument('--wait-baseline-pid',type=int)
    args=ap.parse_args();out=args.experiment.resolve();os.chdir(ROOT)
    work=out/'dual_baseline';work.mkdir(exist_ok=True)
    def state(phase,**kw):
        atomic_json(work/'pipeline_status.json',dict(status='running',phase=phase,pid=os.getpid(),updated=time.time(),**kw))
    def run(stage,command):
        done=work/(stage+'.done.json')
        if done.exists():
            previous=json.loads(done.read_text())
            if previous['command']!=command:raise ValueError('stage command drift')
            return
        with (work/(stage+'.log')).open('a') as f:
            child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT)
            while child.poll() is None:
                state(stage,child_pid=child.pid,command=command);time.sleep(10)
        if child.returncode:
            atomic_json(work/'pipeline_status.json',dict(status='failed',phase=stage,exit_code=child.returncode,updated=time.time()))
            raise RuntimeError(f'{stage} failed; see log')
        atomic_json(done,dict(command=command,exit_code=0,completed=time.time()))
    base=yaml.safe_load((out/'baseline.yaml').read_text())
    base_run=Path(base['paths']['runs'])/(base['training']['run_name']+'_frozen_seed42')
    if args.wait_baseline_pid:
        proc=Path(f'/proc/{args.wait_baseline_pid}')
        if proc.exists():
            command=(proc/'cmdline').read_bytes().decode().replace('\0',' ')
            if 'phgeofuse.train' not in command or 'dual_phoptnn_20260924/baseline.yaml' not in command:
                raise ValueError('attached PID is not the expected baseline training process')
            start=(proc/'stat').read_text().split()[21]
            while proc.exists():
                try:
                    current=(proc/'stat').read_text().split()
                except FileNotFoundError:break
                if current[21]!=start or current[2]=='Z':break
                state('attached_baseline',child_pid=args.wait_baseline_pid,process_start=start)
                time.sleep(10)
    import torch
    if not (base_run/'last.pt').exists():raise RuntimeError('baseline has no checkpoint')
    cp=torch.load(base_run/'last.pt',map_location='cpu')
    if not (cp['epoch']>=base['training']['epochs']-1 or cp.get('stale_epochs',0)>=base['training']['early_stopping_patience']):
        raise RuntimeError('baseline stopped before its terminal criterion; resume it')
    v3=yaml.safe_load((ROOT/'configs/phgeofuse_phopt_homology_gate_v3.yaml').read_text())
    v3['paths']=base['paths'];v3['data']=base['data'];v3['structure']=base['structure']
    v3['retrieval'].update(base['retrieval']);v3['training']['seed']=42
    v3['training']['diagnostics']={'evaluate_train':True}
    config=out/'v3.yaml';atomic_text(config,yaml.safe_dump(v3,sort_keys=False))
    v3run=Path(v3['paths']['runs'])/(v3['training']['run_name']+'_frozen_seed42')
    py=sys.executable
    run('v3_train',[py,'-u','-m','phgeofuse.train','--config',str(config),'--dataset','phopt','--init-checkpoint',str(base_run/'best.pt')])
    run('calibrate',[py,'-u','-m','phgeofuse.calibrate','--config',str(config),'--dataset','phopt','--checkpoint',str(v3run/'best.pt')])
    for split in ['validation','test']:
        run('evaluate_'+split,[py,'-u','-m','phgeofuse.evaluate','--config',str(config),'--dataset','phopt',
                              '--checkpoint',str(v3run/'best_calibrated.pt'),'--split',split,'--output',str(work/f'{split}.csv')])
    run('refit_dual',[py,'-u',str(ROOT/'scripts/refit_corrected_dual.py'),'--experiment',str(out)])
    for split in ['validation','test']:
        if not (out/'dual_refit'/f'{split}.csv').exists():raise RuntimeError('full Dual predictions missing')
    atomic_json(work/'pipeline_status.json',dict(status='complete',pid=os.getpid(),updated=time.time(),
                baseline_sha256=sha256_file(base_run/'best.pt'),dual_bundle_sha256=sha256_file(out/'dual_refit/bundle/model.json')))


if __name__=='__main__':main()
