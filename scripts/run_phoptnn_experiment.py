"""Resume graph preparation, serialize GPU work, train three seeds, and compare."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from phgeofuse.phoptnn_adapter.graphs import atomic_json,digest


def live_identity(pid):
    try:
        base=Path(f'/proc/{pid}');stat=(base/'stat').read_text().split()
        if stat[2]=='Z':return None
        return stat[21],(base/'cmdline').read_bytes().decode().replace('\0',' ')
    except FileNotFoundError:return None


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--experiment',type=Path,required=True)
    ap.add_argument('--wait-graph-pid',type=int)
    ap.add_argument('--wait-training-pid',type=int)
    args=ap.parse_args();out=args.experiment.resolve();os.chdir(ROOT)
    work=out/'phoptnn';work.mkdir(exist_ok=True)
    def state(phase,**extra):
        atomic_json(work/'pipeline_status.json',dict(status='running',phase=phase,pid=os.getpid(),updated=time.time(),**extra))
    def run(phase,command):
        with (work/(phase+'.log')).open('a') as f:
            f.write('\nCOMMAND '+json.dumps(command)+'\n');f.flush()
            child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT)
            while child.poll() is None:
                state(phase,child_pid=child.pid,command=command);time.sleep(10)
        if child.returncode:
            atomic_json(work/'pipeline_status.json',dict(status='failed',phase=phase,exit_code=child.returncode,updated=time.time()))
            raise RuntimeError(f'{phase} failed; see log')
    if args.wait_graph_pid:
        identity=live_identity(args.wait_graph_pid)
        if identity and 'phoptnn_adapter.graphs' not in identity[1]:raise ValueError('unexpected graph process')
        while identity and live_identity(args.wait_graph_pid)==identity:
            state('attached_graph_preparation',child_pid=args.wait_graph_pid);time.sleep(10)
    py=sys.executable
    if args.wait_training_pid:
        identity=live_identity(args.wait_training_pid)
        if identity and ('phoptnn_adapter.train' not in identity[1] or 'seed42' not in identity[1]):
            raise ValueError('unexpected attached training process')
        while identity and live_identity(args.wait_training_pid)==identity:
            state('attached_seed42_training',child_pid=args.wait_training_pid);time.sleep(10)
        if not (work/'seed42/results.json').exists():
            raise RuntimeError('attached seed42 exited without completed training; inspect its error log')
    run('graph_retry',[py,'-u','-m','phgeofuse.phoptnn_adapter.graphs','--manifest',str(out/'manifest.csv'),
                       '--output',str(out/'atoms'),'--workers','12'])
    rows=list(csv.DictReader((out/'atoms/atom_manifest.csv').open()))
    if len(rows)!=9855 or any(r['status']!='ready' for r in rows):
        raise ValueError('review incomplete graph coverage before starting same-split training')
    # Wait on the actual live GPU producer, not on a stale progress marker.
    while True:
        pipeline=json.loads((out/'dual_baseline/pipeline_status.json').read_text())
        if pipeline['status']=='complete':break
        if pipeline['status']=='failed':raise RuntimeError('Dual pipeline failed')
        pid=pipeline.get('child_pid')
        parent=live_identity(pipeline['pid'])
        child=live_identity(pid) if pid else None
        if parent is None:raise RuntimeError('Dual pipeline producer is no longer alive')
        if pipeline.get('phase')=='refit_dual' and child and 'refit_corrected_dual.py' in child[1]:
            # Its remaining work is CPU only; the prior evaluation child has exited.
            if all((out/'dual_baseline'/f'{s}.csv').exists() for s in ['validation','test']):break
        state('waiting_for_dual_gpu',producer_pid=pipeline['pid'],child_pid=pid);time.sleep(10)
    seeds=[42,0,1]
    atomic_json(work/'protocol.json',dict(seeds=seeds,manifest_sha256=digest(out/'atoms/atom_manifest.csv'),
                model='upstream EGNN Best_hp row 6',selection='validation RMSE',test_used_for_selection=False,
                precision='float32',epochs=1000,patience=30,split_counts={'train':7124,'validation':760,'test':1971}))
    for seed in seeds:
        target=work/f'seed{seed}'
        if (target/'results.json').exists():
            completed=json.loads((target/'results.json').read_text())
            if completed['config']['manifest_sha256']!=digest(out/'atoms/atom_manifest.csv'):
                raise ValueError('completed seed belongs to another manifest')
            run(f'compare_through_seed{seed}',[py,'-u',str(ROOT/'scripts/analyze_dual_phoptnn.py'),
                '--experiment',str(out),'--seeds',*[str(s) for s in seeds[:seeds.index(seed)+1]],'--dual-version','historical'])
            continue
        command=[py,'-u','-m','phgeofuse.phoptnn_adapter.train','--manifest',str(out/'atoms/atom_manifest.csv'),
                 '--output',str(target),'--seed',str(seed),'--workers','0']
        if (target/'last.pt').exists():command.append('--resume')
        run(f'train_seed{seed}',command)
        run(f'compare_through_seed{seed}',[py,'-u',str(ROOT/'scripts/analyze_dual_phoptnn.py'),
            '--experiment',str(out),'--seeds',*[str(s) for s in seeds[:seeds.index(seed)+1]],'--dual-version','historical'])
    while not all((out/'dual_refit'/f'{s}.csv').exists() for s in ['validation','test']):
        pipeline=json.loads((out/'dual_baseline/pipeline_status.json').read_text())
        if pipeline['status']=='failed' or live_identity(pipeline['pid']) is None:
            raise RuntimeError('corrected Dual outputs missing and producer is not running')
        state('waiting_for_dual_predictions',producer_pid=pipeline['pid']);time.sleep(10)
    run('complementarity',[py,'-u',str(ROOT/'scripts/analyze_dual_phoptnn.py'),'--experiment',str(out)])
    atomic_json(work/'pipeline_status.json',dict(status='complete',pid=os.getpid(),updated=time.time(),seeds=seeds,
                report=str(out/'complementarity_historical/REPORT_ZH.md')))


if __name__=='__main__':main()
