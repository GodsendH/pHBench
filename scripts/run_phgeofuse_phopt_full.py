"""Five independent PHOPT baseline -> v3 -> calibration -> test experiments."""
import argparse, copy, csv, hashlib, json, os, shutil, statistics, subprocess, sys, time
from collections import Counter
from pathlib import Path
import yaml
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phgeofuse.io import read_manifest, read_fasta
from phgeofuse.retrieval import record_key
from phgeofuse.cache import atomic_json
SEEDS = [0, 1, 2, 3, 42]

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def signature(r):
    return (r.split, r.protein_id, r.sequence, r.ph_opt, r.sample_weight)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    report = Path(args.report).resolve()
    report.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / 'artifacts/phgeofuse/manifest.csv'
    records = read_manifest(manifest)
    fasta = []
    sources = {'train':'training','validation':'validation','test':'testing'}
    for split, suffix in sources.items():
        fasta.extend(read_fasta(ROOT / f'data/phopt_{suffix}.fasta', split))
    assert sorted(map(signature, records)) == sorted(map(signature, fasta)), 'manifest differs from PHOPT'
    assert Counter(r.split for r in records) == {'train':7124,'validation':760,'test':1971}
    assert all(r.status == 'ready' for r in records)
    for r in records:
        assert all(Path(p).is_file() for p in (r.graph_path, r.embedding_path, r.structure_path)), r.protein_id
    cache = ROOT / 'artifacts/phgeofuse/retrieval.pt'
    payload = torch.load(cache, map_location='cpu')
    train = [r for r in records if r.split == 'train']
    assert set(payload['rows']) == {record_key(r) for r in records}
    assert payload['training_keys'] == [record_key(r) for r in train]
    assert payload['training_sequences'] == [r.sequence for r in train]
    assert torch.allclose(payload['training_labels'], torch.tensor([r.ph_opt for r in train],dtype=torch.float32))
    preflight = {'dataset':'phopt','counts':dict(Counter(r.split for r in records)), 'manifest_sha256':digest(manifest),'retrieval_sha256':digest(cache),'python':sys.executable,'seeds':SEEDS,'baseline_initialization':'fresh downstream parameters, frozen pretrained SaProt; no task checkpoint','v3_initialization':'same seed PHOPT baseline only'}
    atomic_json(report / 'preflight.json', preflight)
    if args.check_only:
        print(json.dumps(preflight), flush=True)
        return
    inputs = report / 'inputs'; inputs.mkdir(exist_ok=True)
    for source in (manifest, cache):
        destination = inputs / source.name
        if not destination.exists(): shutil.copy2(source, destination)
        assert digest(destination) == digest(source)
    shutil.copy2(__file__, report / 'runner.py')
    subprocess.run(['git','diff'],stdout=(report/'source.diff').open('w'),check=True)
    results = []
    for seed in SEEDS:
        seed_dir = report / f'seed{seed}'; seed_dir.mkdir(exist_ok=True)
        configs = {}
        runs = {}
        for kind, name in [('baseline','phgeofuse_phopt_tuned_v1.yaml'),('v3','phgeofuse_phopt_homology_gate_v3.yaml')]:
            cfg = yaml.safe_load((ROOT/'configs'/name).read_text())
            cfg['data']['splits'] = {k:str(ROOT/f'data/phopt_{v}.fasta') for k,v in sources.items()}
            cfg['data'].pop('subsets',None)
            for key,val in cfg['paths'].items(): cfg['paths'][key] = str(ROOT/val)
            cfg['paths']['manifest'] = str(inputs/'manifest.csv')
            cfg['paths']['retrieval'] = str(inputs/'retrieval.pt')
            cfg['paths']['runs'] = str(seed_dir/'runs')
            cfg['training']['seed'] = seed
            cfg['training']['diagnostics'] = {'evaluate_train':True}
            cfg['training']['trainable_scope'] = 'all' if kind == 'baseline' else 'homology_gate'
            cp = seed_dir / f'{kind}.yaml'
            content = yaml.safe_dump(cfg,sort_keys=False)
            if cp.exists(): assert cp.read_text() == content, 'config drift'
            else: cp.write_text(content)
            configs[kind] = cp
            runs[kind] = seed_dir/'runs'/f'{cfg["training"]["run_name"]}_frozen_seed{seed}'
        stages = [('baseline_train','train','baseline',[]),('v3_train','train','v3',['--init-checkpoint',str(runs['baseline']/'best.pt')]),('calibrate','calibrate','v3',['--checkpoint',str(runs['v3']/'best.pt')]),('test','evaluate','v3',['--checkpoint',str(runs['v3']/'best_calibrated.pt'),'--split','test','--output',str(seed_dir/'test_predictions.csv')])]
        for stage,module,kind,extra in stages:
            marker = seed_dir/f'{stage}.done.json'
            if marker.exists(): continue
            if stage == 'v3_train':
                atomic_json(seed_dir/'initialization.json', {'checkpoint':str(runs['baseline']/'best.pt'),'sha256':digest(runs['baseline']/'best.pt'),'dataset':'phopt','seed':seed})
            last = runs[kind]/'last.pt'
            if module == 'train' and last.exists(): extra = ['--resume',str(last)]
            command = [sys.executable,'-u','-m',f'phgeofuse.{module}','--config',str(configs[kind]),'--dataset','phopt',*extra]
            started = time.time()
            with (seed_dir/f'{stage}.log').open('a') as log:
                log.write('\nCOMMAND '+json.dumps(command)+'\n'); log.flush()
                child = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                while True:
                    code = child.poll()
                    state = {'status':'running' if code is None else ('stage_complete' if code == 0 else 'failed'),'seed':seed,'stage':stage,'pid':child.pid,'runner_pid':os.getpid(),'started':started,'updated':time.time(),'returncode':code,'command':command}
                    atomic_json(report/'status.json',state)
                    if code is not None: break
                    time.sleep(15)
            if code: raise RuntimeError(f'{stage} seed {seed} exited {code}')
            atomic_json(marker,state)
        preds = list(csv.DictReader((seed_dir/'test_predictions.csv').open()))
        assert len(preds) == 1971
        metrics = json.loads((seed_dir/'test_predictions.metrics.json').read_text())
        results.append({'seed':seed,**{k:metrics[k] for k in ('rmse','mae','r2','spearman')}})
        atomic_json(report/'results.json',results)
        with (report/'results.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    aggregate = {k:{'mean':statistics.mean(r[k] for r in results),'std':statistics.stdev(r[k] for r in results)} for k in ('rmse','mae','r2','spearman')}
    atomic_json(report/'summary.json',aggregate)
    atomic_json(report/'status.json',{'status':'complete','seeds':SEEDS,'updated':time.time()})
    print('ALL_FIVE_COMPLETE',flush=True)

if __name__ == '__main__':
    main()
