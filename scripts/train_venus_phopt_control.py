"""Fixed five-seed PHOPT Venus-DREAM control with audited train-only supports.

Uses the repaired Reptile implementation. Cached label-free ESM outputs replace
the frozen encoder in memory; task-head inference and support adaptation match
the normal model. Test labels are evaluated only in the separate release phase.
"""
import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import reptile
from models import pHPredictionModel, ProteinpHDataset
from utils import DistributedMetaBatchSampler, NullSummaryWriter, seed_everything
from utils.distributed import DistributedContext
from phgeofuse.delta_ref.metrics import metrics, seed_summary


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def frozen_json(path, payload):
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError(f'Frozen Venus protocol changed: {path}')
    else:
        write_json(path, payload)


class RequiredCachedEncoder(nn.Module):
    def forward(self, *args, **kwargs):
        raise RuntimeError('A required certified ESM cache item is missing.')


class ReportingReptile(reptile.ReptileTrainer):
    status_file = None
    run_seed = None

    def _broadcast_early_stopping(self, valid_loss, global_step):
        if self.status_file is not None:
            write_json(self.status_file, {
                'state': 'validation', 'pid': os.getpid(), 'updated': time.time(),
                'seed': self.run_seed, 'step': global_step, 'validation_mse': valid_loss,
            })
        return super()._broadcast_early_stopping(valid_loss, global_step)


def verify_inputs(args, config):
    complete = json.loads((args.cache / 'complete.json').read_text())
    protocol = json.loads((args.cache / 'protocol.json').read_text())
    if sha(args.cache / 'protocol.json') != complete['protocol_sha256']:
        raise ValueError('ESM cache protocol changed.')
    if protocol['manifest_sha256'] != sha(args.manifest) or protocol['labels_consumed']:
        raise ValueError('ESM cache uses a different manifest or labels.')
    spec = importlib.util.spec_from_file_location('venus_inputs_audit', ROOT / 'scripts/audit_venus_phopt_inputs.py')
    audit_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit_module)
    audit = audit_module.audit(args.manifest, args.retrieval, args.cache / 'tokens')
    if any(row['missing_representation_keys'] for row in audit['splits'].values()):
        raise ValueError('Some query/support representations are missing.')
    for key, digest in complete['token_hashes'].items():
        if sha(args.cache / 'tokens' / (key + '.pt')) != digest:
            raise ValueError('Certified token content changed: ' + key)
    if config['seeds'] != [0, 1, 2, 3, 42]:
        raise ValueError('Five prescribed seeds required.')
    sources = [Path(__file__), args.config, ROOT / 'reptile.py',
               ROOT / 'scripts/audit_venus_phopt_inputs.py',
               ROOT / 'baseline/EpHod/ephod/models.py',
               ROOT / 'baseline/EpHod/ephod/saved_models/RLAT/RLAT.pt',
               ROOT / 'baseline/EpHod/ephod/saved_models/RLAT/params.json',
               args.manifest, *args.retrieval.glob('*.json'),
               args.cache / 'protocol.json', args.cache / 'complete.json',
               *sorted((ROOT / 'models').glob('*.py')), *sorted((ROOT / 'utils').glob('*.py'))]
    return {
        'method': 'Venus-DREAM Reptile; repaired initialization/evaluation implementation',
        'information_condition': 'extra-task-pretraining' if config['pretrained_task_head'] else 'PHOPT-only',
        'config': config, 'test_used_for_selection': False, 'support_audit': audit,
        'cache': str(args.cache.resolve()), 'retrieval': str(args.retrieval.resolve()),
        'manifest': str(args.manifest.resolve()),
        'source_sha256': {str(p.resolve()): sha(p) for p in sources},
        'cache_only_execution': 'frozen ESM unloaded after model construction; all token content verified',
    }


def make_trainer(args, config, seed, directory, testing=False):
    seed_everything(seed)
    model = pHPredictionModel(
        pretrained=config['pretrained_task_head'], embedding_cache_dir=str(args.cache / 'tokens'),
        embedding_memory_cache_size=config['embedding_memory_cache_size'],
        device='cuda', use_data_parallel=False,
    ).cuda()
    model.ephod_model.esm1v_model = RequiredCachedEncoder()
    torch.cuda.empty_cache()
    seed_everything(seed)
    train = ProteinpHDataset(args.retrieval / 'retrieval_train.json', verbose=False)
    validation = ProteinpHDataset(args.retrieval / 'retrieval_valid.json', verbose=False)
    sampler = DistributedMetaBatchSampler(
        len(train), config['meta_batch_size'], rank=0, world_size=1, shuffle=True, seed=seed,
    )
    common = {'collate_fn': reptile.collate_tasks, 'num_workers': config['num_workers']}
    train_loader = DataLoader(train, batch_sampler=sampler, **common)
    valid_loader = DataLoader(validation, batch_size=config['eval_batch_size'], shuffle=False, **common)
    test_loader = None
    if testing:
        test_loader = DataLoader(
            ProteinpHDataset(args.retrieval / 'retrieval_test.json', verbose=False),
            batch_size=config['eval_batch_size'], shuffle=False, **common,
        )
    trainer = ReportingReptile(
        model, train_loader, valid_loader, test_loader,
        meta_lr=config['meta_lr'], inner_lr=config['inner_lr'], num_epochs=config['num_epochs'],
        writer=NullSummaryWriter(), save_dir=str(directory),
        distributed_context=DistributedContext(False, 0, 0, 1, torch.device('cuda:0')),
        train_batch_sampler=sampler, patience=config['patience'], min_delta=config['min_delta'],
        validate_every=config['validate_every'], support_batch_size=config['support_batch_size'],
        inner_steps=config['inner_steps'], seed=seed,
    )
    trainer.status_file, trainer.run_seed = args.output / 'status.json', seed
    return trainer


def train(args, config):
    protocol = verify_inputs(args, config)
    frozen_json(args.output / 'protocol.json', protocol)
    packages = []
    for seed in config['seeds']:
        directory = args.output / f'seed{seed}'
        directory.mkdir(exist_ok=True)
        completion = directory / 'complete.json'
        if completion.exists():
            info = json.loads(completion.read_text())
            if sha(info['checkpoint']) != info['checkpoint_sha256']:
                raise ValueError('Completed Venus checkpoint changed.')
            packages.append(info)
            continue
        # Preserve incomplete attempts; legacy Reptile does not save exact resume state.
        attempt = directory / f'attempt{len(list(directory.glob("attempt*"))):02d}'
        attempt.mkdir()
        started = time.time()
        write_json(args.output / 'status.json', {'state': 'seed_start', 'seed': seed,
                   'pid': os.getpid(), 'updated': started, 'attempt': str(attempt)})
        trainer = make_trainer(args, config, seed, attempt)
        trainer.train()
        checkpoint = attempt / 'best_model.pth'
        if not checkpoint.exists():
            raise RuntimeError('Training did not save a validation-selected checkpoint.')
        info = {'seed': seed, 'checkpoint': str(checkpoint), 'checkpoint_sha256': sha(checkpoint),
                'best_step': trainer.best_step, 'best_validation_mse': trainer.best_loss,
                'seconds': time.time() - started}
        frozen_json(completion, info)
        packages.append(info)
        del trainer
        torch.cuda.empty_cache()
    frozen_json(args.output / 'release.json', {
        'models': packages, 'protocol_sha256': sha(args.output / 'protocol.json'),
        'test_used_for_selection': False, 'seeds': config['seeds'],
        'information_condition': protocol['information_condition'],
    })
    write_json(args.output / 'status.json', {'state': 'frozen_before_test', 'pid': os.getpid(), 'updated': time.time()})
    print('VENUS_FIVE_SEED_CONTROL_FROZEN', flush=True)


def profile(args, config):
    verify_inputs(args, config)
    trainer = make_trainer(args, config, 42, args.output)
    dataset = trainer.train_loader.dataset
    order = sorted(range(len(dataset)), key=lambda i: sum(map(len, [dataset[i]['opt_seq'], *dataset[i]['env_seqs']])))
    chosen = [order[0], order[len(order) // 2], order[-1]]
    rows = []
    for index in chosen:
        task = dataset[index]
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.time()
        result = trainer._run_local_meta_batch([task])
        torch.cuda.synchronize()
        assert torch.isfinite(result[1]).all()
        rows.append({'query': task['opt_id'], 'query_residues': len(task['opt_seq']),
                     'support_residues': list(map(len, task['env_seqs'])),
                     'seconds': time.time() - started,
                     'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                     'inner_steps': config['inner_steps']})
    write_json(args.output / 'resource_profile.json', {'scope': 'training-task resource probe only', 'tasks': rows})
    print(json.dumps(rows), flush=True)


def evaluate(args, config):
    release = json.loads((args.output / 'release.json').read_text())
    if sha(args.output / 'protocol.json') != release['protocol_sha256']:
        raise ValueError('Venus release protocol changed.')
    protocol = json.loads((args.output / 'protocol.json').read_text())
    if verify_inputs(args, config) != protocol:
        raise ValueError('Venus inputs or code changed before evaluation.')
    destination = args.output / 'followup_test'
    destination.mkdir(exist_ok=False)
    with args.manifest.open() as stream:
        labels = {r['protein_id']: float(r['ph_opt']) for r in csv.DictReader(stream) if r['split'] == 'test'}
    predictions = []
    keys = sorted(labels)
    for item in release['models']:
        if sha(item['checkpoint']) != item['checkpoint_sha256']:
            raise ValueError('Venus model changed after freeze.')
        trainer = make_trainer(args, config, item['seed'], args.output, testing=True)
        trainer.load_model_checkpoint(item['checkpoint'])
        rows = trainer.test()
        values = {r['opt_id']: r['predicted_pH'] for r in rows}
        if len(rows) != len(values) or set(values) != set(labels):
            raise ValueError('Incomplete PHOPT prediction coverage.')
        prediction = np.array([values[k] for k in keys])
        score = metrics(np.array([labels[k] for k in keys]), prediction)
        predictions.append(prediction)
        with (destination / f'seed{item["seed"]}.csv').open('w') as stream:
            writer = csv.writer(stream)
            writer.writerow(['key', 'label', 'prediction'])
            writer.writerows(('test::' + k, labels[k], values[k]) for k in keys)
        write_json(destination / f'seed{item["seed"]}.metrics.json', score)
        del trainer
        torch.cuda.empty_cache()
    write_json(destination / 'results.json', {
        'seeds': config['seeds'], 'information_condition': release['information_condition'],
        'test_role': 'follow-up; historically inspected',
        'summary': seed_summary(np.array([labels[k] for k in keys]), np.array(predictions)),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/venus_phopt_control.json')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'artifacts/phgeofuse/manifest.csv')
    parser.add_argument('--retrieval', type=Path, default=ROOT / 'data/processed/top5/esm2_opt_retrieval')
    parser.add_argument('--phase', choices=['train', 'profile', 'evaluate'], default='train')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.cache = args.cache.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    torch.set_num_threads(config['threads'])
    with (args.output / 'writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        {'train': train, 'profile': profile, 'evaluate': evaluate}[args.phase](args, config)


if __name__ == '__main__':
    main()
