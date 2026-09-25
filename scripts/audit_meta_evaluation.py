"""Measure old stochastic and corrected deterministic evaluation on one train task."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from unittest import mock

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import reptile
from models import base_model
from models.embedding_cache import EmbeddingCache
from utils.distributed import DistributedContext, NullSummaryWriter


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output directory to preserve evidence.')
    started = time.time()
    torch.set_num_threads(2)
    retrieval = ROOT / 'data/processed/top5/esm2_opt_retrieval/retrieval_train.json'
    cache = ROOT / 'data/features/esm1v_t33_650M_UR90S_1'

    def cache_file(sequence):
        sequence = base_model.ephod_utils.replace_noncanonical(sequence, 'X')
        return cache / (EmbeddingCache.make_key(base_model.ESM1V_MODEL_NAME, sequence) + '.pt')

    candidates = [
        row for row in json.loads(retrieval.read_text())
        if all(cache_file(s).is_file() for s in [row['opt_sequence'], *row['env_sequences']])
    ]
    row = min(candidates, key=lambda r: (sum(map(len, [r['opt_sequence'], *r['env_sequences']])), r['opt_id']))
    task = {
        'opt_id': row['opt_id'], 'opt_seq': row['opt_sequence'],
        'opt_pH': torch.tensor(float(row['opt_pH'])),
        'env_ids': row['env_ids'], 'env_seqs': row['env_sequences'],
        'env_pHs': torch.tensor([float(v) for v in row['env_pHs']]),
    }
    torch.manual_seed(42)
    with mock.patch.object(base_model.models.EpHodModel, 'load_ESM1v_model', return_value=(nn.Identity(), None)):
        model = base_model.pHPredictionModel(
            pretrained=True, embedding_cache_dir=str(cache), device='cpu', use_data_parallel=False,
        )
    with tempfile.TemporaryDirectory() as temp:
        trainer = reptile.ReptileTrainer(
            model, None, [[task]], [[task]], meta_lr=1.0, inner_lr=0.001,
            num_epochs=1, writer=NullSummaryWriter(), save_dir=temp,
            distributed_context=DistributedContext(False, 0, 0, 1, torch.device('cpu')),
            train_batch_sampler=None, support_batch_size=1, inner_steps=1, seed=42,
        )
        initial = trainer._snapshot_trainable_vector()
        old_predictions = []
        for seed in [0, 1, 42]:
            torch.manual_seed(seed)
            trainer.support_generator.manual_seed(seed)
            trainer._load_trainable_vector(initial)
            model.train()
            trainer._adapt_current_model(task)
            old_predictions.append(trainer._query_task(task)[1])
        trainer._load_trainable_vector(initial)

        corrected_predictions = []
        checks = []
        # Single-task metrics are not performance estimates; only capture output.
        with mock.patch.object(reptile, 'calculate_metrics', return_value={}), mock.patch.object(reptile, 'print_metrics'):
            for seed in [0, 1, 42]:
                torch.manual_seed(seed)
                trainer.support_generator.manual_seed(seed)
                rng = torch.random.get_rng_state()
                generator = trainer.support_generator.get_state()
                modes = [m.training for m in model.modules()]
                prediction = trainer.test()[0]['predicted_pH']
                corrected_predictions.append(prediction)
                check = {
                    'parameters_restored': torch.equal(initial, trainer._snapshot_trainable_vector()),
                    'global_rng_restored': torch.equal(rng, torch.random.get_rng_state()),
                    'support_rng_restored': torch.equal(generator, trainer.support_generator.get_state()),
                    'module_modes_restored': modes == [m.training for m in model.modules()],
                }
                assert all(check.values()), check
                checks.append(check)
        assert len(set(corrected_predictions)) == 1
        assert len(set(old_predictions)) > 1
        assert torch.isfinite(torch.tensor(old_predictions + corrected_predictions)).all()

    source_files = [
        Path(__file__), ROOT / 'models/meta_evaluation.py', ROOT / 'models/base_model.py',
        ROOT / 'reptile.py', ROOT / 'maml.py',
        ROOT / 'baseline/EpHod/ephod/saved_models/RLAT/RLAT.pt', retrieval,
        *[cache_file(s) for s in [row['opt_sequence'], *row['env_sequences']]],
    ]
    result = {
        'scope': 'real RLAT implementation audit on one training task; not a performance experiment',
        'device': 'cpu', 'torch': torch.__version__, 'elapsed_seconds': time.time() - started,
        'task': {'key': 'train::' + row['opt_id'], 'support_ids': row['env_ids'],
                 'query_residues': len(row['opt_sequence']), 'inner_steps': 1},
        'information_condition': 'original task-pretrained RLAT; cached ESM1v; pretrained behavior',
        'random_seeds': [0, 1, 42],
        'legacy_train_mode_predictions': old_predictions,
        'corrected_eval_mode_predictions': corrected_predictions,
        'legacy_prediction_range': max(old_predictions) - min(old_predictions),
        'corrected_prediction_range': max(corrected_predictions) - min(corrected_predictions),
        'training_state_checks': checks,
        'source_sha256': {str(p): sha(p) for p in source_files},
    }
    args.output.mkdir(parents=True)
    (args.output / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
