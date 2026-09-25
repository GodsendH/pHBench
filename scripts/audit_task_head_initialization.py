"""Verify fresh/legacy RLAT state using real local weights and a cached input.

This is an implementation audit on CPU, not a performance experiment. ESM
loading is replaced with an identity module because its cached output is used.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest import mock
import warnings

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import base_model
from models.embedding_cache import EmbeddingCache


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def head_hash(model):
    digest = hashlib.sha256()
    for key, tensor in model.ephod_model.rlat_model.state_dict().items():
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output directory to preserve evidence.')
    started = time.time()
    torch.set_num_threads(2)
    checkpoint = ROOT / 'baseline/EpHod/ephod/saved_models/RLAT/RLAT.pt'
    original = torch.load(checkpoint, map_location='cpu')['model_state_dict']
    protocol_file = ROOT / 'experiments/delta_ref_phopt_20260916/protocol.json'
    frozen_hashes = json.loads(protocol_file.read_text())['source_hashes']
    assert all(file_hash(ROOT / f) == h for f, h in frozen_hashes.items())

    manifest = ROOT / 'artifacts/phgeofuse/manifest.csv'
    cache = ROOT / 'data/features/esm1v_t33_650M_UR90S_1'
    with manifest.open() as stream:
        records = sorted(
            (r for r in csv.DictReader(stream) if r['split'] == 'train'),
            key=lambda r: (len(r['sequence']), r['protein_id']),
        )
    for record in records:
        normalized = base_model.ephod_utils.replace_noncanonical(record['sequence'], 'X')
        key = EmbeddingCache.make_key(base_model.ESM1V_MODEL_NAME, normalized)
        cache_file = cache / (key + '.pt')
        if cache_file.exists():
            break
    else:
        raise FileNotFoundError('No PHOPT training sequence has a residue cache.')
    embedding = torch.load(cache_file, map_location='cpu').float().unsqueeze(0)
    assert embedding.shape == (1, 1280, len(normalized) + 2)
    assert torch.isfinite(embedding).all()
    mask = torch.zeros(1, embedding.shape[-1], dtype=torch.int32)
    mask[:, :len(normalized)] = 1

    def build(pretrained, seed):
        torch.manual_seed(seed)
        with mock.patch.object(
            base_model.models.EpHodModel, 'load_ESM1v_model',
            return_value=(nn.Identity(), None),
        ):
            return base_model.pHPredictionModel(
                pretrained=pretrained, device='cpu', use_data_parallel=False,
            ).eval()

    def prediction(model):
        with torch.inference_mode():
            output = model.ephod_model.rlat_model(embedding, mask)[0]
        assert torch.isfinite(output).all()
        return output

    pretrained = build(True, 0)
    for name, value in pretrained.ephod_model.rlat_model.state_dict().items():
        torch.testing.assert_close(value, original['module.' + name], rtol=0, atol=0)
    original_prediction = prediction(pretrained)
    legacy_state = {
        name: value.detach().cpu().clone()
        for name, value in pretrained.named_parameters() if value.requires_grad
    }
    original_buffers = {
        name: {
            'tracked_batches': int(module.num_batches_tracked),
            'mean_abs': float(module.running_mean.abs().mean()),
            'variance_mean': float(module.running_var.mean()),
        }
        for name, module in pretrained.ephod_model.rlat_model.named_modules()
        if isinstance(module, nn.BatchNorm1d)
    }
    del pretrained, original

    fresh = build(False, 0)
    fresh_hash = head_hash(fresh)
    fresh_prediction = prediction(fresh)
    for module in fresh.ephod_model.rlat_model.modules():
        if isinstance(module, nn.BatchNorm1d):
            assert module.num_batches_tracked.item() == 0
            assert torch.equal(module.running_mean, torch.zeros_like(module.running_mean))
            assert torch.equal(module.running_var, torch.ones_like(module.running_var))
    task_state = fresh.trainable_state_dict()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fresh.load_trainable_state_dict(legacy_state)
    assert any('legacy' in str(w.message) for w in caught)
    torch.testing.assert_close(prediction(fresh), original_prediction, rtol=0, atol=0)
    fresh.load_trainable_state_dict(task_state)
    torch.testing.assert_close(prediction(fresh), fresh_prediction, rtol=0, atol=0)
    del fresh, legacy_state, task_state

    repeat = build(False, 0)
    assert head_hash(repeat) == fresh_hash
    del repeat
    different = build(False, 1)
    different_hash = head_hash(different)
    assert different_hash != fresh_hash
    del different
    assert all(file_hash(ROOT / f) == h for f, h in frozen_hashes.items())
    result = {
        'scope': 'real RLAT checkpoint and cached training-input implementation audit; no performance inference',
        'device': 'cpu', 'torch': torch.__version__,
        'elapsed_seconds': time.time() - started,
        'checks': {
            'pretrained_all_parameters_and_buffers_unchanged': True,
            'fresh_all_batchnorm_statistics_reset': True,
            'same_seed_identical_all_head_state': True,
            'different_seed_different_head_state': True,
            'legacy_parameter_only_prediction_exact': True,
            'fresh_task_state_prediction_exact': True,
            'frozen_main_source_hashes_unchanged': True,
        },
        'original_batchnorm': original_buffers,
        'fresh_head_hashes': {'seed0': fresh_hash, 'seed1': different_hash},
        'input': {'key': 'train::' + record['protein_id'], 'residues': len(normalized)},
        'source_sha256': {
            str(p): file_hash(p) for p in [
                Path(__file__), ROOT / 'models/base_model.py',
                ROOT / 'baseline/EpHod/ephod/models.py', checkpoint,
                checkpoint.parent / 'params.json', cache_file, manifest, protocol_file,
            ]
        },
    }
    args.output.mkdir(parents=True)
    (args.output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
