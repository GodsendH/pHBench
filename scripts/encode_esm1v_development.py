"""Fresh label-free ESM1v features from local general pretrained weights."""
import os
os.environ['OMP_NUM_THREADS'] = '4'
import argparse
import hashlib
import json
import time
import sys
from pathlib import Path

import numpy as np
import torch
import esm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phgeofuse.io import read_manifest
from phgeofuse.cache import atomic_json


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=None, help='Smoke check only; does not write final features')
    parser.add_argument('--test', action='store_true', help='Encode test only after verified candidate freeze')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = root / 'experiments/phgeofuse_redesign_20260914/esm1v_masked'
    out.mkdir(exist_ok=True)
    if args.test:
        bundle = root / 'experiments/phgeofuse_redesign_20260914/dual_candidate_float64'
        verified = json.loads((bundle / 'verification.json').read_text())
        if verified['model_sha256'] != file_hash(bundle / 'model.json'):
            raise ValueError('candidate freeze verification differs')
    output_name = 'features_test.npz' if args.test else 'features.npz'
    status_name = 'status_test.json' if args.test else 'status.json'
    provenance_name = 'provenance_test.json' if args.test else 'provenance.json'
    splits = ['test'] if args.test else ['train', 'validation']
    torch.set_num_threads(4)
    checkpoint = Path.home() / '.cache/torch/hub/checkpoints/esm1v_t33_650M_UR90S_1.pt'
    identity = {'model': 'esm1v_t33_650M_UR90S_1', 'checkpoint_sha256': file_hash(checkpoint),
                'schema': 'residue_mean_std_acid_DE_basic_HKR_chunk1022_v1',
                'model_dtype': 'float32_autocast_float16', 'labels_used': False, 'splits': ['train', 'validation']}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = out / fingerprint
    cache.mkdir(exist_ok=True)
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(checkpoint))
    model = model.cuda().eval()
    model.requires_grad_(False)
    convert = alphabet.get_batch_converter()
    records = [r for r in read_manifest(root / 'artifacts/phgeofuse/manifest.csv')
               if r.split in splits]
    selected = sorted(records, key=lambda r: len(r.sequence))
    if args.limit:
        selected = selected[:args.limit]
    for i, record in enumerate(selected):
        target = cache / f'{record.sequence_sha256}.npz'
        if not target.exists():
            chunks = []
            for start in range(0, len(record.sequence), 1022):
                chunk = record.sequence[start:start+1022]
                _, _, tokens = convert([('protein', chunk)])
                assert tokens.shape[1] == len(chunk) + 2
                with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float16):
                    h = model(tokens.cuda(), repr_layers=[33], return_contacts=False)['representations'][33]
                    chunks.append(h[0, 1:len(chunk)+1].float().cpu())
            residues = torch.cat(chunks).numpy().astype(np.float64)
            assert len(residues) == len(record.sequence) and np.isfinite(residues).all()
            features = {'mean': residues.mean(0), 'std': residues.std(0)}
            for name, letters in [('acid', 'DE'), ('basic', 'HKR')]:
                mask = np.array([a in letters for a in record.sequence])
                features[name] = residues[mask].mean(0) if mask.any() else np.zeros(residues.shape[1])
            with target.with_suffix('.tmp').open('wb') as handle:
                np.savez(handle, **{k: v.astype(np.float32) for k, v in features.items()})
            target.with_suffix('.tmp').replace(target)
        if i % 100 == 0 or i + 1 == len(selected):
            status = {'status': 'running', 'pid': os.getpid(), 'updated': time.time(),
                      'complete': i + 1, 'count': len(selected), 'length': len(record.sequence)}
            atomic_json(out / status_name, status)
            print(json.dumps(status), flush=True)
    if args.limit:
        print('ESM1V_SMOKE_COMPLETE', flush=True)
        return
    features = {k: [] for k in ['mean', 'std', 'acid', 'basic']}
    for r in records:
        with np.load(cache / f'{r.sequence_sha256}.npz') as data:
            for k in features:
                features[k].append(data[k])
    np.savez(out / output_name, **{k: np.array(v) for k, v in features.items()},
             keys=np.array([r.split + '::' + r.protein_id for r in records]))
    atomic_json(out / provenance_name, {**identity, 'splits': splits, 'count': len(records),
                'checkpoint': str(checkpoint), 'script_sha256': file_hash(Path(__file__)),
                'note': 'Only residue tokens; no CLS/EOS/padding. All long-sequence residues '
                        'are covered by consecutive chunks. Acid/basic pooling is label-free.'})
    atomic_json(out / status_name, {'status': 'complete', 'pid': os.getpid(), 'updated': time.time()})
    print('ESM1V_FEATURES_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
