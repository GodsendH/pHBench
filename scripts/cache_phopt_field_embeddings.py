"""Cache full-precision ESM1v tokens for controlled EpHod/Venus comparisons.

Only accession, split and sequence columns are consumed. The output includes
BOS/EOS token means used by official EpHod-SVR and per-token Venus inputs.
"""
import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import esm
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.embedding_cache import EmbeddingCache

MODEL = 'esm1v_t33_650M_UR90S_1'


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


def records_from_manifest(manifest):
    with manifest.open() as stream:
        rows = [
            {'key': r['split'] + '::' + r['protein_id'], 'split': r['split'],
             'sequence': r['sequence']}
            for r in csv.DictReader(stream)
        ]
    counts = {s: sum(r['split'] == s for r in rows) for s in ['train', 'validation', 'test']}
    if counts != {'train': 7124, 'validation': 760, 'test': 1971}:
        raise ValueError('Expected original PHOPT 7124/760/1971 manifest.')
    if len({r['key'] for r in rows}) != len(rows):
        raise ValueError('Duplicate sample keys.')
    for row in rows:
        normalized = row['sequence'].translate(str.maketrans({c: 'X' for c in 'BJOUZ'}))
        if not 1 <= len(normalized) <= 1022:
            raise ValueError(f"Unsupported sequence length; no truncation or filtering: {row['key']}")
        if not set(normalized) <= set('ACDEFGHIKLMNPQRSTVWYX'):
            raise ValueError(f"Unexpected sequence characters: {row['key']}")
        row['normalized'] = normalized
        row['cache_key'] = EmbeddingCache.make_key(MODEL, normalized)
    return rows


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = records_from_manifest(args.manifest)
        expected = {
            'model': MODEL, 'checkpoint_sha256': sha(args.checkpoint),
            'manifest_sha256': sha(args.manifest),
            'source_sha256': {str(Path(__file__).resolve()): sha(Path(__file__))},
            'dtype': 'float32', 'autocast': False, 'allow_tf32': False,
            'batch_size': 1, 'token_policy': 'BOS + all residues + EOS; no padding',
            'SVR_pooling': 'mean of every token, matching official batch-size-one inference',
            'labels_consumed': False, 'max_residues': 1022,
            'torch': torch.__version__, 'fair_esm': importlib.metadata.version('fair-esm'),
        }
        protocol = output / 'protocol.json'
        if protocol.exists():
            if json.loads(protocol.read_text()) != expected:
                raise ValueError('Representation protocol changed; use a new output directory.')
        else:
            write_json(protocol, expected)
        unique = {r['cache_key']: r for r in rows}
        ordered = sorted(unique.values(), key=lambda r: (len(r['normalized']), r['cache_key']))
        selected = ordered
        if args.profile:
            selected = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
        elif args.limit is not None:
            selected = ordered[:args.limit]
        torch.set_num_threads(args.threads)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(42)
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(args.checkpoint))
        model = model.to(args.device).float().eval().requires_grad_(False)
        convert = alphabet.get_batch_converter()
        cache = EmbeddingCache(str(output / 'tokens'), max_memory_items=0)
        entries = []
        start = time.time()
        emitted = start
        try:
            for i, row in enumerate(selected):
                step_start = time.time()
                sequence = row['normalized']
                key = row['cache_key']
                tensor = cache.get(key)
                reused = tensor is not None
                if reused:
                    if tensor.shape != (1280, len(sequence) + 2) or not torch.isfinite(tensor).all():
                        raise ValueError(f'Invalid existing token cache: {key}')
                else:
                    _, _, tokens = convert([(row['key'], sequence)])
                    assert tokens.shape == (1, len(sequence) + 2)
                    if args.device == 'cuda':
                        torch.cuda.reset_peak_memory_stats()
                    with torch.inference_mode():
                        tensor = model(
                            tokens.to(args.device), repr_layers=[33], return_contacts=False,
                        )['representations'][33][0].T.contiguous().cpu()
                    if tensor.shape != (1280, len(sequence) + 2) or not torch.isfinite(tensor).all():
                        raise ValueError(f'Invalid encoded input: {row["key"]}')
                    cache.put(key, tensor)
                elapsed = time.time() - step_start
                entry = {
                    'cache_key': key, 'sample_key': row['key'], 'residues': len(sequence),
                    'reused': reused, 'seconds': elapsed,
                    'sha256': sha(output / 'tokens' / (key + '.pt')),
                }
                if args.device == 'cuda' and not reused:
                    entry['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
                    entry['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
                entries.append(entry)
                with (output / 'encoding_events.jsonl').open('a') as stream:
                    stream.write(json.dumps(entry) + '\n')
                now = time.time()
                if args.profile or i == 0 or now - emitted >= 30 or i + 1 == len(selected):
                    status = {
                        'state': 'encoding', 'pid': os.getpid(), 'updated': now,
                        'processed': i + 1, 'selected': len(selected), 'unique_sequences': len(ordered),
                        'elapsed_seconds': now - start, 'last_residues': len(sequence),
                        'last_encode_seconds': elapsed,
                    }
                    write_json(output / 'status.json', status)
                    print(json.dumps(status), flush=True)
                    emitted = now
            if args.profile or args.limit is not None:
                report = {'profile': args.profile, 'partial': True, 'elapsed_seconds': time.time() - start,
                          'encoded': entries, 'unique_sequences': len(ordered)}
                destination = output / f'profile_{os.getpid()}.json'
                write_json(destination, report)
                write_json(output / 'status.json', {'state': 'profile_complete', 'pid': os.getpid(),
                                                   'updated': time.time(), 'report': str(destination)})
                print(json.dumps(report), flush=True)
                return
            del model
            if args.device == 'cuda':
                torch.cuda.empty_cache()
            means = {}
            for row in ordered:
                key = row['cache_key']
                means[key] = cache.get(key).numpy().mean(axis=-1)
            temporary = output / f'features.{os.getpid()}.tmp'
            with temporary.open('wb') as stream:
                np.savez(stream, keys=np.array([r['key'] for r in rows]),
                         token_mean=np.array([means[r['cache_key']] for r in rows]))
            temporary.replace(output / 'features.npz')
            write_json(output / 'complete.json', {
                'sample_count': len(rows), 'unique_sequences': len(ordered), 'labels_consumed': False,
                'protocol_sha256': sha(protocol), 'features_sha256': sha(output / 'features.npz'),
                'token_hashes': {e['cache_key']: e['sha256'] for e in entries},
                'elapsed_seconds': time.time() - start, 'pid': os.getpid(),
            })
            write_json(output / 'status.json', {'state': 'complete', 'pid': os.getpid(), 'updated': time.time()})
            print('FIELD_ESM1V_CACHE_COMPLETE', flush=True)
        except BaseException as exc:
            write_json(output / 'status.json', {'state': 'failed', 'pid': os.getpid(), 'updated': time.time(),
                                               'exception': type(exc).__name__, 'reason': str(exc)})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'artifacts/phgeofuse/manifest.csv')
    parser.add_argument('--checkpoint', type=Path, default=Path.home() / '.cache/torch/hub/checkpoints' / (MODEL + '.pt'))
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error('--limit must be positive')
    run(args)


if __name__ == '__main__':
    main()
