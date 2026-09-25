"""Fresh label-free Ankh-base encoding from pinned local general weights."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, T5EncoderModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phgeofuse.cache import atomic_json
from phgeofuse.io import read_manifest


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = root / 'experiments/phgeofuse_redesign_20260914/ankh_masked'
    out.mkdir(exist_ok=True)
    snapshot = Path.home() / '.cache/huggingface/hub/models--ElnaggarLab--ankh-base/snapshots/d99cb6b966530dfc2ae96bc69d9255c2a07308b0'
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, use_fast=True)
    model = T5EncoderModel.from_pretrained(str(snapshot), local_files_only=True,
                                           torch_dtype=torch.float32).cuda().eval()
    model.requires_grad_(False)
    identity = {'repo': 'ElnaggarLab/ankh-base', 'revision': snapshot.name,
        'weights_sha256': digest(snapshot / 'pytorch_model.bin'),
        'tokenizer_sha256': digest(snapshot / 'tokenizer.json'),
        'config_sha256': digest(snapshot / 'config.json'),
        'pooling': 'per-residue mean and population std; excludes special/padding tokens; '
                   'consecutive chunks <=1022 residues; token word IDs checked',
        'dtype': 'float32_with_bfloat16_autocast', 'labels_used': False}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = out / fingerprint
    cache.mkdir(exist_ok=True)
    records = [r for r in read_manifest(root / 'artifacts/phgeofuse/manifest.csv')
               if r.split in ('train', 'validation')]
    selected = sorted(records, key=lambda r: len(r.sequence))
    if args.smoke:
        selected = [selected[0], selected[-1]]
    for i, r in enumerate(selected):
        path = cache / (r.sequence_sha256 + '.npz')
        if not path.exists():
            hidden = []
            for start in range(0, len(r.sequence), 1022):
                chunk = r.sequence[start:start+1022]
                tokens = tokenizer(list(chunk), is_split_into_words=True, add_special_tokens=True,
                                   return_tensors='pt', return_special_tokens_mask=True)
                words = tokens.word_ids()
                indices = [j for j, word in enumerate(words) if word is not None]
                # Any tokenizer that splits/merges residues must be handled explicitly.
                assert [words[j] for j in indices] == list(range(len(chunk))), words
                inputs = {k: v.cuda() for k, v in tokens.items()
                          if k in ['input_ids', 'attention_mask']}
                with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    h = model(**inputs).last_hidden_state[0, indices].float().cpu()
                hidden.append(h)
            values = torch.cat(hidden)
            assert len(values) == len(r.sequence) and torch.isfinite(values).all()
            with path.with_suffix('.tmp').open('wb') as handle:
                np.savez(handle, mean=values.mean(0).numpy(), std=values.std(0, unbiased=False).numpy())
            path.with_suffix('.tmp').replace(path)
        if i % 100 == 0 or i+1 == len(selected):
            row = {'status': 'running', 'pid': os.getpid(), 'updated': time.time(),
                   'count': len(selected), 'complete': i+1, 'length': len(r.sequence)}
            atomic_json(out / 'status.json', row)
            print(json.dumps(row), flush=True)
    if args.smoke:
        print('ANKH_SMOKE_COMPLETE', flush=True)
        return
    mean, std = [], []
    for r in records:
        with np.load(cache / (r.sequence_sha256 + '.npz')) as f:
            mean.append(f['mean'])
            std.append(f['std'])
    np.savez(out / 'features.npz', mean=np.array(mean), std=np.array(std),
             keys=np.array([r.split + '::' + r.protein_id for r in records]))
    atomic_json(out / 'provenance.json', {**identity, 'count': len(records),
        'splits': ['train', 'validation'], 'script_sha256': digest(Path(__file__))})
    atomic_json(out / 'status.json', {'status': 'complete', 'pid': os.getpid(), 'updated': time.time()})
    print('ANKH_FEATURES_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
