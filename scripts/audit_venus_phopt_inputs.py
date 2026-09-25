"""Check original PHOPT query/support provenance and representation coverage."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.data_loader import SEQUENCE_MAX_LENGTH
from models.embedding_cache import EmbeddingCache
from models.base_model import ESM1V_MODEL_NAME, ephod_utils


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(manifest, inputs, cache):
    with manifest.open() as stream:
        records = list(csv.DictReader(stream))
    by_id = {r['protein_id']: r for r in records}
    if len(by_id) != len(records):
        raise ValueError('Manifest accessions must be unique.')
    train_ids = {r['protein_id'] for r in records if r['split'] == 'train'}
    expected_counts = {'train': 7124, 'validation': 760, 'test': 1971}
    if Counter(r['split'] for r in records) != expected_counts:
        raise ValueError('Expected original PHOPT 7124/760/1971 split.')
    report = {}
    files = [manifest]
    for split, suffix in [('train', 'train'), ('validation', 'valid'), ('test', 'test')]:
        filename = inputs / f'retrieval_{suffix}.json'
        queries = json.loads(filename.read_text())
        expected = {r['protein_id'] for r in records if r['split'] == split}
        ids = [q['opt_id'] for q in queries]
        if len(ids) != len(set(ids)) or set(ids) != expected:
            raise ValueError(f'{split}: query coverage differs from the manifest.')
        supports = set()
        support_edges = 0
        needed = set()
        counts = Counter()
        missing_cache_keys = set()
        for query in queries:
            rid = query['opt_id']
            if (query['opt_sequence'] != by_id[rid]['sequence'] or
                    abs(float(query['opt_pH']) - float(by_id[rid]['ph_opt'])) > 1e-9):
                raise ValueError(f'{split}: query sequence or label mismatch: {rid}')
            support_ids = query['env_ids']
            if (len(support_ids) != 5 or len(set(support_ids)) != 5 or
                    len(query['env_sequences']) != 5 or len(query['env_pHs']) != 5):
                raise ValueError(f'{split}: expected five distinct supports: {rid}')
            if rid in support_ids or not set(support_ids).issubset(train_ids):
                raise ValueError(f'{split}: supports must be other training accessions: {rid}')
            for sid, sequence, ph in zip(support_ids, query['env_sequences'], query['env_pHs']):
                if (sequence != by_id[sid]['sequence'] or
                        abs(float(ph) - float(by_id[sid]['ph_opt'])) > 1e-9):
                    raise ValueError(f'{split}: support sequence or label mismatch: {sid}')
                supports.add(sid)
                support_edges += 1
            label = float(query['opt_pH'])
            group = 'acid_le4' if label <= 4 else 'alkaline_ge10' if label >= 10 else 'core'
            counts[group] += 1
            if len(query['opt_sequence']) > SEQUENCE_MAX_LENGTH:
                counts['truncated_' + group] += 1
            for sequence in [query['opt_sequence'], *query['env_sequences']]:
                normalized = ephod_utils.replace_noncanonical(sequence[:SEQUENCE_MAX_LENGTH], 'X')
                key = EmbeddingCache.make_key(ESM1V_MODEL_NAME, normalized)
                needed.add(key)
                if not (cache / (key + '.pt')).is_file():
                    missing_cache_keys.add(key)
            normalized = ephod_utils.replace_noncanonical(
                query['opt_sequence'][:SEQUENCE_MAX_LENGTH], 'X',
            )
            key = EmbeddingCache.make_key(ESM1V_MODEL_NAME, normalized)
            if (cache / (key + '.pt')).is_file():
                counts['cached_queries'] += 1
        report[split] = {
            'queries': len(queries), 'sample_coverage': 1.0,
            'support_edges': support_edges, 'unique_support_accessions': len(supports),
            'all_supports_training_only': True, 'self_support_edges': 0,
            'label_and_sequence_matches': True, 'counts': dict(counts),
            'unique_representation_keys_needed': len(needed),
            'missing_representation_keys': len(missing_cache_keys),
            'max_query_length': max(len(q['opt_sequence']) for q in queries),
        }
        files.append(filename)
    return {'splits': report, 'input_sha256': {str(p): sha(p) for p in files}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output directory to preserve evidence.')
    result = audit(
        ROOT / 'artifacts/phgeofuse/manifest.csv',
        ROOT / 'data/processed/top5/esm2_opt_retrieval',
        ROOT / 'data/features/esm1v_t33_650M_UR90S_1',
    )
    result.update({
        'scope': 'original-split support audit; not grouped-fold validation or performance evidence',
        'representation_policy': {
            'truncation_residues': SEQUENCE_MAX_LENGTH,
            'full_sample_coverage_is_not_full_residue_coverage': True,
            'cache_counts_verify_file_presence_only': True,
        },
        'source_sha256': {
            str(p): sha(p) for p in [
                Path(__file__), ROOT / 'models/data_loader.py',
                ROOT / 'models/embedding_cache.py', ROOT / 'models/base_model.py',
                ROOT / 'reptile.py', ROOT / 'dataset_registry.py',
            ]
        },
    })
    args.output.mkdir(parents=True)
    (args.output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
