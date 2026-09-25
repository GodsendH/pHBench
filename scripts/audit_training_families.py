"""Audit training label ambiguity and observed cross-fold sequence neighbors."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phgeofuse.cache import atomic_json
from phgeofuse.io import read_fasta
from phgeofuse.retrieval import _mmseqs_hits, _safe_key


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / 'experiments/phgeofuse_redesign_20260914/homology_oof'
    records = read_fasta(root / 'data/phopt_training.fasta', 'train')
    foldrows = json.loads((out / 'folds.json').read_text())['rows']
    folds = {r['key']: r['fold'] for r in foldrows}
    groups = {r['key']: r['group'] for r in foldrows}
    by_sequence = defaultdict(list)
    by_family = defaultdict(list)
    for record in records:
        by_sequence[record.sequence].append(record)
        by_family[groups[record.protein_id]].append(record)
    duplicate = []
    irreducible_sse = 0.
    for rs in by_sequence.values():
        labels = np.array([r.ph_opt for r in rs])
        irreducible_sse += float(np.sum((labels - labels.mean()) ** 2))
        if len(rs) > 1:
            duplicate.append({'ids': [r.protein_id for r in rs], 'labels': labels.tolist(),
                              'range': float(np.ptp(labels))})
    group_sizes = np.array([len(rs) for rs in by_family.values()])
    stats = {'train_count': len(records), 'unique_sequences': len(by_sequence),
             'families': len(by_family), 'singleton_families': int((group_sizes == 1).sum()),
             'largest_family': int(group_sizes.max()), 'duplicate_sequence_groups': duplicate,
             'sequence_only_in_sample_rmse_lower_bound': float(np.sqrt(irreducible_sse / len(records))),
             'note': 'Conflicting labels on an identical sequence may reflect different conditions; '
                     'this is a sequence-only in-sample lower bound, not an estimated test noise floor.'}
    stats['families_by_ph_group'] = {}
    for name, predicate in [('acidic', lambda y: y < 6), ('neutral', lambda y: 6 <= y < 8),
                            ('alkaline', lambda y: y >= 8)]:
        selected = [r for r in records if predicate(r.ph_opt)]
        stats['families_by_ph_group'][name] = {'samples': len(selected),
            'families': len({groups[r.protein_id] for r in selected})}
    atomic_json(out / 'training_label_audit.json', stats)
    print(json.dumps(stats), flush=True)

    path = out / 'crossfold_neighbors.json'
    config = {'retrieval': {'require_mmseqs': True, 'candidate_k': len(records),
                           'search_threads': 8, 'search_sensitivity': 7.5}}
    hits = _mmseqs_hits(records, records, config)
    keys = {_safe_key(r): r.protein_id for r in records}
    cross = []
    all_qualified = []
    for (q, t), h in hits.items():
        q, t = keys[q], keys[t]
        if q == t or h.query_coverage < .8 or h.target_coverage < .8:
            continue
        if h.identity >= .3:
            all_qualified.append((q, t))
        if folds[q] == folds[t]:
            continue
        if h.identity >= .2:
            cross.append({'query': q, 'target': t, 'identity': h.identity,
                          'qcov': h.query_coverage, 'tcov': h.target_coverage,
                          'query_fold': folds[q], 'target_fold': folds[t]})
    report = {'config': config, 'hits': len(hits), 'crossfold_hits': cross,
              'thresholds': {str(threshold): {'directed_pairs': sum(h['identity'] >= threshold for h in cross),
                  'queries': len({h['query'] for h in cross if h['identity'] >= threshold})}
                  for threshold in [.2, .3, .5, .9]},
              'note': 'Observed MMseqs hits with >=80% coverage on both sequences. '
                      'Search is heuristic; absent hits do not prove all pairs are below threshold.'}
    atomic_json(path, report)
    parent = {r.protein_id: r.protein_id for r in records}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for rs in by_family.values():
        for r in rs[1:]:
            union(rs[0].protein_id, r.protein_id)
    for q, t in all_qualified:
        union(q, t)
    strict_groups = np.array([find(r.protein_id) for r in records])
    strict_fold = np.empty(len(records), dtype=int)
    for f, (_, indices) in enumerate(GroupKFold(5).split(np.zeros((len(records), 1)), groups=strict_groups)):
        strict_fold[indices] = f
    mapping = {r.protein_id: int(strict_fold[i]) for i, r in enumerate(records)}
    assert all(mapping[q] == mapping[t] for q, t in all_qualified)
    atomic_json(out / 'strict_folds.json', {
        'groups': len(set(strict_groups)), 'source': 'Union of original clusters and all observed '
        'MMseqs >=30% identity, >=80% bidirectional coverage links, including within-fold links.',
        'qualified_directed_pairs': len(all_qualified), 'observed_crossfold_violations': 0,
        'caveat': 'Absence of heuristic search hits does not prove mathematical pairwise separation.',
        'rows': [{'key': r.protein_id, 'group': str(strict_groups[i]), 'fold': int(strict_fold[i])}
                 for i, r in enumerate(records)]})
    atomic_json(out / 'observed_30_links.json', all_qualified)
    print('TRAINING_FAMILY_AUDIT_COMPLETE', json.dumps(report['thresholds']), flush=True)
    print('STRICT_GROUPS', len(set(strict_groups)), flush=True)


if __name__ == '__main__':
    main()
