"""Version PHOPT confidence, migrating only confidence-dependent artifacts."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phgeofuse.cache import atomic_json, atomic_text, atomic_torch_save, sha256_file, sha256_text
from phgeofuse.confidence import CONFIDENCE_VERSION, confidence_scale, normalize_pdb_confidence
from phgeofuse.config import load_config
from phgeofuse.io import read_manifest, read_fasta, write_manifest
from phgeofuse.retrieval import record_key
from phgeofuse.saprot import embedding_cache_path
from phgeofuse.structures import select_chain, foldseek_three_di


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT/'experiments/dual_phoptnn_20260924')
    args = ap.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = ROOT/'artifacts/phgeofuse/manifest.csv'
    original = read_manifest(source)
    signature = lambda rs: sorted((r.split, r.protein_id, r.sequence, r.ph_opt, r.sample_weight) for r in rs)
    official = []
    for split, suffix in [('train', 'training'), ('validation', 'validation'), ('test', 'testing')]:
        official.extend(read_fasta(ROOT/f'data/phopt_{suffix}.fasta', split))
    if signature(original) != signature(official):
        raise ValueError('source manifest does not match original PHOPT splits')
    frozen = out/'historical_manifest.csv'
    if frozen.exists() and sha256_file(frozen) != sha256_file(source):
        raise ValueError('historical manifest has changed since experiment freeze')
    if not frozen.exists():
        shutil.copy2(source, frozen)
    config = load_config(ROOT/'configs/phgeofuse_phopt_tuned_v1.yaml')
    for k, v in list(config['paths'].items()):
        config['paths'][k] = str((ROOT/v).resolve())
    for k, v in list(config['data']['splits'].items()):
        config['data']['splits'][k] = str((ROOT/v).resolve())
    config['data'].pop('subsets', None)
    config['data']['dataset_fingerprint'] = sha256_text(sha256_file(source)+CONFIDENCE_VERSION)
    config['structure'].update(foldseek_binary='/home/hetianci/envs/miniforge3/envs/phbench/bin/foldseek')
    config['retrieval'].update(mmseqs_binary='/home/hetianci/envs/miniforge3/envs/phbench/bin/mmseqs', search_threads=8,
                               candidate_k=64,search_sensitivity=7.5)
    config['paths'].update(manifest=str(out/'manifest.csv'), structures=str(out/'structures'),
                           graphs=str(out/'residue_graphs'), embeddings=str(out/'embeddings'),
                           retrieval=str(out/'retrieval.pt'), runs=str(out/'dual_baseline/runs'),
                           predictions=str(out/'dual_baseline/predictions'))
    config['training']['diagnostics'] = {'evaluate_train': True}
    config['training']['trainable_scope'] = 'all'
    atomic_text(out/'baseline.yaml', yaml.safe_dump({k:v for k,v in config.items() if not k.startswith('_')}, sort_keys=False))
    quality, migrated, seen = [], [], {}
    started = time.time()
    for i, old in enumerate(original):
        cachekey = (old.sequence_sha256, old.structure_sha256, old.graph_path)
        if cachekey in seen:
            template, q = seen[cachekey]
            record = copy.copy(old)
            for k in ['structure_path', 'structure_sha256', 'mean_plddt', 'three_di', 'graph_path', 'embedding_path']:
                setattr(record, k, getattr(template, k))
        else:
            record = copy.copy(old)
            source_path = Path(old.structure_path)
            if sha256_file(source_path) != old.structure_sha256:
                raise ValueError(f'structure hash mismatch: {source_path}')
            provenance = json.loads(source_path.with_suffix('.pdb.json').read_text())
            scale = confidence_scale(provenance)
            chain, residues = select_chain(source_path, old.sequence)
            raw = np.array([r.plddt for r in residues])
            if scale == '0_1':
                target = out/'structures'/old.sequence_sha256[:2]/(old.sequence_sha256+'.pdb')
                corrected = normalize_pdb_confidence(source_path.read_text(), scale)
                if not target.exists():
                    atomic_text(target, corrected)
                elif target.read_text() != corrected:
                    raise ValueError(f'versioned structure differs: {target}')
                record.structure_path = str(target)
                record.structure_sha256 = sha256_file(target)
                provenance.update(plddt_scale='0_100', confidence_version=CONFIDENCE_VERSION,
                                  legacy_structure_sha256=old.structure_sha256,
                                  structure_sha256=record.structure_sha256,
                                  raw_plddt_scale=scale, mean_plddt=float((raw*100).mean()))
                atomic_json(target.with_suffix('.pdb.json'), provenance)
                graph = torch.load(old.graph_path, map_location='cpu')
                if graph['metadata']['schema_version'] != '1' or graph['node_features'].shape[1] != 35:
                    raise ValueError('unexpected graph schema; migration must be reviewed')
                if not torch.allclose(graph['plddt'], torch.tensor(raw, dtype=torch.float32), atol=1e-5):
                    raise ValueError('original graph confidence does not match structure')
                graph['plddt'] = torch.tensor(raw*100, dtype=torch.float32)
                graph['node_features'] = graph['node_features'].clone()
                if not torch.allclose(graph['node_features'][:,21], torch.tensor(raw/100, dtype=torch.float32), atol=1e-6):
                    raise ValueError('confidence feature index/schema mismatch')
                graph['node_features'][:,21] = graph['plddt']/100
                graph['metadata'].update(structure_sha256=record.structure_sha256,
                                         confidence_version=CONFIDENCE_VERSION,
                                         migrated_from=sha256_file(old.graph_path))
                gkey = sha256_text(sha256_file(old.graph_path)+record.structure_sha256+CONFIDENCE_VERSION)
                graph_path = out/'residue_graphs'/gkey[:2]/(gkey+'.pt')
                if not graph_path.exists():
                    atomic_torch_save(graph_path, graph)
                record.graph_path = str(graph_path)
                _, record.three_di = foldseek_three_di(target, old.sequence, graph['plddt'], config)
                if record.three_di != old.three_di:
                    record.embedding_path = str(embedding_cache_path(out/'embeddings', old.sequence, record.three_di, config))
                normalized = raw*100
            else:
                if not np.isfinite(raw).all() or (raw<0).any() or (raw>100).any():
                    raise ValueError('invalid confidence range')
                normalized = raw
            record.mean_plddt = float(normalized.mean())
            q = dict(sequence_sha256=record.sequence_sha256, structure_sha256=record.structure_sha256,
                     source=record.structure_source, confidence_version=CONFIDENCE_VERSION,
                     raw_scale=scale, raw_mean=float(raw.mean()), mean_plddt=record.mean_plddt,
                     median_plddt=float(np.median(normalized)), p10_plddt=float(np.quantile(normalized,.1)),
                     fraction_lt50=float((normalized<50).mean()), fraction_lt70=float((normalized<70).mean()),
                     length=len(normalized), unknown_fraction=old.sequence.count('X')/len(old.sequence),
                     chain=chain, legacy_structure_sha256=old.structure_sha256,
                     original_all_3di_masked=set(old.three_di)=={'#'},
                     corrected_all_3di_masked=set(record.three_di)=={'#'})
            seen[cachekey] = (record, q)
        migrated.append(record)
        quality.append(dict(key=record_key(record), split=record.split, **q))
        if i%250 == 0:
            atomic_json(out/'prepare_status.json', dict(pid=os.getpid(), status='running', completed=i,
                        total=len(original), seconds=time.time()-started, updated=time.time()))
            print('CONFIDENCE', i, len(original), flush=True)
    write_manifest(out/'manifest.csv', migrated)
    with (out/'quality.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(quality[0]));writer.writeheader();writer.writerows(quality)
    changes = [(a,b) for a,b in zip(original,migrated) if a.structure_sha256!=b.structure_sha256]
    report = dict(status='complete', pid=os.getpid(), updated=time.time(), confidence_version=CONFIDENCE_VERSION,
                  historical_manifest_sha256=sha256_file(frozen), manifest_sha256=sha256_file(out/'manifest.csv'),
                  quality_sha256=sha256_file(out/'quality.csv'), total=len(migrated), unique_structures=len(seen),
                  corrected_records=len(changes), corrected_unique_structures=len({b.structure_sha256 for a,b in changes}),
                  changed_3di_records=sum(a.three_di!=b.three_di for a,b in changes),
                  pending_embeddings=len({r.embedding_path for r in migrated if not Path(r.embedding_path).exists()}),
                  seconds=time.time()-started)
    atomic_json(out/'confidence_audit.json', report)
    atomic_json(out/'prepare_status.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
