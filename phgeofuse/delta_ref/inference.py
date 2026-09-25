"""Prediction without query labels, using cached features or sequence input.

The preserved full predictor remains an explicit dependency of the package.
Its sequence, structure and retrieval computations are not replaced by the
difference head. All new sequences retain every residue during PLM pooling.
"""
from __future__ import annotations

import copy
import gc
import json
import os
from pathlib import Path
import time
import numpy as np
import torch

from phgeofuse.cache import atomic_json, sha256_file
from phgeofuse.config import load_config, path
from phgeofuse.robust_fusion import pool_features, chemistry_features
from .data import atomic_npz, freeze_json, read_predictions, stable_hash, write_predictions
from .model import finite_array
from .training import load_bundle


def baseline_dependencies(config, seed):
    root = Path(config['_root'])
    research = path(config, 'paths.source_experiment')
    original = path(config, 'paths.baseline_experiment')
    checkpoint = original / f'seed{seed}/runs' / f'phgeofuse_phopt_homology_gate_v3_shrinkage_frozen_seed{seed}/best_calibrated.pt'
    bundle = research / 'dual_candidate_float64'
    files = [checkpoint, bundle / 'model.json', root / 'artifacts/phgeofuse/retrieval.pt']
    model = json.loads((bundle / 'model.json').read_text())
    files += [bundle / name for name in model['file_hashes']]
    return {'seed': seed, 'checkpoint': str(checkpoint), 'dual_bundle': str(bundle),
            'retrieval': str(files[2]), 'files': {str(p): sha256_file(p) for p in files}}


def verify_dependencies(dependencies):
    for name, digest in dependencies['files'].items():
        if sha256_file(Path(name)) != digest:
            raise ValueError(f'preserved baseline dependency changed: {name}')


def attach_baseline(bundle, config, seed):
    bundle = Path(bundle)
    dependencies = baseline_dependencies(config, seed)
    freeze_json(bundle / 'baseline.json', dependencies)
    saved = json.loads((bundle / 'model.json').read_text())
    saved['baseline_manifest_sha256'] = sha256_file(bundle / 'baseline.json')
    atomic_json(bundle / 'model.json', saved)


def _prediction_metadata(bundle, keys, source, elapsed, result):
    return {'model_sha256': sha256_file(Path(bundle) / 'model.json'),
            'keys_sha256': stable_hash(list(map(str, keys))), 'count': len(keys),
            'query_labels_used': False, 'source': source, 'wall_seconds': elapsed,
            'fallback_count': int(result['fallback'].sum()),
            'agreement_is_calibrated_probability': False}


def predict_cached(bundle, features, baseline_csv, output, device='cpu'):
    started = time.monotonic()
    predictor, saved = load_bundle(bundle, device)
    with np.load(features, allow_pickle=False) as z:
        # A label column is rejected rather than accidentally exposed to inference.
        if {'label', 'labels', 'y', 'ph_opt'} & set(z.files):
            raise ValueError('prediction feature file must be label-free')
        keys = np.asarray(z['keys']).astype(str)
        x = finite_array(z['features'], 2)
        if len(keys) != len(x) or len(set(keys)) != len(keys):
            raise ValueError('unique aligned feature keys required')
        if saved['recipe'].get('representation') == 'lora':
            if 'adapter_sha256' not in z or str(z['adapter_sha256'].item()) != saved['adapter_sha256']:
                raise ValueError('LoRA prediction requires features encoded with this fitted adapter')
    baseline = read_predictions(baseline_csv, keys)
    result = predictor.predict(x, baseline, saved['strength'], keys)
    write_predictions(output, keys, result)
    atomic_json(Path(output).with_suffix('.provenance.json'), _prediction_metadata(
        bundle, keys, {'features_sha256': sha256_file(features),
                      'baseline_csv_sha256': sha256_file(baseline_csv)}, time.monotonic()-started, result))
    return result


def frozen_pools(records, bundle_config, cache, device='cuda'):
    """Reproduce the original ESM1v/ESM2 pooling precision and token rules."""
    from transformers import AutoModel, AutoTokenizer
    from transformers.utils.hub import cached_file
    from phgeofuse.sequence_pooling import encode_residue_pools
    import esm
    device = torch.device(device)
    cache = Path(cache)
    torch.set_num_threads(4)
    outputs = []
    provenance = {}
    for name in ('esm1v', 'esm2'):
        if name == 'esm1v':
            identity = bundle_config['esm1v_provenance']
            checkpoint = Path(identity['checkpoint'])
            if sha256_file(checkpoint) != identity['checkpoint_sha256']:
                raise ValueError('ESM1v pretrained weights differ from baseline')
            model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(checkpoint))
            model = model.to(device).eval()
            converter = alphabet.get_batch_converter()
        else:
            repo = bundle_config['esm2_provenance']['repo']
            model_source=Path(cached_file(repo,'config.json',local_files_only=True)).parent
            tokenizer = AutoTokenizer.from_pretrained(str(model_source), local_files_only=True)
            model = AutoModel.from_pretrained(str(model_source), local_files_only=True).to(device).eval()
            # CPU supports float32; record this numerical difference explicitly.
            if device.type == 'cuda':
                model.half()
            identity = {'repo': repo, 'revision': getattr(model.config, '_commit_hash', None) or model_source.name,
                        'schema': 'residue_mean_population_std_chunk1022_v1'}
            if identity['revision'] is None:
                raise ValueError('ESM2 pretrained revision could not be pinned')
        model.requires_grad_(False)
        identity = {**identity, 'inference_device_type': device.type,
                    'precision': 'original_cuda' if device.type == 'cuda' else 'float32_cpu'}
        provenance[name] = identity
        directory = cache / name / stable_hash(identity)
        directory.mkdir(parents=True, exist_ok=True)
        freeze_json(directory / 'provenance.json', identity)
        means, stds = [], []
        for record in records:
            target = directory / (record.sequence_sha256 + '.npz')
            if not target.exists():
                sequence = record.sequence
                if name == 'esm1v':
                    chunks = []
                    for start in range(0, len(sequence), 1022):
                        part = sequence[start:start+1022]
                        _, _, tokens = converter([('query', part)])
                        if tokens.shape[1] != len(part)+2:
                            raise ValueError('ESM1v token/residue mismatch')
                        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                            h = model(tokens.to(device), repr_layers=[33], return_contacts=False)['representations'][33]
                        chunks.append(h[0, 1:len(part)+1].float().cpu().numpy())
                    residues = np.concatenate(chunks).astype(np.float64)
                    if len(residues) != len(sequence):
                        raise ValueError('incomplete ESM1v residue coverage')
                    mean, std = residues.mean(0).astype(np.float32), residues.std(0).astype(np.float32)
                else:
                    mean, std = encode_residue_pools(sequence, tokenizer, model, device)
                atomic_npz(target, mean=mean, std=std, length=len(sequence))
            with np.load(target, allow_pickle=False) as z:
                if int(z['length']) != len(record.sequence):
                    raise ValueError('pooled cache residue coverage differs')
                means.append(z['mean']); stds.append(z['std'])
        outputs.extend([finite_array(means, 2), finite_array(stds, 2)])
        del model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return outputs, provenance


def adapt_features(bundle, records, features, cache, device='cuda'):
    from .lora import PrefixEncoder
    bundle = Path(bundle)
    saved = json.loads((bundle / 'model.json').read_text())
    if sha256_file(bundle / 'adapter.pt') != saved['adapter_sha256']:
        raise ValueError('adapter checksum differs')
    payload = torch.load(bundle / 'adapter.pt', map_location='cpu')
    encoder = PrefixEncoder(payload['settings'], cache, device)
    if encoder.identity != payload['identity']:
        raise ValueError('adapter pretrained revision or precision differs')
    encoder.load_adapter_state(payload['state'])
    encoder.set_adapter_training(False)
    result = np.asarray(features, dtype=float).copy()
    with torch.no_grad():
        for i, record in enumerate(records):
            result[i, 2560:5120] = encoder.encode(record.sequence).cpu().numpy()
    encoder._load_prefix.cache_clear()
    del encoder
    gc.collect()
    if torch.device(device).type == 'cuda':
        torch.cuda.empty_cache()
    return finite_array(result, 2)


def predict_fasta(fasta, bundle, config_path, output, device='cuda', online=False):
    from dataclasses import replace
    from torch.utils.data import DataLoader
    from phgeofuse.io import read_fasta, write_manifest
    from phgeofuse.prepare import prepare_records
    from phgeofuse.saprot import encode_records
    from phgeofuse.retrieval import RetrievalStore, record_key
    from phgeofuse.dataset import ProteinGraphDataset, collate_graphs, move_batch
    from phgeofuse.model import PHGeoFuse
    from phgeofuse.engine import load_checkpoint, _amp_dtype
    from phgeofuse.dual_fusion import DualFusion
    started = time.monotonic()
    os.environ['PATH'] = str(Path(os.sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
    bundle, output = Path(bundle), Path(output)
    predictor, saved = load_bundle(bundle, device)
    if 'baseline_manifest_sha256' not in saved:
        raise ValueError('FASTA inference needs a final bundle with baseline.json; use cached predictions for subset models')
    if sha256_file(bundle / 'baseline.json') != saved['baseline_manifest_sha256']:
        raise ValueError('baseline manifest checksum differs')
    dependencies = json.loads((bundle / 'baseline.json').read_text())
    verify_dependencies(dependencies)
    config = load_config(config_path)
    checkpoint = torch.load(dependencies['checkpoint'], map_location='cpu')
    base_config = copy.deepcopy(checkpoint['config'])
    base_config['_root'] = config['_root']
    base_config.setdefault('runtime', {})['offline'] = not online
    records = [replace(r, ph_opt=float('nan'), ec='', organism='', sample_weight=1.)
               for r in read_fasta(fasta, 'predict', require_labels=False)]
    if not records:
        raise ValueError('empty FASTA')
    keys = [record_key(r) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate FASTA identifiers')
    work = output.parent / (output.stem + '.inputs')
    work.mkdir(parents=True, exist_ok=True)
    manifest = work / 'manifest.csv'
    records, failures = prepare_records(records, base_config, offline=not online, manifest_path=manifest)
    write_manifest(manifest, records)
    if failures:
        raise RuntimeError(f'{len(failures)} sequences failed structure preparation; no rows silently dropped: {manifest}')
    dev = torch.device(device)
    encode_records(records, base_config, dev)
    gc.collect()
    if dev.type == 'cuda':
        torch.cuda.empty_cache()
    retrieval = RetrievalStore.load(dependencies['retrieval'])
    retrieval.add_queries(records, base_config)
    graph = PHGeoFuse(base_config, dev).to(dev)
    load_checkpoint(dependencies['checkpoint'], graph)
    graph.eval()
    loader = DataLoader(ProteinGraphDataset(records, 'predict', retrieval, 'frozen'),
                        batch_size=1, shuffle=False, collate_fn=collate_graphs)
    raw = {}
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, dev)
            with torch.autocast(dev.type, dtype=_amp_dtype(base_config, dev), enabled=dev.type == 'cuda'):
                prediction = graph(batch)['mean'].float().cpu().numpy()
            raw.update(zip(batch['keys'], prediction.tolist()))
    del graph, checkpoint
    gc.collect()
    if dev.type == 'cuda':
        torch.cuda.empty_cache()
    if set(raw) != set(keys):
        raise ValueError('graph predictor changed sequence coverage')
    fusion = DualFusion(dependencies['dual_bundle'])
    pools, pool_provenance = frozen_pools(records, fusion.config, work / 'pools', device)
    r = np.array([retrieval.features(k).numpy() for k in keys], dtype=float)
    baseline = fusion.predict([raw[k] for k in keys], *pools, r, [v.sequence for v in records])['prediction']
    x = np.column_stack([pool_features(*pools[:2], 'mean_std'),
                         pool_features(*pools[2:], 'mean_std'), chemistry_features([v.sequence for v in records])])
    feature_meta = {}
    if saved['recipe'].get('representation') == 'lora':
        x = adapt_features(bundle, records, x, work / 'prefix', device)
        feature_meta['adapter_sha256'] = saved['adapter_sha256']
    atomic_npz(work / 'features.npz', keys=np.array(keys), features=x, **feature_meta)
    write_predictions(work / 'baseline.csv', keys, {'prediction': baseline})
    result = predictor.predict(x, baseline, saved['strength'], keys)
    write_predictions(output, keys, result)
    atomic_json(output.with_suffix('.provenance.json'), _prediction_metadata(bundle, keys,
        {'fasta_sha256': sha256_file(fasta), 'baseline': dependencies, 'pooling': pool_provenance,
         'all_residues_covered': True, 'structure_assisted': True}, time.monotonic()-started, result))
    return result
