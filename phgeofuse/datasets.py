from __future__ import annotations

from pathlib import Path
import hashlib
import json

from dataset_registry import dataset_fasta_paths, normalize_dataset_name


def apply_dataset(config: dict, dataset: str) -> dict:
    name = normalize_dataset_name(dataset)
    config["_dataset"] = name
    if name == "phopt":
        return config

    project_root = Path(config["_root"])
    split_paths = dataset_fasta_paths(project_root, name)
    missing = [str(source) for source in split_paths.values() if not source.is_file()]
    if missing:
        raise FileNotFoundError(
            f"dataset {name} has not been generated; missing files: {missing}"
        )

    data = config.setdefault("data", {})
    data["manifest"] = None
    data["splits"] = {
        split: str(source.relative_to(project_root))
        for split, source in split_paths.items()
    }
    data.pop("subsets", None)
    metadata_path = split_paths['train'].parent / 'metadata.json'
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get('protocol') in {'fixed_test_removal_v1', 'fixed_test_removal_v2', 'fixed_test_removal_v3'}:
            hashes = {s: hashlib.sha256(p.read_bytes()).hexdigest() for s, p in split_paths.items()}
            if hashes != metadata['file_hashes']:
                raise ValueError(f'Dataset {name} FASTAs differ from locked metadata; regenerate the dataset')
            fingerprint_data = hashes if metadata['protocol'] == 'fixed_test_removal_v1' else {'protocol': metadata['protocol'], 'hashes': hashes}
            data['dataset_fingerprint'] = hashlib.sha256(json.dumps(fingerprint_data, sort_keys=True).encode()).hexdigest()

    artifact_root = Path("artifacts") / "phgeofuse" / "datasets" / name
    paths = config.setdefault("paths", {})
    paths["manifest"] = str(artifact_root / "manifest.csv")
    paths["retrieval"] = str(artifact_root / "retrieval.pt")
    paths["runs"] = str(artifact_root / "runs")
    paths["predictions"] = str(artifact_root / "predictions")
    return config


def validate_fixed_test_inputs(records, config, checkpoint=None):
    """Reject stale manifests and task checkpoints for the versioned experiment."""
    expected = config.get('data', {}).get('dataset_fingerprint')
    if not expected:
        return
    from .io import records_from_config
    def signature(items):
        return sorted((r.split, r.protein_id, r.sequence, r.ph_opt, r.sample_weight) for r in items)
    if signature(records) != signature(records_from_config(config)):
        raise ValueError('Manifest does not match the fixed-test dataset FASTAs; rebuild it')
    if checkpoint:
        import torch
        previous = torch.load(checkpoint, map_location='cpu')
        actual = previous.get('config', {}).get('data', {}).get('dataset_fingerprint')
        if actual != expected:
            raise ValueError('Task checkpoint belongs to a different or unversioned dataset. Train a fresh base model on this dataset first.')
