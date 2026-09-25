"""Keyed, training-scoped inputs and immutable experiment provenance."""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np

from phgeofuse.cache import atomic_json, sha256_file
from phgeofuse.config import load_config, path
from phgeofuse.io import read_manifest
from phgeofuse.robust_fusion import pool_features, chemistry_features
from .model import finite_array, ph_bins


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_npz(destination, **arrays):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.npz")
    np.savez(temporary, **arrays)
    temporary.replace(destination)


def freeze_json(destination, payload):
    destination = Path(destination)
    if destination.exists():
        if json.loads(destination.read_text()) != payload:
            raise ValueError(f"immutable experiment metadata differs: {destination}")
    else:
        atomic_json(destination, payload)


def assert_disjoint(train_keys, query_keys, train_groups=None, query_groups=None):
    if set(map(str, train_keys)) & set(map(str, query_keys)):
        raise ValueError("query key entered a training/reference set")
    if train_groups is not None and query_groups is not None:
        if set(map(str, train_groups)) & set(map(str, query_groups)):
            raise ValueError("held-out family entered a training/reference set")


def validate_certificate(certificate, fit_keys, query_keys, excluded):
    expected = {"fit_keys": list(map(str, fit_keys)), "query_keys": list(map(str, query_keys)),
                "excluded_folds": sorted(map(int, excluded))}
    for key, value in expected.items():
        if certificate.get(key) != value:
            raise ValueError(f"cached fit provenance differs: {key}")
    assert_disjoint(fit_keys, query_keys)


@dataclass
class DevelopmentData:
    records: list
    keys: np.ndarray
    x: np.ndarray
    embeddings: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    folds: np.ndarray
    splits: np.ndarray
    config: dict
    provenance: dict

    @classmethod
    def load(cls, config_path, test=False):
        config = load_config(config_path)
        # Load only requested feature files. Training never opens test features.
        requested = {"test"} if test else {"train", "validation"}
        records = [r for r in read_manifest(path(config, "paths.manifest")) if r.split in requested]
        if not records or not all(r.status == "ready" for r in records):
            raise ValueError("requested PHOPT records are not all ready")
        for split in requested:
            count = sum(r.split == split for r in records)
            if count != config["protocol"]["expected_counts"][split]:
                raise ValueError(f"PHOPT {split} count differs: {count}")
        keys = np.asarray([r.split + "::" + r.protein_id for r in records])
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate manifest keys")
        feature_files, parts = [], []
        for name in ("esm1v", "esm2"):
            source = path(config, "paths." + name)
            if test:
                source = source.with_name("features_test.npz")
            feature_files.append(source)
            with np.load(source, allow_pickle=False) as z:
                order = {str(k): i for i, k in enumerate(z["keys"])}
                if len(order) != len(z["keys"]):
                    raise ValueError(f"duplicate feature keys: {source}")
                if not set(keys) <= set(order):
                    raise ValueError(f"missing feature rows: {source}")
                index = [order[k] for k in keys]
                parts.append(pool_features(z["mean"][index], z["std"][index], "mean_std"))
        embedding = np.column_stack(parts)
        if embedding.shape != (len(keys), 5120):
            raise ValueError("expected frozen ESM1v+ESM2 5120-dimensional pooling")
        x = np.column_stack((embedding, chemistry_features([r.sequence for r in records])))
        finite_array(x, 2)
        y = np.asarray([r.ph_opt for r in records])
        ph_bins(y)
        folds_path = path(config, "paths.folds")
        f = json.loads(folds_path.read_text())
        if f.get("observed_crossfold_violations") not in (0, [], None):
            raise ValueError("homology folds have reported violations")
        mapping = {"train::" + r["key"]: r for r in f["rows"]}
        groups = np.array([mapping[k]["group"] if k in mapping else "unassigned::" + k for k in keys])
        folds = np.array([mapping[k]["fold"] if k in mapping else -1 for k in keys])
        splits = np.array([r.split for r in records])
        if not test:
            train = splits == "train"
            if set(keys[train]) != set(mapping) or set(folds[train]) != set(range(5)):
                raise ValueError("strict folds do not cover PHOPT training exactly")
            for group in np.unique(groups[train]):
                if len(set(folds[train & (groups == group)])) != 1:
                    raise ValueError("a family occurs in multiple folds")
        files = [path(config, "paths.manifest"), folds_path, *feature_files, Path(config_path).resolve()]
        provenance = {"files": {str(p): sha256_file(p) for p in files},
                      "keys_sha256": stable_hash(keys.tolist()), "test_features_opened": test,
                      "feature_schema": "dual_residue_mean_std_independent_l2_plus_chem25_v1"}
        return cls(records, keys, x, embedding, y, groups, folds, splits, config, provenance)

    @property
    def train(self):
        return np.flatnonzero(self.splits == "train")

    @property
    def validation(self):
        return np.flatnonzero(self.splits == "validation")

    def partition(self, excluded):
        excluded = sorted(set(map(int, excluded)))
        if not excluded or not set(excluded) < set(range(5)):
            raise ValueError("exclude one to four training folds")
        fit = np.flatnonzero((self.splits == "train") & ~np.isin(self.folds, excluded))
        query = np.flatnonzero((self.splits == "train") & np.isin(self.folds, excluded))
        assert_disjoint(self.keys[fit], self.keys[query], self.groups[fit], self.groups[query])
        return fit, query

    def certificate(self, fit, query, excluded):
        assert_disjoint(self.keys[fit], self.keys[query], self.groups[fit], self.groups[query])
        return {"fit_keys": self.keys[fit].tolist(), "query_keys": self.keys[query].tolist(),
                "fit_label_sha256": stable_hash(self.labels[fit].tolist()),
                "fit_groups": self.groups[fit].tolist(), "excluded_folds": sorted(map(int, excluded)),
                "input_provenance": self.provenance}


def read_predictions(source, keys, column="prediction"):
    with Path(source).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    mapping = {r["key"]: float(r[column]) for r in rows}
    if len(mapping) != len(rows) or set(mapping) != set(map(str, keys)):
        raise ValueError(f"prediction keys do not match exactly: {source}")
    return finite_array([mapping[str(k)] for k in keys], 1)


def write_predictions(destination, keys, columns):
    keys = list(map(str, keys))
    if len(set(keys)) != len(keys) or any(len(v) != len(keys) for v in columns.values()):
        raise ValueError("prediction columns must have unique aligned keys")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["key", *columns])
        for i, key in enumerate(keys):
            writer.writerow([key, *[v[i] for v in columns.values()]])
    temporary.replace(destination)
