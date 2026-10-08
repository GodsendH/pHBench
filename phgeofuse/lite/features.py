from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from phgeofuse.config import get
from phgeofuse.io import ProteinRecord
from phgeofuse.retrieval import RETRIEVAL_FEATURE_NAMES, RetrievalStore, record_key, retrieval_build_signature
from phgeofuse.saprot import embedding_key
from .model import feature_names, matrix


POOLING_VERSION = "mean_std_population_l2_each_fp64_v1"


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def encoder_signature(config) -> dict:
    return {"model": get(config, "model.saprot_model", "westlake-repl/SaProt_650M_AF2"),
            "revision": get(config, "model.saprot_revision"),
            "embedding_dim": int(get(config, "model.embedding_dim", 1280)),
            "pooling": POOLING_VERSION}


def pool_embedding(embedding, embedding_dim: int) -> np.ndarray:
    if torch.is_tensor(embedding):
        embedding = embedding.detach().cpu().float().numpy()
    dense = matrix(embedding, embedding_dim)
    parts = [dense.mean(axis=0), dense.std(axis=0, ddof=0)]
    return np.concatenate([part / max(float(np.linalg.norm(part)), 1e-12) for part in parts])


def pool_records(records: list[ProteinRecord], config) -> np.ndarray:
    dimension = int(get(config, "model.embedding_dim", 1280))
    pooled = []
    for index, record in enumerate(records):
        if record.status != "ready" or not record.embedding_path:
            raise ValueError(f"prepared frozen embedding required for {record_key(record)}")
        if len(record.three_di) != len(record.sequence):
            raise ValueError(f"SaProt requires matching sequence and 3Di: {record_key(record)}")
        source = Path(record.embedding_path).expanduser()
        if not source.is_absolute():
            source = Path(config["_root"]) / source
        payload = torch.load(source, map_location="cpu")
        metadata = payload.get("metadata", {})
        if (metadata.get("sequence_sha256") != record.sequence_sha256
                or metadata.get("embedding_key") != embedding_key(record.sequence, record.three_di, config)):
            raise ValueError(f"frozen embedding provenance differs for {record_key(record)}")
        embedding = payload["embedding"]
        if embedding.shape[0] != len(record.sequence):
            raise ValueError(f"embedding residue count differs for {record_key(record)}")
        pooled.append(pool_embedding(embedding, dimension))
        if (index + 1) % 1000 == 0:
            print(f"Pooled {index + 1}/{len(records)} proteins", flush=True)
    return np.asarray(pooled, dtype=np.float64)


def metadata_state(training: list[ProteinRecord]) -> dict:
    if not training or any(record.split != "train" for record in training):
        raise ValueError("metadata statistics must use only training records")
    return {"organism_counts": dict(Counter(record.organism for record in training))}


def metadata_features(records: list[ProteinRecord], state: dict) -> np.ndarray:
    counts = state.get("organism_counts")
    if not isinstance(counts, dict):
        raise ValueError("missing training-only metadata state")
    rows = []
    for record in records:
        if not np.isfinite(record.mean_plddt) or not 0 <= record.mean_plddt <= 100:
            raise ValueError(f"metadata mode requires pLDDT in [0,100]: {record_key(record)}")
        ec = record.ec.split(".")[0]
        ec_index = int(ec) - 1 if ec.isdigit() and 1 <= int(ec) <= 7 else 7
        one_hot = np.zeros(8)
        one_hot[ec_index] = 1
        rows.append([record.mean_plddt / 100, np.log(len(record.sequence)),
                     np.log1p(counts.get(record.organism, 0)),
                     float(record.structure_source != "alphafold_db"), *one_hot])
    return np.asarray(rows, dtype=np.float64)


def reference_signature(store: RetrievalStore) -> str:
    payload = store.payload
    references = payload.get("training_records", [])
    keys = payload.get("training_keys", [])
    if not references or len(references) != len(keys) or len(set(keys)) != len(keys):
        raise ValueError("retrieval cache lacks complete training provenance")
    expected = [f"{row['split']}::{row['protein_id']}" for row in references]
    if expected != keys or any(row["split"] != "train" for row in references):
        raise ValueError("retrieval reference library must contain only training records")
    labels = payload["training_labels"].double().numpy()
    if not np.allclose(labels, [row["ph_opt"] for row in references], atol=1e-6, rtol=0):
        raise ValueError("retrieval reference labels disagree with its provenance")
    vectors = payload["training_vectors"].detach().cpu().contiguous().numpy()
    if vectors.ndim != 2 or len(vectors) != len(keys) or not np.isfinite(vectors).all():
        raise ValueError("invalid retrieval reference vectors")
    return fingerprint({
        "schema_version": payload.get("schema_version"),
        "build_signature": payload.get("build_signature"),
        "references": [{name: row.get(name) for name in (
            "split", "protein_id", "sequence", "ph_opt", "three_di", "structure_sha256")}
            for row in references],
        "vector_shape": list(vectors.shape), "vector_dtype": str(vectors.dtype),
        "vectors_sha256": hashlib.sha256(vectors.tobytes()).hexdigest(),
    })


def validate_references(store: RetrievalStore, training: list[ProteinRecord]) -> str:
    signature = reference_signature(store)
    references = {record_key(ProteinRecord(**row)): row for row in store.payload["training_records"]}
    if set(references) != {record_key(record) for record in training}:
        raise ValueError("retrieval reference set differs from the allowed training split")
    for record in training:
        row = references[record_key(record)]
        if (row["sequence"] != record.sequence or row["three_di"] != record.three_di
                or not np.isclose(row["ph_opt"], record.ph_opt, atol=1e-6, rtol=0)):
            raise ValueError(f"retrieval training provenance differs: {record_key(record)}")
    return signature


@dataclass
class FeatureBatch:
    keys: np.ndarray
    x: np.ndarray
    names: list[str]


def build_features(records: list[ProteinRecord], config, *, design: str, store=None,
                   state=None, pooled=None, build_queries=False) -> FeatureBatch:
    if not records:
        raise ValueError("no records requested")
    keys = np.asarray([record_key(record) for record in records])
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate feature keys")
    dimension = int(get(config, "model.embedding_dim", 1280))
    names = feature_names(design, dimension)
    pooled = pool_records(records, config) if pooled is None else matrix(pooled, 2 * dimension)
    if len(pooled) != len(records):
        raise ValueError("pooled feature row count differs")
    parts = [pooled]
    if "R" in design:
        if store is None:
            raise ValueError("retrieval design requires a reference cache")
        missing = [record for record in records if record_key(record) not in store.rows]
        if missing and build_queries:
            if store.payload.get("build_signature") != retrieval_build_signature(config):
                raise ValueError("query retrieval settings differ from the fitted reference cache")
            store.add_queries(missing, config)
        rows = []
        for key in keys:
            row = store.rows.get(str(key))
            if row is None or any(name not in row for name in RETRIEVAL_FEATURE_NAMES):
                raise ValueError(f"missing complete retrieval row: {key}")
            rows.append([float(row[name]) for name in RETRIEVAL_FEATURE_NAMES])
        parts.append(np.asarray(rows, dtype=np.float64))
    if "M" in design:
        parts.append(metadata_features(records, state or {}))
    return FeatureBatch(keys, matrix(np.column_stack(parts), len(names)), names)
