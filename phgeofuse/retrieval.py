from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable

import torch

from .cache import atomic_torch_save
from .config import get, path
from .io import ProteinRecord


def record_key(record: ProteinRecord) -> str:
    return f"{record.split}::{record.protein_id}"


def _safe_key(record: ProteinRecord) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", f"{record.split}__{record.protein_id}")


def _load_mean_embedding(record: ProteinRecord) -> torch.Tensor:
    payload = torch.load(record.embedding_path, map_location="cpu")
    embedding = payload["embedding"].float()
    if embedding.shape[0] != len(record.sequence):
        raise ValueError(f"embedding length mismatch for {record_key(record)}")
    return torch.nn.functional.normalize(embedding.mean(dim=0), dim=0)


def _weighted_label(labels: torch.Tensor, scores: torch.Tensor, temperature: float = 0.1):
    if labels.numel() == 0:
        return math.nan, 0.0, math.nan
    weights = torch.softmax(scores / max(temperature, 1e-6), dim=0)
    value = float((weights * labels).sum())
    variance = float((weights * (labels - value).square()).sum())
    return value, float(scores.max()), variance


class RetrievalStore:
    def __init__(self, payload: dict[str, Any]):
        self.payload = payload
        self.rows = payload.get("rows", {})

    @classmethod
    def load(cls, source: str | Path) -> "RetrievalStore":
        return cls(torch.load(source, map_location="cpu"))

    def features(self, key: str) -> torch.Tensor:
        row = self.rows.get(key)
        if row is None:
            return torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        return torch.tensor(
            [
                row.get("saprot_value", 0.0), row.get("foldseek_value", 0.0),
                row.get("saprot_similarity", 0.0), row.get("foldseek_similarity", 0.0),
                row.get("identity", 0.0), row.get("saprot_variance", 1.0),
                row.get("foldseek_variance", 1.0),
                float(row.get("saprot_available", False)), float(row.get("foldseek_available", False)),
            ],
            dtype=torch.float32,
        )

    def add_queries(self, records: Iterable[ProteinRecord], config: dict[str, Any]) -> None:
        records = list(records)
        if not records:
            return
        training_rows = self.payload.get("training_records", [])
        training = [ProteinRecord(**row) for row in training_rows]
        if not training:
            raise ValueError("retrieval cache does not contain training provenance")
        train_vectors = self.payload["training_vectors"].float()
        labels = self.payload["training_labels"].float()
        queries = torch.stack([_load_mean_embedding(record) for record in records])
        top_k = int(self.payload.get("top_k", 5))
        neighbors = _cosine_neighbors(queries, train_vectors, min(len(training), top_k))
        foldseek_hits = _foldseek_hits(records, training, config)
        identities = _mmseqs_hits(records, training, config)
        training_by_safe = {_safe_key(record): record for record in training}
        for query_index, record in enumerate(records):
            indices, scores = neighbors[query_index]
            saprot_value, saprot_similarity, saprot_variance = _weighted_label(
                labels[indices], scores
            )
            identity = max(
                identities.get((_safe_key(record), _safe_key(training[index])),
                               _positional_identity(record.sequence, training[index].sequence))
                for index in indices.tolist()
            )
            structural = [
                (training_by_safe[target], score)
                for target, score in foldseek_hits.get(_safe_key(record), [])[:top_k]
                if target in training_by_safe
            ]
            if structural:
                fold_value, fold_similarity, fold_variance = _weighted_label(
                    torch.tensor([target.ph_opt for target, _ in structural]),
                    torch.tensor([score for _, score in structural]),
                    temperature=0.2,
                )
            else:
                fold_value, fold_similarity, fold_variance = math.nan, 0.0, math.nan
            self.rows[record_key(record)] = {
                "saprot_value": saprot_value, "saprot_similarity": saprot_similarity,
                "saprot_variance": saprot_variance, "saprot_available": True,
                "foldseek_value": _finite_or_zero(fold_value),
                "foldseek_similarity": fold_similarity,
                "foldseek_variance": _finite_or_one(fold_variance),
                "foldseek_available": bool(structural), "identity": identity,
            }

    @classmethod
    def build(
        cls,
        records: Iterable[ProteinRecord],
        config: dict[str, Any],
        destination: str | Path,
    ) -> "RetrievalStore":
        records = [record for record in records if record.status == "ready"]
        training = [record for record in records if record.split == "train" and math.isfinite(record.ph_opt)]
        if not training:
            raise ValueError("retrieval requires at least one labeled training protein")
        train_vectors = torch.stack([_load_mean_embedding(record) for record in training])
        query_vectors = torch.stack([_load_mean_embedding(record) for record in records])
        top_k = int(get(config, "retrieval.top_k", 5))
        candidate_k = min(len(training), max(top_k + 1, top_k * 2))
        saprot_neighbors = _cosine_neighbors(query_vectors, train_vectors, candidate_k)
        identities = _mmseqs_hits(records, training, config)
        foldseek_hits = _foldseek_hits(records, training, config)
        train_by_safe = {_safe_key(record): record for record in training}
        rows: dict[str, dict[str, Any]] = {}
        labels = torch.tensor([record.ph_opt for record in training], dtype=torch.float32)
        for query_index, record in enumerate(records):
            neighbor_indices, scores = saprot_neighbors[query_index]
            kept_indices, kept_scores = [], []
            for index, score in zip(neighbor_indices.tolist(), scores.tolist(), strict=True):
                target = training[index]
                if record_key(target) == record_key(record):
                    continue
                kept_indices.append(index)
                kept_scores.append(score)
                if len(kept_indices) == top_k:
                    break
            if kept_indices:
                selected_labels = labels[torch.tensor(kept_indices)]
                selected_scores = torch.tensor(kept_scores)
                saprot_value, saprot_similarity, saprot_variance = _weighted_label(
                    selected_labels, selected_scores
                )
                identity = max(
                    identities.get((_safe_key(record), _safe_key(training[index])),
                                   _positional_identity(record.sequence, training[index].sequence))
                    for index in kept_indices
                )
            else:
                saprot_value, saprot_similarity, saprot_variance, identity = math.nan, 0.0, math.nan, 0.0

            structural = []
            for target_key, score in foldseek_hits.get(_safe_key(record), []):
                target = train_by_safe.get(target_key)
                if target is None or record_key(target) == record_key(record):
                    continue
                structural.append((target, score))
                if len(structural) == top_k:
                    break
            if structural:
                fold_labels = torch.tensor([target.ph_opt for target, _ in structural])
                fold_scores = torch.tensor([score for _, score in structural])
                fold_value, fold_similarity, fold_variance = _weighted_label(
                    fold_labels, fold_scores, temperature=0.2
                )
            else:
                fold_value, fold_similarity, fold_variance = math.nan, 0.0, math.nan
            rows[record_key(record)] = {
                "saprot_value": _finite_or_zero(saprot_value),
                "saprot_similarity": saprot_similarity,
                "saprot_variance": _finite_or_one(saprot_variance),
                "saprot_available": bool(kept_indices),
                "foldseek_value": _finite_or_zero(fold_value),
                "foldseek_similarity": fold_similarity,
                "foldseek_variance": _finite_or_one(fold_variance),
                "foldseek_available": bool(structural),
                "identity": identity,
            }
        payload = {
            "schema_version": "1",
            "rows": rows,
            "training_keys": [record_key(record) for record in training],
            "training_vectors": train_vectors.half(),
            "training_labels": labels,
            "training_sequences": [record.sequence for record in training],
            "training_structure_paths": [record.structure_path for record in training],
            "training_records": [
                {
                    "protein_id": record.protein_id, "sequence": record.sequence,
                    "split": record.split, "ph_opt": record.ph_opt, "ec": record.ec,
                    "organism": record.organism, "sample_weight": record.sample_weight,
                    "sequence_sha256": record.sequence_sha256,
                    "structure_path": record.structure_path,
                    "structure_source": record.structure_source,
                    "structure_sha256": record.structure_sha256,
                    "mean_plddt": record.mean_plddt, "three_di": record.three_di,
                    "graph_path": record.graph_path, "embedding_path": record.embedding_path,
                    "status": record.status, "error": record.error,
                }
                for record in training
            ],
            "top_k": top_k,
        }
        atomic_torch_save(destination, payload)
        return cls(payload)


def _cosine_neighbors(query: torch.Tensor, target: torch.Tensor, top_k: int):
    try:
        import faiss

        index = faiss.IndexFlatIP(target.shape[1])
        index.add(target.numpy())
        scores, indices = index.search(query.numpy(), top_k)
        return [(torch.from_numpy(indices[i]), torch.from_numpy(scores[i])) for i in range(len(query))]
    except ImportError:
        rows = []
        batch_size = 256
        for start in range(0, query.shape[0], batch_size):
            similarity = query[start:start + batch_size] @ target.t()
            scores, indices = torch.topk(similarity, k=top_k, dim=1)
            rows.extend((indices[index], scores[index]) for index in range(indices.shape[0]))
        return rows


def _mmseqs_hits(records, training, config):
    binary = shutil.which(str(get(config, "retrieval.mmseqs_binary", "mmseqs")))
    if binary is None:
        return {}
    with tempfile.TemporaryDirectory(prefix="phgeofuse-mmseqs-") as directory:
        root = Path(directory)
        query_fasta, train_fasta = root / "query.fasta", root / "train.fasta"
        query_fasta.write_text("".join(f">{_safe_key(r)}\n{r.sequence}\n" for r in records))
        train_fasta.write_text("".join(f">{_safe_key(r)}\n{r.sequence}\n" for r in training))
        output = root / "hits.tsv"
        command = [binary, "easy-search", str(query_fasta), str(train_fasta), str(output),
                   str(root / "tmp"), "--max-seqs", str(max(20, int(get(config, "retrieval.top_k", 5)) * 4)),
                   "--format-output", "query,target,fident,bits", "-v", "0"]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            return {}
        hits = {}
        for line in output.read_text().splitlines():
            query, target, identity, _ = line.split("\t")[:4]
            value = float(identity)
            hits[(query, target)] = value if value <= 1.0 else value / 100.0
        return hits


def _foldseek_hits(records, training, config):
    binary = shutil.which(str(get(config, "structure.foldseek_binary", "foldseek")))
    required = bool(get(config, "retrieval.require_foldseek", True))
    if binary is None:
        if required:
            raise FileNotFoundError("Foldseek is required to build the structural retrieval index")
        return {}
    with tempfile.TemporaryDirectory(prefix="phgeofuse-retrieval-") as directory:
        root = Path(directory)
        query_dir, train_dir = root / "query", root / "train"
        query_dir.mkdir()
        train_dir.mkdir()
        for record in records:
            os.symlink(Path(record.structure_path).resolve(), query_dir / f"{_safe_key(record)}.pdb")
        for record in training:
            os.symlink(Path(record.structure_path).resolve(), train_dir / f"{_safe_key(record)}.pdb")
        output = root / "hits.tsv"
        command = [binary, "easy-search", str(query_dir), str(train_dir), str(output), str(root / "tmp"),
                   "--max-seqs", str(max(20, int(get(config, "retrieval.top_k", 5)) * 4)),
                   "--format-output", "query,target,bits", "-v", "0"]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            if required:
                raise RuntimeError(f"Foldseek retrieval failed: {result.stderr.strip()}")
            return {}
        hits: dict[str, list[tuple[str, float]]] = {}
        for line in output.read_text().splitlines():
            query, target, raw_score = line.split("\t")[:3]
            score = float(raw_score)
            normalized = score / (score + 100.0)
            hits.setdefault(Path(query).stem, []).append((Path(target).stem, normalized))
        for values in hits.values():
            values.sort(key=lambda item: item[1], reverse=True)
        return hits


def ensure_retrieval_store(records, config, force: bool = False):
    destination = path(config, "paths.retrieval", "artifacts/phgeofuse/retrieval.pt")
    if destination.is_file() and not force:
        store = RetrievalStore.load(destination)
        expected = {record_key(record) for record in records if record.status == "ready"}
        if expected <= set(store.rows):
            return store
    return RetrievalStore.build(records, config, destination)


def _positional_identity(left: str, right: str) -> float:
    length = max(len(left), len(right))
    return sum(a == b for a, b in zip(left, right)) / length if length else 0.0


def _finite_or_zero(value: float) -> float:
    return value if math.isfinite(value) else 0.0


def _finite_or_one(value: float) -> float:
    return value if math.isfinite(value) else 1.0
