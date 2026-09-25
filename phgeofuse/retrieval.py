from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from .cache import atomic_torch_save
from .config import get, path
from .io import ProteinRecord


RETRIEVAL_SCHEMA_VERSION = "2"
RETRIEVAL_FEATURE_NAMES = (
    "saprot_value",
    "foldseek_value",
    "saprot_similarity",
    "foldseek_similarity",
    "identity",
    "saprot_variance",
    "foldseek_variance",
    "saprot_available",
    "foldseek_available",
    "query_coverage",
    "target_coverage",
    "saprot_hit_fraction",
    "foldseek_hit_fraction",
    "saprot_similarity_margin",
    "foldseek_similarity_margin",
)
RETRIEVAL_FEATURE_DIM = len(RETRIEVAL_FEATURE_NAMES)


@dataclass(frozen=True)
class SequenceHit:
    identity: float
    query_coverage: float
    target_coverage: float
    bits: float


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


def _default_feature_row() -> dict[str, float | bool]:
    return {
        "saprot_value": 0.0,
        "foldseek_value": 0.0,
        "saprot_similarity": 0.0,
        "foldseek_similarity": 0.0,
        "identity": 0.0,
        "saprot_variance": 1.0,
        "foldseek_variance": 1.0,
        "saprot_available": False,
        "foldseek_available": False,
        "query_coverage": 0.0,
        "target_coverage": 0.0,
        "saprot_hit_fraction": 0.0,
        "foldseek_hit_fraction": 0.0,
        "saprot_similarity_margin": 0.0,
        "foldseek_similarity_margin": 0.0,
    }


def _feature_tensor(row: dict[str, Any] | None) -> torch.Tensor:
    values = _default_feature_row()
    if row:
        values.update({name: row.get(name, values[name]) for name in values})
    return torch.tensor(
        [float(values[name]) for name in RETRIEVAL_FEATURE_NAMES],
        dtype=torch.float32,
    )


def _fraction(value: Any, default: float) -> float:
    result = float(default if value is None else value)
    if result > 1.0:
        result /= 100.0
    if not 0.0 <= result <= 1.0:
        raise ValueError("homology identity and coverage must be between 0 and 1")
    return result


def _candidate_k(config: dict[str, Any], training_count: int) -> int:
    top_k = int(get(config, "retrieval.top_k", 5))
    default = 64 if homology_training_enabled(config) else max(20, top_k * 4)
    configured = int(get(config, "retrieval.candidate_k", default))
    return min(training_count, max(top_k + 1, configured))


def retrieval_build_signature(config: dict[str, Any]) -> dict[str, Any]:
    top_k = int(get(config, "retrieval.top_k", 5))
    default_candidate_k = 64 if homology_training_enabled(config) else max(20, top_k * 4)
    return {
        "top_k": top_k,
        "candidate_k": int(get(config, "retrieval.candidate_k", default_candidate_k)),
        "low_homology_identity": _fraction(
            get(config, "retrieval.low_homology_identity", 0.2), 0.2
        ),
        "low_homology_coverage": _fraction(
            get(config, "retrieval.low_homology_coverage", 0.8), 0.8
        ),
        "low_homology_coverage_mode": int(
            get(config, "retrieval.low_homology_coverage_mode", 0)
        ),
        "search_sensitivity": float(get(config, "retrieval.search_sensitivity", 7.5)),
    }


def homology_training_enabled(config: dict[str, Any]) -> bool:
    return bool(get(config, "homology_training.enabled", False))


class RetrievalStore:
    def __init__(self, payload: dict[str, Any]):
        self.payload = payload
        self.rows = payload.get("rows", {})

    @classmethod
    def load(cls, source: str | Path) -> "RetrievalStore":
        return cls(torch.load(source, map_location="cpu"))

    def features(self, key: str, view: str = "normal") -> torch.Tensor:
        if view not in {"normal", "low_homology"}:
            raise ValueError(f"unknown retrieval view: {view}")
        row = self.rows.get(key)
        if row is None:
            return _feature_tensor(None)
        if view == "low_homology":
            row = row.get("low_homology", row)
        return _feature_tensor(row)

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
        self.rows.update(
            _build_retrieval_rows(records, training, train_vectors, labels, config)
        )

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
        labels = torch.tensor([record.ph_opt for record in training], dtype=torch.float32)
        rows = _build_retrieval_rows(records, training, train_vectors, labels, config)
        top_k = int(get(config, "retrieval.top_k", 5))
        payload = {
            "schema_version": RETRIEVAL_SCHEMA_VERSION,
            "dataset_fingerprint": get(config, "data.dataset_fingerprint"),
            "build_signature": retrieval_build_signature(config),
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


def _build_retrieval_rows(
    records: list[ProteinRecord],
    training: list[ProteinRecord],
    train_vectors: torch.Tensor,
    labels: torch.Tensor,
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if not records:
        return {}
    top_k = int(get(config, "retrieval.top_k", 5))
    if top_k < 1:
        raise ValueError("retrieval.top_k must be positive")
    candidate_k = _candidate_k(config, len(training))
    query_vectors = torch.stack([_load_mean_embedding(record) for record in records])
    saprot_neighbors = _cosine_neighbors(query_vectors, train_vectors, candidate_k)
    sequence_hits = _mmseqs_hits(records, training, config)
    foldseek_hits = _foldseek_hits(records, training, config)
    training_by_safe = {_safe_key(record): (index, record) for index, record in enumerate(training)}
    rows: dict[str, dict[str, Any]] = {}
    for query_index, record in enumerate(records):
        indices, scores = saprot_neighbors[query_index]
        saprot_candidates = [
            (int(index), float(score))
            for index, score in zip(indices.tolist(), scores.tolist(), strict=True)
            if record_key(training[int(index)]) != record_key(record)
        ]
        structural_candidates = [
            (target_index, target, float(score))
            for target_key, score in foldseek_hits.get(_safe_key(record), [])
            for match in [training_by_safe.get(target_key)]
            if match is not None
            for target_index, target in [match]
            if record_key(target) != record_key(record)
        ]
        normal = _aggregate_retrieval_view(
            record,
            training,
            labels,
            saprot_candidates,
            structural_candidates,
            sequence_hits,
            top_k,
            config,
            low_homology=False,
        )
        low_homology = _aggregate_retrieval_view(
            record,
            training,
            labels,
            saprot_candidates,
            structural_candidates,
            sequence_hits,
            top_k,
            config,
            low_homology=True,
        )
        rows[record_key(record)] = {**normal, "low_homology": low_homology}
    return rows


def _aggregate_retrieval_view(
    query: ProteinRecord,
    training: list[ProteinRecord],
    labels: torch.Tensor,
    saprot_candidates: list[tuple[int, float]],
    structural_candidates: list[tuple[int, ProteinRecord, float]],
    sequence_hits: dict[tuple[str, str], SequenceHit],
    top_k: int,
    config: dict[str, Any],
    *,
    low_homology: bool,
) -> dict[str, float | bool]:
    selected_saprot: list[tuple[int, float]] = []
    for index, score in saprot_candidates:
        target = training[index]
        if low_homology and not _is_low_homology_pair(
            query, target, sequence_hits, config
        ):
            continue
        selected_saprot.append((index, score))
        if len(selected_saprot) == top_k:
            break

    selected_structural: list[tuple[int, ProteinRecord, float]] = []
    for index, target, score in structural_candidates:
        if low_homology and not _is_low_homology_pair(
            query, target, sequence_hits, config
        ):
            continue
        selected_structural.append((index, target, score))
        if len(selected_structural) == top_k:
            break

    row = _default_feature_row()
    if selected_saprot:
        selected_indices = torch.tensor([index for index, _ in selected_saprot])
        selected_scores = torch.tensor([score for _, score in selected_saprot])
        value, similarity, variance = _weighted_label(
            labels[selected_indices], selected_scores
        )
        best_hit = max(
            (
                _sequence_hit(query, training[index], sequence_hits)
                for index, _ in selected_saprot
            ),
            key=lambda hit: (hit.identity, hit.bits),
        )
        row.update(
            {
                "saprot_value": _finite_or_zero(value),
                "saprot_similarity": similarity,
                "saprot_variance": _finite_or_one(variance),
                "saprot_available": True,
                "identity": best_hit.identity,
                "query_coverage": best_hit.query_coverage,
                "target_coverage": best_hit.target_coverage,
                "saprot_hit_fraction": len(selected_saprot) / top_k,
                "saprot_similarity_margin": _similarity_margin(selected_scores),
            }
        )
    if selected_structural:
        fold_labels = torch.tensor(
            [target.ph_opt for _, target, _ in selected_structural]
        )
        fold_scores = torch.tensor([score for _, _, score in selected_structural])
        value, similarity, variance = _weighted_label(
            fold_labels, fold_scores, temperature=0.2
        )
        row.update(
            {
                "foldseek_value": _finite_or_zero(value),
                "foldseek_similarity": similarity,
                "foldseek_variance": _finite_or_one(variance),
                "foldseek_available": True,
                "foldseek_hit_fraction": len(selected_structural) / top_k,
                "foldseek_similarity_margin": _similarity_margin(fold_scores),
            }
        )
    return row


def _sequence_hit(
    query: ProteinRecord,
    target: ProteinRecord,
    hits: dict[tuple[str, str], SequenceHit],
) -> SequenceHit:
    return hits.get(
        (_safe_key(query), _safe_key(target)),
        SequenceHit(0.0, 0.0, 0.0, 0.0),
    )


def _is_low_homology_pair(
    query: ProteinRecord,
    target: ProteinRecord,
    hits: dict[tuple[str, str], SequenceHit],
    config: dict[str, Any],
) -> bool:
    hit = _sequence_hit(query, target, hits)
    identity = _fraction(get(config, "retrieval.low_homology_identity", 0.2), 0.2)
    coverage = _fraction(get(config, "retrieval.low_homology_coverage", 0.8), 0.8)
    coverage_mode = int(get(config, "retrieval.low_homology_coverage_mode", 0))
    homologous = hit.identity >= identity and _coverage_passes(
        hit, coverage, coverage_mode, len(query.sequence), len(target.sequence)
    )
    return not homologous


def _coverage_passes(
    hit: SequenceHit,
    threshold: float,
    mode: int,
    query_length: int,
    target_length: int,
) -> bool:
    if mode == 0:
        return hit.query_coverage >= threshold and hit.target_coverage >= threshold
    if mode == 1:
        return hit.target_coverage >= threshold
    if mode == 2:
        return hit.query_coverage >= threshold
    if mode == 3:
        return target_length >= threshold * query_length
    if mode == 4:
        return query_length >= threshold * target_length
    if mode == 5:
        return min(query_length, target_length) >= threshold * max(query_length, target_length)
    raise ValueError("retrieval.low_homology_coverage_mode must be between 0 and 5")


def _similarity_margin(scores: torch.Tensor) -> float:
    if scores.numel() <= 1:
        return 0.0
    return float(scores.max() - scores.mean())


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
    required = bool(get(config, "retrieval.require_mmseqs", False))
    if binary is None:
        if required:
            raise FileNotFoundError("MMseqs is required for this homology evaluation")
        return {}
    with tempfile.TemporaryDirectory(prefix="phgeofuse-mmseqs-") as directory:
        root = Path(directory)
        query_fasta, train_fasta = root / "query.fasta", root / "train.fasta"
        query_fasta.write_text("".join(f">{_safe_key(r)}\n{r.sequence}\n" for r in records))
        train_fasta.write_text("".join(f">{_safe_key(r)}\n{r.sequence}\n" for r in training))
        output = root / "hits.tsv"
        candidate_k = _candidate_k(config, len(training))
        command = [binary, "easy-search", str(query_fasta), str(train_fasta), str(output),
                   str(root / "tmp"), "--max-seqs", str(candidate_k),
                   "-s", str(float(get(config, "retrieval.search_sensitivity", 7.5))),
                   "--alignment-mode", "3",
                   "--format-output", "query,target,fident,qcov,tcov,bits", "-v", "0"]
        if get(config, "retrieval.search_threads", None) is not None:
            command.extend(["--threads", str(int(get(config, "retrieval.search_threads")))])
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            if required:
                raise RuntimeError(f"MMseqs retrieval failed: {result.stderr.strip()}")
            return {}
        hits: dict[tuple[str, str], SequenceHit] = {}
        for line in output.read_text().splitlines():
            query, target, identity, qcov, tcov, bits = line.split("\t")[:6]
            hits[(query, target)] = SequenceHit(
                _normalize_reported_fraction(identity),
                _normalize_reported_fraction(qcov),
                _normalize_reported_fraction(tcov),
                float(bits),
            )
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
        candidate_k = _candidate_k(config, len(training))
        command = [binary, "easy-search", str(query_dir), str(train_dir), str(output), str(root / "tmp"),
                   "--max-seqs", str(candidate_k),
                   "--format-output", "query,target,bits", "-v", "0"]
        if get(config, "retrieval.search_threads", None) is not None:
            command.extend(["--threads", str(int(get(config, "retrieval.search_threads")))])
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
        compatible = expected <= set(store.rows)
        fingerprint = get(config, 'data.dataset_fingerprint')
        if fingerprint:
            compatible = (compatible and store.payload.get('dataset_fingerprint') == fingerprint
                          and set(store.payload.get('training_keys', [])) ==
                          {record_key(record) for record in records if record.status == 'ready' and record.split == 'train'})
        if homology_training_enabled(config):
            compatible = (
                compatible
                and store.payload.get("schema_version") == RETRIEVAL_SCHEMA_VERSION
                and store.payload.get("build_signature") == retrieval_build_signature(config)
                and all("low_homology" in store.rows[key] for key in expected)
            )
        if compatible:
            return store
    return RetrievalStore.build(records, config, destination)


def _normalize_reported_fraction(value: str | float) -> float:
    result = float(value)
    return result if result <= 1.0 else result / 100.0


def _finite_or_zero(value: float) -> float:
    return value if math.isfinite(value) else 0.0


def _finite_or_one(value: float) -> float:
    return value if math.isfinite(value) else 1.0
