from __future__ import annotations

import math
from typing import Any

import torch
from torch.utils.data import Dataset

from .io import ProteinRecord
from .retrieval import RetrievalStore, record_key
from .saprot import saprot_text


class ProteinGraphDataset(Dataset):
    def __init__(
        self,
        records: list[ProteinRecord],
        split: str,
        retrieval: RetrievalStore | None,
        mode: str = "frozen",
    ):
        self.records = [record for record in records if record.split == split and record.status == "ready"]
        self.retrieval = retrieval
        self.mode = mode

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        graph = torch.load(record.graph_path, map_location="cpu")
        embedding = None
        if self.mode == "frozen":
            payload = torch.load(record.embedding_path, map_location="cpu")
            embedding = payload["embedding"].float()
            if embedding.shape[0] != len(record.sequence):
                raise ValueError(f"embedding length mismatch for {record_key(record)}")
        retrieval = self.retrieval.features(record_key(record)) if self.retrieval else torch.zeros(9)
        return {
            "key": record_key(record),
            "sequence": record.sequence,
            "saprot_text": saprot_text(record.sequence, record.three_di),
            "embedding": embedding,
            "graph": graph,
            "label": record.ph_opt,
            "weight": record.sample_weight,
            "ec": _ec_class(record.ec),
            "retrieval": retrieval,
        }


def collate_graphs(items: list[dict[str, Any]]) -> dict[str, Any]:
    node_features, coords, edge_indices, edge_features = [], [], [], []
    ionizable_types, pkas, graph_indices, embeddings = [], [], [], []
    offset = 0
    lengths = []
    for graph_index, item in enumerate(items):
        graph = item["graph"]
        length = int(graph["coords"].shape[0])
        lengths.append(length)
        node_features.append(graph["node_features"].float())
        coords.append(graph["coords"].float())
        edge_indices.append(graph["edge_index"].long() + offset)
        edge_features.append(graph["edge_features"].float())
        ionizable_types.append(graph["ionizable_type"].long())
        pkas.append(graph["pka"].float())
        graph_indices.append(torch.full((length,), graph_index, dtype=torch.long))
        if item["embedding"] is not None:
            embeddings.append(item["embedding"])
        offset += length
    labels = torch.tensor([item["label"] for item in items], dtype=torch.float32)
    return {
        "keys": [item["key"] for item in items],
        "sequences": [item["sequence"] for item in items],
        "saprot_texts": [item["saprot_text"] for item in items],
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "node_features": torch.cat(node_features),
        "coords": torch.cat(coords),
        "edge_index": torch.cat(edge_indices, dim=1),
        "edge_features": torch.cat(edge_features),
        "ionizable_type": torch.cat(ionizable_types),
        "pka": torch.cat(pkas),
        "graph_index": torch.cat(graph_indices),
        "embeddings": torch.cat(embeddings) if embeddings else None,
        "labels": labels,
        "weights": torch.tensor([item["weight"] for item in items], dtype=torch.float32),
        "ec_labels": torch.tensor([item["ec"] for item in items], dtype=torch.long),
        "retrieval": torch.stack([item["retrieval"] for item in items]),
    }


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _ec_class(value: str) -> int:
    try:
        number = int(value.split(".", 1)[0])
        return number - 1 if 1 <= number <= 7 else -1
    except (ValueError, AttributeError):
        return -1
