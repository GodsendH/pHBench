"""Fit the closed-form linear expert that P0' freezes into the model.

The graph head is a net loss on its own input: 3,662,475 parameters reach
0.9429 test RMSE, while a Ridge on the same pooled SaProt features -- 2,561
parameters, solved in closed form -- reaches 0.8484.  P0 tried to let SGD learn
that linear map behind a zero initialisation and failed: 1,784 steps moved the
weight norm from 0.112 to 0.109, i.e. it stayed a random walk, and the skip's
contribution was 0.7% of the label spread (experiments/p0p1_bypass_20260929).

So the map is fitted here instead, on the train split only, and frozen.  The
feature layout must stay identical to PHGeoFuse._pooled_summary: per-protein
mean and std of the residue embeddings, each L2-normalised, concatenated.

The output carries the split and manifest fingerprint it was fitted on, so a
stale expert cannot be silently reused against a different data revision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phgeofuse.config import get, load_config, path  # noqa: E402
from phgeofuse.io import read_manifest  # noqa: E402


def l2(value: np.ndarray) -> np.ndarray:
    return value / np.linalg.norm(value, axis=1, keepdims=True)


def pooled_summary(records):
    """Per-protein mean+std of the cached SaProt embeddings, each L2-normalised."""
    means, stds = [], []
    for index, record in enumerate(records):
        payload = torch.load(record.embedding_path, map_location="cpu")
        embedding = payload["embedding"].float().numpy().astype(np.float64)
        if embedding.shape[0] != len(record.sequence):
            raise ValueError(f"embedding/length mismatch for {record.protein_id}")
        means.append(embedding.mean(axis=0))
        stds.append(embedding.std(axis=0))
        if index % 3000 == 0:
            print(f"  pooled {index}", flush=True)
    features = np.column_stack([l2(np.stack(means)), l2(np.stack(stds))])
    labels = np.asarray([r.ph_opt for r in records], dtype=np.float64)
    splits = np.asarray([r.split for r in records])
    return features, labels, splits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", default="phopt")
    parser.add_argument("--output")
    parser.add_argument("--alpha", type=float, default=0.03)
    args = parser.parse_args()

    from phgeofuse.datasets import apply_dataset

    config = apply_dataset(load_config(args.config), args.dataset)
    records = [r for r in read_manifest(path(config, "paths.manifest")) if r.status == "ready"]
    print(f"records: {len(records)}")

    features, labels, splits = pooled_summary(records)
    train = splits == "train"
    if not train.any():
        raise SystemExit("no training records to fit on")
    print(f"fitted on {train.sum()} training records, alpha={args.alpha}")

    model = Ridge(alpha=args.alpha, solver="cholesky")
    model.fit(features[train], labels[train])

    weight = torch.tensor(model.coef_, dtype=torch.float32).reshape(1, -1)
    bias = torch.tensor([float(model.intercept_)], dtype=torch.float32)
    train_keys = sorted(f"{r.split}::{r.protein_id}" for r in records if r.split == "train")
    payload = {
        "weight": weight,
        "bias": bias,
        "alpha": args.alpha,
        "feature_dim": int(weight.shape[1]),
        "pooling": "mean+std-l2",
        "fitted_on": "train",
        "train_records": int(train.sum()),
        "train_key_digest": hashlib.sha256("\n".join(train_keys).encode()).hexdigest(),
        "dataset_fingerprint": get(config, "data.dataset_fingerprint", None),
        "manifest": str(path(config, "paths.manifest")),
    }

    output = Path(args.output) if args.output else (
        Path(__file__).resolve().parents[1]
        / "experiments/p0p1_bypass_20260929/linear_expert.pt"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)

    prediction = model.predict(features[splits == "test"])
    truth = labels[splits == "test"]
    rmse = float(np.sqrt(np.mean((prediction - truth) ** 2)))
    print(f"\nwrote {output}")
    print(f"  ||coef||={np.linalg.norm(model.coef_):.4f}  intercept={model.intercept_:+.4f}")
    print(f"  test RMSE={rmse:.4f}")
    print(json.dumps({"weight_shape": list(weight.shape), "alpha": args.alpha}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
