from __future__ import annotations

import csv
import io
from pathlib import Path
import time

import numpy as np

from phgeofuse.cache import atomic_json, atomic_text, sha256_file
from phgeofuse.config import get, path, save_resolved
from phgeofuse.io import read_manifest
from phgeofuse.retrieval import RetrievalStore
from .features import (
    FeatureBatch, build_features, encoder_signature, fingerprint, metadata_state,
    reference_signature, validate_references,
)
from .model import ALPHAS, PHGeoFuseLite, metrics


def requested_records(manifest, splits):
    records = [record for record in read_manifest(manifest) if record.split in splits]
    if not records or any(record.status != "ready" for record in records):
        raise ValueError("all requested records must be prepared and ready")
    for split in splits:
        if not any(record.split == split for record in records):
            raise ValueError(f"missing requested split: {split}")
    return records


def write_predictions(destination, batch, records, prediction, *, include_labels=False):
    stream = io.StringIO()
    fields = ["key", "protein_id", "predicted_ph_opt"] + (["label"] if include_labels else [])
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for index, record in enumerate(records):
        row = {"key": str(batch.keys[index]), "protein_id": record.protein_id,
               "predicted_ph_opt": float(prediction[index])}
        if include_labels:
            row["label"] = record.ph_opt
        writer.writerow(row)
    atomic_text(destination, stream.getvalue())


def fit(config, *, output=None, design=None):
    started = time.time()
    output = Path(output).expanduser().resolve() if output else path(config, "paths.lite_run")
    if output.exists():
        raise FileExistsError(f"use a new output directory for a Lite fit: {output}")
    design = design or str(get(config, "lite.design", "P+R"))
    manifest = path(config, "paths.manifest")
    records = requested_records(manifest, {"train", "validation"})
    training = [record for record in records if record.split == "train"]
    validation = [record for record in records if record.split == "validation"]
    state = metadata_state(training) if "M" in design else {}
    store = RetrievalStore.load(path(config, "paths.retrieval")) if "R" in design else None
    reference = validate_references(store, training) if store else None
    batch = build_features(records, config, design=design, store=store, state=state)
    train_mask = np.asarray([record.split == "train" for record in records])
    observed = np.asarray([record.ph_opt for record in records], dtype=np.float64)
    provenance = {
        "encoder": encoder_signature(config), "manifest_sha256": sha256_file(manifest),
        "reference_signature": reference,
        "retrieval_sha256": sha256_file(path(config, "paths.retrieval")) if store else None,
        "training_keys_sha256": fingerprint(batch.keys[train_mask].tolist()),
        "training_labels_sha256": fingerprint(observed[train_mask].tolist()),
        "validation_keys_sha256": fingerprint(batch.keys[~train_mask].tolist()),
        "training_count": len(training), "validation_count": len(validation),
        "fit_split": "train", "selection_split": "validation", "test_scored": False,
    }
    model = PHGeoFuseLite.fit(batch.x[train_mask], observed[train_mask],
                             batch.x[~train_mask], observed[~train_mask], design=design,
                             embedding_dim=int(get(config, "model.embedding_dim", 1280)),
                             alphas=get(config, "lite.alphas", ALPHAS),
                             metadata_state=state, provenance=provenance)
    output.mkdir(parents=True)
    model.save(output / "model.json")
    resolved = {**config, "lite": {**config.get("lite", {}), "design": design},
                "paths": {name: str(path(config, f"paths.{name}")) for name in config["paths"]}}
    resolved["paths"]["lite_run"] = str(output)
    save_resolved(resolved, output / "config.resolved.yaml")
    prediction = model.predict(batch.x, names=batch.names)
    validation_indices = np.flatnonzero(~train_mask)
    validation_batch = FeatureBatch(batch.keys[~train_mask], batch.x[~train_mask], batch.names)
    write_predictions(output / "validation_predictions.csv", validation_batch, validation,
                      prediction[validation_indices], include_labels=True)
    result = {"status": "complete", "model": "PHGeoFuse-Lite", "design": design,
              "alpha": model.alpha, "readout_parameters": model.parameter_count,
              "train": metrics(observed[train_mask], prediction[train_mask]),
              "validation": metrics(observed[~train_mask], prediction[~train_mask]),
              "alpha_grid": model.validation_scores, "test_scored": False,
              "seconds": time.time() - started, "provenance": provenance}
    atomic_json(output / "fit_result.json", result)
    return result


def predict(config, model_path, *, manifest=None, split="predict", output,
            evaluate=False, build_queries=False):
    output = Path(output).expanduser().resolve()
    if output.exists() or (evaluate and output.with_suffix(".metrics.json").exists()):
        raise FileExistsError(f"refusing to replace prediction results: {output}")
    model = PHGeoFuseLite.load(model_path)
    if encoder_signature(config) != model.provenance.get("encoder"):
        raise ValueError("prediction encoder/pooling configuration differs from the fitted model")
    records = requested_records(manifest or path(config, "paths.manifest"), {split})
    store = RetrievalStore.load(path(config, "paths.retrieval")) if "R" in model.design else None
    if store and reference_signature(store) != model.provenance.get("reference_signature"):
        raise ValueError("prediction retrieval reference library differs from the fitted model")
    batch = build_features(records, config, design=model.design, store=store,
                           state=model.metadata_state, build_queries=build_queries)
    prediction = model.predict(batch.x, names=batch.names)
    result = metrics([record.ph_opt for record in records], prediction) if evaluate else None
    write_predictions(output, batch, records, prediction, include_labels=evaluate)
    if evaluate:
        atomic_json(output.with_suffix(".metrics.json"), {
            **result, "split": split, "design": model.design,
            "model_sha256": sha256_file(model_path), "model_selection": False,
        })
    return {"output": str(output), "count": len(prediction), "metrics": result}
