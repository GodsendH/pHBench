"""Full Dual with PHGeoFuse-Lite replacing its neural baseline."""
from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

import numpy as np

from .cache import atomic_text
from .config import load_config, path
from .dual_fusion import DualFusion
from .lite import PHGeoFuseLite
from .lite.features import build_features, encoder_signature, reference_signature
from .lite.model import matrix
from .lite.pipeline import requested_records
from .retrieval import RetrievalStore, record_key


def load_sequence_features(records, *sources):
    keys = [record_key(record) for record in records]
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("nonempty unique query keys required")
    features = []
    for source in sources:
        with np.load(source, allow_pickle=False) as data:
            order = {str(key): index for index, key in enumerate(data["keys"])}
            if len(order) != len(data["keys"]):
                raise ValueError(f"duplicate sequence feature keys: {source}")
            if not set(keys) <= set(order):
                raise ValueError(f"missing sequence feature keys: {source}")
            indices = [order[key] for key in keys]
            mean, std = data["mean"], data["std"]
            if (mean.ndim != 2 or std.shape != mean.shape or len(mean) != len(order)
                    or not np.isfinite(mean).all() or not np.isfinite(std).all()):
                raise ValueError(f"invalid sequence feature cache: {source}")
            features.extend([mean[indices], std[indices]])
    return features


class DualLiteFusion:
    def __init__(self, bundle):
        bundle = Path(bundle)
        self.dual = DualFusion(bundle)
        self.config = self.dual.config
        if self.config.get("format") != "dual_lite" or self.config.get("schema_version") != 1:
            raise ValueError("not a supported Dual-Lite bundle")
        if "phgeofuse_lite/model.json" not in self.config["file_hashes"]:
            raise ValueError("bundle does not authenticate its Lite model")
        self.lite = PHGeoFuseLite.load(bundle / "phgeofuse_lite/model.json")
        if self.lite.design != "P+R":
            raise ValueError("this Dual-Lite version requires the P+R design")

    def predict(self, saprot_pooled, esm1v_mean, esm1v_std, esm2_mean, esm2_std,
                retrieval, sequences):
        saprot_pooled = matrix(saprot_pooled, 2 * self.lite.embedding_dim)
        retrieval = matrix(retrieval, 15)
        count = len(sequences)
        if len(saprot_pooled) != count or len(retrieval) != count:
            raise ValueError("Dual-Lite sample counts differ")
        for value, dimension in zip((esm1v_mean, esm1v_std, esm2_mean, esm2_std),
                                    np.repeat(self.config["esm_dimensions"], 2)):
            if len(matrix(value, int(dimension))) != count:
                raise ValueError("Dual-Lite sequence feature counts differ")
        baseline = self.lite.predict(np.column_stack([saprot_pooled, retrieval]))
        result = self.dual.predict(baseline, esm1v_mean, esm1v_std, esm2_mean, esm2_std,
                                   retrieval, sequences)
        result["robust_lite_prediction"] = result.pop("robust_v1_prediction")
        result["phgeofuse_lite_prediction"] = baseline
        return result

    def predict_records(self, records, config, esm1v_features, esm2_features, *,
                        store=None, build_queries=False):
        if encoder_signature(config) != self.lite.provenance.get("encoder"):
            raise ValueError("prediction SaProt configuration differs from the fitted model")
        if store is None:
            store = RetrievalStore.load(path(config, "paths.retrieval"))
        if reference_signature(store) != self.lite.provenance.get("reference_signature"):
            raise ValueError("prediction retrieval reference library differs from the fitted model")
        batch = build_features(records, config, design="P+R", store=store,
                               build_queries=build_queries)
        features = load_sequence_features(records, esm1v_features, esm2_features)
        return self.predict(batch.x[:, :2 * self.lite.embedding_dim], *features,
                            batch.x[:, -15:], [record.sequence for record in records])


def write_predictions(destination, records, result, *, include_labels=False):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"refusing to replace predictions: {destination}")
    stream = io.StringIO()
    fields = ["key", "protein_id", "sequence_sha256", *result]
    if include_labels:
        fields.append("label")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for index, record in enumerate(records):
        row = {"key": record_key(record), "protein_id": record.protein_id,
               "sequence_sha256": record.sequence_sha256,
               **{name: float(values[index]) for name, values in result.items()}}
        if include_labels:
            row["label"] = record.ph_opt
        writer.writerow(row)
    atomic_text(destination, stream.getvalue())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "bundle", "esm1v-features", "esm2-features", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--split", default="predict")
    parser.add_argument("--build-retrieval", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    records = requested_records(args.manifest or path(config, "paths.manifest"), {args.split})
    result = DualLiteFusion(args.bundle).predict_records(
        records, config, args.esm1v_features, args.esm2_features,
        build_queries=args.build_retrieval)
    write_predictions(args.output, records, result)
    print(f"Predicted {len(records)} proteins to {args.output}", flush=True)


if __name__ == "__main__":
    main()
