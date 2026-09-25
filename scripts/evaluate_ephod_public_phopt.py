"""Recompute public EpHod outputs without training or selecting on PHOPT test.

This is an official-output comparison, not a PHOPT-only retraining benchmark.
The public file contains one result per EpHod component, not five seed runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scores(y, p, mask):
    e = p[mask] - y[mask]
    return {"n": int(mask.sum()), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "mae": float(np.mean(abs(e))), "bias": float(e.mean()),
            "abs_bias": float(abs(e.mean()))}


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=root / "docs/extreme_ph_review_20260916")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output directory to preserve previous evidence")
    source = args.sources.resolve()
    tree_file = source / "EpHod_master_tree.json"
    tree = json.loads(tree_file.read_text())
    entries = {entry["path"]: entry for entry in tree["tree"]}
    public_files = {}
    for name in ("example/prediction.csv", "example/test_sequences.fasta", "ephod/run.py", "README.rst"):
        path = source / ("ephod_official_" + name.replace("/", "_"))
        content = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        if blob != entries[name]["sha"]:
            raise ValueError(f"Public blob does not match commit: {name}")
        public_files[name] = path
    records = []
    for block in public_files["example/test_sequences.fasta"].read_text().split(">")[1:]:
        lines = block.splitlines()
        fields = lines[0].split("|")
        records.append({"id": fields[0].strip(), "label": float(fields[3]),
                        "sequence": "".join(lines[1:]).strip()})
    ref = pd.DataFrame(records).set_index("id")
    pred = pd.read_csv(public_files["example/prediction.csv"], index_col=0)
    if not ref.index.is_unique or not pred.index.is_unique or set(ref.index) != set(pred.index):
        raise ValueError("Public prediction IDs must match the public FASTA exactly")
    pred = pred.loc[ref.index]
    if set(pred.columns) != {"RLATtr", "SVR", "Ensemble"} or not np.isfinite(pred.to_numpy()).all():
        raise ValueError("Invalid public prediction schema or values")
    manifest_file = root / "artifacts/phgeofuse/manifest.csv"
    manifest = pd.read_csv(manifest_file)
    test = manifest[manifest.split == "test"].set_index("protein_id")
    if len(ref) != 1971 or not test.index.is_unique or set(test.index) != set(ref.index):
        raise ValueError("Expected complete PHOPT test coverage")
    test = test.loc[ref.index]
    if not (test.sequence == ref.sequence).all():
        raise ValueError("Official and local test sequences differ")
    np.testing.assert_allclose(test.ph_opt, ref.label, rtol=0, atol=1e-12)
    np.testing.assert_allclose(pred.Ensemble, (pred.RLATtr + pred.SVR) / 2, atol=1e-12, rtol=0)
    zenodo_file = source / "EpHod_pHopt_data.csv"
    zenodo = pd.read_csv(zenodo_file)
    ztest = zenodo[zenodo.Split.str.lower().eq("testing")].set_index("Accession")
    if set(ztest.index) != set(ref.index):
        raise ValueError("Official Zenodo testing IDs differ")
    ztest = ztest.loc[ref.index]
    if not (ztest.Sequence == ref.sequence).all():
        raise ValueError("Official Zenodo testing sequences differ")
    np.testing.assert_allclose(ztest.pHopt, ref.label, rtol=0, atol=1e-12)
    keys = ["test::" + key for key in ref.index]
    base, baseline_files = [], []
    for seed in (0, 1, 2, 3, 42):
        path = root / f"experiments/phgeofuse_redesign_20260914/dual_test/seed{seed}.csv"
        df = pd.read_csv(path).set_index("key")
        if not df.index.is_unique or set(df.index) != set(keys):
            raise ValueError("Baseline must cover the same test keys exactly")
        df = df.loc[keys]
        np.testing.assert_allclose(df.label, ref.label, rtol=0, atol=1e-12)
        if not np.isfinite(df.prediction).all():
            raise ValueError("Nonfinite baseline")
        base.append(df.prediction.to_numpy())
        baseline_files.append(path)
    y = ref.label.to_numpy()
    subsets = {"all": np.ones(len(y), bool), "acid_le4": y <= 4, "alkaline_ge10": y >= 10,
               "core": (y > 4) & (y < 10), "alkaline_gt9": y > 9}
    result = {
        "scope": "published-output diagnostic only; not PHOPT-only controlled retraining or proof of field leadership",
        "public_commit": tree["sha"], "official_information": "pHenv-task-pretrained RLATtr plus ESM1v-SVR",
        "public_runs": 1, "baseline_seeds": [0, 1, 2, 3, 42],
        "baseline_aggregation": "mean of per-seed metrics, not metrics of ensemble predictions",
        "verified": {"ids": 1971, "sequences": 1971, "labels": 1971, "zenodo_test_split": True,
                     "github_blob_sha": True, "ensemble_arithmetic": True}, "metrics": {},
    }
    for name, mask in subsets.items():
        per_seed = [scores(y, p, mask) for p in base]
        mean = {k: float(np.mean([row[k] for row in per_seed])) for k in ("rmse", "mae", "bias", "abs_bias")}
        mean["n"] = int(mask.sum())
        result["metrics"][name] = {
            **{k: scores(y, pred[k].to_numpy(), mask) for k in pred},
            "current_full_baseline_mean": mean, "current_full_baseline_per_seed": per_seed,
        }
    result["target_gaps"] = {}
    for group, model in (("acid_le4", "Ensemble"), ("alkaline_ge10", "SVR")):
        ours, reference = result["metrics"][group]["current_full_baseline_mean"], result["metrics"][group][model]
        result["target_gaps"][group] = {"reference": model, "metrics": {
            k: {"absolute_gap": ours[k] - reference[k],
                "reduction_from_current_percent": 100 * (1 - reference[k] / ours[k])}
            for k in ("rmse", "mae", "abs_bias")}}
    sources = [Path(__file__), tree_file, manifest_file, zenodo_file, *public_files.values(), *baseline_files]
    result["sources_sha256"] = {str(path): sha(path) for path in sources}
    args.output.mkdir(parents=True)
    (args.output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    pd.DataFrame({"key": keys, "label": y, **{k: pred[k].to_numpy() for k in pred},
                  **{f"base_seed{s}": p for s, p in zip((0, 1, 2, 3, 42), base)}}).to_csv(
        args.output / "predictions.csv", index=False)
    print(json.dumps({"output": str(args.output), "verified": result["verified"],
                      "target_gaps": result["target_gaps"]}, indent=2))


if __name__ == "__main__":
    main()
