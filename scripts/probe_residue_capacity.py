"""Compare full ESM1v residue width on the same predeclared inner pilot only."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from localph.residue_probes import fit_probe
from phgeofuse.cache import atomic_json, sha256_file
from phgeofuse.delta_ref.data import DevelopmentData, freeze_json, stable_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    cert = json.loads((args.features / "complete.json").read_text())
    for name, digest in cert["hashes"].items():
        if sha256_file(args.features / name) != digest:
            raise ValueError("full-width cache hash differs")
    d = DevelopmentData.load(ROOT / "configs/delta_ref_phopt.yaml")
    t = d.train
    keys, y, folds, groups = d.keys[t], d.labels[t], d.folds[t], d.groups[t]
    fit, query = np.flatnonzero(~np.isin(folds, [0, 1])), np.flatnonzero(folds == 1)
    if set(groups[fit]) & set(groups[query]):
        raise ValueError("family overlap")
    with np.load(args.features / "index.npz", allow_pickle=False) as z:
        if not np.array_equal(z["keys"], keys):
            raise ValueError("feature alignment differs")
        packed = {k: z[k].copy() for k in ("offsets", "ionizable")}
    packed["tokens"] = np.load(args.features / "tokens.npy", mmap_mode="r", allow_pickle=False)
    if packed["tokens"].shape != (packed["offsets"][-1], 1280):
        raise ValueError("full residue dimensions differ")
    bdir = ROOT / "experiments/delta_ref_phopt_20260916/baseline/seed42/excluded_0_1"
    bc = json.loads((bdir / "fit.json").read_text())
    if bc["fit_keys"] != keys[fit].tolist() or bc["fit_label_sha256"] != stable_hash(y[fit].tolist()):
        raise ValueError("baseline certificate differs")
    with np.load(bdir / "predictions.npz", allow_pickle=False) as z:
        lookup = dict(zip(z["keys"], z["prediction"]))
        baseline = np.array([lookup[k] for k in keys[query]])
    sources = [Path(__file__), ROOT / "localph/residue_probes.py", ROOT / "localph/residue_field.py",
               ROOT / "localph/residue_training.py", ROOT / "localph/residue_objectives.py"]
    protocol = {"scope": "outer0 inner1 capacity pilot only; not independent performance estimate",
                "kind": "sparse", "input_dimension": 1280, "hidden_width": 32, "seed": 42,
                "objectives": ["natural", "weighted"], "max_epochs": 40, "patience": 5,
                "fit_keys": keys[fit].tolist(), "query_keys": keys[query].tolist(),
                "fit_labels_sha256": stable_hash(y[fit].tolist()), "source_hashes": {str(p.relative_to(ROOT)): sha256_file(p) for p in sources},
                "feature_complete_sha256": sha256_file(args.features / "complete.json"),
                "feature_hashes": cert["hashes"],
                "baseline_hashes": {str(p.relative_to(ROOT)): sha256_file(p) for p in [bdir / "fit.json", bdir / "predictions.npz"]},
                "optimizer": "AdamW lr0.001 weight_decay0.05 batch32 clip1 dropout0.25",
                "selection": "same frozen residue-training inner selection, no outer0 labels",
                "comparison": "128D natural from nested_v2/outer0/sparse/inner1; 128D weighted from rarity_pilot/sparse",
                "interpretation_limit": "full input width also increases trainable input-projection parameters; this probes the combined bottleneck, not a parameter-matched attribution",
                "test_access": False, "original_validation_used": False, "data_provenance": d.provenance}
    freeze_json(out / "protocol.json", protocol)
    results, start = {}, time.monotonic()
    for objective in protocol["objectives"]:
        atomic_json(out / "status.json", {"state": "running", "objective": objective, "seconds": time.monotonic() - start})
        results[objective] = fit_probe(packed, y, fit, query, baseline, out / objective, weighted=objective == "weighted")
    atomic_json(out / "results.json", {"models": results, "seconds": time.monotonic() - start, "is_confirmatory": False})
    atomic_json(out / "status.json", {"state": "complete", "seconds": time.monotonic() - start, "goal_achieved": False})


if __name__ == "__main__":
    main()
