"""Full nested comparison of direct/global/sparse residue response models."""
import argparse
import itertools
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from localph.residue_training import fit as train_head, choose
from phgeofuse.cache import atomic_json, sha256_file
from phgeofuse.delta_ref.data import DevelopmentData, freeze_json, atomic_npz, stable_hash, write_predictions
from phgeofuse.delta_ref.metrics import metrics, acceptance, paired_family_bootstrap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    features = args.features.resolve()
    started = time.monotonic()
    data = DevelopmentData.load(ROOT / "configs/delta_ref_phopt.yaml")
    train = data.train
    keys, y, groups, folds = data.keys[train], data.labels[train], data.groups[train], data.folds[train]
    feature_complete = json.loads((features.parent / "complete.json").read_text())
    if sha256_file(features) != feature_complete["sha256"]:
        raise ValueError("packed feature hash differs")
    with np.load(features, allow_pickle=False) as z:
        if not np.array_equal(z["keys"], keys):
            raise ValueError("feature keys differ")
        packed = {k: z[k].copy() for k in ("tokens", "ionizable", "offsets")}
    if not (packed["offsets"][0] == 0 and packed["offsets"][-1] == len(packed["tokens"])
            and len(packed["offsets"]) == len(keys) + 1 and (np.diff(packed["offsets"]) > 0).all()
            and len(packed["tokens"]) == len(packed["ionizable"])):
        raise ValueError("invalid packed coverage")
    baseline_root = ROOT / "experiments/delta_ref_phopt_20260916/baseline/seed42"
    baseline_files = [p for d in baseline_root.glob("excluded_*") for p in (d / "fit.json", d / "predictions.npz")]
    sources = [Path(__file__), ROOT / "localph/residue_field.py", ROOT / "localph/residue_training.py",
               ROOT / "phgeofuse/delta_ref/metrics.py"]
    kinds = ["direct", "global", "sparse"]
    protocol = {"scope": "PHOPT-only 5x4 grouped comparison of three fixed residue heads",
                "kinds": kinds, "seed": 42, "hidden_width": 32, "dropout": .25,
                "optimizer": "AdamW", "lr": .001, "weight_decay": .05, "batch_size": 32,
                "gradient_clip": 1., "max_epochs": 40, "patience": 5,
                "minimum_stopping_improvement_standalone_objective": .002,
                "standalone_stopping_objective": "global MSE + 0.05*(acid MSE + alkaline MSE); natural-prior mean for field",
                "checkpoint_ties": "fused tail/all/strength rank, then standalone objective; prevents first-epoch lock at lambda=0",
                "refit_epochs": "median of four inner best epochs, lower integer via int(np.median)",
                "prior": "current training subset Gaussian-smoothed pH prior only",
                "loss": "direct MSE; field soft-label NLL with log training prior plus 0.25 mean-MSE and 0.01 curvature",
                "selection": "inner-only tail minimax with unchanged +0.01 all/core RMSE/MAE and +0.005 false extreme guardrails",
                "selection_decisions": ["direct"] + [f"prior{p:g}_{d}" for p in [1., .5, 0.] for d in ["mean", "mode"]],
                "strengths": [0., .25, .5, .75, 1.], "test_access": False, "validation_used": False,
                "source_hashes": {str(p.relative_to(ROOT)): sha256_file(p) for p in sources},
                "baseline_hashes": {str(p.relative_to(ROOT)): sha256_file(p) for p in baseline_files},
                "features_sha256": feature_complete["sha256"], "data_provenance": data.provenance,
                "retained_model": "history plus state_dict and training-only prior for every fit",
                "local_runtime_cap_hours": 8, "no_LoRA": True}
    freeze_json(out / "protocol.json", protocol)
    atomic_json(out / "status.json", {"state": "running", "started": time.time()})

    def baseline(excluded):
        fit = np.flatnonzero(~np.isin(folds, excluded))
        q = np.flatnonzero(np.isin(folds, excluded))
        folder = baseline_root / ("excluded_" + "_".join(map(str, excluded)))
        cert = json.loads((folder / "fit.json").read_text())
        if (cert["fit_keys"] != keys[fit].tolist() or cert["query_keys"] != keys[q].tolist()
                or cert["fit_label_sha256"] != stable_hash(y[fit].tolist())
                or cert["excluded_folds"] != list(excluded) or set(groups[fit]) & set(groups[q])):
            raise ValueError("baseline exclusion certificate differs")
        with np.load(folder / "predictions.npz", allow_pickle=False) as z:
            if not np.array_equal(z["keys"], keys[q]):
                raise ValueError("baseline query alignment differs")
            return fit, q, z["prediction"].copy()

    def fitted(fit, query, base, kind, target, epochs=None):
        if set(groups[fit]) & set(groups[query]):
            raise ValueError("family overlap")
        certificate = {"fit_keys": keys[fit].tolist(), "query_keys": keys[query].tolist(), "kind": kind,
                       "fixed_epochs": epochs, "fit_labels_sha256": stable_hash(y[fit].tolist()),
                       "protocol_sha256": sha256_file(out / "protocol.json")}
        target.mkdir(parents=True, exist_ok=True)
        freeze_json(target / "fit.json", certificate)
        if (target / "verified_complete.json").exists():
            saved = json.loads((target / "verified_complete.json").read_text())
            for name, digest in saved["hashes"].items():
                if sha256_file(target / name) != digest:
                    raise ValueError("completed fit hash differs")
            with np.load(target / "predictions.npz", allow_pickle=False) as z:
                predictions = {k: z[k].copy() for k in z.files}
            return predictions, json.loads((target / "complete.json").read_text())
        # Preserve interrupted fits, then start the same fixed seed afresh.
        if (target / "history.json").exists():
            archive = target / ("interrupted_" + str(time.time_ns()))
            archive.mkdir()
            for name in ["history.json", "complete.json", "predictions.npz", "weights.pt"]:
                if (target / name).exists():
                    (target / name).rename(archive / name)
        predictions, report = train_head(packed, y, fit, query, base, kind, target, fixed_epochs=epochs)
        if any(p.shape != (len(query),) or not np.isfinite(p).all() for p in predictions.values()):
            raise ValueError("invalid saved predictions")
        atomic_json(target / "verified_complete.json", {"hashes": {name: sha256_file(target / name)
                    for name in ["history.json", "complete.json", "predictions.npz", "weights.pt"]}})
        return predictions, report

    arrays = {k: np.full(len(y), np.nan) for k in ["baseline", "nested_selected", *kinds]}
    outer_rows = []
    for outer in range(5):
        fit, q, b = baseline((outer,))
        arrays["baseline"][q] = b
        kind_choices, outer_predictions = {}, {}
        for kind in kinds:
            inner_base = np.full(len(y), np.nan)
            inner_predictions = {}
            epochs = []
            for inner in sorted(set(range(5)) - {outer}):
                excluded = tuple(sorted([outer, inner]))
                subfit, combined, subb = baseline(excluded)
                keep = folds[combined] == inner
                query, base = combined[keep], subb[keep]
                atomic_json(out / "status.json", {"state": "inner_fit", "outer": outer, "inner": inner, "kind": kind,
                                                   "elapsed_seconds": time.monotonic() - started})
                p, report = fitted(subfit, query, base, kind, out / f"outer{outer}" / kind / f"inner{inner}")
                inner_base[query] = base
                for name, values in p.items():
                    inner_predictions.setdefault(name, np.full(len(y), np.nan))[query] = values
                epochs.append(report["selected_epoch"])
            if not np.isfinite(inner_base[fit]).all() or not np.isnan(inner_base[q]).all():
                raise ValueError("outer isolation failed")
            decision = choose(y[fit], inner_base[fit], {k: p[fit] for k, p in inner_predictions.items()})
            refit_epochs = max(1, int(np.median(epochs)))
            p, report = fitted(fit, q, None, kind, out / f"outer{outer}" / kind / "refit", refit_epochs)
            atomic_npz(out / f"outer{outer}" / kind / "inner_predictions.npz", keys=keys[fit],
                       baseline=inner_base[fit], **{k: p[fit] for k, p in inner_predictions.items()})
            prediction = np.clip(b + decision["strength"] * (p[decision["decision"]] - b), 0, 14)
            arrays[kind][q] = prediction
            kind_choices[kind] = {"choice": decision, "epochs": refit_epochs, "inner_epochs": epochs,
                                  "train_metrics": report["train_metrics"][decision["decision"]],
                                  "query_metrics": report["query_metrics"][decision["decision"]]}
            outer_predictions[kind] = prediction
            print(json.dumps({"event": "outer_kind_complete", "outer": outer, "kind": kind,
                              "choice": decision, "refit_epochs": refit_epochs}), flush=True)
        chosen_kind = min(kinds, key=lambda k: (*kind_choices[k]["choice"]["rank"], kinds.index(k)))
        arrays["nested_selected"][q] = outer_predictions[chosen_kind]
        row = {"outer": outer, "selected_kind": chosen_kind, "models": kind_choices,
               "metrics": {k: metrics(y[q], arrays[k][q], groups[q]) for k in arrays}}
        outer_rows.append(row)
        atomic_json(out / "outer_results.json", outer_rows)
        print(json.dumps({"event": "outer_complete", "outer": outer, "selected_kind": chosen_kind,
                          "elapsed_seconds": time.monotonic() - started}), flush=True)
    if any(not np.isfinite(p).all() for p in arrays.values()):
        raise ValueError("incomplete OOF coverage")
    scores = {k: metrics(y, p, groups) for k, p in arrays.items()}
    result = {**scores, "outer": outer_rows, "acceptance": {k: acceptance(scores[k], scores["baseline"])
              for k in ["nested_selected", *kinds]}, "seconds": time.monotonic() - started,
              "default_model_replaced": False, "test_access": False,
              "family_bootstrap": paired_family_bootstrap(y, arrays["nested_selected"][None],
                  {k: arrays[k][None] for k in ["baseline", "direct", "global"]}, groups, 10000)}
    atomic_json(out / "results.json", result)
    atomic_npz(out / "predictions.npz", keys=keys, y=y, groups=groups, folds=folds, **arrays)
    write_predictions(out / "predictions.csv", keys, {"label": y, "group": groups, "fold": folds, **arrays})
    atomic_json(out / "status.json", {"state": "complete", "seconds": time.monotonic() - started,
                                      "acceptance": result["acceptance"], "default_model_replaced": False})
    print(json.dumps({"event": "complete", "seconds": result["seconds"], "acceptance": result["acceptance"]}), flush=True)


if __name__ == "__main__":
    main()
