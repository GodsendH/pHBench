"""Five-by-four grouped development experiment for ion context ridge.

All transfer models, scaling, recipe/strength selection exclude the outer
family fold. Uses certified existing complete-baseline exclusion predictions.
No original validation/test scores select a recipe in this command.
"""
from __future__ import annotations
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
from localph.kernel import ContextRidge
from phgeofuse.cache import atomic_json, sha256_file
from phgeofuse.delta_ref.data import DevelopmentData, atomic_npz, freeze_json, stable_hash, write_predictions
from phgeofuse.delta_ref.metrics import metrics, selection_score, acceptance, paired_family_bootstrap


def recipes():
    return [{"name": f"site{s:g}_a{a:g}_p{p:g}", "site_weight": s, "alpha": a, "power": p}
            for s, a, p in itertools.product([0., .5, 2.], [.2, 2.], [0., .5])]


def select(y, baseline, predictions, choices, groups):
    bm = metrics(y, baseline, groups)
    candidates = []
    for recipe in choices:
        for strength in [0., .25, .5, .75, 1.]:
            p = np.clip(baseline + strength * (predictions[recipe["name"]] - baseline), 0, 14)
            m = metrics(y, p, groups)
            rank = selection_score(m, bm, strength)
            candidates.append({"recipe": recipe, "strength": strength, "rank": rank,
                               "metrics": m})
    return min(candidates, key=lambda c: (*c["rank"], c["recipe"]["site_weight"], c["recipe"]["name"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    args.features = args.features.resolve()
    args.output = args.output.resolve()
    torch.set_num_threads(4)
    start = time.monotonic()
    data = DevelopmentData.load(ROOT / "configs/delta_ref_phopt.yaml")
    ids = data.train
    keys, y, folds, groups = data.keys[ids], data.labels[ids], data.folds[ids], data.groups[ids]
    global_x = data.embeddings[ids]
    with np.load(args.features, allow_pickle=False) as z:
        if not np.array_equal(z["keys"], keys):
            raise ValueError("feature keys differ")
        site = z["features"].astype(float)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    baseline_root = ROOT / "experiments/delta_ref_phopt_20260916/baseline/seed42"
    choices = recipes()
    inputs = {args.features: sha256_file(args.features), ROOT / "configs/delta_ref_phopt.yaml":
              sha256_file(ROOT / "configs/delta_ref_phopt.yaml")}
    baseline_sources = {}
    for n in (1, 2):
        for ex in itertools.combinations(range(5), n):
            folder = baseline_root / ("excluded_" + "_".join(map(str, ex)))
            for name in ("fit.json", "predictions.npz"):
                baseline_sources[str((folder / name).relative_to(ROOT))] = sha256_file(folder / name)
    protocol = {"scope": "exploratory nested PHOPT training only", "test_access": False,
                "validation_used": False, "recipes": choices, "strengths": [0., .25, .5, .75, 1.],
                "outer_folds": 5, "inner_folds": 4, "device": args.device,
                "source_hashes": {str(p.relative_to(ROOT)): sha256_file(p) for p in
                                  [Path(__file__), ROOT / "localph/kernel.py", ROOT / "localph/features.py"]},
                "input_hashes": {str(p.relative_to(ROOT)): v for p, v in inputs.items()},
                "baseline_source_hashes": baseline_sources,
                "development_provenance": data.provenance,
                "seeds": "deterministic ridge, fixed label-free projection; not five independent fits",
                "selection": "minimize worst tail RMSE ratio subject to +0.01 all/core RMSE and MAE; +0.005 false extremes"}
    freeze_json(output / "protocol.json", protocol)

    def baseline(excluded):
        fit = np.flatnonzero(~np.isin(folds, excluded))
        q = np.flatnonzero(np.isin(folds, excluded))
        folder = baseline_root / ("excluded_" + "_".join(map(str, excluded)))
        cert = json.loads((folder / "fit.json").read_text())
        if (cert["fit_keys"] != keys[fit].tolist() or cert["query_keys"] != keys[q].tolist()
                or cert["excluded_folds"] != list(excluded)
                or cert["fit_label_sha256"] != stable_hash(y[fit].tolist())
                or set(groups[fit]) & set(groups[q])):
            raise ValueError("baseline exclusion certificate mismatch")
        with np.load(folder / "predictions.npz", allow_pickle=False) as z:
            if not np.array_equal(z["keys"], keys[q]):
                raise ValueError("baseline query alignment mismatch")
            return fit, q, z["prediction"].copy()

    fit_results = {}
    for n in (2, 1):
        for excluded in itertools.combinations(range(5), n):
            fit, q, b = baseline(excluded)
            folder = output / ("excluded_" + "_".join(map(str, excluded)))
            folder.mkdir(exist_ok=True)
            estimates = {}
            for recipe in choices:
                dest = folder / (recipe["name"] + ".npz")
                if dest.exists() and not args.profile:
                    with np.load(dest, allow_pickle=False) as z:
                        if not np.array_equal(z["keys"], keys[q]):
                            raise ValueError("cached query alignment mismatch")
                        estimates[recipe["name"]] = z["prediction"].copy()
                    continue
                before = time.monotonic()
                model = ContextRidge(**{k: recipe[k] for k in ("site_weight", "alpha", "power")}, device=args.device)
                model.fit(global_x[fit], site[fit], y[fit])
                p = model.predict(global_x[q], site[q])
                train_p = model.predict(global_x[fit], site[fit])
                seconds = time.monotonic() - before
                if not np.isfinite(p).all():
                    raise ValueError("nonfinite predictions")
                report = {"excluded": excluded, "recipe": recipe, "seconds": seconds,
                          "train_metrics": metrics(y[fit], train_p, groups[fit]),
                          "query_metrics": metrics(y[q], p, groups[q])}
                if args.profile:
                    atomic_json(output / "resource_profile.json", {**report, "projected_180_fit_seconds": seconds * 180,
                                "scope": "resource probe; not model selection", "train": len(fit), "query": len(q)})
                    print(json.dumps({"fit_seconds": seconds, "projected_hours": seconds * 180 / 3600}), flush=True)
                    return
                atomic_npz(dest, keys=keys[q], prediction=p)
                atomic_json(dest.with_suffix(".json"), report)
                estimates[recipe["name"]] = p
                print(json.dumps({"event": "fit", "excluded": excluded, "recipe": recipe["name"],
                                  "seconds": seconds, "elapsed": time.monotonic() - start}), flush=True)
            fit_results[excluded] = (q, b, estimates)

    chosen_p, control_p, base_p = (np.zeros(len(y)) for _ in range(3))
    outer_reports = []
    for outer in range(5):
        fit = np.flatnonzero(folds != outer)
        inner_b = np.full(len(y), np.nan)
        inner_p = {r["name"]: np.full(len(y), np.nan) for r in choices}
        for inner in sorted(set(range(5)) - {outer}):
            q, b, estimates = fit_results[tuple(sorted([outer, inner]))]
            keep = folds[q] == inner
            inner_b[q[keep]] = b[keep]
            for name, p in estimates.items():
                inner_p[name][q[keep]] = p[keep]
        if not np.isfinite(inner_b[fit]).all() or not np.isnan(inner_b[folds == outer]).all():
            raise ValueError("outer label/input isolation failure")
        selected = select(y[fit], inner_b[fit], {k: v[fit] for k, v in inner_p.items()}, choices, groups[fit])
        control = select(y[fit], inner_b[fit], {k: v[fit] for k, v in inner_p.items()},
                         [r for r in choices if r["site_weight"] == 0], groups[fit])
        q, b, estimates = fit_results[(outer,)]
        base_p[q] = b
        for choice, dest in [(selected, chosen_p), (control, control_p)]:
            p = estimates[choice["recipe"]["name"]]
            dest[q] = np.clip(b + choice["strength"] * (p - b), 0, 14)
        outer_reports.append({"outer": outer, "selected": selected, "global_only_control": control,
                              "baseline": metrics(y[q], b, groups[q]),
                              "candidate": metrics(y[q], chosen_p[q], groups[q])})
        print(json.dumps({"event": "outer_selected", "outer": outer, "recipe": selected["recipe"],
                          "strength": selected["strength"]}), flush=True)
    bm, cm = metrics(y, base_p, groups), metrics(y, chosen_p, groups)
    result = {"baseline": bm, "candidate": cm, "global_only_control": metrics(y, control_p, groups),
              "acceptance": acceptance(cm, bm), "outer": outer_reports, "seconds": time.monotonic() - start,
              "default_model_replaced": False, "test_access": False,
              "family_bootstrap": paired_family_bootstrap(y, chosen_p[None],
                                      {"baseline": base_p[None], "global_only": control_p[None]}, groups, 10000)}
    write_predictions(output / "predictions.csv", keys, {"label": y, "group": groups, "fold": folds,
                      "baseline": base_p, "prediction": chosen_p, "global_only_control": control_p})
    atomic_npz(output / "predictions.npz", keys=keys, y=y, groups=groups, fold=folds,
               baseline=base_p, prediction=chosen_p, global_only_control=control_p)
    atomic_json(output / "results.json", result)
    print(json.dumps({"event": "complete", "acceptance": result["acceptance"], "seconds": result["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
