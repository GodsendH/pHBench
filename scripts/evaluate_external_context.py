"""Nested source-aware context regression with quarantined external labels."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from localph.source_kernel import SourceContextRidge
from phgeofuse.cache import atomic_json, sha256_file
from phgeofuse.delta_ref.data import DevelopmentData, freeze_json, atomic_npz, stable_hash, write_predictions
from phgeofuse.delta_ref.metrics import metrics, acceptance, paired_family_bootstrap
from scripts.evaluate_ion_context import select


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--site", type=Path, default=ROOT / "experiments/ion_context_phopt_20260917/features_train/features.npz")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    started = time.monotonic()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = DevelopmentData.load(ROOT / "configs/delta_ref_phopt.yaml")
    ids = data.train
    keys, y, groups, folds = data.keys[ids], data.labels[ids], data.groups[ids], data.folds[ids]
    audit = json.loads((args.external.parent / "audit.json").read_text())
    if not audit["ready_for_training"] or audit["PHOPT_labels_read"]:
        raise ValueError("external screening not certified")
    complete = json.loads((args.views / "complete.json").read_text())
    for filename, name in [("phopt_global.npz", "phopt_global_sha256"), ("external_features.npz", "external_features_sha256")]:
        if sha256_file(args.views / filename) != complete[name]:
            raise ValueError("feature checksum mismatch")
    with np.load(args.views / "phopt_global.npz", allow_pickle=False) as z:
        if not np.array_equal(z["keys"], keys):
            raise ValueError("PHOPT keys differ")
        gx = z["features"].copy()
    with np.load(args.site, allow_pickle=False) as z:
        if not np.array_equal(z["keys"], keys):
            raise ValueError("site keys differ")
        sx = z["features"].copy()
    with args.external.open() as f:
        rows = list(csv.DictReader(f))
    ey = np.array([float(r["label"]) for r in rows])
    with np.load(args.views / "external_features.npz", allow_pickle=False) as z:
        if z["keys"].tolist() != ["external::" + r["id"] for r in rows]:
            raise ValueError("external keys differ")
        ex, es = z["features"].copy(), z["site"].copy()
    choices = [{"name": f"ext{e:g}_site{s:g}_a{a:g}_p{p:g}", "external_weight": e, "site_weight": s, "alpha": a, "power": p}
               for e, s, a, p in itertools.product([0., .5, 1.], [0., .5], [.2, 2.], [0., .5])]
    baseline_root = ROOT / "experiments/delta_ref_phopt_20260916/baseline/seed42"
    sources = [Path(__file__), ROOT / "localph/source_kernel.py", ROOT / "localph/features.py",
               ROOT / "localph/kernel.py", ROOT / "scripts/evaluate_ion_context.py",
               ROOT / "phgeofuse/delta_ref/metrics.py"]
    inputs = [args.site, args.external, args.external.parent / "audit.json", args.views / "protocol.json",
              args.views / "complete.json", args.views / "external_features.npz", args.views / "phopt_global.npz"]
    baselines = {}
    for n in (2, 1):
        for excluded in itertools.combinations(range(5), n):
            directory = baseline_root / ("excluded_" + "_".join(map(str, excluded)))
            for name in ("fit.json", "predictions.npz"):
                baselines[str((directory / name).relative_to(ROOT))] = sha256_file(directory / name)
    protocol = {"scope": "PHOPT plus screened EnzyBase12k external supervision; exploratory nested development",
                "test_access": False, "validation_used": False, "recipes": choices,
                "selection": "same predeclared tail minimax and all/core guardrails as IonContext PHOPT-only",
                "external_count": len(ey), "external_acid": int((ey <= 4).sum()), "external_alkaline": int((ey >= 10).sum()),
                "PHOPT_esm1v_precision": "float32 moments for all recipes; matched to external ESM1v",
                "source_hashes": {str(p.resolve().relative_to(ROOT)): sha256_file(p) for p in sources},
                "input_hashes": {str(p.resolve().relative_to(ROOT)): sha256_file(p) for p in inputs},
                "baseline_source_hashes": baselines, "development_provenance": data.provenance,
                "outer_folds": 5, "inner_folds": 4, "independent_seeds": 1,
                "external_source_offset": "ridge-regularized scalar; source=0 for every PHOPT prediction",
                "external_weights": "separately normalized capped frequency weights times external_weight"}
    freeze_json(output / "protocol.json", protocol)
    cache = {}
    for n in (2, 1):
        for excluded in itertools.combinations(range(5), n):
            fit = np.flatnonzero(~np.isin(folds, excluded))
            q = np.flatnonzero(np.isin(folds, excluded))
            directory = baseline_root / ("excluded_" + "_".join(map(str, excluded)))
            cert = json.loads((directory / "fit.json").read_text())
            if (cert["fit_keys"] != keys[fit].tolist() or cert["query_keys"] != keys[q].tolist()
                    or cert["fit_label_sha256"] != stable_hash(y[fit].tolist())
                    or cert["excluded_folds"] != list(excluded) or set(groups[fit]) & set(groups[q])):
                raise ValueError("baseline exclusion certificate differs")
            with np.load(directory / "predictions.npz", allow_pickle=False) as z:
                if not np.array_equal(z["keys"], keys[q]):
                    raise ValueError("baseline keys differ")
                b = z["prediction"].copy()
            target = output / directory.name
            target.mkdir(exist_ok=True)
            estimates = {}
            for recipe in choices:
                destination = target / (recipe["name"] + ".npz")
                report_file = destination.with_suffix(".json")
                if destination.exists() and report_file.exists():
                    report = json.loads(report_file.read_text())
                    if sha256_file(destination) != report["predictions_sha256"]:
                        raise ValueError("saved fit checksum differs")
                    with np.load(destination, allow_pickle=False) as z:
                        if not np.array_equal(z["keys"], keys[q]):
                            raise ValueError("saved fit keys differ")
                        estimates[recipe["name"]] = z["prediction"].copy()
                    continue
                begin = time.monotonic()
                model = SourceContextRidge(**{k: v for k, v in recipe.items() if k != "name"}, device=args.device)
                model.fit(gx[fit], sx[fit], y[fit], ex, es, ey)
                p, train_p = model.predict(gx[q], sx[q]), model.predict(gx[fit], sx[fit])
                if not np.isfinite(p).all():
                    raise ValueError("nonfinite prediction")
                atomic_npz(destination, keys=keys[q], prediction=p, coefficient=model.coef, x_mean=model.x_mean,
                           y_mean=np.array(model.y_mean), site_mean=model.scaler.mean, site_scale=model.scaler.scale)
                report = {"recipe": recipe, "excluded": excluded, "seconds": time.monotonic() - begin,
                          "predictions_sha256": sha256_file(destination), "external_offset": float(model.coef[-1]),
                          "fit_keys_sha256": stable_hash(keys[fit].tolist()), "fit_labels_sha256": stable_hash(y[fit].tolist()),
                          "train_metrics": metrics(y[fit], train_p, groups[fit]), "query_metrics": metrics(y[q], p, groups[q])}
                atomic_json(report_file, report)
                estimates[recipe["name"]] = p
                print(json.dumps({"event": "fit", "excluded": excluded, "recipe": recipe["name"],
                                  "seconds": report["seconds"], "elapsed": time.monotonic() - started}), flush=True)
            cache[excluded] = (q, b, estimates)
    result_arrays = {k: np.zeros(len(y)) for k in ("baseline", "candidate", "PHOPT_only", "external_global")}
    choices_by_kind = {"candidate": choices, "PHOPT_only": [r for r in choices if r["external_weight"] == 0],
                       "external_global": [r for r in choices if r["site_weight"] == 0]}
    outer_rows = []
    for outer in range(5):
        fit = np.flatnonzero(folds != outer)
        ib = np.full(len(y), np.nan)
        ip = {r["name"]: np.full(len(y), np.nan) for r in choices}
        for inner in sorted(set(range(5)) - {outer}):
            q, b, estimates = cache[tuple(sorted([outer, inner]))]
            keep = folds[q] == inner
            ib[q[keep]] = b[keep]
            for name, p in estimates.items():
                ip[name][q[keep]] = p[keep]
        if not np.isfinite(ib[fit]).all() or not np.isnan(ib[folds == outer]).all():
            raise ValueError("outer isolation failure")
        q, b, estimates = cache[(outer,)]
        result_arrays["baseline"][q] = b
        row = {"outer": outer}
        for kind, allowed in choices_by_kind.items():
            chosen = select(y[fit], ib[fit], {k: p[fit] for k, p in ip.items()}, allowed, groups[fit])
            result_arrays[kind][q] = np.clip(b + chosen["strength"] * (estimates[chosen["recipe"]["name"]] - b), 0, 14)
            row[kind] = chosen
        outer_rows.append(row)
        print(json.dumps({"event": "outer", "outer": outer, "recipe": row["candidate"]["recipe"],
                          "strength": row["candidate"]["strength"]}), flush=True)
    scores = {k: metrics(y, p, groups) for k, p in result_arrays.items()}
    result = {**scores, "acceptance": acceptance(scores["candidate"], scores["baseline"]), "outer": outer_rows,
              "test_access": False, "default_model_replaced": False, "seconds": time.monotonic() - started,
              "family_bootstrap": paired_family_bootstrap(y, result_arrays["candidate"][None],
                                  {k: result_arrays[k][None] for k in ("baseline", "PHOPT_only", "external_global")}, groups, 10000)}
    atomic_npz(output / "predictions.npz", keys=keys, y=y, groups=groups, folds=folds, **result_arrays)
    write_predictions(output / "predictions.csv", keys, {"label": y, "group": groups, "fold": folds, **result_arrays})
    atomic_json(output / "results.json", result)
    print(json.dumps({"event": "complete", "acceptance": result["acceptance"], "seconds": result["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
