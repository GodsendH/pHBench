"""Nested PHOPT-only conditional-density experiment with training-only priors."""
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
from localph.density import DensityRidge, decode_density
from phgeofuse.cache import atomic_json, sha256_file
from phgeofuse.delta_ref.data import DevelopmentData, freeze_json, atomic_npz, stable_hash, write_predictions
from phgeofuse.delta_ref.metrics import metrics, acceptance, paired_family_bootstrap
from scripts.evaluate_ion_context import select


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    start = time.monotonic()
    data = DevelopmentData.load(ROOT / "configs/delta_ref_phopt.yaml")
    idx = data.train
    keys, y, groups, folds, x = data.keys[idx], data.labels[idx], data.groups[idx], data.folds[idx], data.embeddings[idx]
    recipes = [{"name": f"a{a:g}_bw{bw:g}_prior{power:g}_{decision}", "alpha": a, "bandwidth": bw,
                "prior_power": power, "decision": decision, "site_weight": 0.}
               for a, bw, power, decision in itertools.product([.2, 2.], [.5, 1.], [0., .5], ["mean", "mode"])]
    baseline_root = ROOT / "experiments/delta_ref_phopt_20260916/baseline/seed42"
    files = [p for d in baseline_root.glob("excluded_*") for p in [d / "fit.json", d / "predictions.npz"]]
    sources = [Path(__file__), ROOT / "localph/density.py", ROOT / "scripts/evaluate_ion_context.py",
               ROOT / "phgeofuse/delta_ref/metrics.py"]
    protocol = {"scope": "PHOPT-only smooth conditional density; exploratory 5x4 grouped development",
                "harmonics": 24, "grid": [0, 14, .05], "recipes": recipes,
                "selection": "inner-only worst-tail minimax with unchanged all/core guardrails",
                "strengths": [0, .25, .5, .75, 1.], "test_access": False, "validation_used": False,
                "priors": "mean target cosine moments from the current training subset only",
                "negative_density": "clip to zero; report pre-clipping integrated negative mass",
                "source_hashes": {str(p.relative_to(ROOT)): sha256_file(p) for p in sources},
                "baseline_hashes": {str(p.relative_to(ROOT)): sha256_file(p) for p in files},
                "data_provenance": data.provenance, "deterministic": True,
                "claim_limit": "density is not calibrated predictive uncertainty or a biochemical activity curve"}
    freeze_json(out / "protocol.json", protocol)
    cache = {}
    for n in (2, 1):
        for excluded in itertools.combinations(range(5), n):
            fit, q = np.flatnonzero(~np.isin(folds, excluded)), np.flatnonzero(np.isin(folds, excluded))
            directory = baseline_root / ("excluded_" + "_".join(map(str, excluded)))
            cert = json.loads((directory / "fit.json").read_text())
            if (cert["fit_keys"] != keys[fit].tolist() or cert["query_keys"] != keys[q].tolist()
                    or cert["fit_label_sha256"] != stable_hash(y[fit].tolist())
                    or cert["excluded_folds"] != list(excluded) or set(groups[fit]) & set(groups[q])):
                raise ValueError("exclusion certificate differs")
            with np.load(directory / "predictions.npz", allow_pickle=False) as z:
                if not np.array_equal(z["keys"], keys[q]):
                    raise ValueError("baseline keys differ")
                b = z["prediction"].copy()
            target = out / directory.name
            target.mkdir(exist_ok=True)
            estimates = {}
            for alpha in [.2, 2.]:
                before = time.monotonic()
                model = DensityRidge(alpha, device="cuda").fit(x[fit], y[fit])
                moments, train_moments = model.moments(x[q]), model.moments(x[fit])
                atomic_npz(target / f"density_a{alpha:g}.npz", keys=keys[q], moments=moments, prior=model.target_mean,
                           coef=model.coef, x_mean=model.x_mean)
                for r in [r for r in recipes if r["alpha"] == alpha]:
                    p, negative = decode_density(moments, model.target_mean, r["bandwidth"], r["prior_power"], r["decision"])
                    tr, _ = decode_density(train_moments, model.target_mean, r["bandwidth"], r["prior_power"], r["decision"])
                    estimates[r["name"]] = p
                    atomic_json(target / (r["name"] + ".json"), {"recipe": r, "train_metrics": metrics(y[fit], tr, groups[fit]),
                                "query_metrics": metrics(y[q], p, groups[q]), "mean_negative_mass": float(negative.mean()),
                                "max_negative_mass": float(negative.max()), "fit_label_hash": stable_hash(y[fit].tolist())})
                print(json.dumps({"event": "density_fit", "excluded": excluded, "alpha": alpha,
                                  "seconds": time.monotonic() - before, "elapsed": time.monotonic() - start}), flush=True)
            atomic_npz(target / "predictions.npz", keys=keys[q], **estimates)
            cache[excluded] = (q, b, estimates)
    arrays = {k: np.full(len(y), np.nan) for k in ["baseline", "candidate", "density_mean", "no_prior_correction"]}
    subsets = {"candidate": recipes, "density_mean": [r for r in recipes if r["decision"] == "mean"],
               "no_prior_correction": [r for r in recipes if r["prior_power"] == 0]}
    outer_rows = []
    for outer in range(5):
        fit = np.flatnonzero(folds != outer)
        ib = np.full(len(y), np.nan)
        ip = {r["name"]: np.full(len(y), np.nan) for r in recipes}
        for inner in sorted(set(range(5)) - {outer}):
            q, b, estimates = cache[tuple(sorted([outer, inner]))]
            keep = folds[q] == inner
            ib[q[keep]] = b[keep]
            for name, p in estimates.items():
                ip[name][q[keep]] = p[keep]
        if not np.isfinite(ib[fit]).all() or not np.isnan(ib[folds == outer]).all():
            raise ValueError("outer isolation failed")
        q, b, estimates = cache[(outer,)]
        arrays["baseline"][q] = b
        row = {"outer": outer}
        for kind, allowed in subsets.items():
            choice = select(y[fit], ib[fit], {k: p[fit] for k, p in ip.items()}, allowed, groups[fit])
            arrays[kind][q] = np.clip(b + choice["strength"] * (estimates[choice["recipe"]["name"]] - b), 0, 14)
            row[kind] = choice
        outer_rows.append(row)
        print(json.dumps({"event": "outer", "outer": outer, "recipe": row["candidate"]["recipe"],
                          "strength": row["candidate"]["strength"]}), flush=True)
    scores = {k: metrics(y, p, groups) for k, p in arrays.items()}
    result = {**scores, "outer": outer_rows, "acceptance": acceptance(scores["candidate"], scores["baseline"]),
              "family_bootstrap": paired_family_bootstrap(y, arrays["candidate"][None],
                     {k: arrays[k][None] for k in ["baseline", "density_mean", "no_prior_correction"]}, groups, 10000),
              "test_access": False, "seconds": time.monotonic() - start, "default_model_replaced": False}
    atomic_json(out / "results.json", result)
    atomic_npz(out / "predictions.npz", keys=keys, y=y, groups=groups, folds=folds, **arrays)
    write_predictions(out / "predictions.csv", keys, {"label": y, "groups": groups, "fold": folds, **arrays})
    print(json.dumps({"event": "complete", "acceptance": result["acceptance"], "seconds": result["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
