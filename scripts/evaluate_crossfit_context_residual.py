"""Fixed-recipe residual hypotheses using complete-baseline cross-fitting.

This reports every predeclared recipe, with no recipe selected on the outer
labels. It is exploratory grouped CV, not a nested hyperparameter-selection
estimate. For outer o, training residuals use baselines excluding {o, inner};
query baselines exclude o. No held-out outer label enters either component.
"""
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
from localph.kernel import ContextRidge, training_weights
from phgeofuse.cache import atomic_json, sha256_file
from phgeofuse.delta_ref.data import DevelopmentData, freeze_json, atomic_npz, stable_hash
from phgeofuse.delta_ref.metrics import metrics, acceptance


class ResidualRidge(ContextRidge):
    """Reuse the closed-form solver; weight by true pH, never residual bins."""
    def fit_residual(self, gx, sx, residual, labels):
        from localph.features import SiteScaler
        self.scaler = SiteScaler().fit(sx)
        x = self.features(gx, sx)
        w = training_weights(labels, self.power)
        self.x_mean, self.y_mean = np.average(x, weights=w, axis=0), np.average(residual, weights=w)
        tx = torch.tensor((x - self.x_mean) * np.sqrt(w[:, None]), dtype=torch.float64, device=self.device)
        ty = torch.tensor((residual - self.y_mean) * np.sqrt(w), dtype=torch.float64, device=self.device)
        gram = tx @ tx.T
        gram.diagonal().add_(self.alpha)
        sol = torch.cholesky_solve(ty[:, None], torch.linalg.cholesky(gram))[:, 0]
        self.coef = (tx.T @ sol).cpu().numpy()
        return self


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
    keys, y, groups, folds, gx = data.keys[idx], data.labels[idx], data.groups[idx], data.folds[idx], data.embeddings[idx]
    features = ROOT / "experiments/ion_context_phopt_20260917/features_train/features.npz"
    with np.load(features, allow_pickle=False) as z:
        if not np.array_equal(z["keys"], keys):
            raise ValueError("site keys differ")
        sx = z["features"].copy()
    baseline_root = ROOT / "experiments/delta_ref_phopt_20260916/baseline/seed42"
    recipes = [{"name": f"site{s:g}_a{a:g}_p{p:g}", "site_weight": s, "alpha": a, "power": p}
               for s, a, p in itertools.product([0., .5], [2., 20.], [0., .5])]
    baseline_hashes = {}
    for directory in sorted(baseline_root.glob("excluded_*")):
        for filename in ("fit.json", "predictions.npz"):
            p = directory / filename
            baseline_hashes[str(p.relative_to(ROOT))] = sha256_file(p)
    protocol = {"scope": "fixed-recipe exploratory grouped PHOPT-only CV; no outer-label hyperparameter selection",
                "recipes": recipes, "strengths": [.25, .5], "residual_clip": 2., "test_access": False,
                "source_hashes": {str(p.relative_to(ROOT)): sha256_file(p) for p in
                    [Path(__file__), ROOT / "localph/kernel.py", ROOT / "localph/features.py", ROOT / "phgeofuse/delta_ref/metrics.py"]},
                "features_sha256": sha256_file(features), "baseline_hashes": baseline_hashes,
                "development_provenance": data.provenance, "residual_weights": "true pH bins, not residual bins",
                "selection_warning": "Any best row chosen after viewing these outputs is exploratory and requires fresh confirmation"}
    freeze_json(out / "protocol.json", protocol)

    def load(excluded):
        fit = np.flatnonzero(~np.isin(folds, excluded))
        q = np.flatnonzero(np.isin(folds, excluded))
        directory = baseline_root / ("excluded_" + "_".join(map(str, excluded)))
        cert = json.loads((directory / "fit.json").read_text())
        if (cert["fit_keys"] != keys[fit].tolist() or cert["query_keys"] != keys[q].tolist()
                or cert["fit_label_sha256"] != stable_hash(y[fit].tolist()) or set(groups[fit]) & set(groups[q])):
            raise ValueError("baseline certificate differs")
        with np.load(directory / "predictions.npz", allow_pickle=False) as z:
            if not np.array_equal(z["keys"], keys[q]):
                raise ValueError("baseline query keys differ")
            return q, z["prediction"].copy()

    base = np.full(len(y), np.nan)
    residual_predictions = {r["name"]: np.full(len(y), np.nan) for r in recipes}
    diagnostics = []
    for outer in range(5):
        fit = np.flatnonzero(folds != outer)
        inner_baseline = np.full(len(y), np.nan)
        for inner in sorted(set(range(5)) - {outer}):
            q, b = load(tuple(sorted([outer, inner])))
            keep = folds[q] == inner
            inner_baseline[q[keep]] = b[keep]
        if not np.isfinite(inner_baseline[fit]).all() or not np.isnan(inner_baseline[folds == outer]).all():
            raise ValueError("outer isolation failed")
        q, b = load((outer,))
        base[q] = b
        for recipe in recipes:
            begin = time.monotonic()
            model = ResidualRidge(**{k: v for k, v in recipe.items() if k != "name"}, device="cuda")
            model.fit_residual(gx[fit], sx[fit], y[fit] - inner_baseline[fit], y[fit])
            correction = model.predict(gx[q], sx[q])
            residual_predictions[recipe["name"]][q] = correction
            tr = model.predict(gx[fit], sx[fit])
            diagnostics.append({"outer": outer, "recipe": recipe, "seconds": time.monotonic() - begin,
                                "training_residual_rmse": float(np.mean((tr - y[fit] + inner_baseline[fit]) ** 2) ** .5),
                                "heldout_residual_rmse": float(np.mean((correction - y[q] + b) ** 2) ** .5)})
            atomic_npz(out / f"outer{outer}_{recipe['name']}.npz", coef=model.coef, x_mean=model.x_mean,
                       y_mean=np.array(model.y_mean), site_mean=model.scaler.mean, site_scale=model.scaler.scale,
                       keys=keys[q], correction=correction)
            print(json.dumps({"outer": outer, "recipe": recipe["name"], "elapsed": time.monotonic() - start}), flush=True)
    bm = metrics(y, base, groups)
    results = {}
    saved = {"keys": keys, "y": y, "groups": groups, "folds": folds, "baseline": base}
    for recipe in recipes:
        for strength in [.25, .5]:
            name = recipe["name"] + f"_strength{strength:g}"
            p = np.clip(base + strength * np.clip(residual_predictions[recipe["name"]], -2., 2.), 0, 14)
            m = metrics(y, p, groups)
            results[name] = {"metrics": m, "acceptance": acceptance(m, bm), "recipe": recipe, "strength": strength}
            saved[name] = p
    atomic_npz(out / "predictions.npz", **saved)
    atomic_json(out / "results.json", {"baseline": bm, "fixed_recipes": results, "diagnostics": diagnostics,
                "seconds": time.monotonic() - start, "default_model_replaced": False,
                "any_internal_pass": any(r["acceptance"]["passed"] for r in results.values()), "test_access": False})
    print(json.dumps({"event": "complete", "seconds": time.monotonic() - start,
                      "passes": [k for k, r in results.items() if r["acceptance"]["passed"]]}), flush=True)


if __name__ == "__main__":
    main()
