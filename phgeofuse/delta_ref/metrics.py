"""Predeclared tail metrics, constrained selection and paired family inference."""
from __future__ import annotations

import numpy as np
from .model import finite_array, blend_predictions


def masks(y):
    y = finite_array(y, 1)
    return {"all": np.ones(len(y), bool), "acid": y <= 4,
            "alkaline": y >= 10, "core": (y > 4) & (y < 10)}


def metrics(y, prediction, groups=None, low_homology=None):
    y, p = finite_array(y, 1), finite_array(prediction, 1)
    if y.shape != p.shape or not len(y):
        raise ValueError("nonempty aligned label/prediction arrays required")
    subsets = masks(y)
    if low_homology is not None:
        low = np.asarray(low_homology, bool)
        if low.shape != y.shape:
            raise ValueError("low-homology mask mismatch")
        subsets["low_homology"] = low
    result = {}
    for name, m in subsets.items():
        error = p[m] - y[m]
        row = {"count": int(m.sum()), "families": None if groups is None else len(set(np.asarray(groups)[m]))}
        row.update({k: None for k in ("rmse", "mae", "bias", "abs_bias", "within1")})
        if m.any():
            row.update(rmse=float(np.mean(error ** 2) ** .5), mae=float(np.mean(abs(error))),
                       bias=float(error.mean()), abs_bias=float(abs(error.mean())),
                       within1=float((abs(error) <= 1).mean()))
        result[name] = row
    core = subsets["core"]
    result["false_extreme_rate"] = float(((p[core] <= 4) | (p[core] >= 10)).mean()) if core.any() else None
    if np.ptp(y) and len(y) > 1:
        from scipy.stats import spearmanr
        result["r2"] = float(1 - np.sum((p-y)**2) / np.sum((y-y.mean())**2))
        result["pearson"] = float(np.corrcoef(y, p)[0, 1]) if np.ptp(p) else None
        result["spearman"] = float(spearmanr(y, p).statistic) if np.ptp(p) else None
    return result


def guardrails(candidate, baseline, tolerance=.01, false_extreme_tolerance=.005):
    failures = []
    for group in ("all", "core"):
        for metric in ("rmse", "mae"):
            a, b = candidate[group][metric], baseline[group][metric]
            if a is None or b is None or a > b + tolerance + 1e-12:
                failures.append(f"{group}.{metric}")
    a, b = candidate["false_extreme_rate"], baseline["false_extreme_rate"]
    if a is None or b is None or a > b + false_extreme_tolerance + 1e-12:
        failures.append("false_extreme_rate")
    return failures


def acceptance(candidate, baseline):
    failures = guardrails(candidate, baseline)
    for group in ("acid", "alkaline"):
        for metric, ratio in (("rmse", .9), ("mae", .9), ("abs_bias", .8)):
            a, b = candidate[group][metric], baseline[group][metric]
            if a is None or b is None or a > ratio*b + 1e-12:
                failures.append(f"{group}.{metric}")
    return {"passed": not failures, "failures": failures}


def selection_score(candidate, baseline, strength):
    if guardrails(candidate, baseline):
        return (float("inf"), float("inf"), float(strength))
    ratios = []
    for group in ("acid", "alkaline"):
        a, b = candidate[group]["rmse"], baseline[group]["rmse"]
        if a is None or b is None or b <= 0:
            return (float("inf"), float("inf"), float(strength))
        ratios.append(a / b)
    return (max(ratios), candidate["all"]["rmse"], float(strength))


def select_strength(y, baseline, transfer, dispersion, valid, groups=None, consistency=True):
    base_metrics = metrics(y, baseline, groups)
    rows = []
    for strength in np.linspace(0., 1., 11):
        result = blend_predictions(baseline, transfer, dispersion, float(strength), valid, consistency)
        score = metrics(y, result["prediction"], groups)
        rows.append({"strength": float(strength), "metrics": score,
                     "rank": selection_score(score, base_metrics, strength)})
    best = min(rows, key=lambda r: r["rank"])
    if not np.isfinite(best["rank"][0]):
        raise ValueError("selection requires both tails and a valid zero-strength baseline")
    return best, rows


def seed_summary(y, predictions, groups=None):
    a = finite_array(predictions, 2)
    rows = [metrics(y, p, groups) for p in a]
    result = {"per_seed": rows, "mean": {}, "std": {}}
    for group in masks(y):
        result["mean"][group], result["std"][group] = {}, {}
        for key in ("rmse", "mae", "bias", "abs_bias", "within1"):
            values = [r[group][key] for r in rows]
            result["mean"][group][key] = float(np.mean(values)) if all(v is not None for v in values) else None
            result["std"][group][key] = float(np.std(values, ddof=1)) if len(values)>1 and all(v is not None for v in values) else None
        for key in ("count", "families"):
            result["mean"][group][key] = rows[0][group][key]
    values = [r["false_extreme_rate"] for r in rows]
    result["mean"]["false_extreme_rate"] = float(np.mean(values)) if all(v is not None for v in values) else None
    return result


def paired_family_bootstrap(y, candidate, baselines, groups, draws=10000, seed=42):
    """Compare the MEAN of per-seed metrics, never metrics of mean predictions.

    Each resample draws families once and applies the identical indices to all
    seeds, models and endpoints. Bonferroni simultaneous intervals cover all
    reported acid/alkaline RMSE, MAE and absolute-bias comparisons.
    """
    y, candidate = finite_array(y, 1), finite_array(candidate, 2)
    groups = np.asarray(groups).astype(str)
    if candidate.shape[1] != len(y) or len(groups) != len(y) or draws < 100:
        raise ValueError("invalid bootstrap inputs")
    baselines = {k: finite_array(v, 2) for k, v in baselines.items()}
    if any(v.shape != candidate.shape for v in baselines.values()) or not baselines:
        raise ValueError("comparisons require matched sample and seed axes")
    unique, inverse = np.unique(groups, return_inverse=True)
    if len(unique) < 2:
        raise ValueError("family inference needs at least two families")
    subsets = masks(y)
    subsets.pop("core")
    models = {"candidate": candidate, **baselines}
    totals = {}
    for group, mask in subsets.items():
        count = np.bincount(inverse, weights=mask, minlength=len(unique))
        for name, prediction in models.items():
            e = prediction - y
            arrays = [np.array([np.bincount(inverse, weights=w * mask, minlength=len(unique)) for w in weights])
                      for weights in (e ** 2, abs(e), e)]
            totals[(name, group)] = (count, *arrays)
    samples = {(name, group, metric): [] for name in baselines for group in subsets
               for metric in ("rmse", "mae", "abs_bias")}
    rng = np.random.default_rng(seed)
    for start in range(0, draws, 128):
        size = min(128, draws-start)
        sampled = rng.integers(len(unique), size=(size, len(unique)))
        multiplicity = np.array([np.bincount(row, minlength=len(unique)) for row in sampled])
        scores = {}
        for (name, group), (count, sse, sae, signed) in totals.items():
            denominator = multiplicity @ count
            safe = np.where(denominator > 0, denominator, np.nan)
            scores[(name, group)] = {
                "rmse": np.sqrt((multiplicity @ sse.T) / safe[:, None]).mean(1),
                "mae": ((multiplicity @ sae.T) / safe[:, None]).mean(1),
                "abs_bias": abs((multiplicity @ signed.T) / safe[:, None]).mean(1)}
        for key in samples:
            name, group, metric = key
            samples[key].extend(scores[("candidate", group)][metric] - scores[(name, group)][metric])
    corrected_alpha = .05 / (len(baselines) * 2 * 3)
    result = {"draws": draws, "seed": seed, "family_count": len(unique),
              "unit": "family", "seed_aggregation": "mean of metrics", "comparisons": {}}
    for (name, group, metric), values in samples.items():
        v = np.asarray(values)
        valid = v[np.isfinite(v)]
        alpha = corrected_alpha if group != "all" else .05
        row = {"valid_draws": len(valid), "ci95": None, "simultaneous_ci": None,
               "supported_improvement": False}
        if len(valid) >= .95 * draws:
            row["ci95"] = np.quantile(valid, [.025, .975]).tolist()
            row["simultaneous_ci"] = np.quantile(valid, [alpha/2, 1-alpha/2]).tolist()
            row["supported_improvement"] = row["simultaneous_ci"][1] < 0
        result["comparisons"].setdefault(name, {}).setdefault(group, {})[metric] = row
    return result
