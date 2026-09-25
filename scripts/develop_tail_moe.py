"""Leakage-controlled tail-aware mixture-of-experts search.

The base predictions and retrieval rows come from the strict homology outer
fold cache.  For each outer fold, all classifiers/regressors are fitted only
on the other four folds and evaluated on the held-out fold.  This script is a
development experiment on PHOPT train rows; it never reads PHOPT test labels.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))
from develop_phgeofuse_regression import OUT, ROOT, metrics
from phgeofuse.cache import atomic_json
from phgeofuse.io import read_manifest
from phgeofuse.robust_fusion import chemistry_features


def load_data():
    source = OUT / "nested_homology_strict"
    pred = np.load(source / "predictions.npz", allow_pickle=False)
    records = [r for r in read_manifest(ROOT / "artifacts/phgeofuse/manifest.csv") if r.split == "train"]
    assert len(records) == 7124
    keys = [f"train::{r.protein_id}" for r in records]
    assert np.array_equal(pred["keys"].astype(str), np.asarray(keys))
    y = pred["y"].astype(float)
    fold = pred["fold"].astype(int)
    groups = pred["groups"].astype(str)
    retrieval = pred["retrieval"].astype(float)
    chem = chemistry_features([r.sequence for r in records])
    # Strict outer predictions are all train-only base outputs.  Keep the
    # independent sequence and retrieval experts as explicit inputs.
    names = ["sequence", "anchor", "unweighted50", "unweighted150",
             "phweighted50", "family_phweighted50"]
    for n in names:
        assert np.isfinite(pred[n]).all()
    return pred, y, fold, groups, retrieval, chem, names


def base_features(pred, retrieval, chem, names):
    p = np.column_stack([pred[n] for n in names])
    # Reliability and disagreement features are label-free and available at
    # inference.  The raw retrieval values are retained for the gate.
    seq = pred["sequence"]
    anchor = pred["anchor"]
    r = retrieval
    avail = r[:, 7:9]
    mean_retr = np.where(avail.sum(1) > 0,
                         (r[:, :2] * avail).sum(1) / np.maximum(avail.sum(1), 1),
                         seq)
    stack = np.column_stack([
        p, r, chem,
        mean_retr, mean_retr - seq, anchor - seq,
        np.abs(mean_retr - seq), np.abs(anchor - seq),
        r[:, 4] * r[:, 9] * r[:, 10],
        r[:, 2] * r[:, 3],
        r[:, 5] + r[:, 6],
    ])
    assert np.isfinite(stack).all()
    return stack


def make_classifier(kind):
    if kind == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.2, max_iter=500, multi_class="multinomial"))
    if kind == "hgb":
        return HistGradientBoostingClassifier(max_leaf_nodes=5, max_iter=60, learning_rate=.05,
                                              min_samples_leaf=100, l2_regularization=20,
                                              early_stopping=False, random_state=42)
    raise ValueError(kind)


def make_regressor(kind, seed=42):
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=30.0))
    if kind == "hgb":
        return HistGradientBoostingRegressor(max_leaf_nodes=5, max_iter=60, learning_rate=.05,
                                             min_samples_leaf=100, l2_regularization=40,
                                             early_stopping=False, random_state=seed)
    if kind == "extra":
        return ExtraTreesRegressor(n_estimators=160, max_features=.55, min_samples_leaf=80,
                                   max_depth=10, random_state=seed, n_jobs=4)
    if kind == "forest":
        return RandomForestRegressor(n_estimators=160, max_features=.55, min_samples_leaf=80,
                                     max_depth=10, random_state=seed, n_jobs=4)
    raise ValueError(kind)


def fit_predict_tail(Xtr, ytr, Xte, clf_kind, reg_kind, transform, smooth, weight_power):
    """Fit a probability gate and three conditional experts."""
    # Three broad regimes are deliberately defined from training labels only;
    # these boundaries are fixed before evaluation and are not tuned per fold.
    cls = np.where(ytr < 6, 0, np.where(ytr < 8, 1, 2))
    clf = make_classifier(clf_kind).fit(Xtr, cls)
    prob = clf.predict_proba(Xte)
    # Ensure all classes exist in the fit set and probabilities have a stable
    # column order.
    classes = np.asarray(clf.classes_)
    fullprob = np.full((len(Xte), 3), 0.0)
    for j, c in enumerate(classes):
        fullprob[:, int(c)] = prob[:, j]
    experts = []
    train_prob = clf.predict_proba(Xtr)
    full_train_prob = np.zeros((len(Xtr), 3))
    for j, c in enumerate(classes):
        full_train_prob[:, int(c)] = train_prob[:, j]
    # Soft responsibilities prevent the hard class gate from amplifying
    # classification mistakes and retain enough tail data for each expert.
    for c in range(3):
        if transform == "asinh":
            target = np.arcsinh((ytr - 7.0) / 2.0)
        elif transform == "signed_log":
            target = np.sign(ytr - 7.0) * np.log1p(np.abs(ytr - 7.0))
        else:
            target = ytr
        # Mild responsibility weighting is clipped to avoid overfitting rare
        # acidic examples.  It is fixed before seeing held-out labels.
        w = np.clip((full_train_prob[:, c] + 0.05) ** weight_power, .25, 2.0)
        reg = make_regressor(reg_kind)
        reg.fit(Xtr, target, **({"sample_weight": w} if reg_kind in {"hgb", "extra", "forest"} else {}))
        q = reg.predict(Xte)
        if transform == "asinh":
            q = 7.0 + 2.0 * np.sinh(q)
        elif transform == "signed_log":
            q = 7.0 + np.sign(q) * np.expm1(np.abs(q))
        experts.append(q)
    expert_pred = np.column_stack(experts)
    # The mild smoothing avoids extreme gate probabilities.  If smooth=0 it
    # is a standard probability mixture; otherwise mix with uniform prior.
    fullprob = (1.0 - smooth) * fullprob + smooth / 3.0
    return (fullprob * expert_pred).sum(1), fullprob, expert_pred, clf


def main():
    pred, y, fold, groups, retrieval, chem, names = load_data()
    X = base_features(pred, retrieval, chem, names)
    out = OUT / "tail_moe"
    out.mkdir(exist_ok=True)
    configs = []
    # Keep the first search deliberately small and low-capacity.  Tree
    # ensembles and large grids are reserved for a later, pre-registered
    # ablation because thousands of fits would obscure the generalization
    # comparison.
    for clf in ["logistic", "hgb"]:
        for reg in ["ridge", "hgb"]:
            for transform in ["raw", "asinh"]:
                for smooth in [0.0, .10]:
                    for power in [0.0, .5]:
                        configs.append({"name": f"{clf}_{reg}_{transform}_s{smooth}_w{power}",
                                        "clf": clf, "reg": reg, "transform": transform,
                                        "smooth": smooth, "weight_power": power})
    # Include simpler monotone calibrators as references.
    results = {c["name"]: np.full(len(y), np.nan) for c in configs}
    perfold = []
    for outer in range(5):
        tr = np.flatnonzero(fold != outer)
        te = np.flatnonzero(fold == outer)
        for i, c in enumerate(configs):
            p, prob, ep, model = fit_predict_tail(X[tr], y[tr], X[te], c["clf"], c["reg"],
                                                   c["transform"], c["smooth"], c["weight_power"])
            results[c["name"]][te] = p
            # Save only a few representative models; predictions remain the
            # authoritative nested evidence for model selection.
            if c["name"] in {"logistic_hgb_asinh_s0.1_w0.5", "hgb_hgb_asinh_s0.1_w0.5"}:
                joblib.dump(model, out / f"outer{outer}_{c['name']}.joblib")
        print("TAIL_MOE_OUTER_COMPLETE", outer, flush=True)
    # Compute metrics, with a composite rank that rewards improvements in all
    # requested axes without allowing a large RMSE regression.
    low = ~((retrieval[:, 4] >= .2) & (retrieval[:, 9] >= .8) & (retrieval[:, 10] >= .8))
    summary = {}
    for c in configs:
        p = results[c["name"]]
        m = metrics(y, p, low)
        score = m["rmse"] + .04 * abs(m["acidic"]["bias"]) + .04 * abs(m["alkaline"]["bias"]) + .02 * m["low_homology"]["rmse"]
        summary[c["name"]] = {"config": c, "metrics": m, "composite": float(score)}
    best = sorted(summary.values(), key=lambda d: (d["composite"], d["metrics"]["rmse"]))
    np.savez(out / "predictions.npz", **results, y=y, fold=fold, groups=groups,
             keys=pred["keys"], retrieval=retrieval)
    atomic_json(out / "results.json", {"protocol": {"dataset": "PHOPT train only", "outer_folds": 5,
                                                    "base": "strict nested homology predictions", "test_access": False,
                                                    "configs": len(configs)},
                                       "results": summary, "best_by_composite": best[:20],
                                       "best_by_rmse": sorted(summary.values(), key=lambda d: d["metrics"]["rmse"])[:20]})
    print("TAIL_MOE_COMPLETE", json.dumps(best[:5], indent=2), flush=True)


if __name__ == "__main__":
    main()
