"""Strict nested evaluation of label-free titration and local-charge features.

The experiment uses only sequence-derived features.  For every outer homology
fold, retrieval rows and the ESM sequence prediction are loaded from caches
whose reference set excludes that outer fold.  The meta model is fitted only
on inner out-of-fold features from the remaining folds.  No PHOPT test labels
are read.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from develop_phgeofuse_regression import OUT, ROOT, metrics
from phgeofuse.cache import atomic_json
from phgeofuse.dual_fusion import retrieval_sequence_anchor
from phgeofuse.io import read_manifest
from phgeofuse.robust_fusion import chemistry_features


def _sigmoid_charge(pka: float, ph: np.ndarray, acidic: bool) -> np.ndarray:
    if acidic:
        return -1.0 / (1.0 + np.power(10.0, pka - ph))
    return 1.0 / (1.0 + np.power(10.0, ph - pka))


def titration_features(sequence: str) -> np.ndarray:
    """Return compact, label-free titration and local-charge descriptors.

    Standard pKa values are used only to produce a physically motivated prior;
    no measured pH or sample metadata enters these features.  Features include
    charge curves, buffering slopes, residue counts, terminal composition and
    local signed charge around titratable residues.
    """
    seq = np.asarray(list(sequence))
    n = max(len(seq), 1)
    pka = {"D": 3.9, "E": 4.3, "C": 8.3, "Y": 10.1,
           "H": 6.0, "K": 10.5, "R": 12.5}
    acidic = set("DECY")
    basic = set("HKR")
    grid = np.arange(2.0, 12.01, 0.5)
    charge = np.zeros_like(grid)
    buffering = np.zeros_like(grid)
    for aa in seq:
        aa = str(aa)
        if aa in pka:
            q = _sigmoid_charge(pka[aa], grid, aa in acidic)
            charge += q
            buffering += np.abs(np.gradient(q, grid))
    # Protein termini are useful but deliberately kept as a weak prior.
    charge += 1.0 / (1.0 + np.power(10.0, grid - 8.0))
    charge -= 1.0 / (1.0 + np.power(10.0, 3.1 - grid))
    charge /= n
    buffering /= n
    zero = float(grid[np.argmin(np.abs(charge))])
    counts = np.array([np.sum(seq == aa) / n for aa in "DE CYHKR".replace(" ", "")], dtype=float)
    # The string above is intentionally expanded to the seven ionizable types.
    counts = np.array([np.sum(seq == aa) / n for aa in "DECYHKR"], dtype=float)
    acidic_frac = np.array([np.mean(seq == aa) for aa in "DE"], dtype=float)
    basic_frac = np.array([np.mean(seq == aa) for aa in "KRH"], dtype=float)
    terminal = np.array([
        float(seq[0] in acidic), float(seq[0] in basic),
        float(seq[-1] in acidic), float(seq[-1] in basic),
    ])
    # Local windows summarize whether titratable residues sit in strongly
    # charged neighborhoods rather than only counting them globally.
    signed = np.array([1.0 if aa in basic else -1.0 if aa in acidic else 0.0 for aa in seq])
    local = []
    for width in (5, 11, 21):
        kernel = np.ones(width) / width
        smoothed = np.convolve(signed, kernel, mode="same")
        local.extend([float(np.mean(smoothed)), float(np.std(smoothed)),
                      float(np.min(smoothed)), float(np.max(smoothed)),
                      float(np.mean(np.abs(smoothed)))])
    thirds = []
    for part in np.array_split(seq, 3):
        m = max(len(part), 1)
        thirds.extend([float(np.mean(np.isin(part, list(acidic)))),
                       float(np.mean(np.isin(part, list(basic))))])
    values = np.concatenate([
        charge, buffering, [zero, charge[10], charge[12], charge[14],
                            charge[16], charge[18], charge[20]],
        counts, acidic_frac, basic_frac, terminal, local, thirds,
        [np.log1p(n), float(np.mean(signed)), float(np.std(signed))],
    ])
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("nonfinite titration features")
    return values


def load_cache(source: Path, excluded: list[int], folds: np.ndarray,
               keys: list[str]):
    tag = "_".join(map(str, sorted(excluded)))
    payload = __import__("torch").load(source / f"excluded_{tag}.pt", map_location="cpu")
    query = np.flatnonzero(np.isin(folds, excluded))
    if payload["metadata"]["query_keys"] != [keys[i] for i in query]:
        raise ValueError("strict cache query order mismatch")
    return query, np.asarray(payload["retrieval"], dtype=np.float64), np.asarray(payload["sequence"], dtype=np.float64)


def ridge_fit_predict(x, y, xt, alpha):
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(x, y)
    return model.predict(xt), model


def main() -> None:
    out = OUT / "titration_nested"
    out.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / "artifacts/phgeofuse/manifest.csv"
    records = [r for r in read_manifest(manifest) if r.split == "train"]
    if len(records) != 7124:
        raise ValueError(f"expected 7124 training records, found {len(records)}")
    keys = [f"train::{r.protein_id}" for r in records]
    folds_rows = json.loads((OUT / "homology_oof/strict_folds.json").read_text())["rows"]
    if [r["key"] for r in folds_rows] != [r.protein_id for r in records]:
        raise ValueError("strict fold key order mismatch")
    folds = np.asarray([r["fold"] for r in folds_rows], dtype=int)
    groups = np.asarray([r["group"] for r in folds_rows])
    sequences = [r.sequence for r in records]
    old_chem = chemistry_features(sequences)
    titration = np.asarray([titration_features(s) for s in sequences], dtype=np.float64)
    if not np.isfinite(titration).all():
        raise ValueError("invalid titration matrix")
    y = np.asarray([r.ph_opt for r in records], dtype=np.float64)
    source = OUT / "nested_homology_strict"
    predictions = {name: np.full(len(y), np.nan) for name in (
        "anchor", "dual", "aug_hgb", "aug_ridge10", "aug_ridge100",
        "blend75", "blend50", "physics_ridge")}
    fold_results = []
    for outer in range(5):
        tr = np.flatnonzero(folds != outer)
        te, r_outer, s_outer = load_cache(source, [outer], folds, keys)
        inner_r = np.full((len(y), 15), np.nan)
        inner_s = np.full(len(y), np.nan)
        for inner in range(5):
            if inner == outer:
                continue
            q, rr, ss = load_cache(source, [outer, inner], folds, keys)
            keep = folds[q] == inner
            inner_r[q[keep]] = rr[keep]
            inner_s[q[keep]] = ss[keep]
        if not np.isfinite(inner_s[tr]).all() or np.isfinite(inner_s[te]).any():
            raise ValueError("inner cache leakage")
        anchor_tr = retrieval_sequence_anchor(inner_r[tr], inner_s[tr])
        anchor_te = retrieval_sequence_anchor(r_outer, s_outer)
        predictions["anchor"][te] = anchor_te
        base_tr = np.column_stack([inner_r[tr], inner_s[tr], old_chem[tr]])
        base_te = np.column_stack([r_outer, s_outer, old_chem[te]])
        aug_tr = np.column_stack([base_tr, titration[tr]])
        aug_te = np.column_stack([base_te, titration[te]])
        hgb = HistGradientBoostingRegressor(
            max_leaf_nodes=7, max_iter=50, min_samples_leaf=80,
            l2_regularization=40, learning_rate=.05,
            early_stopping=False, random_state=42,
        ).fit(aug_tr, y[tr] - anchor_tr)
        dual_hgb = HistGradientBoostingRegressor(
            max_leaf_nodes=7, max_iter=50, min_samples_leaf=80,
            l2_regularization=30, learning_rate=.05,
            early_stopping=False, random_state=42,
        ).fit(base_tr, y[tr] - anchor_tr)
        p_dual = anchor_te + dual_hgb.predict(base_te)
        p_hgb = anchor_te + hgb.predict(aug_te)
        predictions["dual"][te] = p_dual
        predictions["aug_hgb"][te] = p_hgb
        for alpha, name in [(10., "aug_ridge10"), (100., "aug_ridge100")]:
            p_res, _ = ridge_fit_predict(aug_tr, y[tr] - anchor_tr, aug_te, alpha)
            predictions[name][te] = anchor_te + p_res
        # Physics-only sequence expert; this is intentionally independent of
        # retrieval and provides a safe fallback when no homolog is available.
        p_phys, _ = ridge_fit_predict(titration[tr], y[tr], titration[te], 100.)
        predictions["physics_ridge"][te] = p_phys
        predictions["blend75"][te] = .75 * p_dual + .25 * p_hgb
        predictions["blend50"][te] = .50 * p_dual + .50 * p_hgb
        row = {"outer": outer}
        low = ~((r_outer[:, 4] >= .2) & (r_outer[:, 9] >= .8) & (r_outer[:, 10] >= .8))
        for name in predictions:
            if np.isfinite(predictions[name][te]).all():
                row[name] = metrics(y[te], predictions[name][te], low)
        fold_results.append(row)
        atomic_json(out / f"outer{outer}_status.json", {"status": "complete", "outer": outer, "updated": time.time()})
        print(json.dumps(row), flush=True)
    if not all(np.isfinite(p).all() for p in predictions.values()):
        raise ValueError("incomplete outer predictions")
    low_all = np.zeros(len(y), dtype=bool)
    # Reconstruct the outer retrieval quality from each cache.
    retrieval = np.full((len(y), 15), np.nan)
    for outer in range(5):
        q, rr, _ = load_cache(source, [outer], folds, keys)
        retrieval[q] = rr
    low_all = ~((retrieval[:, 4] >= .2) & (retrieval[:, 9] >= .8) & (retrieval[:, 10] >= .8))
    result = {
        "protocol": {
            "dataset": "PHOPT train only",
            "count": len(y), "outer_folds": 5,
            "features": "standard-pKa titration curves, buffering slopes, local signed-charge windows, termini and composition",
            "retrieval_and_sequence": "strict nested caches exclude outer and inner folds",
            "test_access": False,
            "titration_dim": int(titration.shape[1]),
        },
        "metrics": {name: metrics(y, p, low_all) for name, p in predictions.items()},
        "fold_results": fold_results,
    }
    _, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    rng = np.random.default_rng(42)
    draws = rng.integers(len(counts), size=(1000, len(counts)))
    denom = counts[draws].sum(1)
    ref_sse = np.bincount(inv, weights=(predictions["dual"] - y) ** 2)
    result["cluster_bootstrap_delta_vs_dual"] = {}
    for name, p in predictions.items():
        sse = np.bincount(inv, weights=(p - y) ** 2)
        delta = np.sqrt(sse[draws].sum(1) / denom) - np.sqrt(ref_sse[draws].sum(1) / denom)
        result["cluster_bootstrap_delta_vs_dual"][name] = np.quantile(delta, [.025, .975]).tolist()
    np.savez(out / "predictions.npz", **predictions, y=y, fold=folds, groups=groups,
             keys=np.asarray(keys), retrieval=retrieval, titration=titration)
    atomic_json(out / "results.json", result)
    atomic_json(out / "status.json", {"status": "complete", "updated": time.time()})
    print("TITRATION_NESTED_COMPLETE", json.dumps(result["metrics"]), flush=True)


if __name__ == "__main__":
    main()
