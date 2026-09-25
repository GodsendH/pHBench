"""Read-only diagnostics of completed DeltaRef outer folds; never selects a model.

Only development data are opened. Output must be new and outside the frozen
implementation. The runner can continue while this CPU-only analysis executes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from phgeofuse.delta_ref.data import DevelopmentData, write_predictions
from phgeofuse.delta_ref.metrics import acceptance, masks, metrics
from phgeofuse.delta_ref.model import blend_predictions, huber_location, ph_bins, weighted_median
from phgeofuse.delta_ref.training import frequency_weights, load_bundle, sample_pairs


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def correlation(a, b):
    return float(spearmanr(a, b).statistic) if len(a) > 1 and np.ptp(a) and np.ptp(b) else None


@torch.inference_mode()
def panel_estimates(predictor, data, index):
    """Evaluate the fixed panel with same-family references removed on both sets."""
    net = predictor.network
    x = predictor.scaler.transform(data.x[index])
    ref = net.encode(torch.from_numpy(predictor.scaler.transform(predictor.features)))
    estimates = []
    for start in range(0, len(x), 128):
        q = net.encode(torch.from_numpy(x[start:start + 128]))
        d = net.difference(q[:, None, :].expand(-1, len(ref), -1),
                           ref[None, :, :].expand(len(q), -1, -1))
        estimates.append(d.numpy().astype(float) + predictor.labels)
    t = np.concatenate(estimates)
    w = np.broadcast_to(predictor.weights, t.shape).copy()
    bins = ph_bins(predictor.labels)
    for i, idx in enumerate(index):
        w[i, (predictor.keys == data.keys[idx]) | (predictor.groups == data.groups[idx])] = 0
    for b in np.unique(bins):
        m = bins == b
        mass = w[:, m].sum(1)
        w[:, m] *= np.divide(predictor.weights[m].sum(), mass,
                             out=np.zeros(len(mass)), where=mass > 0)[:, None]
    valid = np.array([len(set(predictor.groups[row > 0])) >= 3 for row in w])
    if not valid.all():
        raise ValueError("diagnostics require at least three foreign reference families per query")
    w /= w.sum(1, keepdims=True)
    pooled = huber_location(t, w)
    median = weighted_median(t, w)
    dispersion = weighted_median(abs(t - median[:, None]), w)
    reference_check = predictor.transfer(data.x[index], data.keys[index], data.groups[index])
    np.testing.assert_allclose(pooled, reference_check[0], atol=1e-10, rtol=0)
    np.testing.assert_allclose(dispersion, reference_check[1], atol=1e-10, rtol=0)
    np.testing.assert_array_equal(valid, reference_check[2])
    return t, w, pooled, dispersion


def fit_diagnostics(data, fit, recipe, predictor):
    y, groups = data.labels[fit], data.groups[fit]
    estimates, ref_weights, transfer, dispersion = panel_estimates(predictor, data, fit)
    pair_mse = (ref_weights * (estimates - y[:, None]) ** 2).sum(1)
    result = {
        "scope": "in-sample queries with self and same-family references excluded; not generalization evidence",
        "transfer_metrics": metrics(y, transfer, groups),
        "panel_pair_rmse": {name: float(np.sqrt(pair_mse[m].mean())) for name, m in masks(y).items()},
    }
    qi, ri = sample_pairs(y, groups, np.random.default_rng(42), count=8, natural=4)
    fw = frequency_weights(y, recipe["power"])
    weights = np.sqrt(fw[qi] * fw[ri]).astype(float)
    unique, inverse = np.unique(groups, return_inverse=True)
    endpoint = np.bincount(inverse[qi], weights=weights, minlength=len(unique))
    endpoint += np.bincount(inverse[ri], weights=weights, minlength=len(unique))
    shares = endpoint / endpoint.sum()
    counts = Counter(groups)
    result["family_exposure"] = {
        "queries": len(fit), "families": len(unique), "pairs_per_epoch": len(qi),
        "largest_family_queries": max(counts.values()),
        "largest_family_pair_endpoint_share": float(shares.max()),
        "largest_10_family_pair_endpoint_share": float(np.sort(shares)[-10:].sum()),
        "kish_family_count_from_endpoint_weights": float(1 / np.sum(shares ** 2)),
        "note": "Kish count describes endpoint-weight concentration; it is not an estimate of independent observations",
    }
    result["label_bin_exposure"] = []
    bins = ph_bins(y)
    for b in np.unique(bins):
        m = bins == b
        result["label_bin_exposure"].append({
            "bin": int(b), "queries": int(m.sum()), "families": len(set(groups[m])),
            "weighted_query_share": float(weights[m[qi]].sum() / weights.sum()),
            "weighted_reference_share": float(weights[m[ri]].sum() / weights.sum()),
        })
    return result


def learning_diagnostics(outer_dir):
    rows, curves = [], {}
    for path in sorted(outer_dir.glob("*/inner*/history.json")):
        history = json.loads(path.read_text())
        info = json.loads((path.parent / "complete.json").read_text())
        name = str(path.parent.relative_to(outer_dir))
        curves[name] = history
        rows.append({
            "name": name, "best_epoch": info["best_epoch"], "epochs": len(history),
            "first_training_loss": history[0]["training_loss"],
            "last_training_loss": history[-1]["training_loss"],
            "loss_reduction_percent": 100 * (1 - history[-1]["training_loss"] / history[0]["training_loss"]),
            "raw_validation_rmse_increase": history[-1]["standalone"]["all"]["rmse"] - history[0]["standalone"]["all"]["rmse"],
        })
    return {"fits": rows, "first_epoch_selected": sum(r["best_epoch"] == 1 for r in rows),
            "last_raw_rmse_worse": sum(r["raw_validation_rmse_increase"] > 0 for r in rows),
            "median_loss_reduction_percent": float(np.median([r["loss_reduction_percent"] for r in rows]))}, curves


def make_plot(output, result, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    base, pred = result["baseline"], result["candidate"]
    names = ["all", "core", "acid", "alkaline"]
    pos = np.arange(4)
    axs[0, 0].bar(pos - .18, [base[g]["rmse"] for g in names], .36, label="Matched baseline", color="#8b98a8")
    axs[0, 0].bar(pos + .18, [pred[g]["rmse"] for g in names], .36, label="DeltaRef", color="#266b99")
    axs[0, 0].set(xticks=pos, xticklabels=["All", "Core", "Acid", "Alkaline"], ylabel="RMSE (pH)", title="A. Held-out errors, outer fold 0")
    axs[0, 0].legend(frameon=False, fontsize=9)
    for i, g in enumerate(names):
        change = 100 * (pred[g]["rmse"] / base[g]["rmse"] - 1)
        axs[0, 0].text(i, max(base[g]["rmse"], pred[g]["rmse"]) + .08, f"{change:+.2f}%", ha="center", fontsize=9)
    colors = ["#c87436", "#416ba5"]
    for color, name in zip(colors, ["acid", "alkaline"]):
        d = result["transfer_diagnostics"][name]
        values = [d[k] for k in ["mean_true", "mean_base", "mean_transfer", "mean_final"]]
        axs[0, 1].plot(range(4), values, "o-", color=color, label=name.capitalize())
        for i, v in enumerate(values): axs[0, 1].annotate(f"{v:.2f}", (i, v), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9)
    axs[0, 1].set(xticks=range(4), xticklabels=["True mean", "Baseline", "Transfer T", "Final"],
                  ylabel="Mean pH", ylim=(2.7, 11.5), title="B. Substantial tail shrinkage remains")
    axs[0, 1].legend(frameon=False, fontsize=9)
    for history in curves.values():
        epochs = [h["epoch"] for h in history]
        initial_loss = history[0]["training_loss"]
        initial_rmse = history[0]["standalone"]["all"]["rmse"]
        axs[1, 0].plot(epochs, [h["training_loss"] / initial_loss for h in history], color="#8b98a8", alpha=.35)
        axs[1, 1].plot(epochs, [h["standalone"]["all"]["rmse"] - initial_rmse for h in history], color="#266b99", alpha=.4)
    axs[1, 0].set(xlabel="Epoch", ylabel="Training pair loss / first-epoch loss", title="C. Training loss drops in all 16 inner fits", yscale="log")
    axs[1, 1].axhline(0, color="#777777", linewidth=.8, linestyle="--")
    axs[1, 1].set(xlabel="Epoch", ylabel="Transfer validation RMSE minus first epoch", title="D. Last-epoch error rises in all 16 fits")
    fig.suptitle("DeltaRef-pH interim diagnosis | seed 42 | descriptive only", fontsize=15)
    fig.savefig(output / "diagnostics.png", dpi=170)
    fig.savefig(output / "diagnostics.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--outer", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("preserve prior analyses; use a new output directory")
    if args.outer != 0: raise ValueError("current visualization is specific to outer0")
    torch.set_num_threads(2)
    if hasattr(os, "nice"): os.nice(10)
    run = args.run.resolve()
    protocol = json.loads((run / "protocol.json").read_text())
    for name, digest in protocol["source_hashes"].items():
        if sha(ROOT / name) != digest: raise ValueError(f"frozen source changed: {name}")
    raw_outer = json.loads((run / "frozen/outer_results.json").read_text())
    outer = next(row for row in raw_outer if row["outer"] == args.outer)
    data = DevelopmentData.load(ROOT / "configs/delta_ref_phopt.yaml")
    if data.provenance != protocol["inputs"]: raise ValueError("development data changed since freeze")
    fit, query = data.partition([args.outer])
    winner = outer["winner"]
    predictor, metadata = load_bundle(winner["refit"], "cpu")
    baseline_file = run / "baseline/seed42" / f"excluded_{args.outer}/predictions.npz"
    with np.load(baseline_file, allow_pickle=False) as z:
        order = {k: i for i, k in enumerate(z["keys"])}
        index = [order[k] for k in data.keys[query]]
        baseline = z["prediction"][index]
        low = z["low_homology"][index]
    estimates, weights, transfer, dispersion = panel_estimates(predictor, data, query)
    pred = blend_predictions(baseline, transfer, dispersion, winner["strength"])
    y, groups = data.labels[query], data.groups[query]
    base = metrics(y, baseline, groups, low)
    candidate = metrics(y, pred["prediction"], groups, low)
    for subset in masks(y):
        for metric in ("rmse", "mae", "bias"):
            np.testing.assert_allclose(base[subset][metric], outer["baseline"][subset][metric], atol=1e-10, rtol=0)
            np.testing.assert_allclose(candidate[subset][metric], outer["metrics"][subset][metric], atol=2e-6, rtol=0)
    result = {"observed_at": datetime.now().astimezone().isoformat(), "seed": 42, "outer": args.outer,
              "scope": "interim outer-fold diagnostics; no fitting or hyperparameter selection; no test-feature access",
              "baseline": base, "candidate": candidate, "acceptance_diagnostic": acceptance(candidate, base),
              "winner": {k: winner[k] for k in ("recipe", "strength", "epochs")},
              "frozen_source_files_verified": len(protocol["source_hashes"]),
              "transfer_diagnostics": {}, "reference_bin_bias": []}
    pair_mse = (weights * (estimates - y[:, None]) ** 2).sum(1)
    for name, m in masks(y).items():
        gain = (baseline - y) ** 2 - (pred["prediction"] - y) ** 2
        result["transfer_diagnostics"][name] = {
            "count": int(m.sum()), "mean_true": float(y[m].mean()),
            "mean_base": float(baseline[m].mean()), "mean_transfer": float(transfer[m].mean()),
            "mean_final": float(pred["prediction"][m].mean()), "mean_correction": float(pred["correction"][m].mean()),
            "mean_agreement": float(pred["agreement"][m].mean()), "mean_effective_strength": float((winner["strength"] * pred["agreement"][m]).mean()),
            "raw_rmse": float(np.sqrt(np.mean((transfer[m] - y[m]) ** 2))),
            "panel_pair_rmse": float(np.sqrt(pair_mse[m].mean())),
            "sse_change": float(-gain[m].sum()), "improved_samples": int((gain[m] > 0).sum()),
            "low_dispersion_queries": int((dispersion[m] <= .5).sum()),
            "low_dispersion_but_transfer_error_over1": int(((dispersion[m] <= .5) & (abs(transfer[m] - y[m]) > 1)).sum()),
            "dispersion_vs_abs_transfer_error_spearman": correlation(dispersion[m], abs(transfer[m] - y[m])),
            "clip4_active_count": int((abs(transfer[m] - baseline[m]) >= 4).sum()),
        }
        for b in np.unique(ph_bins(predictor.labels)):
            refmask = ph_bins(predictor.labels) == b
            result["reference_bin_bias"].append({"subset": name, "reference_bin": int(b),
                "references": int(refmask.sum()),
                "mean_prediction": float(estimates[m][:, refmask].mean()),
                "signed_bias": float((estimates[m][:, refmask] - y[m, None]).mean())})
    result["in_sample_fit_diagnostics"] = fit_diagnostics(data, fit, winner["recipe"], predictor)
    result["learning_curves"], curves = learning_diagnostics(run / "frozen" / f"outer{args.outer}")
    result["inner_recipe_comparison"] = []
    for choice in outer["choices"]:
        selection = json.loads((run / "frozen" / f"outer{args.outer}" / choice["recipe"]["name"] / "selection.json").read_text())
        b = selection["grid"][0]["metrics"]
        c = choice["metrics"]
        result["inner_recipe_comparison"].append({"recipe": choice["recipe"]["name"], "strength": choice["strength"],
            "acid_rmse_improvement_percent": 100 * (1 - c["acid"]["rmse"] / b["acid"]["rmse"]),
            "alkaline_rmse_improvement_percent": 100 * (1 - c["alkaline"]["rmse"] / b["alkaline"]["rmse"]),
            "all_rmse_change": c["all"]["rmse"] - b["all"]["rmse"],
            "core_rmse_change": c["core"]["rmse"] - b["core"]["rmse"]})
    args.output.mkdir(parents=True)
    dump(args.output / "analysis.json", result)
    pd.DataFrame(result["reference_bin_bias"]).to_csv(args.output / "reference_bin_bias.csv", index=False)
    pd.DataFrame(result["learning_curves"]["fits"]).to_csv(args.output / "learning_curves.csv", index=False)
    write_predictions(args.output / "predictions.csv", data.keys[query],
                      {"label": y, "group": groups, "low_homology": low, **pred})
    make_plot(args.output, result, curves)
    sources = [Path(__file__), run / "protocol.json", run / "frozen/outer_results.json", baseline_file,
               Path(winner["refit"]) / "model.json", Path(winner["refit"]) / "weights.pt"]
    sources += sorted((run / "frozen" / f"outer{args.outer}").glob("*/inner*/history.json"))
    sources += sorted((run / "frozen" / f"outer{args.outer}").glob("*/selection.json"))
    dump(args.output / "provenance.json", {"sources_sha256": {str(p): sha(p) for p in sources},
        "data_provenance": data.provenance, "frozen_source_hashes": protocol["source_hashes"],
        "command": sys.argv, "python": sys.executable,
        "notes": "CPU inference differences from original GPU checked to 2e-6; query labels used for metrics only"})
    print(json.dumps({"output": str(args.output), "acceptance": result["acceptance_diagnostic"],
        "fit_transfer_rmse": result["in_sample_fit_diagnostics"]["transfer_metrics"]["all"]["rmse"],
        "heldout_transfer_rmse": result["transfer_diagnostics"]["all"]["raw_rmse"]}, indent=2))


if __name__ == "__main__": main()
