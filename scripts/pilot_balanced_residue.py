"""Bounded inner-fold diagnostic for scarce-label underfitting; no outer score.

This is a pilot, not a nested performance estimate. It uses exactly outer 0,
inner 1 of the already declared grouping. Natural sparse/global controls are
retained by the main experiment; this pilot does not replace or interrupt it.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader
from localph.residue_field import ResidueField, label_prior
from localph.residue_training import PackedResidues, collate, predict, choose, standalone_objective, light_metrics
from localph.residue_objectives import smooth_rarity_weights, weighted_label_prior, rarity_loss
from phgeofuse.cache import atomic_json, sha256_file
from phgeofuse.delta_ref.data import DevelopmentData, freeze_json, atomic_npz, stable_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    data = DevelopmentData.load(ROOT / "configs/delta_ref_phopt.yaml")
    t = data.train
    keys, y, folds, groups = data.keys[t], data.labels[t], data.folds[t], data.groups[t]
    fit, query = np.flatnonzero(~np.isin(folds, [0, 1])), np.flatnonzero(folds == 1)
    if set(groups[fit]) & set(groups[query]):
        raise ValueError("family overlap")
    with np.load(args.features, allow_pickle=False) as z:
        if not np.array_equal(keys, z["keys"]):
            raise ValueError("feature alignment differs")
        packed = {k: z[k].copy() for k in ("tokens", "ionizable", "offsets")}
    source = ROOT / "experiments/delta_ref_phopt_20260916/baseline/seed42/excluded_0_1"
    cert = json.loads((source / "fit.json").read_text())
    if cert["fit_keys"] != keys[fit].tolist() or cert["fit_label_sha256"] != stable_hash(y[fit].tolist()):
        raise ValueError("baseline fit differs")
    with np.load(source / "predictions.npz", allow_pickle=False) as z:
        lookup = dict(zip(z["keys"], z["prediction"]))
        base = np.array([lookup[k] for k in keys[query]])
    source_files = [Path(__file__), ROOT / "localph/residue_objectives.py", ROOT / "localph/residue_field.py", ROOT / "localph/residue_training.py"]
    protocol = {"scope": "one inner fold pilot, never outer 0 labels for training or selection",
                "excluded_training_folds": [0, 1], "query_fold": 1, "seed": 42,
                "kinds": ["direct", "sparse"], "max_epochs": 40, "patience": 5,
                "weight": "training Gaussian density sigma0.5 to power -0.5; mean-normalize, cap8, re-normalize",
                "training_prior": "Gaussian targets averaged with training rarity weights",
                "decoding_prior": "natural unweighted training Gaussian prior",
                "optimizer": "same AdamW lr0.001 weight_decay0.05, batch32, dropout0.25",
                "checkpoint": "same inner fused ranking plus standalone objective tie-break",
                "source_hashes": {str(p.relative_to(ROOT)): sha256_file(p) for p in source_files},
                "features_sha256": sha256_file(args.features),
                "baseline_hashes": {str(p.relative_to(ROOT)): sha256_file(p) for p in [source / "fit.json", source / "predictions.npz"]},
                "fit_keys": keys[fit].tolist(), "query_keys": keys[query].tolist(),
                "fit_labels_sha256": stable_hash(y[fit].tolist()), "test_access": False,
                "original_validation_used": False, "data_provenance": data.provenance}
    freeze_json(args.output / "protocol.json", protocol)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    weights = smooth_rarity_weights(y[fit])
    full_weights = np.zeros(len(y), dtype=np.float32)
    full_weights[fit] = weights
    weight_tensor = torch.tensor(full_weights, device="cuda")
    start, results = time.monotonic(), {}
    for kind in protocol["kinds"]:
        folder = args.output / kind
        folder.mkdir()
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        np.random.seed(42)
        model = ResidueField(kind=kind).cuda()
        natural_prior = label_prior(y[fit], model.grid)
        training_prior = weighted_label_prior(y[fit], weights, model.grid)
        loader = DataLoader(PackedResidues(packed, fit, y), batch_size=32, shuffle=True,
                            generator=torch.Generator().manual_seed(42), collate_fn=collate, num_workers=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.05)
        history, best, best_state, stable_best, stale = [], None, None, float("inf"), 0
        for epoch in range(1, 41):
            model.train()
            total, count = 0., 0
            for tokens, mask, ion, labels, indices in loader:
                optimizer.zero_grad(set_to_none=True)
                weights_batch = weight_tensor[torch.as_tensor(indices, device="cuda")]
                loss = rarity_loss(model(tokens.cuda(), mask.cuda(), ion.cuda()), labels.cuda(), weights_batch,
                                   model.grid, training_prior, kind)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                total += float(loss.detach()) * len(labels)
                count += len(labels)
            p = predict(model, packed, query, y, natural_prior)
            choice = choose(y[query], base, p)
            objective = standalone_objective(y[query], p["direct" if kind == "direct" else "prior1_mean"])
            rank = [*choice["rank"], objective]
            row = {"epoch": epoch, "train_loss": total / count, "choice": choice,
                   "standalone_validation_objective": objective, "checkpoint_rank": rank}
            history.append(row)
            if best is None or tuple(rank) < tuple(best["checkpoint_rank"]):
                best, best_state = copy.deepcopy(row), {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if objective < stable_best - .002:
                stable_best, stale = objective, 0
            else:
                stale += 1
            atomic_json(folder / "history.json", history)
            atomic_json(args.output / "status.json", {"state": "training", "kind": kind, "epoch": epoch,
                                                     "seconds": time.monotonic() - start})
            print(json.dumps({"kind": kind, **row}), flush=True)
            if stale >= 5:
                break
        model.load_state_dict(best_state)
        p = predict(model, packed, query, y, natural_prior)
        train_p = predict(model, packed, fit, y, natural_prior)
        torch.save({"state_dict": best_state, "kind": kind, "prior": natural_prior.cpu(),
                    "training_prior": training_prior.cpu(), "epoch": best["epoch"], "seed": 42}, folder / "weights.pt")
        atomic_npz(folder / "predictions.npz", keys=keys[query], **p)
        report = {"selected_epoch": best["epoch"], "epochs_run": len(history), "choice": choose(y[query], base, p),
                  "train_metrics": {k: light_metrics(y[fit], v) for k, v in train_p.items()},
                  "query_metrics": {k: light_metrics(y[query], v) for k, v in p.items()}}
        atomic_json(folder / "complete.json", report)
        results[kind] = report
    atomic_npz(args.output / "training_weights.npz", keys=keys[fit], weights=weights)
    atomic_json(args.output / "results.json", {"models": results, "baseline": light_metrics(y[query], base),
                                               "seconds": time.monotonic() - start, "is_confirmatory": False,
                                               "weights_min_max": [float(weights.min()), float(weights.max())],
                                               "weights_tail_mean": {g: float(weights[m].mean()) for g, m in
                                                                     {"acid": y[fit] <= 4, "alkaline": y[fit] >= 10}.items()}})
    atomic_json(args.output / "status.json", {"state": "complete", "seconds": time.monotonic() - start,
                                              "is_confirmatory": False, "goal_achieved": False})


if __name__ == "__main__":
    main()
