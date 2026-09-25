"""Deterministic pair sampling, fitted-panel packaging and subset-only fitting."""
from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path
import numpy as np
import torch

from phgeofuse.cache import atomic_json, atomic_torch_save, sha256_file
from .data import freeze_json, stable_hash, write_predictions, assert_disjoint
from .model import DeltaNetwork, Standardizer, ReferencePredictor, select_panel, ph_bins
from .metrics import select_strength, metrics, selection_score


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def frequency_weights(labels, power):
    bins = ph_bins(labels)
    count = np.bincount(bins, minlength=14)
    weights = (len(labels) / count[bins]) ** power
    weights /= weights.mean()
    return np.minimum(weights, 3.).astype(np.float32)


def sample_pairs(labels, groups, rng, count=8, natural=4):
    """Eight distinct foreign families per query; no label enters the network."""
    groups = np.asarray(groups).astype(str)
    bins = ph_bins(labels)
    if not 0 <= natural <= count or len(set(groups)) <= count:
        raise ValueError("pair sampling needs at least count+1 different families")
    members = {b: np.flatnonzero(bins == b) for b in np.unique(bins)}
    bin_values = np.array(sorted(members))
    q, r = [], []
    for i in rng.permutation(len(labels)):
        used = {groups[i]}
        for k in range(count):
            selected = None
            for _ in range(200):
                j = int(rng.integers(len(groups))) if k < natural else int(rng.choice(members[int(rng.choice(bin_values))]))
                if groups[j] not in used:
                    selected = j
                    break
            if selected is None:
                eligible = np.flatnonzero(~np.isin(groups, list(used)))
                if k < natural:
                    selected = int(rng.choice(eligible))
                else:
                    b = int(rng.choice(np.unique(bins[eligible])))
                    selected = int(rng.choice(eligible[bins[eligible] == b]))
            used.add(groups[selected])
            q.append(i)
            r.append(selected)
    order = rng.permutation(len(q))
    return np.asarray(q)[order], np.asarray(r)[order]


def recipes(config):
    return [{"name": f"pair_w{w}_p{p:g}", "kind": "pair", "width": w, "power": p}
            for w in config["model"]["widths"] for p in config["model"]["frequency_powers"]]


def save_bundle(destination, predictor, recipe, strength, metadata):
    destination = Path(destination)
    payload = {"network_spec": predictor.network.spec,
               "state_dict": {k: v.detach().cpu() for k, v in predictor.network.state_dict().items()},
               "scaler_mean": predictor.scaler.mean, "scaler_scale": predictor.scaler.scale,
               "reference_features": predictor.features, "reference_labels": predictor.labels,
               "reference_groups": predictor.groups, "reference_keys": predictor.keys,
               "reference_weights": predictor.weights, "balanced_panel": predictor.balanced}
    atomic_torch_save(destination / "weights.pt", payload)
    atomic_json(destination / "model.json", {"schema": 1, "architecture": "DeltaRefPH",
        "recipe": recipe, "strength": float(strength), "metadata": metadata,
        "weights_sha256": sha256_file(destination / "weights.pt"),
        "agreement_is_probability": False, "default_model_replaced": False})


def load_bundle(source, device="cpu"):
    source = Path(source)
    config = json.loads((source / "model.json").read_text())
    if config.get("schema") != 1 or sha256_file(source / "weights.pt") != config["weights_sha256"]:
        raise ValueError("invalid model schema or checksum")
    p = torch.load(source / "weights.pt", map_location="cpu")
    model = DeltaNetwork(**p["network_spec"])
    model.load_state_dict(p["state_dict"], strict=True)
    predictor = ReferencePredictor(model, Standardizer(p["scaler_mean"], p["scaler_scale"]),
        p["reference_features"], p["reference_labels"], p["reference_groups"], p["reference_keys"],
        p["reference_weights"], device, balanced=p.get("balanced_panel", True))
    return predictor, config


def fit_subset(data, fit, validation, baseline_validation, recipe, output, seed=42,
               fixed_epochs=None, excluded=(), device=None, progress=None):
    """Validation can select epochs only when it is inside the outer train set.

    A fixed-epoch refit never looks at validation labels, even when a caller
    supplies query indices for immediate prediction after fitting.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = data.config
    settings = config["training"]
    device = device or settings["device"]
    torch.set_num_threads(settings.get("threads", 4))
    assert_disjoint(data.keys[fit], data.keys[validation], data.groups[fit], data.groups[validation])
    metadata = {**data.certificate(fit, validation, excluded), "recipe": recipe, "seed": seed,
                "fixed_epochs": fixed_epochs, "training_settings": settings,
                "panel_settings": config["model"], "epoch_selection_keys": [] if fixed_epochs else data.keys[validation].tolist(),
                "epoch_selection_label_sha256": None if fixed_epochs else stable_hash(data.labels[validation].tolist())}
    freeze_json(output / "fit.json", metadata)
    if (output / "complete.json").exists():
        predictor, saved = load_bundle(output, device)
        if saved["metadata"]["fit_hash"] != stable_hash(metadata):
            raise ValueError("completed fit belongs to another protocol")
        return predictor, json.loads((output / "complete.json").read_text())
    seed_all(seed)
    scaler = Standardizer.fit(data.x[fit])
    x = torch.as_tensor(scaler.transform(data.x[fit]), device=device)
    y = torch.as_tensor(data.labels[fit], dtype=torch.float32, device=device)
    weights = torch.as_tensor(frequency_weights(data.labels[fit], recipe["power"]), device=device)
    network = DeltaNetwork(data.x.shape[1], recipe["width"], config["model"]["dropout"], recipe["kind"]).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    panel, panel_weights = select_panel(data.embeddings[fit], data.labels[fit], data.groups[fit], data.keys[fit],
                                        config["model"]["references_per_bin"])
    panel = fit[panel]
    def predictor_now():
        return ReferencePredictor(network, scaler, data.x[panel], data.labels[panel], data.groups[panel], data.keys[panel], panel_weights, device)
    rng = np.random.default_rng(seed)
    best_rank, best_epoch, stale = (float("inf"),)*3, 0, 0
    best_state, best_strength = None, 0.
    history = []
    started = time.monotonic()
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    epochs = fixed_epochs if fixed_epochs is not None else settings["max_epochs"]
    if epochs < 1:
        raise ValueError("positive number of epochs required")
    for epoch in range(1, epochs + 1):
        network.train()
        loss_sum, pair_count = 0., 0
        if network.kind == "absolute":
            # Match optimizer updates with the pair model without adding data.
            qi = rng.permutation(np.tile(np.arange(len(fit)), settings["pairs_per_query"])); ri = qi
        else:
            qi, ri = sample_pairs(data.labels[fit], data.groups[fit], rng,
                                  settings["pairs_per_query"], settings["natural_partners"])
        for start in range(0, len(qi), settings["batch_size"]):
            q = torch.as_tensor(qi[start:start+settings["batch_size"]], device=device)
            r = torch.as_tensor(ri[start:start+settings["batch_size"]], device=device)
            optimizer.zero_grad(set_to_none=True)
            prediction = network(x[q]) if network.kind == "absolute" else network(x[q], x[r])
            target = y[q] if network.kind == "absolute" else y[q] - y[r]
            w = weights[q] if network.kind == "absolute" else torch.sqrt(weights[q] * weights[r])
            loss = (w * (prediction-target).square()).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite pair loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), settings["gradient_clip"])
            optimizer.step()
            loss_sum += float(loss.detach()) * len(q)
            pair_count += len(q)
        row = {"epoch": epoch, "training_loss": loss_sum/pair_count, "pairs": pair_count,
               "elapsed_seconds": time.monotonic()-started}
        if fixed_epochs is None:
            predictor = predictor_now()
            t, d, valid = predictor.transfer(data.x[validation], data.keys[validation])
            selected, _ = select_strength(data.labels[validation], baseline_validation, t, d, valid, data.groups[validation])
            # Pure transfer breaks zero-strength ties for epoch selection only;
            # it never relaxes constrained candidate/strength selection.
            raw = metrics(data.labels[validation], t, data.groups[validation])
            rank = (*selected["rank"], raw["all"]["rmse"])
            row.update(validation=selected["metrics"], strength=selected["strength"], standalone=raw)
            if best_state is None or rank < best_rank:
                best_rank, best_epoch, stale = rank, epoch, 0
                best_strength = selected["strength"]
                best_state = copy.deepcopy({k: v.detach().cpu() for k,v in network.state_dict().items()})
            else:
                stale += 1
        else:
            best_epoch = epoch
        history.append(row)
        atomic_json(output / "history.json", history)
        if progress:
            progress({"event": "delta_epoch", "output": str(output), **row})
        elif epoch == 1 or epoch % 5 == 0:
            print(json.dumps({"event": "delta_epoch", "epoch": epoch, "output": str(output),
                              "loss": row["training_loss"], "seconds": row["elapsed_seconds"]}), flush=True)
        if fixed_epochs is None and stale >= settings["patience"]:
            break
    if fixed_epochs is None:
        network.load_state_dict(best_state)
    predictor = predictor_now()
    summary = {"best_epoch": best_epoch, "strength": best_strength, "epochs_run": len(history),
               "wall_seconds": time.monotonic()-started, "parameter_count": sum(p.numel() for p in network.parameters()),
               "peak_gpu_bytes": torch.cuda.max_memory_allocated() if torch.device(device).type == "cuda" else 0,
               "panel_size": len(panel), "panel_families": len(set(data.groups[panel])), "fit_hash": stable_hash(metadata)}
    save_bundle(output, predictor, recipe, best_strength, {**summary, "provenance": metadata})
    reloaded, _ = load_bundle(output, device)
    # Check serialization without querying evaluation labels.
    probe = data.x[validation[:32]]
    original = predictor.transfer(probe)
    restored = reloaded.transfer(probe)
    if not all(np.allclose(a, b, atol=1e-7, rtol=0) for a,b in zip(original, restored)):
        raise ValueError("packaged predictor differs from training predictor")
    atomic_json(output / "complete.json", summary)
    return predictor, summary
