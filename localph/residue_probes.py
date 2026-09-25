"""Small, auditable capacity probes; never an outer performance estimator."""
import copy
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from phgeofuse.cache import atomic_json
from phgeofuse.delta_ref.data import atomic_npz
from .residue_field import ResidueField, field_loss, label_prior
from .residue_training import PackedResidues, collate, predict, choose, standalone_objective, light_metrics
from .residue_objectives import smooth_rarity_weights, weighted_label_prior, rarity_loss


def fit_probe(packed, labels, fit_indices, query_indices, baseline, output, weighted=False,
              kind="sparse", max_epochs=40, seed=42, device="cuda"):
    if set(map(int, fit_indices)) & set(map(int, query_indices)):
        raise ValueError("training/query overlap")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    width = packed["tokens"].shape[1]
    model = ResidueField(input_dim=width, kind=kind).to(device)
    prior = label_prior(labels[fit_indices], model.grid)
    weights = smooth_rarity_weights(labels[fit_indices]) if weighted else np.ones(len(fit_indices), dtype=np.float32)
    train_prior = weighted_label_prior(labels[fit_indices], weights, model.grid) if weighted else prior
    full_weights = np.zeros(len(labels), dtype=np.float32)
    full_weights[fit_indices] = weights
    all_weights = torch.as_tensor(full_weights, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.05)
    loader = DataLoader(PackedResidues(packed, fit_indices, labels), batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(seed), collate_fn=collate, num_workers=0)
    history, best, best_state, stable_best, stale = [], None, None, float("inf"), 0
    start = time.monotonic()
    for epoch in range(1, max_epochs + 1):
        total, count = 0., 0
        model.train()
        for tokens, mask, ion, y, idx in loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(tokens.to(device), mask.to(device), ion.to(device))
            y = y.to(device)
            loss = (rarity_loss(output, y, all_weights[torch.as_tensor(idx, device=device)], model.grid, train_prior, kind)
                    if weighted else field_loss(output, y, model.grid, prior, kind))
            if not torch.isfinite(loss):
                raise ValueError("nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += float(loss.detach()) * len(y)
            count += len(y)
        p = predict(model, packed, query_indices, labels, prior, device=device)
        choice = choose(labels[query_indices], baseline, p)
        objective = standalone_objective(labels[query_indices], p["direct" if kind == "direct" else "prior1_mean"])
        rank = [*choice["rank"], objective]
        row = {"epoch": epoch, "train_loss": total / count, "choice": choice,
               "checkpoint_rank": rank, "standalone_validation_objective": objective,
               "seconds": time.monotonic() - start}
        history.append(row)
        if best is None or tuple(rank) < tuple(best["checkpoint_rank"]):
            best = copy.deepcopy(row)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if objective < stable_best - .002:
            stable_best, stale = objective, 0
        else:
            stale += 1
        atomic_json(out / "history.json", history)
        print({"input_dim": width, "weighted": weighted, "epoch": epoch, "objective": objective,
               "choice": choice["rank"], "seconds": row["seconds"]}, flush=True)
        if stale >= 5:
            break
    model.load_state_dict(best_state)
    predictions = predict(model, packed, query_indices, labels, prior, device=device)
    train_p = predict(model, packed, fit_indices, labels, prior, device=device)
    torch.save({"kind": kind, "input_dim": width, "hidden_width": 32, "dropout": .25,
                "state_dict": best_state, "prior": prior.cpu(), "training_prior": train_prior.cpu(),
                "epoch": best["epoch"], "seed": seed}, out / "weights.pt")
    atomic_npz(out / "predictions.npz", **predictions)
    report = {"kind": kind, "input_dim": width, "weighted": weighted,
              "parameters": sum(p.numel() for p in model.parameters()), "selected_epoch": best["epoch"],
              "epochs_run": len(history), "seconds": time.monotonic() - start,
              "choice": choose(labels[query_indices], baseline, predictions),
              "train_metrics": {k: light_metrics(labels[fit_indices], p) for k, p in train_p.items()},
              "query_metrics": {k: light_metrics(labels[query_indices], p) for k, p in predictions.items()}}
    atomic_json(out / "complete.json", report)
    return report
