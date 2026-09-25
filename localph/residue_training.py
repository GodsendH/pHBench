"""Bounded small-head fitting on immutable packed residue sketches."""
import copy
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .residue_field import ResidueField, label_prior, field_loss, decode


class PackedResidues(Dataset):
    def __init__(self, packed, indices, labels):
        self.packed, self.indices, self.labels = packed, np.asarray(indices), labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        a, b = self.packed["offsets"][j:j + 2]
        return self.packed["tokens"][a:b], self.packed["ionizable"][a:b], float(self.labels[j]), int(j)


def collate(rows):
    length = max(len(r[0]) for r in rows)
    tokens = torch.zeros(len(rows), length, rows[0][0].shape[1], dtype=torch.float32)
    mask, ion = torch.zeros(len(rows), length, dtype=torch.bool), torch.zeros(len(rows), length, dtype=torch.bool)
    for i, (x, ions, _, _) in enumerate(rows):
        tokens[i, :len(x)] = torch.from_numpy(x.astype(np.float32))
        mask[i, :len(x)] = True
        ion[i, :len(x)] = torch.from_numpy(ions.copy())
    return tokens, mask, ion, torch.tensor([r[2] for r in rows]), np.array([r[3] for r in rows])


def light_metrics(y, p):
    result = {}
    for name, mask in {"all": np.ones(len(y), bool), "acid": y <= 4, "alkaline": y >= 10,
                       "core": (y > 4) & (y < 10)}.items():
        e = p[mask] - y[mask]
        result[name] = {"rmse": float(np.mean(e ** 2) ** .5), "mae": float(abs(e).mean())} if len(e) else {"rmse": None, "mae": None}
    core = (y > 4) & (y < 10)
    result["false_extreme_rate"] = float(((p[core] <= 4) | (p[core] >= 10)).mean())
    return result


def choose(y, baseline, decisions):
    bm = light_metrics(y, baseline)
    choices = []
    for name, p in decisions.items():
        for strength in (0., .25, .5, .75, 1.):
            q = np.clip(baseline + strength * (p - baseline), 0, 14)
            m = light_metrics(y, q)
            eligible = all(m[g][k] is not None and m[g][k] <= bm[g][k] + .01 + 1e-12
                           for g in ("all", "core") for k in ("rmse", "mae"))
            eligible &= m["false_extreme_rate"] <= bm["false_extreme_rate"] + .005 + 1e-12
            if not eligible or any(m[g]["rmse"] is None for g in ("acid", "alkaline")):
                continue
            worst = max(m[g]["rmse"] / bm[g]["rmse"] for g in ("acid", "alkaline"))
            choices.append({"decision": name, "strength": strength, "metrics": m,
                            "rank": [worst, m["all"]["rmse"], strength]})
    if not choices:
        raise ValueError("no valid baseline or both tails not represented")
    return min(choices, key=lambda r: (*r["rank"], r["decision"]))


def standalone_objective(y, prediction):
    """Tie-break and stopping signal when all fused candidates fall back.

    Both tails get a modest fixed contribution; this is evaluated only on the
    current inner validation fold. It cannot turn a guardrail failure into an
    eligible blend, and it never consumes any outer query labels.
    """
    m = light_metrics(y, prediction)
    return m["all"]["rmse"] ** 2 + .05 * sum(m[g]["rmse"] ** 2 for g in ("acid", "alkaline"))


@torch.inference_mode()
def predict(model, packed, indices, y, prior, batch_size=32, device="cuda"):
    loader = DataLoader(PackedResidues(packed, indices, y), batch_size=batch_size, collate_fn=collate,
                        shuffle=False, num_workers=0)
    model.eval()
    rows = {}
    for tokens, mask, ion, _, _ in loader:
        output = model(tokens.to(device), mask.to(device), ion.to(device))
        predictions = {"direct": output["prediction"].cpu().numpy()} if model.kind == "direct" else decode(output["logits"], prior, model.grid)
        for name, p in predictions.items():
            rows.setdefault(name, []).append(p)
    return {name: np.concatenate(values) for name, values in rows.items()}


def fit(packed, labels, fit_indices, query_indices, baseline, kind, output, fixed_epochs=None, max_epochs=40,
        seed=42, device="cuda", patience=5):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if set(map(int, fit_indices)) & set(map(int, query_indices)):
        raise ValueError("training/query overlap")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = ResidueField(kind=kind).to(device)
    prior = label_prior(labels[fit_indices], model.grid)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(PackedResidues(packed, fit_indices, labels), batch_size=32, shuffle=True,
                        generator=generator, collate_fn=collate, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.05)
    start = time.monotonic()
    history, best, best_state, stable_best, stale = [], None, None, float("inf"), 0
    for epoch in range(1, (fixed_epochs or max_epochs) + 1):
        model.train()
        sum_loss, count = 0., 0
        for tokens, mask, ion, y, _ in loader:
            tokens, mask, ion, y = tokens.to(device), mask.to(device), ion.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = field_loss(model(tokens, mask, ion), y, model.grid, prior, kind)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            sum_loss += float(loss.detach()) * len(y)
            count += len(y)
        row = {"epoch": epoch, "train_loss": sum_loss / count, "seconds": time.monotonic() - start}
        if fixed_epochs is None:
            predictions = predict(model, packed, query_indices, labels, prior, device=device)
            choice = choose(labels[query_indices], baseline, predictions)
            row["choice"] = choice
            default_decision = "direct" if kind == "direct" else "prior1_mean"
            objective = standalone_objective(labels[query_indices], predictions[default_decision])
            row["standalone_validation_objective"] = objective
            row["checkpoint_rank"] = [*choice["rank"], objective]
            if best is None or tuple(row["checkpoint_rank"]) < tuple(best["checkpoint_rank"]):
                best = copy.deepcopy(row)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if objective < stable_best - .002:
                stable_best, stale = objective, 0
            else:
                stale += 1
        history.append(row)
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps({"event": "residue_epoch", "kind": kind, "output": str(output), **row}), flush=True)
        if fixed_epochs is None and stale >= patience:
            break
    if fixed_epochs is None:
        model.load_state_dict(best_state)
        selected_epoch = best["epoch"]
    else:
        selected_epoch = fixed_epochs
    predictions = predict(model, packed, query_indices, labels, prior, device=device)
    train_predictions = predict(model, packed, fit_indices, labels, prior, device=device)
    torch.save({"state_dict": model.cpu().state_dict(), "prior": prior.cpu(), "kind": kind,
                "seed": seed, "epoch": selected_epoch}, output / "weights.pt")
    report = {"kind": kind, "seed": seed, "selected_epoch": selected_epoch, "epochs_run": len(history),
              "seconds": time.monotonic() - start, "parameters": sum(p.numel() for p in model.parameters()),
              "max_epochs": max_epochs, "fit_rows": len(fit_indices), "query_rows": len(query_indices),
              "fit_indices": list(map(int, fit_indices)), "query_indices": list(map(int, query_indices)),
              "train_metrics": {k: light_metrics(labels[fit_indices], p) for k, p in train_predictions.items()},
              "query_metrics": {k: light_metrics(labels[query_indices], p) for k, p in predictions.items()}}
    (output / "complete.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez(output / "predictions.npz", **predictions)
    return predictions, report
