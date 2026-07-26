from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler

from .cache import atomic_json, atomic_torch_save
from .config import config_hash, get, path, save_resolved
from .dataset import ProteinGraphDataset, collate_graphs, move_batch
from .model import PHGeoFuse, compute_loss
from .retrieval import RetrievalStore, ensure_retrieval_store


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loaders(records, retrieval, config, context):
    mode = str(get(config, "model.mode", "frozen"))
    datasets = {
        split: ProteinGraphDataset(records, split, retrieval, mode)
        for split in ("train", "validation", "test")
    }
    batch_size = int(get(config, "training.per_device_batch_size", 2))
    workers = int(get(config, "training.num_workers", 2))
    loaders = {}
    train_sampler = None
    for split, dataset in datasets.items():
        if context.distributed and split == "train":
            sampler = DistributedSampler(
                dataset, num_replicas=context.world_size, rank=context.rank,
                shuffle=True, seed=int(get(config, "training.seed", 42)), drop_last=False,
            )
            train_sampler = sampler
        elif context.distributed:
            from utils.distributed import DistributedShardSampler
            sampler = DistributedShardSampler(len(dataset), context.rank, context.world_size)
        else:
            sampler = None
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=split == "train" and sampler is None,
            num_workers=workers,
            pin_memory=context.device.type == "cuda",
            persistent_workers=workers > 0,
            collate_fn=collate_graphs,
            drop_last=False,
        )
    return loaders, train_sampler


def train_model(records, config, context, resume: str | Path | None = None):
    seed = int(get(config, "training.seed", 42))
    seed_everything(seed + context.rank)
    retrieval_path = path(config, "paths.retrieval", "artifacts/phgeofuse/retrieval.pt")
    if context.is_main:
        ensure_retrieval_store(records, config)
    context.barrier()
    retrieval = RetrievalStore.load(retrieval_path)
    loaders, train_sampler = build_loaders(records, retrieval, config, context)
    if len(loaders["train"].dataset) == 0 or len(loaders["validation"].dataset) == 0:
        raise ValueError("training and validation splits must both contain ready proteins")

    model = PHGeoFuse(config, context.device).to(context.device)
    if context.distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[context.local_rank] if context.device.type == "cuda" else None,
            output_device=context.local_rank if context.device.type == "cuda" else None,
            find_unused_parameters=False,
        )
    optimizer = _optimizer(model, config)
    global_batch = int(get(config, "training.global_batch_size", 32))
    per_device = int(get(config, "training.per_device_batch_size", 2))
    accumulation = max(1, math.ceil(global_batch / (per_device * context.world_size)))
    epochs = int(get(config, "training.epochs", 100))
    updates_per_epoch = max(1, math.ceil(len(loaders["train"]) / accumulation))
    total_steps = epochs * updates_per_epoch
    warmup_steps = int(total_steps * float(get(config, "training.warmup_fraction", 0.05)))
    scheduler, scheduler_interval = _scheduler(
        optimizer, config, warmup_steps, total_steps
    )
    amp_dtype = _amp_dtype(config, context.device)
    scaler = torch.cuda.amp.GradScaler(enabled=context.device.type == "cuda" and amp_dtype == torch.float16)
    start_epoch, global_step, best_rmse, stale = 0, 0, math.inf, 0
    if resume:
        state = load_checkpoint(resume, model, optimizer, scheduler, scaler)
        start_epoch = int(state.get("epoch", -1)) + 1
        global_step = int(state.get("global_step", 0))
        best_rmse = float(state.get("best_rmse", math.inf))
        stale = int(state.get("stale_epochs", 0))

    run_dir = _run_dir(config)
    if context.is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        save_resolved(config, run_dir / "config.resolved.yaml")
    patience = int(get(config, "training.early_stopping_patience", 15))
    clip = float(get(config, "training.gradient_clip", 1.0))
    for epoch in range(start_epoch, epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss, sample_count = 0.0, 0
        for batch_index, raw_batch in enumerate(loaders["train"]):
            batch = move_batch(raw_batch, context.device)
            synchronize = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loaders["train"])
            sync_context = nullcontext() if synchronize or not context.distributed else model.no_sync()
            with sync_context:
                with _autocast(context.device, amp_dtype):
                    outputs = model(batch)
                    loss, _ = compute_loss(outputs, batch, config)
                    scaled_loss = loss / accumulation
                scaler.scale(scaled_loss).backward()
            running_loss += float(loss.detach()) * len(batch["labels"])
            sample_count += len(batch["labels"])
            if synchronize:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler_interval == "step":
                    scheduler.step()
                global_step += 1
        train_loss = _global_average(running_loss, sample_count, context.device)
        validation = evaluate_loader(model, loaders["validation"], config, context, include_loss=True)
        validation_rmse = torch.tensor(
            validation["metrics"].get("rmse", 0.0), device=context.device
        )
        context.broadcast(validation_rmse, source=0)
        validation["metrics"]["rmse"] = float(validation_rmse)
        if scheduler_interval == "epoch":
            scheduler.step(float(validation_rmse))
        improved = float(validation_rmse) < best_rmse
        if improved:
            best_rmse = float(validation_rmse)
            stale = 0
        else:
            stale += 1
        if context.is_main:
            row = {
                "epoch": epoch, "global_step": global_step, "train_loss": train_loss,
                "validation": validation["metrics"], "best_rmse": best_rmse,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            checkpoint = checkpoint_payload(
                model, optimizer, scheduler, scaler, config, epoch, global_step, best_rmse, stale, context.world_size
            )
            atomic_torch_save(run_dir / "last.pt", checkpoint)
            if improved:
                atomic_torch_save(run_dir / "best.pt", checkpoint)
            print(
                f"epoch={epoch + 1} train_loss={train_loss:.4f} "
                f"val_rmse={validation['metrics']['rmse']:.4f} best={best_rmse:.4f}"
            )
        stop = torch.tensor(float(stale >= patience), device=context.device)
        context.broadcast(stop, source=0)
        if bool(stop.item()):
            break
    context.barrier()
    return run_dir / "best.pt"


def evaluate_loader(model, loader, config, context, include_loss: bool = False):
    model.eval()
    local_rows = []
    local_loss, local_count = 0.0, 0
    with torch.inference_mode():
        for raw_batch in loader:
            batch = move_batch(raw_batch, context.device)
            with _autocast(context.device, _amp_dtype(config, context.device)):
                outputs = model(batch)
                if include_loss:
                    loss, _ = compute_loss(outputs, batch, config)
                    local_loss += float(loss) * len(batch["labels"])
                    local_count += len(batch["labels"])
            for index, key in enumerate(batch["keys"]):
                local_rows.append(
                    {
                        "key": key,
                        "label": float(batch["labels"][index]),
                        "prediction": float(outputs["mean"][index]),
                        "global_prediction": float(outputs["global_mean"][index]),
                        "uncertainty": float(outputs["variance"][index].sqrt()),
                        "global_gate": float(outputs["gate_weights"][index, 0]),
                        "saprot_gate": float(outputs["gate_weights"][index, 1]),
                        "foldseek_gate": float(outputs["gate_weights"][index, 2]),
                    }
                )
    rows = _gather_rows(local_rows, context)
    result = {"rows": rows if context.is_main else []}
    if context.is_main:
        labels = np.asarray([row["label"] for row in rows])
        predictions = np.asarray([row["prediction"] for row in rows])
        result["metrics"] = regression_metrics(labels, predictions)
        result["metrics"].update(ph_bin_metrics(labels, predictions))
    else:
        result["metrics"] = {}
    if include_loss:
        result["loss"] = _global_average(local_loss, local_count, context.device)
    return result


def evaluate_checkpoint(records, config, checkpoint_path, context, split="test", output=None):
    retrieval = RetrievalStore.load(path(config, "paths.retrieval"))
    loaders, _ = build_loaders(records, retrieval, config, context)
    model = PHGeoFuse(config, context.device).to(context.device)
    load_checkpoint(checkpoint_path, model)
    if context.distributed:
        model = DistributedDataParallel(
            model, device_ids=[context.local_rank] if context.device.type == "cuda" else None,
            output_device=context.local_rank if context.device.type == "cuda" else None,
        )
    result = evaluate_loader(model, loaders[split], config, context)
    if context.is_main:
        destination = Path(output) if output else Path(checkpoint_path).parent / f"{split}_predictions.csv"
        _write_predictions(destination, result["rows"])
        atomic_json(destination.with_suffix(".metrics.json"), result["metrics"])
        print(json.dumps(result["metrics"], indent=2, sort_keys=True))
    context.barrier()
    return result


def regression_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mse = mean_squared_error(labels, predictions)
    correlation = spearmanr(labels, predictions).statistic
    return {
        "mse": float(mse), "rmse": float(math.sqrt(mse)),
        "mae": float(mean_absolute_error(labels, predictions)),
        "r2": float(r2_score(labels, predictions)),
        "spearman": float(correlation),
    }


def ph_bin_metrics(labels, predictions):
    result = {}
    for name, lower, upper in (("acidic", -math.inf, 6.0), ("neutral", 6.0, 8.0), ("alkaline", 8.0, math.inf)):
        mask = (labels < upper) & (labels >= lower)
        if mask.any():
            result[f"rmse_{name}"] = float(np.sqrt(np.mean((labels[mask] - predictions[mask]) ** 2)))
            result[f"count_{name}"] = int(mask.sum())
    return result


def checkpoint_payload(model, optimizer, scheduler, scaler, config, epoch, global_step, best_rmse, stale, world_size):
    module = model.module if hasattr(model, "module") else model
    model_state = module.state_dict()
    adapter_only = getattr(module, "mode", "frozen") == "lora"
    if adapter_only:
        model_state = {
            name: value
            for name, value in model_state.items()
            if not name.startswith("saprot_model.") or "lora_" in name
        }
    model_state = {name: value.detach().cpu() for name, value in model_state.items()}
    return {
        "schema_version": "1", "model_state_dict": model_state,
        "saprot_adapter_only": adapter_only,
        "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(), "epoch": epoch, "global_step": global_step,
        "best_rmse": best_rmse, "stale_epochs": stale, "config": config,
        "config_hash": config_hash(config), "world_size": world_size,
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
    }


def load_checkpoint(source, model, optimizer=None, scheduler=None, scaler=None):
    payload = torch.load(source, map_location="cpu")
    module = model.module if hasattr(model, "module") else model
    incompatible = module.load_state_dict(
        payload["model_state_dict"], strict=not payload.get("saprot_adapter_only", False)
    )
    if payload.get("saprot_adapter_only", False):
        unexpected = list(incompatible.unexpected_keys)
        invalid_missing = [
            name for name in incompatible.missing_keys
            if not name.startswith("saprot_model.") or "lora_" in name
        ]
        if unexpected or invalid_missing:
            raise RuntimeError(
                f"invalid adapter checkpoint: missing={invalid_missing}, unexpected={unexpected}"
            )
    if optimizer is not None and payload.get("optimizer_state_dict"):
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict"):
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if scaler is not None and payload.get("scaler_state_dict"):
        scaler.load_state_dict(payload["scaler_state_dict"])
    return payload


def apply_ablation(config: dict[str, Any], name: str) -> None:
    ablation = config.setdefault("ablation", {})
    ablation.update({"disable_structural_features": False, "disable_ph_conditioning": False,
                     "disable_retrieval": False, "disable_foldseek": False})
    if name == "saprot_only":
        ablation.update({"disable_structural_features": True, "disable_ph_conditioning": True, "disable_retrieval": True})
    elif name == "geometry":
        ablation.update({"disable_ph_conditioning": True, "disable_retrieval": True})
    elif name == "ph_conditioned":
        ablation["disable_retrieval"] = True
    elif name == "saprot_retrieval":
        ablation["disable_foldseek"] = True
    elif name != "full":
        raise ValueError(f"unknown ablation: {name}")


def _optimizer(model, config):
    lora, other = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (lora if "lora_" in name else other).append(parameter)
    groups = [{"params": other, "lr": float(get(config, "training.learning_rate", 3e-4))}]
    if lora:
        groups.append({"params": lora, "lr": float(get(config, "training.lora_learning_rate", 1e-5))})
    return AdamW(groups, weight_decay=float(get(config, "training.weight_decay", 1e-2)))


def _scheduler(optimizer, config, warmup_steps, total_steps):
    scheduler_type = str(get(config, "training.scheduler.type", "cosine")).lower()
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: _cosine_schedule(step, warmup_steps, total_steps)
        )
        return scheduler, "step"
    if scheduler_type in {"plateau", "reduce_on_plateau"}:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(get(config, "training.scheduler.factor", 0.5)),
            patience=int(get(config, "training.scheduler.patience", 1)),
            threshold=float(get(config, "training.scheduler.threshold", 1e-3)),
            threshold_mode=str(get(config, "training.scheduler.threshold_mode", "abs")),
            cooldown=int(get(config, "training.scheduler.cooldown", 0)),
            min_lr=float(get(config, "training.scheduler.min_lr", 1e-6)),
        )
        return scheduler, "epoch"
    raise ValueError(f"unknown training scheduler: {scheduler_type}")


def _amp_dtype(config, device):
    if device.type != "cuda":
        return torch.float32
    requested = str(get(config, "training.precision", "bf16")).lower()
    if requested == "bf16" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _autocast(device, dtype):
    return torch.autocast(device_type="cuda", dtype=dtype) if device.type == "cuda" else nullcontext()


def _cosine_schedule(step, warmup, total):
    if warmup and step < warmup:
        return max(1e-8, step / warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _global_average(value, count, device):
    tensor = torch.tensor([value, count], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor)
    return float(tensor[0] / tensor[1].clamp_min(1))


def _gather_rows(rows, context):
    if not context.distributed:
        return rows
    gathered = [None for _ in range(context.world_size)] if context.is_main else None
    dist.gather_object(rows, gathered, dst=0)
    return [row for shard in gathered for row in shard] if context.is_main else []


def _run_dir(config):
    name = str(get(config, "training.run_name", "phgeofuse"))
    seed = int(get(config, "training.seed", 42))
    mode = str(get(config, "model.mode", "frozen"))
    return path(config, "paths.runs", "artifacts/phgeofuse/runs") / f"{name}_{mode}_seed{seed}"


def _write_predictions(destination, rows):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["key", "prediction"])
        writer.writeheader()
        writer.writerows(rows)
