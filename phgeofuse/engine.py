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
from .io import read_fasta
from .model import PHGeoFuse, compute_dual_view_loss, compute_loss
from .retrieval import RetrievalStore, ensure_retrieval_store, homology_training_enabled


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def build_loaders(records, retrieval, config, context, split_filters=None):
    mode = str(get(config, "model.mode", "frozen"))
    split_filters = split_filters or {}
    datasets = {
        split: ProteinGraphDataset(
            records,
            split,
            retrieval,
            mode,
            allowed_protein_ids=split_filters.get(split),
        )
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
    if bool(get(config, "training.diagnostics.evaluate_train", False)):
        if context.distributed:
            from utils.distributed import DistributedShardSampler
            evaluation_sampler = DistributedShardSampler(
                len(datasets["train"]), context.rank, context.world_size
            )
        else:
            evaluation_sampler = None
        loaders["train_evaluation"] = DataLoader(
            datasets["train"],
            batch_size=batch_size,
            sampler=evaluation_sampler,
            shuffle=False,
            num_workers=workers,
            pin_memory=context.device.type == "cuda",
            persistent_workers=workers > 0,
            collate_fn=collate_graphs,
            drop_last=False,
        )
    return loaders, train_sampler


def resolve_evaluation_split(records, config, requested_split):
    if requested_split in {"train", "validation", "test"}:
        return requested_split, None, {}

    subset_config = get(config, f"data.subsets.{requested_split}")
    if not isinstance(subset_config, dict):
        raise ValueError(f"evaluation subset is not configured: {requested_split}")
    source_split = str(subset_config.get("source_split", "test"))
    if source_split not in {"train", "validation", "test"}:
        raise ValueError(
            f"invalid source split for {requested_split}: {source_split}"
        )
    subset_records = read_fasta(
        path(config, f"data.subsets.{requested_split}.fasta"),
        source_split,
    )
    expected_count = int(subset_config.get("expected_count", len(subset_records)))
    if len(subset_records) != expected_count:
        raise ValueError(
            f"{requested_split} must contain {expected_count} records, "
            f"found {len(subset_records)}"
        )

    source_records = {
        record.protein_id: record
        for record in records
        if record.split == source_split
    }
    unknown_ids = [
        record.protein_id
        for record in subset_records
        if record.protein_id not in source_records
    ]
    if unknown_ids:
        raise ValueError(
            f"{requested_split} contains IDs outside {source_split}: {unknown_ids[:10]}"
        )
    for subset_record in subset_records:
        source_record = source_records[subset_record.protein_id]
        if subset_record.sequence != source_record.sequence:
            raise ValueError(
                f"sequence mismatch in {requested_split}: {subset_record.protein_id}"
            )
        if not math.isclose(
            subset_record.ph_opt, source_record.ph_opt, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(
                f"pHopt mismatch in {requested_split}: {subset_record.protein_id}"
            )

    requested_ids = {record.protein_id for record in subset_records}
    unavailable_ids = [
        record.protein_id
        for record in subset_records
        if source_records[record.protein_id].status != "ready"
    ]
    evaluated_count = len(requested_ids) - len(unavailable_ids)
    if evaluated_count == 0:
        raise ValueError(f"{requested_split} has no ready records to evaluate")
    metadata = {
        "evaluation_split": requested_split,
        "source_split": source_split,
        "subset_expected_count": expected_count,
        "subset_requested_count": len(requested_ids),
        "subset_evaluated_count": evaluated_count,
        "subset_unavailable_count": len(unavailable_ids),
        "subset_coverage": evaluated_count / len(requested_ids),
        "subset_complete": not unavailable_ids,
        "subset_unavailable_ids": unavailable_ids,
    }
    return source_split, requested_ids, metadata


def train_model(
    records,
    config,
    context,
    resume: str | Path | None = None,
    init_checkpoint: str | Path | None = None,
):
    if resume and init_checkpoint:
        raise ValueError("resume and init_checkpoint are mutually exclusive")
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
    if init_checkpoint:
        load_initial_checkpoint(init_checkpoint, model)
    _configure_trainable_scope(model, config, resume=resume, init_checkpoint=init_checkpoint)
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
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _cosine_schedule(step, warmup_steps, total_steps)
    )
    amp_dtype = _amp_dtype(config, context.device)
    scaler = torch.cuda.amp.GradScaler(enabled=context.device.type == "cuda" and amp_dtype == torch.float16)
    start_epoch, global_step = 0, 0
    best_rmse, patience_rmse, stale = math.inf, math.inf, 0
    if resume:
        state = load_checkpoint(resume, model, optimizer, scheduler, scaler)
        start_epoch = int(state.get("epoch", -1)) + 1
        global_step = int(state.get("global_step", 0))
        best_rmse = float(state.get("best_rmse", math.inf))
        patience_rmse = float(state.get("patience_rmse", best_rmse))
        stale = int(state.get("stale_epochs", 0))

    run_dir = _run_dir(config)
    if context.is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        save_resolved(config, run_dir / "config.resolved.yaml")
    patience = int(get(config, "training.early_stopping_patience", 15))
    min_delta = float(get(config, "training.early_stopping_min_delta", 0.0))
    if min_delta < 0:
        raise ValueError("training.early_stopping_min_delta must be non-negative")
    clip = float(get(config, "training.gradient_clip", 1.0))
    for epoch in range(start_epoch, epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        _set_training_mode(model, config)
        optimizer.zero_grad(set_to_none=True)
        running_loss, sample_count = 0.0, 0
        running_component_sums: dict[str, float] = {}
        for batch_index, raw_batch in enumerate(loaders["train"]):
            batch = move_batch(raw_batch, context.device)
            synchronize = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loaders["train"])
            sync_context = nullcontext() if synchronize or not context.distributed else model.no_sync()
            with sync_context:
                with _autocast(context.device, amp_dtype):
                    if homology_training_enabled(config):
                        outputs = model(
                            batch,
                            retrieval_views={
                                "normal": batch["retrieval"],
                                "low_homology": batch["low_homology_retrieval"],
                            },
                        )
                        loss, components = compute_dual_view_loss(
                            outputs, batch, config
                        )
                    else:
                        outputs = model(batch)
                        loss, components = compute_loss(outputs, batch, config)
                    scaled_loss = loss / accumulation
                scaler.scale(scaled_loss).backward()
            running_loss += float(loss.detach()) * len(batch["labels"])
            sample_count += len(batch["labels"])
            for name, value in components.items():
                running_component_sums[name] = (
                    running_component_sums.get(name, 0.0)
                    + float(value) * len(batch["labels"])
                )
            if synchronize:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
        train_loss = _global_average(running_loss, sample_count, context.device)
        train_loss_components = {
            name: _global_average(value, sample_count, context.device)
            for name, value in running_component_sums.items()
        }
        validation = evaluate_loader(model, loaders["validation"], config, context, include_loss=True)
        validation = select_homology_residual_scale(
            model, validation, config, context
        )
        train_evaluation = None
        if "train_evaluation" in loaders:
            rng_state = _capture_rng_state()
            train_evaluation = evaluate_loader(
                model, loaders["train_evaluation"], config, context, include_loss=True
            )
            _restore_rng_state(rng_state)
        validation_rmse = torch.tensor(
            validation["metrics"].get("rmse", 0.0), device=context.device
        )
        context.broadcast(validation_rmse, source=0)
        validation["metrics"]["rmse"] = float(validation_rmse)
        (
            best_rmse,
            patience_rmse,
            stale,
            checkpoint_improved,
        ) = _update_validation_tracking(
            float(validation_rmse), best_rmse, patience_rmse, stale, min_delta
        )
        if context.is_main:
            row = {
                "epoch": epoch, "global_step": global_step, "train_loss": train_loss,
                "train_loss_components": train_loss_components,
                "validation": validation["metrics"], "best_rmse": best_rmse,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "validation_loss": validation["loss"],
                "validation_loss_components": validation["loss_components"],
            }
            if train_evaluation is not None:
                row.update(
                    {
                        "train_evaluation": train_evaluation["metrics"],
                        "train_evaluation_loss": train_evaluation["loss"],
                        "train_evaluation_loss_components": train_evaluation["loss_components"],
                        "generalization_gap_rmse": (
                            validation["metrics"]["rmse"]
                            - train_evaluation["metrics"]["rmse"]
                        ),
                    }
                )
            with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            checkpoint = checkpoint_payload(
                model, optimizer, scheduler, scaler, config, epoch, global_step,
                best_rmse, patience_rmse, stale, context.world_size
            )
            atomic_torch_save(run_dir / "last.pt", checkpoint)
            if checkpoint_improved:
                atomic_torch_save(run_dir / "best.pt", checkpoint)
            train_rmse = (
                f" train_rmse={train_evaluation['metrics']['rmse']:.4f}"
                if train_evaluation is not None else ""
            )
            print(
                f"epoch={epoch + 1} train_loss={train_loss:.4f}{train_rmse} "
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
    local_component_sums: dict[str, float] = {}
    with torch.inference_mode():
        for raw_batch in loader:
            batch = move_batch(raw_batch, context.device)
            with _autocast(context.device, _amp_dtype(config, context.device)):
                outputs = model(batch)
                if include_loss:
                    loss, components = compute_loss(outputs, batch, config)
                    local_loss += float(loss) * len(batch["labels"])
                    local_count += len(batch["labels"])
                    for name, value in components.items():
                        local_component_sums[name] = (
                            local_component_sums.get(name, 0.0)
                            + float(value) * len(batch["labels"])
                        )
            for index, key in enumerate(batch["keys"]):
                retrieval = batch["retrieval"][index]
                row = {
                        "key": key,
                        "label": float(batch["labels"][index]),
                        "prediction": float(outputs["mean"][index]),
                        "global_prediction": float(outputs["global_mean"][index]),
                        "saprot_prediction": float(retrieval[0]),
                        "foldseek_prediction": float(retrieval[1]),
                        "saprot_available": bool(retrieval[7]),
                        "foldseek_available": bool(retrieval[8]),
                        "uncertainty": float(outputs["variance"][index].sqrt()),
                        "global_gate": float(outputs["gate_weights"][index, 0]),
                        "saprot_gate": float(outputs["gate_weights"][index, 1]),
                        "foldseek_gate": float(outputs["gate_weights"][index, 2]),
                        "sequence_identity": float(retrieval[4]),
                        "query_coverage": float(retrieval[9]),
                        "target_coverage": float(retrieval[10]),
                        "saprot_hit_fraction": float(retrieval[11]),
                        "foldseek_hit_fraction": float(retrieval[12]),
                        "saprot_similarity_margin": float(retrieval[13]),
                        "foldseek_similarity_margin": float(retrieval[14]),
                    }
                if "homology_residual_logits" in outputs:
                    row.update(
                        {
                            "_expert_means": outputs["expert_means"][index]
                            .float()
                            .cpu()
                            .tolist(),
                            "_expert_variances": torch.stack(
                                [
                                    outputs["global_variance"][index],
                                    retrieval[5],
                                    retrieval[6],
                                ]
                            )
                            .float()
                            .cpu()
                            .tolist(),
                            "_expert_available": outputs["expert_available"][index]
                            .cpu()
                            .tolist(),
                            "_baseline_gate_weights": outputs[
                                "baseline_gate_weights"
                            ][index]
                            .float()
                            .cpu()
                            .tolist(),
                            "_homology_residual_logits": outputs[
                                "homology_residual_logits"
                            ][index]
                            .float()
                            .cpu()
                            .tolist(),
                        }
                    )
                local_rows.append(row)
    rows = _gather_rows(local_rows, context)
    result = {"rows": rows if context.is_main else []}
    if context.is_main:
        result["metrics"] = prediction_metrics(rows, config)
        module = model.module if hasattr(model, "module") else model
        if module.fusion_mode == "homology_residual":
            result["metrics"]["homology_residual_scale"] = float(
                module.homology_residual_scale
            )
    else:
        result["metrics"] = {}
    if include_loss:
        result["loss"] = _global_average(local_loss, local_count, context.device)
        result["loss_components"] = {
            name: _global_average(value, local_count, context.device)
            for name, value in local_component_sums.items()
        }
    return result


def evaluate_checkpoint(records, config, checkpoint_path, context, split="test", output=None):
    retrieval = RetrievalStore.load(path(config, "paths.retrieval"))
    source_split, allowed_ids, evaluation_metadata = resolve_evaluation_split(
        records, config, split
    )
    split_filters = {source_split: allowed_ids} if allowed_ids is not None else None
    loaders, _ = build_loaders(
        records, retrieval, config, context, split_filters=split_filters
    )
    model = PHGeoFuse(config, context.device).to(context.device)
    load_checkpoint(checkpoint_path, model)
    if context.distributed:
        model = DistributedDataParallel(
            model, device_ids=[context.local_rank] if context.device.type == "cuda" else None,
            output_device=context.local_rank if context.device.type == "cuda" else None,
        )
    result = evaluate_loader(model, loaders[source_split], config, context)
    if context.is_main:
        result["metrics"].update(evaluation_metadata)
        destination = Path(output) if output else Path(checkpoint_path).parent / f"{split}_predictions.csv"
        _write_predictions(destination, result["rows"])
        atomic_json(destination.with_suffix(".metrics.json"), result["metrics"])
        print(json.dumps(result["metrics"], indent=2, sort_keys=True))
    context.barrier()
    return result


def calibrate_checkpoint(records, config, checkpoint_path, context, output=None):
    """Select deployment-time residual shrinkage using validation data."""
    retrieval = RetrievalStore.load(path(config, "paths.retrieval"))
    loaders, _ = build_loaders(records, retrieval, config, context)
    if len(loaders["validation"].dataset) == 0:
        raise ValueError("validation split must contain ready proteins")
    model = PHGeoFuse(config, context.device).to(context.device)
    payload = load_checkpoint(checkpoint_path, model)
    if context.distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[context.local_rank] if context.device.type == "cuda" else None,
            output_device=context.local_rank if context.device.type == "cuda" else None,
        )
    validation = evaluate_loader(model, loaders["validation"], config, context)
    validation = select_homology_residual_scale(
        model, validation, config, context
    )
    destination = None
    if context.is_main:
        destination = (
            Path(output)
            if output
            else Path(checkpoint_path).with_name(
                f"{Path(checkpoint_path).stem}_calibrated.pt"
            )
        )
        calibrated = dict(payload)
        calibrated["homology_residual_scale"] = float(
            validation["metrics"]["homology_residual_scale"]
        )
        calibrated["calibration"] = {
            "split": "validation",
            "metrics": validation["metrics"],
        }
        atomic_torch_save(destination, calibrated)
        atomic_json(
            destination.with_suffix(".calibration.metrics.json"),
            validation["metrics"],
        )
        print(json.dumps(validation["metrics"], indent=2, sort_keys=True))
        print(f"Calibrated checkpoint: {destination}")
    context.barrier()
    return destination


def regression_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mse = mean_squared_error(labels, predictions)
    has_correlation = (
        labels.size > 1
        and float(np.ptp(labels)) > 0.0
        and float(np.ptp(predictions)) > 0.0
    )
    correlation = spearmanr(labels, predictions).statistic if has_correlation else 0.0
    correlation = float(correlation) if math.isfinite(float(correlation)) else 0.0
    r2 = float(r2_score(labels, predictions)) if float(np.ptp(labels)) > 0.0 else 0.0
    return {
        "mse": float(mse), "rmse": float(math.sqrt(mse)),
        "mae": float(mean_absolute_error(labels, predictions)),
        "r2": r2,
        "spearman": correlation,
        "bias": float(np.mean(predictions - labels)),
    }


def ph_bin_metrics(labels, predictions):
    result = {}
    for name, lower, upper in (("acidic", -math.inf, 6.0), ("neutral", 6.0, 8.0), ("alkaline", 8.0, math.inf)):
        mask = (labels < upper) & (labels >= lower)
        if mask.any():
            errors = predictions[mask] - labels[mask]
            result[f"rmse_{name}"] = float(np.sqrt(np.mean(errors ** 2)))
            result[f"mae_{name}"] = float(np.mean(np.abs(errors)))
            result[f"bias_{name}"] = float(np.mean(errors))
            result[f"count_{name}"] = int(mask.sum())
    return result


def prediction_metrics(rows, config=None):
    labels = np.asarray([row["label"] for row in rows])
    predictions = np.asarray([row["prediction"] for row in rows])
    metrics = regression_metrics(labels, predictions)
    metrics.update(ph_bin_metrics(labels, predictions))

    expert_specs = (
        ("global", "global_prediction", None),
        ("saprot", "saprot_prediction", "saprot_available"),
        ("foldseek", "foldseek_prediction", "foldseek_available"),
    )
    expert_predictions = []
    expert_availability = []
    for prefix, prediction_key, availability_key in expert_specs:
        available = np.ones(len(rows), dtype=bool) if availability_key is None else np.asarray(
            [row[availability_key] for row in rows], dtype=bool
        )
        values = np.asarray([row[prediction_key] for row in rows])
        expert_predictions.append(values)
        expert_availability.append(available)
        metrics[f"{prefix}_count"] = int(available.sum())
        if available.any():
            for name, value in regression_metrics(labels[available], values[available]).items():
                metrics[f"{prefix}_{name}"] = value

    bins = (
        ("acidic", labels < 6.0),
        ("neutral", (labels >= 6.0) & (labels < 8.0)),
        ("alkaline", labels >= 8.0),
    )
    for gate_name in ("global_gate", "saprot_gate", "foldseek_gate"):
        gate_values = np.asarray([row[gate_name] for row in rows])
        metrics[f"mean_{gate_name}"] = float(gate_values.mean())
        for bin_name, mask in bins:
            if mask.any():
                metrics[f"mean_{gate_name}_{bin_name}"] = float(gate_values[mask].mean())

    experts = np.stack(expert_predictions, axis=1)
    availability = np.stack(expert_availability, axis=1)
    errors = np.where(availability, np.abs(experts - labels[:, None]), np.inf)
    best_expert = errors.argmin(axis=1)
    oracle = experts[np.arange(len(rows)), best_expert]
    gates = np.stack(
        [np.asarray([row[name] for row in rows]) for name in ("global_gate", "saprot_gate", "foldseek_gate")],
        axis=1,
    )
    metrics["oracle_rmse"] = float(np.sqrt(np.mean((oracle - labels) ** 2)))
    metrics["gate_best_expert_rate"] = float(np.mean(gates.argmax(axis=1) == best_expert))
    global_errors = np.abs(expert_predictions[0] - labels)
    for index, prefix in ((1, "saprot"), (2, "foldseek")):
        available = expert_availability[index]
        advantage = global_errors - np.abs(expert_predictions[index] - labels)
        metrics[f"gate_advantage_corr_{prefix}"] = _safe_pearson(
            gates[available, index], advantage[available]
        )
    uncertainty = np.asarray([row["uncertainty"] for row in rows])
    metrics["uncertainty_absolute_error_corr"] = _safe_pearson(
        uncertainty, np.abs(predictions - labels)
    )
    identities = np.asarray([row.get("sequence_identity", 0.0) for row in rows])
    query_coverage = np.asarray([row.get("query_coverage", 0.0) for row in rows])
    target_coverage = np.asarray([row.get("target_coverage", 0.0) for row in rows])
    metrics["mean_sequence_identity"] = float(identities.mean())
    metrics["mean_query_coverage"] = float(query_coverage.mean())
    metrics["mean_target_coverage"] = float(target_coverage.mean())
    for name in (
        "saprot_hit_fraction",
        "foldseek_hit_fraction",
        "saprot_similarity_margin",
        "foldseek_similarity_margin",
    ):
        metrics[f"mean_{name}"] = float(
            np.asarray([row.get(name, 0.0) for row in rows]).mean()
        )
    config = config or {}
    if homology_training_enabled(config):
        low_identity = float(get(config, "retrieval.low_homology_identity", 0.2))
        low_coverage = float(get(config, "retrieval.low_homology_coverage", 0.8))
        if low_identity > 1.0:
            low_identity /= 100.0
        if low_coverage > 1.0:
            low_coverage /= 100.0
        low_mask = ~(
            (identities >= low_identity)
            & (query_coverage >= low_coverage)
            & (target_coverage >= low_coverage)
        )
        metrics["low_homology_count"] = int(low_mask.sum())
        if low_mask.any():
            for name, value in regression_metrics(
                labels[low_mask], predictions[low_mask]
            ).items():
                metrics[f"low_homology_{name}"] = value
    return metrics


def select_homology_residual_scale(model, evaluation, config, context):
    """Shrink a learned residual gate using validation predictions only."""
    if str(get(config, "fusion.mode", "learned")).lower() != "homology_residual":
        return evaluation
    raw_grid = get(config, "homology_training.residual_scale_grid")
    if raw_grid is None:
        return evaluation
    if not isinstance(raw_grid, list) or not raw_grid:
        raise ValueError(
            "homology_training.residual_scale_grid must be a non-empty list"
        )
    grid = sorted({float(value) for value in raw_grid})
    if any(not 0.0 <= value <= 1.0 for value in grid) or 0.0 not in grid:
        raise ValueError(
            "homology_training.residual_scale_grid must contain zero and values "
            "between zero and one"
        )

    selected_scale = 0.0
    if context.is_main:
        rows = evaluation["rows"]
        if rows and "_homology_residual_logits" not in rows[0]:
            raise ValueError(
                "residual-scale selection requires homology residual diagnostics"
            )
        candidates = []
        for scale in grid:
            candidate_rows = _apply_homology_residual_scale(rows, scale, config)
            candidate_metrics = prediction_metrics(candidate_rows, config)
            candidates.append((candidate_metrics["rmse"], scale, candidate_rows, candidate_metrics))
        best_candidate = min(
            candidates, key=lambda item: (item[0], item[1])
        )
        tolerance = float(
            get(config, "homology_training.residual_scale_simplicity_tolerance", 0.0)
        )
        if tolerance < 0:
            raise ValueError(
                "homology_training.residual_scale_simplicity_tolerance must be non-negative"
            )
        eligible = [
            item for item in candidates if item[0] <= best_candidate[0] + tolerance
        ]
        _, selected_scale, selected_rows, selected_metrics = min(
            eligible, key=lambda item: item[1]
        )
        baseline_rmse = next(item[0] for item in candidates if item[1] == 0.0)
        minimum_improvement = float(
            get(config, "homology_training.residual_scale_min_improvement", 0.0)
        )
        if minimum_improvement < 0:
            raise ValueError(
                "homology_training.residual_scale_min_improvement must be non-negative"
            )
        if baseline_rmse - selected_metrics["rmse"] < minimum_improvement:
            _, selected_scale, selected_rows, selected_metrics = next(
                item for item in candidates if item[1] == 0.0
            )
        selected_metrics["homology_residual_scale"] = selected_scale
        selected_metrics["homology_residual_baseline_rmse"] = baseline_rmse
        selected_metrics["homology_residual_best_candidate_rmse"] = best_candidate[0]
        selected_metrics["homology_residual_scale_simplicity_tolerance"] = tolerance
        selected_metrics["homology_residual_scale_rmse"] = {
            str(scale): rmse for rmse, scale, _, _ in candidates
        }
        evaluation["rows"] = selected_rows
        evaluation["metrics"] = selected_metrics

    scale_tensor = torch.tensor(selected_scale, device=context.device)
    context.broadcast(scale_tensor, source=0)
    _set_homology_residual_scale(model, float(scale_tensor))
    return evaluation


def _update_validation_tracking(rmse, best_rmse, patience_rmse, stale, min_delta):
    checkpoint_improved = rmse < best_rmse
    if checkpoint_improved:
        best_rmse = rmse
    if rmse < patience_rmse - min_delta:
        patience_rmse = rmse
        stale = 0
    else:
        stale += 1
    return best_rmse, patience_rmse, stale, checkpoint_improved


def _apply_homology_residual_scale(rows, scale, config):
    temperature = float(get(config, "fusion.gate_temperature", 1.0))
    adjusted = []
    for source in rows:
        row = dict(source)
        baseline = np.asarray(row["_baseline_gate_weights"], dtype=np.float64)
        residual = np.asarray(row["_homology_residual_logits"], dtype=np.float64)
        available = np.asarray(row["_expert_available"], dtype=bool)
        logits = np.log(np.clip(baseline, 1e-30, None)) + scale * residual / temperature
        logits[~available] = -np.inf
        finite = np.isfinite(logits)
        shifted = logits - np.max(logits[finite])
        gates = np.where(finite, np.exp(shifted), 0.0)
        gates /= gates.sum()
        means = np.asarray(row["_expert_means"], dtype=np.float64)
        variances = np.asarray(row["_expert_variances"], dtype=np.float64)
        prediction = float(np.dot(gates, means))
        variance = float(
            np.dot(gates, variances + np.square(means - prediction))
        )
        row.update(
            {
                "prediction": prediction,
                "uncertainty": math.sqrt(max(0.0, variance)),
                "global_gate": float(gates[0]),
                "saprot_gate": float(gates[1]),
                "foldseek_gate": float(gates[2]),
            }
        )
        adjusted.append(row)
    return adjusted


def _safe_pearson(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 2 or left.std() == 0.0 or right.std() == 0.0:
        return 0.0
    value = float(np.corrcoef(left, right)[0, 1])
    return value if math.isfinite(value) else 0.0


def checkpoint_payload(
    model, optimizer, scheduler, scaler, config, epoch, global_step,
    best_rmse, patience_rmse, stale, world_size,
):
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
        "best_rmse": best_rmse, "patience_rmse": patience_rmse,
        "stale_epochs": stale, "config": config,
        "config_hash": config_hash(config), "world_size": world_size,
        "homology_residual_scale": float(module.homology_residual_scale),
        "rng": _capture_rng_state(),
    }


def load_checkpoint(source, model, optimizer=None, scheduler=None, scaler=None):
    payload = torch.load(source, map_location="cpu")
    module = model.module if hasattr(model, "module") else model
    incompatible = module.load_state_dict(
        payload["model_state_dict"], strict=not payload.get("saprot_adapter_only", False)
    )
    _set_homology_residual_scale(
        module, float(payload.get("homology_residual_scale", 1.0))
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


def load_initial_checkpoint(source, model):
    payload = torch.load(source, map_location="cpu")
    if payload.get("saprot_adapter_only", False):
        raise RuntimeError("init_checkpoint does not support adapter-only checkpoints")
    module = model.module if hasattr(model, "module") else model
    if module.fusion_mode == "homology_residual" and any(
        name.startswith("homology_gate.") for name in payload["model_state_dict"]
    ):
        raise RuntimeError(
            "homology_residual must be initialized from a baseline checkpoint "
            "without homology_gate weights; use --resume for a v2 checkpoint"
        )
    incompatible = module.load_state_dict(payload["model_state_dict"], strict=False)
    invalid_missing = [
        name for name in incompatible.missing_keys
        if not name.startswith("homology_gate.")
    ]
    if invalid_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "invalid initialization checkpoint: "
            f"missing={invalid_missing}, unexpected={list(incompatible.unexpected_keys)}"
        )
    return payload


def _configure_trainable_scope(model, config, *, resume=None, init_checkpoint=None):
    scope = str(get(config, "training.trainable_scope", "all")).lower()
    if scope == "all":
        return
    if scope != "homology_gate":
        raise ValueError("training.trainable_scope must be 'all' or 'homology_gate'")
    if not resume and not init_checkpoint:
        raise ValueError(
            "training.trainable_scope=homology_gate requires --init-checkpoint or --resume"
        )
    module = model.module if hasattr(model, "module") else model
    if (
        module.fusion_mode not in {"homology_reliability", "homology_residual"}
        or module.homology_gate is None
    ):
        raise ValueError(
            "homology_gate scope requires fusion.mode=homology_reliability "
            "or homology_residual"
        )
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    module.homology_gate.requires_grad_(True)


def _set_training_mode(model, config):
    scope = str(get(config, "training.trainable_scope", "all")).lower()
    if scope == "all":
        model.train()
        return
    model.eval()
    module = model.module if hasattr(model, "module") else model
    module.homology_gate.train()


def _set_homology_residual_scale(model, scale):
    module = model.module if hasattr(model, "module") else model
    if not 0.0 <= scale <= 1.0:
        raise ValueError("homology residual scale must be between zero and one")
    module.homology_residual_scale.fill_(scale)


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
    if not other and not lora:
        raise ValueError("training configuration does not leave any trainable parameters")
    groups = []
    if other:
        groups.append(
            {"params": other, "lr": float(get(config, "training.learning_rate", 3e-4))}
        )
    if lora:
        groups.append({"params": lora, "lr": float(get(config, "training.lora_learning_rate", 1e-5))})
    return AdamW(groups, weight_decay=float(get(config, "training.weight_decay", 1e-2)))


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
    public_rows = [
        {name: value for name, value in row.items() if not name.startswith("_")}
        for row in rows
    ]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(public_rows[0]) if public_rows else ["key", "prediction"],
        )
        writer.writeheader()
        writer.writerows(public_rows)
