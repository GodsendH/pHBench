from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .chemistry import henderson_hasselbalch_charge
from .config import get
from .graph import graph_feature_dim
from .saprot import SaProtEncoder


def scatter_sum(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    output = values.new_zeros((size,) + values.shape[1:])
    output.index_add_(0, index, values)
    return output


def scatter_mean(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    output = scatter_sum(values, index, size)
    counts = torch.bincount(index, minlength=size).to(values.dtype)
    return output / counts.clamp_min(1).reshape((-1,) + (1,) * (values.dim() - 1))


class EGNNLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.coordinate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1, bias=False))
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden, coords, edge_index, edge_features):
        source, target = edge_index
        relative = coords[source] - coords[target]
        squared_distance = relative.square().sum(dim=-1, keepdim=True)
        messages = self.message(
            torch.cat([hidden[source], hidden[target], edge_features, squared_distance], dim=-1)
        )
        aggregated = scatter_sum(messages, target, hidden.shape[0])
        coordinate_scale = self.coordinate(messages).tanh()
        delta = relative / squared_distance.sqrt().clamp_min(1e-3) * coordinate_scale
        coords = coords + scatter_mean(delta, target, hidden.shape[0])
        hidden = self.norm(hidden + self.update(torch.cat([hidden, aggregated], dim=-1)))
        return hidden, coords


class GeometryEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, edge_dim: int, layers: int, dropout: float):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.layers = nn.ModuleList([EGNNLayer(hidden_dim, edge_dim, dropout) for _ in range(layers)])

    def forward(self, node_features, coords, edge_index, edge_features):
        hidden = self.project(node_features)
        for layer in self.layers:
            hidden, coords = layer(hidden, coords, edge_index, edge_features)
        return hidden, coords


class PHConditionedDecoder(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, ph_min: float, ph_max: float, ph_step: float, dropout: float, use_charge: bool = True):
        super().__init__()
        grid = torch.arange(ph_min, ph_max + ph_step / 2, ph_step)
        self.register_buffer("ph_grid", grid)
        self.use_charge = use_charge
        self.ph_embedding = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.attention = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, hidden, graph_index, lengths, ionizable_type, pka):
        batch_size = int(lengths.shape[0])
        charges = henderson_hasselbalch_charge(ionizable_type, pka, self.ph_grid)
        if not self.use_charge:
            charges = torch.zeros_like(charges)
        weighted = hidden.unsqueeze(1) * charges.unsqueeze(-1)
        charge_context = scatter_sum(weighted, graph_index, batch_size)
        denominator = scatter_sum(charges.abs(), graph_index, batch_size).unsqueeze(-1).clamp_min(1.0)
        charge_context = charge_context / denominator
        normalized_ph = ((self.ph_grid - 7.0) / 5.0).reshape(1, -1, 1)
        queries = self.ph_embedding(normalized_ph).expand(batch_size, -1, -1) + charge_context
        padded, padding_mask = _pad_hidden(hidden, graph_index, lengths)
        attended, _ = self.attention(queries, padded, padded, key_padding_mask=padding_mask, need_weights=False)
        logits = self.output(attended).squeeze(-1)
        probabilities = torch.softmax(logits, dim=-1)
        mean = (probabilities * self.ph_grid).sum(dim=-1)
        variance = (probabilities * (self.ph_grid.unsqueeze(0) - mean.unsqueeze(-1)).square()).sum(dim=-1)
        return logits, probabilities, mean, variance


class PHGeoFuse(nn.Module):
    def __init__(self, config: dict[str, Any], device: torch.device | None = None):
        super().__init__()
        self.config = config
        self.mode = str(get(config, "model.mode", "frozen"))
        self.disable_structural_features = bool(get(config, "ablation.disable_structural_features", False))
        self.disable_retrieval = bool(get(config, "ablation.disable_retrieval", False))
        self.disable_foldseek = bool(get(config, "ablation.disable_foldseek", False))
        hidden_dim = int(get(config, "model.hidden_dim", 256))
        embedding_dim = int(get(config, "model.embedding_dim", 1280))
        edge_dim = int(get(config, "graph.rbf_bins", 16)) + 2
        self.retrieval_dropout = float(get(config, "retrieval.dropout", 0.4))
        self.fusion_mode = str(get(config, "fusion.mode", "learned")).lower()
        if self.fusion_mode not in {
            "learned", "fixed", "reliability", "homology_reliability",
            "homology_residual",
        }:
            raise ValueError(
                "fusion.mode must be 'learned', 'fixed', 'reliability', "
                "'homology_reliability', or 'homology_residual'"
            )
        fixed_weights = torch.tensor(
            get(config, "fusion.fixed_weights", [0.25, 0.25, 0.5]),
            dtype=torch.float32,
        )
        if fixed_weights.shape != (3,) or bool((fixed_weights < 0).any()) or float(fixed_weights.sum()) <= 0:
            raise ValueError("fusion.fixed_weights must contain three non-negative values")
        self.register_buffer(
            "fixed_gate_weights", fixed_weights / fixed_weights.sum(), persistent=False
        )
        self.saprot_model = None
        self.saprot_tokenizer = None
        if self.mode == "lora":
            if device is None:
                raise ValueError("LoRA mode requires a target device during model construction")
            self._initialize_lora(config, device)
        elif self.mode != "frozen":
            raise ValueError("model.mode must be 'frozen' or 'lora'")
        self.geometry = GeometryEncoder(
            embedding_dim + graph_feature_dim(), hidden_dim, edge_dim,
            int(get(config, "model.egnn_layers", 6)), float(get(config, "model.dropout", 0.1)),
        )
        self.decoder = PHConditionedDecoder(
            hidden_dim, int(get(config, "model.attention_heads", 4)),
            float(get(config, "model.ph_min", 2.0)), float(get(config, "model.ph_max", 12.0)),
            float(get(config, "model.ph_step", 0.25)), float(get(config, "model.dropout", 0.1)),
            not bool(get(config, "ablation.disable_ph_conditioning", False)),
        )
        self.ec_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 7))
        self.gate = nn.Sequential(nn.Linear(6, 64), nn.SiLU(), nn.Dropout(0.1), nn.Linear(64, 3))
        self.reliability_gate = None
        self.homology_gate = None
        self.gate_temperature = float(get(config, "fusion.gate_temperature", 1.0))
        if self.gate_temperature <= 0:
            raise ValueError("fusion.gate_temperature must be positive")
        if self.fusion_mode == "reliability":
            gate_hidden_dim = int(get(config, "fusion.gate_hidden_dim", 32))
            gate_dropout = float(get(config, "fusion.gate_dropout", 0.1))
            if gate_hidden_dim <= 0 or not 0 <= gate_dropout < 1:
                raise ValueError("reliability gate hidden dimension and dropout are invalid")
            self.reliability_gate = nn.Sequential(
                nn.Linear(15, gate_hidden_dim),
                nn.SiLU(),
                nn.Dropout(gate_dropout),
                nn.Linear(gate_hidden_dim, 3),
            )
        elif self.fusion_mode in {"homology_reliability", "homology_residual"}:
            gate_hidden_dim = int(get(config, "fusion.gate_hidden_dim", 64))
            gate_dropout = float(get(config, "fusion.gate_dropout", 0.1))
            if gate_hidden_dim <= 0 or not 0 <= gate_dropout < 1:
                raise ValueError("homology gate hidden dimension and dropout are invalid")
            self.homology_gate = nn.Sequential(
                nn.Linear(21, gate_hidden_dim),
                nn.SiLU(),
                nn.Dropout(gate_dropout),
                nn.Linear(gate_hidden_dim, 3),
            )
            if self.fusion_mode == "homology_residual":
                nn.init.zeros_(self.homology_gate[-1].weight)
                nn.init.zeros_(self.homology_gate[-1].bias)
        if self.fusion_mode != "learned":
            self.gate.requires_grad_(False)

    def _initialize_lora(self, config, device):
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise RuntimeError("LoRA mode requires the peft package") from exc
        encoder = SaProtEncoder(config, device, trainable=True)
        target_modules = [
            name for name, module in encoder.model.named_modules()
            if isinstance(module, nn.Linear) and name.endswith(
                ("attention.self.query", "attention.self.key", "attention.self.value", "attention.output.dense")
            )
        ]
        if not target_modules:
            raise RuntimeError("could not locate SaProt attention modules for LoRA")
        lora = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=int(get(config, "model.lora.rank", 8)),
            lora_alpha=int(get(config, "model.lora.alpha", 16)),
            lora_dropout=float(get(config, "model.lora.dropout", 0.05)),
            target_modules=target_modules,
            bias="none",
        )
        self.saprot_model = get_peft_model(encoder.model, lora)
        self.saprot_tokenizer = encoder.tokenizer

    def _lora_embeddings(self, texts, lengths):
        tokenized = self.saprot_tokenizer(
            texts, add_special_tokens=True, padding=True, return_attention_mask=True,
            return_special_tokens_mask=True, return_tensors="pt",
        )
        device = next(self.saprot_model.parameters()).device
        special = tokenized.pop("special_tokens_mask").to(device).bool()
        tokenized = {key: value.to(device) for key, value in tokenized.items()}
        hidden = self.saprot_model(**tokenized).last_hidden_state
        mask = tokenized["attention_mask"].bool() & ~special
        rows = []
        for index, length in enumerate(lengths.tolist()):
            row = hidden[index][mask[index]]
            if row.shape[0] != length:
                raise ValueError("SaProt token/residue mismatch in LoRA mode")
            rows.append(row)
        return torch.cat(rows, dim=0)

    def forward(
        self,
        batch: dict[str, Any],
        retrieval_views: dict[str, torch.Tensor] | None = None,
    ):
        embeddings = batch["embeddings"]
        if self.mode == "lora":
            embeddings = self._lora_embeddings(batch["saprot_texts"], batch["lengths"])
        if embeddings is None:
            raise ValueError("frozen mode requires cached SaProt embeddings")
        structural = batch["node_features"]
        if self.disable_structural_features:
            structural = torch.zeros_like(structural)
        features = torch.cat([embeddings.float(), structural], dim=-1)
        hidden, updated_coords = self.geometry(
            features, batch["coords"], batch["edge_index"], batch["edge_features"]
        )
        logits, probabilities, global_mean, global_variance = self.decoder(
            hidden, batch["graph_index"], batch["lengths"], batch["ionizable_type"], batch["pka"]
        )
        pooled = scatter_mean(hidden, batch["graph_index"], int(batch["lengths"].shape[0]))
        ec_logits = self.ec_head(pooled)
        shared = {
            "logits": logits,
            "probabilities": probabilities,
            "global_mean": global_mean,
            "global_variance": global_variance,
            "ec_logits": ec_logits,
            "coords": updated_coords,
        }
        if retrieval_views is not None:
            return {
                name: self._fuse(shared, retrieval)
                for name, retrieval in retrieval_views.items()
            }
        return self._fuse(shared, batch["retrieval"])

    def _fuse(self, shared: dict[str, torch.Tensor], retrieval: torch.Tensor):
        global_mean = shared["global_mean"]
        global_variance = shared["global_variance"]
        reliability_context = torch.stack(
            [retrieval[:, 2], retrieval[:, 3], retrieval[:, 4], retrieval[:, 5], retrieval[:, 6], global_variance], dim=-1
        )
        expert_means = torch.stack([global_mean, retrieval[:, 0], retrieval[:, 1]], dim=-1)
        available = torch.stack(
            [torch.ones_like(retrieval[:, 7]), retrieval[:, 7], retrieval[:, 8]], dim=-1
        ).bool()
        if self.disable_retrieval:
            available[:, 1:] = False
        elif self.disable_foldseek:
            available[:, 2] = False
        if self.training and self.retrieval_dropout > 0:
            dropped = torch.rand_like(available[:, 1:].float()) < self.retrieval_dropout
            available[:, 1:] &= ~dropped
        fusion_details = {}
        if self.fusion_mode == "fixed":
            gate_weights = self.fixed_gate_weights.to(expert_means).expand(len(available), -1)
            gate_weights = gate_weights * available.to(gate_weights.dtype)
            gate_weights = gate_weights / gate_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        elif self.fusion_mode == "reliability":
            disagreements = torch.stack(
                [
                    (expert_means[:, 0] - expert_means[:, 1]).abs(),
                    (expert_means[:, 0] - expert_means[:, 2]).abs(),
                    (expert_means[:, 1] - expert_means[:, 2]).abs(),
                ],
                dim=-1,
            )
            gate_features = torch.cat(
                [
                    reliability_context,
                    expert_means,
                    disagreements,
                    available.to(expert_means.dtype),
                ],
                dim=-1,
            ).detach()
            gate_logits = self.reliability_gate(gate_features)
            gate_logits = gate_logits.masked_fill(
                ~available, torch.finfo(gate_logits.dtype).min
            )
            gate_weights = torch.softmax(gate_logits / self.gate_temperature, dim=-1)
        elif self.fusion_mode in {"homology_reliability", "homology_residual"}:
            extended_context = torch.cat(
                [
                    retrieval[:, [2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14]],
                    global_variance.unsqueeze(-1),
                ],
                dim=-1,
            )
            disagreements = torch.stack(
                [
                    (expert_means[:, 0] - expert_means[:, 1]).abs(),
                    (expert_means[:, 0] - expert_means[:, 2]).abs(),
                    (expert_means[:, 1] - expert_means[:, 2]).abs(),
                ],
                dim=-1,
            )
            gate_features = torch.cat(
                [
                    extended_context,
                    expert_means,
                    disagreements,
                    available.to(expert_means.dtype),
                ],
                dim=-1,
            ).detach()
            residual_logits = self.homology_gate(gate_features)
            if self.fusion_mode == "homology_residual":
                baseline_logits = self.gate(reliability_context)
                baseline_logits = baseline_logits.masked_fill(
                    ~available, torch.finfo(baseline_logits.dtype).min
                )
                baseline_gate_weights = torch.softmax(baseline_logits, dim=-1)
                baseline_mean = (baseline_gate_weights * expert_means).sum(dim=-1)
                gate_logits = baseline_logits + residual_logits / self.gate_temperature
                fusion_details = {
                    "baseline_mean": baseline_mean,
                    "baseline_gate_weights": baseline_gate_weights,
                    "homology_residual_logits": residual_logits,
                }
            else:
                gate_logits = residual_logits / self.gate_temperature
            gate_logits = gate_logits.masked_fill(
                ~available, torch.finfo(gate_logits.dtype).min
            )
            gate_weights = torch.softmax(gate_logits, dim=-1)
        else:
            gate_logits = self.gate(reliability_context)
            gate_logits = gate_logits.masked_fill(~available, torch.finfo(gate_logits.dtype).min)
            gate_weights = torch.softmax(gate_logits, dim=-1)
        mean = (gate_weights * expert_means).sum(dim=-1)
        expert_variances = torch.stack([global_variance, retrieval[:, 5], retrieval[:, 6]], dim=-1)
        variance = (gate_weights * (expert_variances + (expert_means - mean.unsqueeze(-1)).square())).sum(dim=-1)
        return {
            **shared,
            "mean": mean,
            "variance": variance,
            "gate_weights": gate_weights,
            "expert_means": expert_means,
            "expert_available": available,
            **fusion_details,
        }


def compute_loss(outputs: dict[str, torch.Tensor], batch: dict[str, Any], config: dict[str, Any]):
    labels = batch["labels"]
    grid = torch.arange(
        float(get(config, "model.ph_min", 2.0)),
        float(get(config, "model.ph_max", 12.0)) + float(get(config, "model.ph_step", 0.25)) / 2,
        float(get(config, "model.ph_step", 0.25)),
        device=labels.device,
    )
    sigma = float(get(config, "loss.soft_label_sigma", 0.35))
    soft_targets = torch.exp(-0.5 * ((grid.unsqueeze(0) - labels.unsqueeze(-1)) / sigma).square())
    soft_targets = soft_targets / soft_targets.sum(dim=-1, keepdim=True)
    distribution_loss = -(soft_targets * F.log_softmax(outputs["logits"], dim=-1)).sum(dim=-1)
    mse_loss = F.mse_loss(outputs["mean"], labels, reduction="none")
    weights = batch["weights"].clamp(
        float(get(config, "loss.min_sample_weight", 0.5)), float(get(config, "loss.max_sample_weight", 3.0))
    )
    weighted_distribution = (weights * distribution_loss).sum() / weights.sum().clamp_min(1e-8)
    primary = (
        float(get(config, "loss.mse_weight", 1.0)) * mse_loss.mean()
        + float(get(config, "loss.distribution_weight", 0.2)) * weighted_distribution
    )
    valid_ec = batch["ec_labels"] >= 0
    ec_loss = outputs["logits"].new_zeros(())
    if valid_ec.any():
        ec_loss = F.cross_entropy(outputs["ec_logits"][valid_ec], batch["ec_labels"][valid_ec])
    total = primary + float(get(config, "loss.ec_weight", 0.1)) * ec_loss
    components = {
        "mse": mse_loss.mean().detach(),
        "distribution": weighted_distribution.detach(),
        "ec": ec_loss.detach(),
    }
    gate_loss, gate_components = _gate_regularization(outputs, labels, config)
    total = total + gate_loss
    components.update(gate_components)
    return total, components


def compute_low_homology_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    config: dict[str, Any],
):
    labels = batch["labels"]
    mse = F.mse_loss(outputs["mean"], labels)
    gate_loss, gate_components = _gate_regularization(outputs, labels, config)
    total = float(get(config, "loss.mse_weight", 1.0)) * mse + gate_loss
    return total, {"mse": mse.detach(), **gate_components}


def compute_dual_view_loss(
    outputs: dict[str, dict[str, torch.Tensor]],
    batch: dict[str, Any],
    config: dict[str, Any],
):
    if set(outputs) != {"normal", "low_homology"}:
        raise ValueError("dual-view outputs must contain normal and low_homology")
    normal_loss, normal_components = compute_loss(outputs["normal"], batch, config)
    low_loss, low_components = compute_low_homology_loss(
        outputs["low_homology"], batch, config
    )
    consistency = F.smooth_l1_loss(
        outputs["low_homology"]["mean"],
        outputs["normal"]["mean"].detach(),
    )
    normal_weight = float(get(config, "homology_training.normal_loss_weight", 1.0))
    low_weight = float(get(config, "homology_training.low_homology_loss_weight", 0.5))
    consistency_weight = float(get(config, "homology_training.consistency_weight", 0.1))
    preservation_weight = float(
        get(config, "homology_training.preservation_weight", 0.0)
    )
    preservation = batch["labels"].new_zeros(())
    if preservation_weight > 0:
        normal = outputs["normal"]
        if "baseline_mean" not in normal:
            raise ValueError(
                "homology preservation requires fusion.mode=homology_residual"
            )
        preservation_mask = _homology_preservation_mask(batch, config)
        if preservation_mask.any():
            preservation = F.smooth_l1_loss(
                normal["mean"][preservation_mask],
                normal["baseline_mean"][preservation_mask].detach(),
            )
    total = (
        normal_weight * normal_loss
        + low_weight * low_loss
        + consistency_weight * consistency
        + preservation_weight * preservation
    )
    components = {
        **{f"normal_{name}": value for name, value in normal_components.items()},
        **{f"low_homology_{name}": value for name, value in low_components.items()},
        "consistency": consistency.detach(),
        "preservation": preservation.detach(),
    }
    return total, components


def _homology_preservation_mask(batch, config):
    scope = str(
        get(config, "homology_training.preservation_scope", "high_homology")
    ).lower()
    retrieval = batch["retrieval"]
    if scope == "all":
        return torch.ones(len(retrieval), dtype=torch.bool, device=retrieval.device)
    if scope != "high_homology":
        raise ValueError(
            "homology_training.preservation_scope must be 'high_homology' or 'all'"
        )
    identity = _fraction_threshold(
        get(
            config,
            "homology_training.preservation_identity",
            get(config, "retrieval.low_homology_identity", 0.2),
        ),
        "homology_training.preservation_identity",
    )
    coverage = _fraction_threshold(
        get(
            config,
            "homology_training.preservation_coverage",
            get(config, "retrieval.low_homology_coverage", 0.8),
        ),
        "homology_training.preservation_coverage",
    )
    return (
        (retrieval[:, 4] >= identity)
        & (retrieval[:, 9] >= coverage)
        & (retrieval[:, 10] >= coverage)
    )


def _fraction_threshold(value, name):
    threshold = float(value)
    if threshold > 1:
        threshold /= 100.0
    if not 0 <= threshold <= 1:
        raise ValueError(f"{name} must be a fraction or percentage")
    return threshold


def _gate_regularization(outputs, labels, config):
    gate_supervision_weight = float(get(config, "loss.gate_supervision_weight", 0.0))
    gate_prior_weight = float(get(config, "loss.gate_prior_weight", 0.0))
    zero = labels.new_zeros(())
    if gate_supervision_weight <= 0 and gate_prior_weight <= 0:
        return zero, {}
    if "expert_means" not in outputs or "expert_available" not in outputs:
        raise ValueError("supervised gate loss requires expert predictions and availability")
    gate_weights = outputs["gate_weights"].clamp_min(1e-8)
    available = outputs["expert_available"].bool()
    target_temperature = float(get(config, "loss.gate_target_temperature", 0.3))
    if target_temperature <= 0:
        raise ValueError("loss.gate_target_temperature must be positive")
    expert_errors = (outputs["expert_means"].detach() - labels.unsqueeze(-1)).abs()
    target_logits = (-expert_errors / target_temperature).masked_fill(
        ~available, torch.finfo(expert_errors.dtype).min
    )
    gate_targets = torch.softmax(target_logits, dim=-1)
    gate_supervision = -(gate_targets * gate_weights.log()).sum(dim=-1).mean()

    gate_prior = torch.as_tensor(
        get(config, "fusion.gate_prior", [0.25, 0.25, 0.5]),
        dtype=gate_weights.dtype,
        device=gate_weights.device,
    )
    if gate_prior.shape != (3,) or bool((gate_prior < 0).any()) or float(gate_prior.sum()) <= 0:
        raise ValueError("fusion.gate_prior must contain three non-negative values")
    gate_prior = gate_prior.expand_as(gate_weights) * available.to(gate_weights.dtype)
    gate_prior = gate_prior / gate_prior.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    gate_prior_loss = (
        gate_weights * (gate_weights.log() - gate_prior.clamp_min(1e-8).log())
    ).sum(dim=-1).mean()
    total = (
        gate_supervision_weight * gate_supervision
        + gate_prior_weight * gate_prior_loss
    )
    return total, {
        "gate_supervision": gate_supervision.detach(),
        "gate_prior": gate_prior_loss.detach(),
    }


def _pad_hidden(hidden, graph_index, lengths):
    batch_size = int(lengths.shape[0])
    maximum = int(lengths.max().item())
    padded = hidden.new_zeros((batch_size, maximum, hidden.shape[-1]))
    padding_mask = torch.ones((batch_size, maximum), dtype=torch.bool, device=hidden.device)
    for index, length in enumerate(lengths.tolist()):
        row = hidden[graph_index == index]
        padded[index, :length] = row
        padding_mask[index, :length] = False
    return padded, padding_mask
