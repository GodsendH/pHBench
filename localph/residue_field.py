"""Sparse residue-conditioned pH response field on fixed PLM sketches.

The response is statistical evidence over pH, not a mechanistic activity curve.
There is no task-trained self-attention, retrieval, expert router or PLM update.
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class ResidueField(nn.Module):
    def __init__(self, input_dim=128, width=32, kind="sparse", dropout=.25):
        super().__init__()
        if kind not in ("sparse", "global", "direct"):
            raise ValueError("invalid field kind")
        self.kind = kind
        self.norm = nn.LayerNorm(input_dim)
        self.encoder = nn.Sequential(nn.Linear(input_dim, width), nn.GELU(), nn.Dropout(dropout))
        self.global_head = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Dropout(dropout),
                                         nn.Linear(width, 1 if kind == "direct" else 13))
        self.local_head = nn.Linear(width, 13, bias=False) if kind == "sparse" else None
        grid = torch.linspace(0, 14, 57)
        centers = torch.linspace(1, 13, 13)
        basis = torch.exp(-.5 * ((grid[None] - centers[:, None]) / 1.25) ** 2)
        self.register_buffer("grid", grid)
        self.register_buffer("basis", basis)
        # Begin near the center with moderate predictions, without saturating.
        nn.init.zeros_(self.global_head[-1].weight)
        nn.init.zeros_(self.global_head[-1].bias)
        if self.local_head is not None:
            nn.init.normal_(self.local_head.weight, std=.01)

    def forward(self, tokens, mask, ionizable):
        if tokens.ndim != 3 or mask.shape != tokens.shape[:2] or ionizable.shape != mask.shape:
            raise ValueError("residue/mask shapes differ")
        if not mask.any(1).all() or (ionizable & ~mask).any():
            raise ValueError("invalid residue mask")
        h = self.encoder(self.norm(tokens.float()))
        weights = mask[..., None].to(h.dtype)
        n = weights.sum(1)
        mean = (h * weights).sum(1) / n
        second = (h.square() * weights).sum(1) / n
        std = (second - mean.square()).clamp_min(1e-6).sqrt()
        pooled = torch.cat([mean, std], -1)
        coefficients = self.global_head(pooled)
        if self.kind == "direct":
            return {"prediction": 7 + coefficients[:, 0]}
        logits = coefficients @ self.basis
        if self.kind == "sparse":
            # Hard top-k is over residues for each pH; no normalized attention.
            # No ionizable residue is treated as zero local response.
            local = self.local_head(h) @ self.basis
            eligible = ionizable & mask
            local = local.masked_fill(~eligible[..., None], -1e4)
            k = min(8, tokens.shape[1])
            top = local.topk(k, dim=1).values
            counts = eligible.sum(1).clamp_max(k)
            valid_top = torch.arange(k, device=tokens.device)[None, :, None] < counts[:, None, None]
            contribution = torch.where(valid_top, top, 0).sum(1) / counts.clamp_min(1)[:, None]
            logits = logits + contribution
        return {"logits": logits, "prediction": logits.softmax(-1) @ self.grid}


def label_prior(labels, grid, sigma=.5):
    labels = torch.as_tensor(labels, dtype=torch.float32, device=grid.device)
    targets = torch.exp(-.5 * ((grid[None] - labels[:, None]) / sigma) ** 2)
    targets /= targets.sum(1, keepdim=True)
    prior = targets.mean(0).clamp_min(1e-4)
    return prior / prior.sum()


def field_loss(output, labels, grid, prior, kind):
    if kind == "direct":
        return F.mse_loss(output["prediction"], labels)
    soft = torch.exp(-.5 * ((grid[None] - labels[:, None]) / .5) ** 2)
    soft /= soft.sum(1, keepdim=True)
    # The log training prior is explicit; the network learns evidence in
    # addition to this prior. A mean-error term protects central accuracy.
    logits = output["logits"] + prior.log()[None]
    nll = -(soft * logits.log_softmax(-1)).sum(1).mean()
    mean = logits.softmax(-1) @ grid
    curvature = output["logits"][:, 2:] - 2 * output["logits"][:, 1:-1] + output["logits"][:, :-2]
    return nll + .25 * F.mse_loss(mean, labels) + .01 * curvature.square().mean()


def decode(logits, prior, grid):
    """Predeclared point decisions; never uses query labels."""
    if logits.ndim != 2 or logits.shape[1] != len(grid):
        raise ValueError("pH grid mismatch")
    predictions = {}
    for power in (1., .5, 0.):
        probabilities = (logits + power * prior.log()[None]).softmax(-1)
        predictions[f"prior{power:g}_mean"] = (probabilities @ grid).cpu().numpy()
        predictions[f"prior{power:g}_mode"] = grid[probabilities.argmax(-1)].cpu().numpy()
    return predictions
