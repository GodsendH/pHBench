"""Label-free sketches of ionizable residue environments.

These are sequence-context features, not inferred catalytic sites or measured
pKa values. Residue-specific contrasts preserve information erased by a single
whole-protein mean. The projection is fixed before seeing any task labels.
"""
import numpy as np
import torch
import torch.nn.functional as F

IONIZABLE = "DEHCKRY"
PROJECTION_SEED = 17092026


def projection(input_dim=1280, width=64):
    if not 1 <= width <= input_dim:
        raise ValueError("projection width must be in [1, input_dim]")
    rng = np.random.default_rng(PROJECTION_SEED)
    q, _ = np.linalg.qr(rng.standard_normal((input_dim, width)))
    return torch.tensor(q, dtype=torch.float32)


@torch.inference_mode()
def ion_context(sequence, residue_embeddings, basis):
    h = residue_embeddings
    if h.ndim != 2 or len(sequence) != len(h) or not len(sequence):
        raise ValueError("one embedding per residue is required; strip BOS/EOS first")
    if basis.ndim != 2 or basis.shape[0] != h.shape[1]:
        raise ValueError("projection dimension mismatch")
    if not torch.isfinite(h).all() or not torch.isfinite(basis).all():
        raise ValueError("nonfinite representation")
    z = h.float() @ basis.to(h.device)
    # Include all real residues at termini; zero padding has zero weight.
    n = len(sequence)
    windows = F.avg_pool1d(z.T[None], 9, stride=1, padding=4, count_include_pad=False)[0].T
    global_mean = z.mean(0)
    rows = []
    counts = []
    for aa in IONIZABLE:
        mask = torch.tensor([c == aa for c in sequence], device=h.device)
        count = int(mask.sum())
        if count:
            selected = z[mask]
            rows.extend([selected.mean(0) - global_mean,
                         selected.std(0, unbiased=False),
                         windows[mask].mean(0) - selected.mean(0)])
        else:
            rows.extend([z.new_zeros(z.shape[1]) for _ in range(3)])
        counts.extend([count / n, float(count > 0)])
    return torch.cat([*rows, z.new_tensor(counts)]).cpu().numpy()


class SiteScaler:
    """Training-only standardization, variance floor, and fixed block scale."""
    def fit(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
            raise ValueError("nonempty finite features required")
        self.mean = x.mean(0)
        self.scale = np.maximum(x.std(0), .05)
        self.width = x.shape[1]
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.width or not np.isfinite(x).all():
            raise ValueError("feature dimension or finiteness mismatch")
        return np.clip((x - self.mean) / self.scale, -6, 6) / np.sqrt(self.width)
