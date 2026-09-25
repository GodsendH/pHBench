"""Small environment-supervised encoder and continuous enzyme residual.

pHenv and enzyme pHopt have separate heads. No pHenv label or organism
identifier is required at enzyme inference; the PLM remains frozen upstream.
"""
import numpy as np
import torch
from torch import nn


class EnvironmentEncoder(nn.Module):
    def __init__(self, input_dim=1280, width=32, dropout=.1):
        super().__init__()
        self.input_dim, self.width = input_dim, width
        self.norm = nn.LayerNorm(input_dim, elementwise_affine=False)
        self.map = nn.Sequential(nn.Linear(input_dim, width), nn.GELU(), nn.Dropout(dropout))

    def forward(self, tokens, mask):
        if tokens.ndim != 3 or tokens.shape[-1] != self.input_dim or mask.shape != tokens.shape[:2]:
            raise ValueError('invalid token/mask shapes')
        if mask.dtype != torch.bool or not bool(mask.any(1).all()):
            raise ValueError('each protein needs a boolean mask and at least one residue')
        # Padding does not enter normalization or either moment.
        h = self.map(self.norm(tokens.float().masked_fill(~mask[...,None], 0)))
        w = mask[...,None].to(h.dtype)
        mean = (h*w).sum(1)/w.sum(1)
        variance = ((h-mean[:,None]).square()*w).sum(1)/w.sum(1)
        return torch.cat([mean, variance.clamp_min(1e-6).sqrt()], dim=-1)


class EnvironmentTransfer(nn.Module):
    def __init__(self, input_dim=1280, width=32, dropout=.1, affine=True):
        super().__init__()
        self.encoder = EnvironmentEncoder(input_dim, width, dropout)
        self.environment_head = nn.Linear(2*width, 1)
        self.enzyme_head = nn.Linear(2*width, 2 if affine else 1)
        self.affine = affine
        self.encoder_frozen = False
        self.reset_enzyme_head()

    def reset_enzyme_head(self):
        nn.init.zeros_(self.enzyme_head.weight)
        nn.init.zeros_(self.enzyme_head.bias)

    def freeze_environment(self):
        self.encoder_frozen = True
        self.encoder.requires_grad_(False).eval()
        self.environment_head.requires_grad_(False).eval()
        return self

    def train(self, mode=True):
        super().train(mode)
        if self.encoder_frozen:
            self.encoder.eval()
            self.environment_head.eval()
        return self

    def predict_environment(self, tokens, mask):
        return 7 + self.environment_head(self.encoder(tokens,mask)).squeeze(-1)

    def forward(self, tokens, mask, baseline):
        z = self.encoder(tokens,mask)
        if baseline.shape != z.shape[:1]:
            raise ValueError('baseline must have one value per protein')
        coefficients = self.enzyme_head(z)
        correction = coefficients[:,0]
        if self.affine:
            correction = correction + coefficients[:,1]*(baseline-7)
        return baseline + correction


def training_bin_weights(labels, power=1., mix=.5):
    """Weights fitted only to the explicit training vector, mean normalized.

    mix=0 is natural MSE; mix=1 uses full inverse-frequency weights.
    Return the counts and table so that the fit can be independently audited.
    """
    y = np.asarray(labels,dtype=np.float64)
    if y.ndim!=1 or len(y)==0 or not np.isfinite(y).all() or not 0<=mix<=1 or not 0<=power<=1:
        raise ValueError('invalid training labels or weighting recipe')
    bins = np.digitize(y,[5.,9.])
    counts = np.bincount(bins,minlength=3)
    table = np.zeros(3)
    present = counts>0
    table[present] = counts[present].astype(float)**(-power)
    table /= (table*counts).sum()/len(y)
    weights = (1-mix)+mix*table[bins]
    return weights.astype(np.float32), {'counts':counts.tolist(),'table':table.tolist(),'power':power,'mix':mix,
                                     'bins':['<5','[5,9)','>=9'],'fit_rows':len(y)}


def enzyme_objective(prediction, baseline, labels, weights, core_strength=1.):
    if not (prediction.shape==baseline.shape==labels.shape==weights.shape) or prediction.ndim!=1:
        raise ValueError('unaligned enzyme loss inputs')
    if not bool(torch.isfinite(weights).all()) or bool((weights<=0).any()) or core_strength<0:
        raise ValueError('invalid enzyme loss weights')
    mse = ((prediction-labels).square()*weights).sum()/weights.sum()
    core = (labels>4)&(labels<10)
    preservation = (prediction[core]-baseline[core]).square().mean() if bool(core.any()) else prediction.sum()*0
    return mse+core_strength*preservation, {'weighted_mse':mse.detach(),'core_preservation':preservation.detach()}
