"""Sequence-only inference with train-only EC auxiliary supervision and nested early stopping."""
import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '8'
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from develop_phgeofuse_regression import ROOT, OUT, metrics
from phgeofuse.io import read_manifest
from phgeofuse.robust_fusion import pool_features
from phgeofuse.cache import atomic_json, atomic_torch_save


class FunctionalHead(nn.Module):
    def __init__(self, width, classes):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Dropout(.2))
        self.ph = nn.Linear(64, 1)
        self.ec = nn.Linear(64, classes)

    def forward(self, x):
        z = self.shared(x)
        return self.ph(z).squeeze(-1), self.ec(z)


def fit_head(x, y, ec, aux, seed, epochs, validation=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    scaler = StandardScaler().fit(x)
    z = torch.tensor(np.clip(scaler.transform(x), -8, 8), dtype=torch.float32, device='cuda')
    mean = float(y.mean())
    target = torch.tensor(y - mean, dtype=torch.float32, device='cuda')
    vocabulary = sorted(set(ec) - {'unknown'})
    mapping = {v: i for i, v in enumerate(vocabulary)}
    labels = torch.tensor([mapping.get(e, -1) for e in ec], device='cuda', dtype=torch.long)
    model = FunctionalHead(x.shape[1], max(1, len(vocabulary))).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    valid = None
    if validation is not None:
        vx, vy = validation
        valid = torch.tensor(np.clip(scaler.transform(vx), -8, 8), dtype=torch.float32, device='cuda')
    best = float('inf')
    best_epoch = 1
    history = []
    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        p, logits = model(z)
        loss_ph = (p-target).square().mean()
        loss = loss_ph
        if aux and (labels >= 0).any():
            loss = loss + aux * nn.functional.cross_entropy(logits, labels, ignore_index=-1)
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite multitask loss')
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.inference_mode():
                train_rmse = float(torch.mean((model(z)[0] - target)**2).sqrt().item())
                if valid is not None:
                    pv = model(valid)[0].cpu().numpy() + mean
                    score = float(np.sqrt(np.mean((pv-vy)**2)))
                else:
                    score = None
            history.append({'epoch': epoch, 'training_rmse': train_rmse, 'inner_validation_rmse': score})
            if score is not None:
                if score < best - .0001:
                    best, best_epoch = score, epoch
                if epoch - best_epoch >= 40:
                    break
    model.eval()
    return model, scaler, mean, vocabulary, best_epoch if validation is not None else epochs, history


def predict(fitted, x):
    model, scaler, mean = fitted[:3]
    z = torch.tensor(np.clip(scaler.transform(x), -8, 8), dtype=torch.float32, device='cuda')
    with torch.inference_mode():
        return model(z)[0].cpu().numpy().astype(float) + mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    out = OUT / 'function_multitask'
    out.mkdir(exist_ok=True)
    records = [r for r in read_manifest(ROOT / 'artifacts/phgeofuse/manifest.csv')
               if r.split in ('train', 'validation')]
    train = np.array([r.split == 'train' for r in records])
    keys = np.array([r.split + '::' + r.protein_id for r in records])
    labels = np.array([r.ph_opt for r in records])
    y, yv = labels[train], labels[~train]
    ec = np.array(['.'.join(r.ec.strip().split('.')[:2])
                   if re.fullmatch(r'[1-7](?:\.(?:\d+|-)){3}', r.ec.strip()) else 'unknown'
                   for r in records])[train]
    parts = []
    for encoder in ['esm1v', 'esm2']:
        with np.load(OUT / f'{encoder}_masked/features.npz') as f:
            mapping = {str(k): i for i, k in enumerate(f['keys'])}
            idx = [mapping[k] for k in keys]
            parts.append(pool_features(f['mean'][idx], f['std'][idx], 'mean_std'))
    xall = np.column_stack(parts)
    x, xv = xall[train], xall[~train]
    foldrows = json.loads((OUT / 'homology_oof/strict_folds.json').read_text())['rows']
    assert [r['key'] for r in foldrows] == [r.protein_id for r in records if r.split == 'train']
    fold = np.array([r['fold'] for r in foldrows])
    groups = np.array([r['group'] for r in foldrows])
    protocol = {'features': 'frozen ESM1v/ESM2 residue mean/std', 'hidden': 64, 'dropout': .2,
        'aux_weights': [0., .1], 'seed': 42, 'epochs_max': 300, 'early_stopping_patience': 40,
        'learning_rate': .001, 'weight_decay': .01, 'gradient_clip': 1.,
        'early_stopping': '20% inner group holdout selects epochs; refit from scratch on all outer train',
        'EC': 'training-only auxiliary EC level2; not an inference input', 'test_access': False,
        'evaluation': 'Five strict outer groups; original validation used only after full-train refit. '
                      'Architecture exploration remains development, not independent final confirmation.'}
    if args.smoke:
        fitted = fit_head(x[:300], y[:300], ec[:300], .1, 42, 5, (x[300:350], y[300:350]))
        assert np.isfinite(predict(fitted, x[350:370])).all()
        print('FUNCTION_MULTITASK_SMOKE_COMPLETE', flush=True)
        return
    atomic_json(out / 'protocol.json', protocol)
    lowv = np.load(OUT / 'development_features.npz')['lowv']
    reference = np.load(OUT / 'nested_homology_strict/predictions.npz')
    assert np.array_equal(reference['keys'], keys[train])
    retr = reference['retrieval']
    low = ~((retr[:, 4] >= .2) & (retr[:, 9] >= .8) & (retr[:, 10] >= .8))
    results = []
    for aux in [0., .1]:
        oof = np.full(len(y), np.nan)
        audits = []
        for outer in range(6):
            tr = np.flatnonzero(fold != outer) if outer < 5 else np.arange(len(y))
            te = np.flatnonzero(fold == outer) if outer < 5 else None
            it, iv = next(GroupShuffleSplit(1, test_size=.2, random_state=42+outer).split(x[tr], groups=groups[tr]))
            inner_train, inner_val = tr[it], tr[iv]
            assert not set(groups[inner_train]) & set(groups[inner_val])
            if outer < 5:
                assert not set(groups[tr]) & set(groups[te])
            atomic_json(out / 'status.json', {'status': 'running', 'pid': os.getpid(), 'aux': aux,
                'outer': outer, 'phase': 'inner_selection', 'updated': time.time()})
            selected = fit_head(x[inner_train], y[inner_train], ec[inner_train], aux, 42, 300,
                                (x[inner_val], y[inner_val]))
            epochs = selected[4]
            history = selected[5]
            del selected
            torch.cuda.empty_cache()
            atomic_json(out / 'status.json', {'status': 'running', 'pid': os.getpid(), 'aux': aux,
                'outer': outer, 'phase': 'outer_refit', 'epochs': epochs, 'updated': time.time()})
            fitted = fit_head(x[tr], y[tr], ec[tr], aux, 42, epochs)
            pred = predict(fitted, x[te] if outer < 5 else xv)
            audit = {'outer': outer, 'selected_epochs': epochs,
                'training_rmse': float(np.sqrt(np.mean((predict(fitted, x[tr])-y[tr])**2))),
                'metrics': metrics(y[te] if outer < 5 else yv, pred, low[te] if outer < 5 else lowv),
                'inner_history': history, 'refit_history': fitted[5],
                'train_keys': keys[train][tr].tolist(), 'inner_validation_keys': keys[train][inner_val].tolist()}
            audits.append(audit)
            if outer < 5:
                oof[te] = pred
            else:
                pv = pred
                model, scaler, mean, vocabulary = fitted[:4]
                atomic_torch_save(out / f'aux{aux}.pt', {'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                    'scaler_mean': scaler.mean_, 'scaler_scale': scaler.scale_, 'label_mean': mean,
                    'ec_vocabulary': vocabulary, 'input_width': x.shape[1], 'epochs': epochs, 'protocol': protocol})
            atomic_json(out / f'aux{aux}.fold_audit.json', audits)
            print('FUNCTION_MULTITASK_FOLD', aux, outer, 'epochs', epochs, json.dumps(audit['metrics']), flush=True)
            del fitted
            torch.cuda.empty_cache()
        row = {'aux_weight': aux, 'strict_oof': metrics(y, oof, low), 'validation': metrics(yv, pv, lowv),
               'outer_train_rmse_mean': float(np.mean([a['training_rmse'] for a in audits[:5]]))}
        results.append(row)
        np.savez(out / f'aux{aux}.predictions.npz', oof=oof, validation=pv, keys=keys[train])
        atomic_json(out / 'results.json', results)
        print(json.dumps(row), flush=True)
    atomic_json(out / 'status.json', {'status': 'complete', 'pid': os.getpid(), 'updated': time.time()})
    print('FUNCTION_MULTITASK_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
