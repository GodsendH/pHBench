# pH-GeoFuse v3 shrinkage results

## Scope

The v3 experiment trains a zero-initialized residual on top of the frozen
tuned-v1 retrieval gate. SaProt, EGNN, the pH decoder, and the tuned-v1 gate
remain frozen. A validation-only calibration stage selects the residual scale
from `[0, 0.125, 0.25, 0.5, 0.75, 1]`.

The calibration has two safeguards against gate overfitting:

- Disable the residual unless it improves validation RMSE by at least `0.002`.
- Among scales within `0.0011` RMSE of the validation optimum, select the
  smallest scale.

Checkpoint selection always preserves the numerically lowest validation RMSE.
Early stopping separately uses `min_delta=0.001`, so BF16 evaluation jitter
does not reset patience.

## Final test results

All results use seed 42 and the calibrated checkpoint selected without test
labels. Lower RMSE and MAE are better.

| Dataset | tuned-v1 RMSE | v3 RMSE | Delta | tuned-v1 MAE | v3 MAE | Delta | Scale |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PHOPT test | 0.788768 | 0.786807 | -0.001960 | 0.562897 | 0.562495 | -0.000403 | 0.00 |
| identity20 test | 0.883364 | 0.883061 | -0.000303 | 0.653217 | 0.649548 | -0.003669 | 0.75 |
| PHOPT official low-identity subset | 0.839247 | 0.836984 | -0.002263 | 0.601979 | 0.602150 | +0.000171 | 0.00 |

The two required full datasets improve in both RMSE and MAE. The additional
PHOPT low-identity diagnostic improves RMSE; its MAE changes by `+0.000171`.
The subset evaluation covers 997 of 999 requested proteins because `P97364`
and `P22352` do not have ready structure artifacts.

## Overfitting evidence

On PHOPT, the unshrunk residual validation RMSE worsened from `0.827751` at
epoch 1 to `0.838501` at epoch 6. Calibration selected scale 0 throughout,
holding deployed validation RMSE near `0.8274` and train RMSE near `0.7319`.
The learned correction was therefore rejected instead of widening the
train-validation gap.

On identity20, the residual produced stable validation gains. At the selected
epoch, validation calibration measured:

| Scale | Validation RMSE |
| ---: | ---: |
| 0.00 | 0.946336 |
| 0.25 | 0.942503 |
| 0.50 | 0.939633 |
| 0.75 | 0.937845 |
| 1.00 | 0.937226 |

Scale 0.75 is within `0.0011` of the optimum and is the smallest eligible
residual, reducing deployment variance while retaining most validation gain.

## Reproduction

Train separate checkpoints because PHOPT and identity20 splits overlap:

```bash
python -m phgeofuse.train \
  --config configs/phgeofuse_phopt_homology_gate_v3.yaml \
  --dataset phopt \
  --init-checkpoint \
    artifacts/phgeofuse/runs/phgeofuse_phopt_tuned_mse_v1_frozen_seed42/best.pt

python -m phgeofuse.train \
  --config configs/phgeofuse_phopt_homology_gate_v3.yaml \
  --dataset identity20 \
  --init-checkpoint \
    artifacts/phgeofuse/datasets/identity20/runs/phgeofuse_phopt_tuned_mse_v1_frozen_seed42/best.pt
```

Calibrate each `best.pt` with its own validation split before test evaluation:

```bash
python -m phgeofuse.calibrate \
  --config configs/phgeofuse_phopt_homology_gate_v3.yaml \
  --dataset identity20 \
  --checkpoint \
    artifacts/phgeofuse/datasets/identity20/runs/phgeofuse_phopt_homology_gate_v3_shrinkage_frozen_seed42/best.pt
```

The final metrics are stored at:

- `artifacts/phgeofuse/runs/phgeofuse_phopt_homology_gate_v3_shrinkage_frozen_seed42/test_predictions_calibrated.metrics.json`
- `artifacts/phgeofuse/runs/phgeofuse_phopt_homology_gate_v3_shrinkage_frozen_seed42/test_low_identity_predictions_calibrated.metrics.json`
- `artifacts/phgeofuse/datasets/identity20/runs/phgeofuse_phopt_homology_gate_v3_shrinkage_frozen_seed42/test_predictions_calibrated.metrics.json`

Validation calibration metrics and selected scales are stored beside each
checkpoint as `best_calibrated.calibration.metrics.json`.
