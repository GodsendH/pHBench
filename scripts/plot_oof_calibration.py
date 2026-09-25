"""Render saved calibration audit; can run in a separate plotting environment."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = Path(__file__).resolve().parents[1] / 'experiments/phgeofuse_redesign_20260914/oof_calibration'
report = json.loads((out / 'diagnostic.json').read_text())
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
labels = {'unweighted50': 'Unweighted', 'phweighted50': 'pH weighted', 'family_phweighted50': 'Family + pH weighted'}
for name, label in labels.items():
    row = report['models'][name]
    bins = row['predicted_deciles']
    axes[0].plot([b['prediction_mean'] for b in bins], [b['label_mean'] for b in bins], 'o-', label=label)
    axes[1].plot([5, 7, 9], [row['observed'][g]['bias'] for g in ['acidic', 'neutral', 'alkaline']], 'o-', label=label)
axes[0].plot([5, 9], [5, 9], '--', color='gray')
axes[0].set(xlabel='Mean prediction in predicted decile', ylabel='Mean true pH', title='Calibration by predicted value')
axes[1].axhline(0, linestyle='--', color='gray')
axes[1].set(xticks=[5, 7, 9], xticklabels=['Acidic <6', 'Neutral [6,8)', 'Alkaline >=8'],
            ylabel='Mean prediction minus label', title='Bias by true pH group')
axes[0].legend(fontsize=8)
fig.suptitle('Strict grouped OOF: calibration and tail bias measure different properties')
fig.savefig(out / 'calibration.png', dpi=160)
fig.savefig(out / 'calibration.svg')
plt.close(fig)
print(out / 'calibration.png')
