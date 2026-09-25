"""Real train-subset two-epoch resource probe; not a performance experiment."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phgeofuse.delta_ref.data import DevelopmentData
from localph.residue_training import fit
from phgeofuse.cache import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = DevelopmentData.load(ROOT / "configs/delta_ref_phopt.yaml")
    train = data.train
    keys, labels, folds = data.keys[train], data.labels[train], data.folds[train]
    with np.load(args.features, allow_pickle=False) as z:
        if not np.array_equal(z["keys"], keys):
            raise ValueError("cache keys differ")
        packed = {k: z[k].copy() for k in ("tokens", "ionizable", "offsets")}
    fit_idx = np.flatnonzero(~np.isin(folds, [0, 1]))
    query = np.flatnonzero(folds == 1)
    torch.cuda.reset_peak_memory_stats()
    _, result = fit(packed, labels, fit_idx, query, None, "sparse", args.output / "two_epoch", fixed_epochs=2)
    summary = {"scope": "two-epoch resource probe only", "result": {k: result[k] for k in
                 ("seconds", "parameters", "fit_rows", "query_rows")},
               "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
               "estimate_75_fits_10_epochs_hours": result["seconds"] / 2 * 10 * 75 / 3600}
    atomic_json(args.output / "resource_profile.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
