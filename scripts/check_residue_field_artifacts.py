"""Verify completed residue fits or final predictions without changing training."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from localph.residue_field import ResidueField
from localph.residue_training import predict
from phgeofuse.cache import atomic_json, sha256_file
from phgeofuse.delta_ref.metrics import metrics, acceptance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--fit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.experiment / "protocol.json").read_text())
    for filename, digest in protocol["source_hashes"].items():
        if sha256_file(ROOT / filename) != digest:
            raise ValueError("training source hash differs")
    if sha256_file(args.features) != protocol["features_sha256"]:
        raise ValueError("feature hash differs")
    result = {"source_hashes_verified": True, "feature_hash_verified": True}
    if args.fit:
        complete = json.loads((args.fit / "verified_complete.json").read_text())
        for name, digest in complete["hashes"].items():
            if sha256_file(args.fit / name) != digest:
                raise ValueError("completed fit hash differs")
        certificate = json.loads((args.fit / "fit.json").read_text())
        with np.load(args.features, allow_pickle=False) as z:
            packed = {k: z[k].copy() for k in ("tokens", "offsets", "ionizable")}
            keys = z["keys"].astype(str)
        mapping = {k: i for i, k in enumerate(keys)}
        indices = np.array([mapping[k] for k in certificate["query_keys"][:32]])
        model_file = torch.load(args.fit / "weights.pt", map_location="cpu", weights_only=True)
        model = ResidueField(kind=model_file["kind"])
        model.load_state_dict(model_file["state_dict"])
        # Dummy labels demonstrate that inference needs no query optimum pH.
        p = predict(model, packed, indices, np.zeros(len(keys)), model_file["prior"], device="cpu")
        with np.load(args.fit / "predictions.npz", allow_pickle=False) as z:
            differences = {k: float(np.max(abs(v - z[k][:len(indices)]))) for k, v in p.items()}
        for k, diff in differences.items():
            # Mode can move one grid point at nearly tied logits across CPU/GPU.
            allowed = .250001 if k.endswith("mode") else 2e-4
            if diff > allowed:
                raise ValueError(f"checkpoint prediction parity failed: {k} {diff}")
        result.update(fit=str(args.fit), checkpoint_reload_verified=True, query_count=len(indices),
                      CPU_GPU_max_differences=differences, query_labels_required=False)
    if (args.experiment / "results.json").exists():
        stored = json.loads((args.experiment / "results.json").read_text())
        with np.load(args.experiment / "predictions.npz", allow_pickle=False) as z:
            recomputed = {k: metrics(z["y"], z[k], z["groups"])
                          for k in ("baseline", "nested_selected", "direct", "global", "sparse")}
        for name, summary in recomputed.items():
            for group in ("all", "core", "acid", "alkaline"):
                for metric in ("rmse", "mae", "bias", "abs_bias"):
                    if abs(summary[group][metric] - stored[name][group][metric]) > 1e-12:
                        raise ValueError("final metric recomputation differs")
        for name in ("nested_selected", "direct", "global", "sparse"):
            if acceptance(recomputed[name], recomputed["baseline"]) != stored["acceptance"][name]:
                raise ValueError("acceptance recomputation differs")
        result.update(final_metrics_recomputed=True, metrics=recomputed, acceptance=stored["acceptance"],
                      predictions_sha256=sha256_file(args.experiment / "predictions.npz"))
    atomic_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "metrics"}, indent=2))


if __name__ == "__main__":
    main()
