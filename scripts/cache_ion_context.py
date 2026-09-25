"""Compute training-only ion context sketches from certified ESM1v tokens."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from models.embedding_cache import EmbeddingCache
from localph.features import ion_context, projection, IONIZABLE, PROJECTION_SEED
from phgeofuse.cache import atomic_json, sha256_file
from phgeofuse.delta_ref.data import freeze_json, atomic_npz


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "experiments/field_comparisons_phopt_20260917/esm1v_full_precision")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split", choices=["train", "validation"], default="train")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    complete = json.loads((args.cache / "complete.json").read_text())
    cache_protocol = json.loads((args.cache / "protocol.json").read_text())
    manifest = ROOT / "artifacts/phgeofuse/manifest.csv"
    if cache_protocol["manifest_sha256"] != sha256_file(manifest):
        raise ValueError("token cache manifest differs")
    if complete["protocol_sha256"] != sha256_file(args.cache / "protocol.json"):
        raise ValueError("token cache protocol differs")
    with manifest.open() as stream:
        records = [{"key": r["split"] + "::" + r["protein_id"], "sequence": r["sequence"]}
                   for r in csv.DictReader(stream) if r["split"] == args.split]
    protocol = {"schema": "ionizable_context_contrast_std_window9_rp64_v1", "labels_read": False,
                "split": args.split, "projection_seed": PROJECTION_SEED, "ionizable": IONIZABLE,
                "token_complete_sha256": sha256_file(args.cache / "complete.json"),
                "source_hashes": {str(p.relative_to(ROOT)): sha256_file(p) for p in
                                  [Path(__file__), ROOT / "localph/features.py"]}}
    freeze_json(args.output / "protocol.json", protocol)
    torch.set_num_threads(4)
    basis = projection().to(args.device)
    values = []
    hashes = {}
    start = time.monotonic()
    for i, row in enumerate(records):
        normalized = row["sequence"].translate(str.maketrans({c: "X" for c in "BJOUZ"}))
        key = EmbeddingCache.make_key("esm1v_t33_650M_UR90S_1", normalized)
        source = args.cache / "tokens" / (key + ".pt")
        digest = sha256_file(source)
        if digest != complete["token_hashes"][key]:
            raise ValueError(f"token hash mismatch: {row['key']}")
        tensor = torch.load(source, map_location="cpu", weights_only=True)
        if tensor.shape != (1280, len(normalized) + 2):
            raise ValueError("token shape mismatch")
        values.append(ion_context(normalized, tensor[:, 1:-1].T.to(args.device), basis))
        hashes[key] = digest
        if i % 250 == 0 or i + 1 == len(records):
            print(json.dumps({"encoded": i + 1, "total": len(records), "seconds": time.monotonic() - start}), flush=True)
    dest = args.output / "features.npz"
    atomic_npz(dest, keys=np.array([r["key"] for r in records]), features=np.asarray(values))
    atomic_json(args.output / "complete.json", {"rows": len(values), "width": len(values[0]),
                "seconds": time.monotonic() - start, "features_sha256": sha256_file(dest),
                "source_token_hashes": hashes})


if __name__ == "__main__":
    main()
