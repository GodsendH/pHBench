"""Project all certified training residue tokens; retain every residue."""
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
from localph.features import projection
from phgeofuse.cache import atomic_json, sha256_file
from phgeofuse.delta_ref.data import freeze_json, atomic_npz


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = ROOT / "experiments/field_comparisons_phopt_20260917/esm1v_full_precision"
    manifest = ROOT / "artifacts/phgeofuse/manifest.csv"
    certification = json.loads((source / "complete.json").read_text())
    source_protocol = json.loads((source / "protocol.json").read_text())
    if sha256_file(manifest) != source_protocol["manifest_sha256"]:
        raise ValueError("manifest differs from certified token cache")
    protocol = {"schema": "all_ESM1v_residue_fixed_orthogonal128_fp16_v1", "labels_read": False,
                "splits": ["train"], "max_length": 1022,
                "source_complete_sha256": sha256_file(source / "complete.json"),
                "source_hashes": {str(p.relative_to(ROOT)): sha256_file(p) for p in
                                  [Path(__file__), ROOT / "localph/features.py"]}}
    freeze_json(args.output / "protocol.json", protocol)
    with manifest.open() as f:
        records = [{"key": "train::" + r["protein_id"], "sequence": r["sequence"]}
                   for r in csv.DictReader(f) if r["split"] == "train"]
    torch.set_num_threads(4)
    basis = projection(width=128).cuda()
    start = time.monotonic()
    values, ionizable, offsets = [], [], [0]
    for i, r in enumerate(records):
        seq = r["sequence"].translate(str.maketrans({c: "X" for c in "BJOUZ"}))
        key = EmbeddingCache.make_key("esm1v_t33_650M_UR90S_1", seq)
        path = source / "tokens" / (key + ".pt")
        if sha256_file(path) != certification["token_hashes"][key]:
            raise ValueError("token checksum differs")
        h = torch.load(path, map_location="cpu", weights_only=True)[:, 1:-1].T
        if len(h) != len(seq):
            raise ValueError("residue coverage differs")
        with torch.inference_mode():
            values.append((h.cuda() @ basis).half().cpu().numpy())
        ionizable.append(np.array([c in "DEHCKRY" for c in seq], dtype=bool))
        offsets.append(offsets[-1] + len(seq))
        if i % 500 == 0 or i + 1 == len(records):
            print(json.dumps({"done": i + 1, "total": len(records), "seconds": time.monotonic() - start}), flush=True)
    dest = args.output / "tokens.npz"
    atomic_npz(dest, keys=np.array([r["key"] for r in records]), tokens=np.concatenate(values),
               ionizable=np.concatenate(ionizable), offsets=np.array(offsets), basis=basis.cpu().numpy())
    atomic_json(args.output / "complete.json", {"keys": len(records), "residues": offsets[-1],
                "sha256": sha256_file(dest), "seconds": time.monotonic() - start,
                "protocol_sha256": sha256_file(args.output / "protocol.json")})


if __name__ == "__main__":
    main()
