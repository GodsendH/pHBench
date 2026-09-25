"""Pack certified full-width ESM1v training tokens without label access.

The mmap cache retains all 1280 dimensions and every real residue. Source
float32 is cast to float16, matching the projected cache's storage precision.
This performs no PLM inference and runs entirely on CPU.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from models.embedding_cache import EmbeddingCache
from phgeofuse.cache import atomic_json, sha256_file
from phgeofuse.delta_ref.data import freeze_json, atomic_npz


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    source = ROOT / "experiments/field_comparisons_phopt_20260917/esm1v_full_precision"
    cert = json.loads((source / "complete.json").read_text())
    manifest = ROOT / "artifacts/phgeofuse/manifest.csv"
    if sha256_file(manifest) != json.loads((source / "protocol.json").read_text())["manifest_sha256"]:
        raise ValueError("manifest differs from certified token source")
    with manifest.open() as f:
        rows = [{"key": "train::" + r["protein_id"], "sequence": r["sequence"].translate(str.maketrans({c: "X" for c in "BJOUZ"}))}
                for r in csv.DictReader(f) if r["split"] == "train"]
    if len(rows) != 7124:
        raise ValueError("PHOPT training count differs")
    offsets = np.concatenate(([0], np.cumsum([len(r["sequence"]) for r in rows])))
    protocol = {"scope": "PHOPT training sequences only; no labels accessed", "input_width": 1280,
                "source_precision": "certified float32 ESM1v", "cache_precision": "float16",
                "projection": None, "truncation": None,
                "source_complete_sha256": sha256_file(source / "complete.json"),
                "manifest_sha256": sha256_file(manifest), "source_sha256": sha256_file(Path(__file__)),
                "keys": len(rows), "residues": int(offsets[-1]), "planned_bytes": int(offsets[-1]) * 1280 * 2}
    freeze_json(out / "protocol.json", protocol)
    torch.set_num_threads(2)
    start = time.monotonic()
    temporary = out / "tokens.tmp.npy"
    packed = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16, shape=(int(offsets[-1]), 1280))
    ion = np.zeros(int(offsets[-1]), dtype=bool)
    for i, row in enumerate(rows):
        seq = row["sequence"]
        key = EmbeddingCache.make_key("esm1v_t33_650M_UR90S_1", seq)
        filename = source / "tokens" / (key + ".pt")
        if sha256_file(filename) != cert["token_hashes"][key]:
            raise ValueError("source token hash differs")
        h = torch.load(filename, map_location="cpu", weights_only=True)[:, 1:-1].T
        if h.shape != (len(seq), 1280) or not torch.isfinite(h).all():
            raise ValueError("invalid residue token coverage")
        a, b = offsets[i:i + 2]
        packed[a:b] = h.half().numpy()
        ion[a:b] = [c in "DEHCKRY" for c in seq]
        if i % 500 == 0 or i + 1 == len(rows):
            row_status = {"state": "packing", "done": i + 1, "total": len(rows), "seconds": time.monotonic() - start}
            atomic_json(out / "status.json", row_status)
            print(json.dumps(row_status), flush=True)
    packed.flush()
    del packed
    temporary.replace(out / "tokens.npy")
    atomic_npz(out / "index.npz", keys=np.array([r["key"] for r in rows]), offsets=offsets, ionizable=ion)
    complete = {"state": "complete", "seconds": time.monotonic() - start,
                "hashes": {name: sha256_file(out / name) for name in ["tokens.npy", "index.npz", "protocol.json"]}}
    atomic_json(out / "complete.json", complete)
    atomic_json(out / "status.json", complete)
    print(json.dumps(complete), flush=True)


if __name__ == "__main__":
    main()
