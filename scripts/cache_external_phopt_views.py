"""Matched frozen representations for PHOPT and screened external enzymes."""
import argparse
import csv
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from phgeofuse.cache import sha256_file, atomic_json
from phgeofuse.delta_ref.data import freeze_json, atomic_npz
from phgeofuse.robust_fusion import pool_features
from models.embedding_cache import EmbeddingCache
from localph.features import ion_context, projection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    audit = json.loads((args.external.parent / "audit.json").read_text())
    if not audit["ready_for_training"]:
        raise ValueError("external homology audit incomplete")
    with args.external.open() as f:
        ext = [{"key": "external::" + r["id"], "sequence": r["sequence"]} for r in csv.DictReader(f)]
    with (ROOT / "artifacts/phgeofuse/manifest.csv").open() as f:
        train = [{"key": "train::" + r["protein_id"], "sequence": r["sequence"]}
                 for r in csv.DictReader(f) if r["split"] == "train"]
    origin = ROOT / "experiments/field_comparisons_phopt_20260917/esm1v_full_precision"
    original_complete = json.loads((origin / "complete.json").read_text())
    checkpoint = Path.home() / ".cache/torch/hub/checkpoints/esm1v_t33_650M_UR90S_1.pt"
    provenance = {"external_file_sha256": sha256_file(args.external),
                  "audit_sha256": sha256_file(args.external.parent / "audit.json"),
                  "labels_consumed": False, "PHOPT_splits": ["train"],
                  "esm1v_precision": "float32, TF32 disabled, all residue moments", "esm2_precision": "float16 model, float32 moments",
                  "esm1v_checkpoint_sha256": sha256_file(checkpoint),
                  "esm1v_PHOPT_complete_sha256": sha256_file(origin / "complete.json"),
                  "source_sha256": sha256_file(Path(__file__)),
                  "site_feature_source_sha256": sha256_file(ROOT / "localph/features.py")}
    freeze_json(args.output / "protocol.json", provenance)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    start = time.monotonic()
    means, stds = [], []
    for row in train:
        sequence = row["sequence"].translate(str.maketrans({c: "X" for c in "BJOUZ"}))
        key = EmbeddingCache.make_key("esm1v_t33_650M_UR90S_1", sequence)
        source = origin / "tokens" / (key + ".pt")
        if sha256_file(source) != original_complete["token_hashes"][key]:
            raise ValueError("PHOPT token hash mismatch")
        h = torch.load(source, map_location="cpu", weights_only=True)[:, 1:-1].T
        if len(h) != len(sequence):
            raise ValueError("residue alignment mismatch")
        means.append(h.mean(0).numpy())
        stds.append(h.std(0, unbiased=False).numpy())
    train_e1 = pool_features(means, stds, "mean_std")
    train_keys = np.array([r["key"] for r in train])
    source = ROOT / "experiments/phgeofuse_redesign_20260914/esm2_masked/features.npz"
    with np.load(source, allow_pickle=False) as z:
        mapping = {str(k): i for i, k in enumerate(z["keys"])}
        idx = [mapping[k] for k in train_keys]
        train_e2 = pool_features(z["mean"][idx], z["std"][idx], "mean_std")
    atomic_npz(args.output / "phopt_global.npz", keys=train_keys, features=np.column_stack([train_e1, train_e2]))
    print(json.dumps({"event": "PHOPT_moments_complete", "seconds": time.monotonic() - start}), flush=True)

    import esm
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(checkpoint))
    model = model.cuda().float().eval().requires_grad_(False)
    convert = alphabet.get_batch_converter()
    basis = projection().cuda()
    rows = []
    site_rows = []
    cache = args.output / "sequence_cache"
    cache.mkdir(exist_ok=True)
    for i, row in enumerate(ext):
        target = cache / (row["key"].split("::")[1] + "_esm1v.npz")
        if not target.exists():
            seq = row["sequence"]
            _, _, tokens = convert([(row["key"], seq)])
            with torch.inference_mode():
                h = model(tokens.cuda(), repr_layers=[33], return_contacts=False)["representations"][33][0, 1:len(seq) + 1]
            atomic_npz(target, mean=h.mean(0).cpu().numpy(), std=h.std(0, unbiased=False).cpu().numpy(),
                       site=ion_context(seq, h, basis))
        with np.load(target, allow_pickle=False) as z:
            rows.append(pool_features(z["mean"][None], z["std"][None], "mean_std")[0])
            site_rows.append(z["site"].copy())
        if i % 100 == 0 or i + 1 == len(ext):
            print(json.dumps({"event": "external_esm1v", "done": i + 1, "total": len(ext), "seconds": time.monotonic() - start}), flush=True)
    external_e1 = np.array(rows)
    del model, h
    gc.collect()
    torch.cuda.empty_cache()
    from transformers import AutoTokenizer, AutoModel
    repo = "facebook/esm2_t33_650M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(repo, local_files_only=True)
    model = AutoModel.from_pretrained(repo, local_files_only=True).cuda().half().eval().requires_grad_(False)
    rows = []
    for i, row in enumerate(ext):
        target = cache / (row["key"].split("::")[1] + "_esm2.npz")
        if not target.exists():
            seq = row["sequence"]
            tokens = {k: v.cuda() for k, v in tokenizer(seq, return_tensors="pt", add_special_tokens=True).items()}
            with torch.inference_mode():
                h = model(**tokens).last_hidden_state[0, 1:len(seq) + 1].float()
            atomic_npz(target, mean=h.mean(0).cpu().numpy(), std=h.std(0, unbiased=False).cpu().numpy())
        with np.load(target, allow_pickle=False) as z:
            rows.append(pool_features(z["mean"][None], z["std"][None], "mean_std")[0])
        if i % 100 == 0 or i + 1 == len(ext):
            print(json.dumps({"event": "external_esm2", "done": i + 1, "total": len(ext), "seconds": time.monotonic() - start}), flush=True)
    dest = args.output / "external_features.npz"
    atomic_npz(dest, keys=np.array([r["key"] for r in ext]), features=np.column_stack([external_e1, np.array(rows)]), site=np.array(site_rows))
    atomic_json(args.output / "complete.json", {"count": len(ext), "seconds": time.monotonic() - start,
                "external_features_sha256": sha256_file(dest), "phopt_global_sha256": sha256_file(args.output / "phopt_global.npz"),
                "esm2_revision": getattr(model.config, "_commit_hash", None),
                "sequence_cache_hashes": {p.name: sha256_file(p) for p in cache.glob("*.npz")}})


if __name__ == "__main__":
    main()
