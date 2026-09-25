"""Repair PHOPT structure coverage and verify full training artifacts."""
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phgeofuse.cache import atomic_json, sha256_file
from phgeofuse.config import load_config, path
from phgeofuse.io import records_from_config, write_manifest
from phgeofuse.prepare import prepare_records
from phgeofuse.retrieval import ensure_retrieval_store, record_key
from phgeofuse.saprot import encode_records


def main():
    config = load_config(ROOT / "configs/phgeofuse_phopt_homology_gate_v3.yaml")
    audit = ROOT / "artifacts/phgeofuse/completion_20260912"
    audit.mkdir(exist_ok=True)
    for name in ("manifest.csv", "manifest.failures.json", "retrieval.pt"):
        source = audit.parent / name
        if source.exists() and not (audit / (name + ".before")).exists():
            shutil.copy2(source, audit / (name + ".before"))
    records = records_from_config(config)
    original = [(r.split, r.protein_id, r.sequence, r.ph_opt, r.sample_weight) for r in records]
    failures = []
    for index, record in enumerate(records, 1):
        _, errors = prepare_records([record], config, offline=False)
        failures.extend(errors)
        if errors or "X" in record.sequence or index % 500 == 0:
            print(f"PREPARE {index}/{len(records)} {record.protein_id} {record.status} {record.error}", flush=True)
        if index % 100 == 0:
            write_manifest(audit / "manifest.inprogress.csv", records)
    write_manifest(audit / "manifest.inprogress.csv", records)
    atomic_json(audit / "prepare.json", {"total": len(records), "failures": failures})
    if failures:
        raise RuntimeError(f"{len(failures)} structure preparation failures")
    print("ENCODING", flush=True)
    written = encode_records(records, config, torch.device("cuda"))
    print(f"ENCODED {written}", flush=True)
    for r in records:
        graph = torch.load(r.graph_path, map_location="cpu")
        embedding = torch.load(r.embedding_path, map_location="cpu")["embedding"]
        assert graph["coords"].shape[0] == embedding.shape[0] == len(r.sequence) == len(r.three_di)
        assert torch.isfinite(embedding).all() and torch.isfinite(graph["coords"]).all()
    assert original == [(r.split, r.protein_id, r.sequence, r.ph_opt, r.sample_weight) for r in records]
    print("REBUILDING RETRIEVAL", flush=True)
    config["paths"]["retrieval"] = str(audit / "retrieval.complete.pt")
    store = ensure_retrieval_store(records, config, force=True)
    assert set(store.rows) == {record_key(r) for r in records}
    assert set(store.payload["training_keys"]) == {record_key(r) for r in records if r.split == "train"}
    shutil.copy2(path(config, "paths.retrieval"), audit.parent / "retrieval.pt")
    write_manifest(audit.parent / "manifest.csv", records)
    atomic_json(audit.parent / "manifest.failures.json", {"total": len(records), "ready": len(records), "failures": [], "offline": False})
    report = {"total": len(records), "ready": len(records), "splits": dict(Counter(r.split for r in records)), "embeddings_written": written, "retrieval_rows": len(store.rows), "training_keys": len(store.payload["training_keys"]), "original_records_unchanged": True, "manifest_sha256": sha256_file(audit.parent / "manifest.csv"), "unknown_residue_policy": "X_to_A_for_folding_only_zero_confidence"}
    atomic_json(audit / "verification.json", report)
    print("DATASET_COMPLETE " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
