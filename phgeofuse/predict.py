from __future__ import annotations

import argparse
import csv
import gc
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import load_config, path
from .dataset import ProteinGraphDataset, collate_graphs, move_batch
from .engine import load_checkpoint
from .io import read_fasta, write_manifest
from .model import PHGeoFuse
from .prepare import prepare_records
from .retrieval import RetrievalStore
from .saprot import encode_records


def main():
    parser = argparse.ArgumentParser(description="Predict optimum pH for an input FASTA")
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--online", action="store_true")
    mode.add_argument("--offline", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if args.config:
        config = load_config(args.config)
    else:
        config = checkpoint.get("config")
        if not isinstance(config, dict):
            raise ValueError("checkpoint has no embedded config; pass --config")
        config.setdefault("_root", str(Path(__file__).resolve().parents[1]))
    config.setdefault("runtime", {})["offline"] = args.offline
    records = read_fasta(args.fasta, "predict", require_labels=False)
    prediction_root = path(config, "paths.predictions", "artifacts/phgeofuse/predictions")
    prediction_root.mkdir(parents=True, exist_ok=True)
    manifest = prediction_root / f"{Path(args.fasta).stem}_{int(time.time())}.manifest.csv"
    records, failures = prepare_records(records, config, offline=args.offline, manifest_path=manifest)
    if failures:
        write_manifest(manifest, records)
        raise RuntimeError(f"{len(failures)} prediction inputs could not be prepared; see {manifest}")
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    # Retrieval always uses the stable frozen SaProt index, including when the
    # prediction model itself is running LoRA adapters.
    encode_records(records, config, device)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    retrieval = RetrievalStore.load(path(config, "paths.retrieval"))
    retrieval.add_queries(records, config)
    dataset = ProteinGraphDataset(records, "predict", retrieval, str(config.get("model", {}).get("mode", "frozen")))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_graphs)
    model = PHGeoFuse(config, device).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()
    rows = []
    with torch.inference_mode():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            outputs = model(batch)
            for index, key in enumerate(batch["keys"]):
                rows.append(
                    {
                        "protein_id": key.split("::", 1)[1],
                        "predicted_ph_opt": float(outputs["mean"][index]),
                        "uncertainty": float(outputs["variance"][index].sqrt()),
                        "global_gate": float(outputs["gate_weights"][index, 0]),
                        "saprot_gate": float(outputs["gate_weights"][index, 1]),
                        "foldseek_gate": float(outputs["gate_weights"][index, 2]),
                    }
                )
    output = Path(args.output).expanduser().resolve() if args.output else prediction_root / f"{Path(args.fasta).stem}.predictions.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} predictions to {output}")


if __name__ == "__main__":
    main()
