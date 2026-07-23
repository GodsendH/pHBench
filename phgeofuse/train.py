from __future__ import annotations

import argparse

from .config import load_config, path
from .engine import apply_ablation, train_model
from .io import read_manifest


def main():
    parser = argparse.ArgumentParser(description="Train pH-GeoFuse with torchrun/DDP")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--resume")
    parser.add_argument("--distributed-backend", default="nccl")
    parser.add_argument("--ablation", default="full", choices=["full", "saprot_only", "geometry", "ph_conditioned", "saprot_retrieval"])
    args = parser.parse_args()
    config = load_config(args.config)
    apply_ablation(config, args.ablation)
    manifest = args.manifest or path(config, "paths.manifest")
    records = read_manifest(manifest)
    from utils.distributed import initialize_distributed
    context = initialize_distributed(args.distributed_backend)
    try:
        checkpoint = train_model(records, config, context, args.resume)
        if context.is_main:
            print(f"Best checkpoint: {checkpoint}")
    finally:
        context.close()


if __name__ == "__main__":
    main()
