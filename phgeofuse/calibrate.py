from __future__ import annotations

import argparse

from dataset_registry import DATASET_CHOICES

from .config import load_config, path
from .datasets import apply_dataset
from .engine import calibrate_checkpoint
from .io import read_manifest


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate pH-GeoFuse residual shrinkage on validation data"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", default="phopt", choices=DATASET_CHOICES)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--output")
    parser.add_argument("--distributed-backend", default="nccl")
    args = parser.parse_args()
    config = apply_dataset(load_config(args.config), args.dataset)
    records = read_manifest(args.manifest or path(config, "paths.manifest"))
    from utils.distributed import initialize_distributed

    context = initialize_distributed(args.distributed_backend)
    try:
        calibrate_checkpoint(
            records, config, args.checkpoint, context, output=args.output
        )
    finally:
        context.close()


if __name__ == "__main__":
    main()
