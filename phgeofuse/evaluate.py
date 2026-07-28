from __future__ import annotations

import argparse

from dataset_registry import DATASET_CHOICES

from .config import load_config, path
from .datasets import apply_dataset
from .engine import evaluate_checkpoint
from .io import read_manifest


def main():
    parser = argparse.ArgumentParser(description="Evaluate a pH-GeoFuse checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", default="phopt", choices=DATASET_CHOICES)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest")
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "validation", "test", "test_low_identity"],
    )
    parser.add_argument("--output")
    parser.add_argument("--distributed-backend", default="nccl")
    args = parser.parse_args()
    config = apply_dataset(load_config(args.config), args.dataset)
    records = read_manifest(args.manifest or path(config, "paths.manifest"))
    from utils.distributed import initialize_distributed
    context = initialize_distributed(args.distributed_backend)
    try:
        evaluate_checkpoint(records, config, args.checkpoint, context, args.split, args.output)
    finally:
        context.close()


if __name__ == "__main__":
    main()
