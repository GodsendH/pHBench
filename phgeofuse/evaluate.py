from __future__ import annotations

import argparse

from .config import load_config, path
from .engine import evaluate_checkpoint
from .io import read_manifest


def main():
    parser = argparse.ArgumentParser(description="Evaluate a pH-GeoFuse checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--output")
    parser.add_argument("--distributed-backend", default="nccl")
    args = parser.parse_args()
    config = load_config(args.config)
    records = read_manifest(args.manifest or path(config, "paths.manifest"))
    from utils.distributed import initialize_distributed
    context = initialize_distributed(args.distributed_backend)
    try:
        evaluate_checkpoint(records, config, args.checkpoint, context, args.split, args.output)
    finally:
        context.close()


if __name__ == "__main__":
    main()
