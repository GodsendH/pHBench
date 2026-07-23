from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phgeofuse.engine import train_model
from phgeofuse.io import read_manifest, write_manifest
from tests.test_phgeofuse import _tiny_config, _tiny_records
from utils.distributed import initialize_distributed


def main():
    context = initialize_distributed("gloo")
    root_value = [tempfile.mkdtemp(prefix="phgeofuse-ddp-") if context.is_main else None]
    dist.broadcast_object_list(root_value, src=0)
    root = Path(root_value[0]).resolve()
    try:
        manifest = root / "manifest.csv"
        if context.is_main:
            write_manifest(manifest, _tiny_records(root))
        context.barrier()
        records = read_manifest(manifest)
        config = _tiny_config()
        config.update(
            {
                "_root": str(root),
                "paths": {"retrieval": str(root / "retrieval.pt"), "runs": str(root / "runs")},
                "training": {
                    "seed": 11, "run_name": "ddp-smoke", "per_device_batch_size": 1,
                    "global_batch_size": 2, "num_workers": 0, "epochs": 1,
                    "learning_rate": 1e-3, "weight_decay": 0.0, "warmup_fraction": 0.0,
                    "early_stopping_patience": 2, "precision": "fp32",
                },
                "structure": {"foldseek_binary": "missing"},
            }
        )
        config["retrieval"].update(
            {"top_k": 1, "require_foldseek": False, "mmseqs_binary": "missing"}
        )
        checkpoint = train_model(records, config, context)
        context.barrier()
        if not checkpoint.is_file():
            raise AssertionError(f"DDP training did not create {checkpoint}")
        if context.is_main:
            print(f"pH-GeoFuse DDP training smoke passed with world_size={context.world_size}")
        context.barrier()
    finally:
        if context.is_main and root.name.startswith("phgeofuse-ddp-"):
            shutil.rmtree(root)
        context.close()


if __name__ == "__main__":
    main()
