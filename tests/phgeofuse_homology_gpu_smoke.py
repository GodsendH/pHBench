from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phgeofuse.cache import atomic_torch_save
from phgeofuse.engine import train_model
from phgeofuse.model import PHGeoFuse
from tests.test_phgeofuse import _tiny_config, _tiny_records
from utils.distributed import initialize_distributed


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("homology GPU smoke requires CUDA")
    context = initialize_distributed("nccl")
    root = Path(tempfile.mkdtemp(prefix="phgeofuse-homology-gpu-")).resolve()
    try:
        source_model = PHGeoFuse(_tiny_config(), torch.device("cpu"))
        initial = root / "initial.pt"
        atomic_torch_save(
            initial,
            {
                "model_state_dict": source_model.state_dict(),
                "saprot_adapter_only": False,
            },
        )
        config = _tiny_config()
        config.update(
            {
                "_root": str(root),
                "paths": {
                    "retrieval": str(root / "retrieval.pt"),
                    "runs": str(root / "runs"),
                },
                "structure": {"foldseek_binary": "missing"},
                "fusion": {
                    "mode": "homology_reliability",
                    "gate_hidden_dim": 8,
                    "gate_dropout": 0.0,
                    "gate_temperature": 1.0,
                    "gate_prior": [0.25, 0.25, 0.5],
                },
                "homology_training": {
                    "enabled": True,
                    "normal_loss_weight": 1.0,
                    "low_homology_loss_weight": 0.5,
                    "consistency_weight": 0.1,
                },
                "training": {
                    "seed": 17,
                    "run_name": "homology-gpu-smoke",
                    "trainable_scope": "homology_gate",
                    "per_device_batch_size": 2,
                    "global_batch_size": 2,
                    "num_workers": 0,
                    "epochs": 1,
                    "learning_rate": 1e-3,
                    "weight_decay": 0.0,
                    "warmup_fraction": 0.0,
                    "early_stopping_patience": 2,
                    "precision": "bf16",
                },
            }
        )
        config["retrieval"].update(
            {
                "top_k": 1,
                "candidate_k": 2,
                "require_foldseek": False,
                "mmseqs_binary": "missing",
            }
        )
        config["loss"].update(
            {"gate_supervision_weight": 0.1, "gate_prior_weight": 0.01}
        )
        checkpoint = train_model(
            _tiny_records(root),
            config,
            context,
            init_checkpoint=initial,
        )
        if not checkpoint.is_file():
            raise AssertionError(f"GPU smoke did not create {checkpoint}")
        print(f"pH-GeoFuse homology GPU smoke passed on {context.device}")
    finally:
        shutil.rmtree(root)
        context.close()


if __name__ == "__main__":
    main()
