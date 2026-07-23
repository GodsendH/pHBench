from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phgeofuse.model import PHGeoFuse, compute_loss
from tests.test_phgeofuse import _tiny_batch, _tiny_config
from utils.distributed import initialize_distributed


def main():
    context = initialize_distributed("gloo")
    try:
        torch.manual_seed(7)
        config = _tiny_config()
        config["ablation"] = {"disable_retrieval": True}
        model = DistributedDataParallel(PHGeoFuse(config, context.device).to(context.device))
        batch = _tiny_batch()
        output = model(batch)
        loss, _ = compute_loss(output, batch, config)
        loss.backward()
        parameter = next(item for item in model.parameters() if item.grad is not None)
        checksum = parameter.grad.float().sum().detach()
        gathered = [torch.zeros_like(checksum) for _ in range(context.world_size)]
        torch.distributed.all_gather(gathered, checksum)
        for value in gathered[1:]:
            torch.testing.assert_close(value, gathered[0])
        context.barrier()
        if context.is_main:
            print(f"pH-GeoFuse DDP smoke passed with world_size={context.world_size}")
    finally:
        context.close()


if __name__ == "__main__":
    main()
