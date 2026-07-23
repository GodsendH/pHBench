import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.distributed import initialize_distributed


def main():
    context = initialize_distributed('gloo')
    try:
        value = torch.tensor(float(context.rank + 1), device=context.device)
        context.all_reduce(value)
        expected = context.world_size * (context.world_size + 1) / 2
        if value.item() != expected:
            raise AssertionError(
                f'Expected all-reduce result {expected}, got {value.item()}.'
            )
        context.barrier()
        if context.is_main:
            print(
                f'Distributed smoke test passed with '
                f'world_size={context.world_size}.'
            )
    finally:
        context.close()


if __name__ == '__main__':
    main()
