import math
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


@dataclass(frozen=True)
class DistributedContext:
    distributed: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self):
        return self.rank == 0

    def barrier(self):
        if self.distributed:
            dist.barrier()

    def all_reduce(self, tensor):
        if self.distributed:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    def broadcast(self, tensor, source=0):
        if self.distributed:
            dist.broadcast(tensor, src=source)
        return tensor

    def close(self):
        if self.distributed and dist.is_initialized():
            dist.destroy_process_group()


def initialize_distributed(backend='nccl'):
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    distributed = world_size > 1

    if distributed:
        local_rank = int(os.environ['LOCAL_RANK'])
        if backend == 'nccl':
            if not torch.cuda.is_available():
                raise RuntimeError('The NCCL backend requires CUDA GPUs.')
            torch.cuda.set_device(local_rank)
            device = torch.device('cuda', local_rank)
        else:
            device = torch.device('cpu')

        dist.init_process_group(backend=backend, init_method='env://')
        return DistributedContext(
            distributed=True,
            rank=dist.get_rank(),
            local_rank=local_rank,
            world_size=dist.get_world_size(),
            device=device,
        )

    device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')
    return DistributedContext(
        distributed=False,
        rank=0,
        local_rank=0,
        world_size=1,
        device=device,
    )


class DistributedMetaBatchSampler(Sampler):
    def __init__(self, dataset_size, global_batch_size, rank=0, world_size=1,
                 shuffle=True, seed=0):
        if global_batch_size < 1:
            raise ValueError('global_batch_size must be at least 1.')
        if world_size < 1:
            raise ValueError('world_size must be at least 1.')
        if world_size > global_batch_size:
            raise ValueError(
                'world_size cannot exceed global_batch_size because some GPUs '
                'would be idle for every meta-batch.'
            )

        self.dataset_size = dataset_size
        self.global_batch_size = global_batch_size
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(
                self.dataset_size,
                generator=generator,
            ).tolist()
        else:
            indices = list(range(self.dataset_size))

        for start in range(0, self.dataset_size, self.global_batch_size):
            global_batch = indices[start:start + self.global_batch_size]
            yield global_batch[self.rank::self.world_size]

    def __len__(self):
        return math.ceil(self.dataset_size / self.global_batch_size)


class DistributedShardSampler(Sampler):
    def __init__(self, dataset_size, rank=0, world_size=1):
        self.dataset_size = dataset_size
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        return iter(range(self.rank, self.dataset_size, self.world_size))

    def __len__(self):
        if self.rank >= self.dataset_size:
            return 0
        return math.ceil((self.dataset_size - self.rank) / self.world_size)


class NullSummaryWriter:
    def add_scalar(self, *args, **kwargs):
        del args, kwargs

    def flush(self):
        pass

    def close(self):
        pass
