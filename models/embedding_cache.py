import hashlib
import os
from collections import OrderedDict

import torch


class EmbeddingCache:
    def __init__(self, cache_dir, max_memory_items=256):
        self.cache_dir = cache_dir
        self.max_memory_items = max(0, max_memory_items)
        self.memory = OrderedDict()

        if self.enabled:
            os.makedirs(self.cache_dir, exist_ok=True)

    @property
    def enabled(self):
        return bool(self.cache_dir)

    @staticmethod
    def make_key(model_name, sequence):
        value = f"{model_name}\0{sequence}".encode("ascii")
        return hashlib.sha256(value).hexdigest()

    def _path(self, key):
        return os.path.join(self.cache_dir, f"{key}.pt")

    def _remember(self, key, tensor):
        if self.max_memory_items == 0:
            return

        self.memory[key] = tensor
        self.memory.move_to_end(key)
        while len(self.memory) > self.max_memory_items:
            self.memory.popitem(last=False)

    def get(self, key):
        if key in self.memory:
            tensor = self.memory.pop(key)
            self.memory[key] = tensor
            return tensor

        if not self.enabled:
            return None

        path = self._path(key)
        if not os.path.exists(path):
            return None

        try:
            tensor = torch.load(path, map_location="cpu")
        except (EOFError, OSError, RuntimeError):
            return None

        if not isinstance(tensor, torch.Tensor) or tensor.dim() != 2:
            return None

        tensor = tensor.detach().to(dtype=torch.float32, device="cpu")
        self._remember(key, tensor)
        return tensor

    def put(self, key, tensor):
        tensor = tensor.detach().to(dtype=torch.float32, device="cpu").contiguous()
        self._remember(key, tensor)

        if not self.enabled:
            return tensor

        path = self._path(key)
        temp_path = f"{path}.{os.getpid()}.tmp"
        torch.save(tensor, temp_path)
        os.replace(temp_path, path)
        return tensor
