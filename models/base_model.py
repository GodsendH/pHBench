import sys
import warnings

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(1, './baseline/EpHod')
from ephod import models, utils as ephod_utils

from .embedding_cache import EmbeddingCache


ESM1V_MODEL_NAME = 'esm1v_t33_650M_UR90S_1'


class pHPredictionModel(nn.Module):
    def __init__(self, pretrained=True, embedding_cache_dir=None,
                 embedding_memory_cache_size=256, device=None,
                 use_data_parallel=True):
        super(pHPredictionModel, self).__init__()
        # EpHod's RLAT constructors reseed PyTorch from their saved configuration.
        # Capture the caller's seed before loading them for a genuinely fresh head.
        self.initialization_seed = None if pretrained else torch.initial_seed()
        self.ephod_model = models.EpHodModel(
            device=device,
            use_data_parallel=use_data_parallel,
        )
        self.embedding_cache = EmbeddingCache(
            embedding_cache_dir,
            max_memory_items=embedding_memory_cache_size,
        )
        
        self._set_parameter_requires_grad()
        self._legacy_task_head_buffers = {
            self._canonical_parameter_name(name): value.detach().cpu().clone()
            for name, value in self._task_head_buffers().items()
        }
        
        if not pretrained:
            self._initialize_trainable_params()
            for module in self.ephod_model.rlat_model.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.reset_running_stats()
        
        self.train()

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, 'ephod_model'):
            self.ephod_model.esm1v_model.eval()
            self._set_batchnorm_to_eval()
        return self

    def _set_parameter_requires_grad(self):
        for name, param in self.ephod_model.named_parameters():
            param.requires_grad = 'rlat_model' in name

    def _initialize_trainable_params(self):
        with torch.random.fork_rng():
            torch.manual_seed(self.initialization_seed)
            for param in self.ephod_model.parameters():
                if param.requires_grad:
                    if param.dim() > 1:
                        nn.init.xavier_uniform_(param)
                    else:
                        nn.init.uniform_(param, -0.1, 0.1)

    def _set_batchnorm_to_eval(self):
        for module in self.ephod_model.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.eval()

    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_state_dict(self):
        """Save the task head, including buffers needed to reproduce inference."""
        state = {
            name: param.detach().to(device='cpu').clone()
            for name, param in self.named_parameters()
            if param.requires_grad
        }
        state.update({
            name: value.detach().cpu().clone()
            for name, value in self._task_head_buffers().items()
        })
        return state

    def _task_head_buffers(self):
        return {
            name: value for name, value in self.named_buffers()
            if name.startswith('ephod_model.rlat_model.')
        }

    def load_trainable_state_dict(self, state):
        canonical_state = {
            self._canonical_parameter_name(name): value
            for name, value in state.items()
        }
        parameters = {
            self._canonical_parameter_name(name): param
            for name, param in self.named_parameters() if param.requires_grad
        }
        buffers = {
            self._canonical_parameter_name(name): value
            for name, value in self._task_head_buffers().items()
        }
        present_buffers = buffers.keys() & canonical_state.keys()
        if present_buffers and present_buffers != buffers.keys():
            raise ValueError('Incomplete task-head buffer state in checkpoint.')
        legacy = bool(buffers) and not present_buffers
        if legacy:
            # Old lightweight checkpoints omitted buffers, even for fresh runs.
            # Their predictions used the original supervised EpHod statistics.
            canonical_state.update(self._legacy_task_head_buffers)
        destinations = {**parameters, **buffers}
        if destinations.keys() != canonical_state.keys():
            raise ValueError('Task-head checkpoint keys do not match the model.')
        for name, value in destinations.items():
            if value.shape != canonical_state[name].shape:
                raise ValueError(f'Task-head checkpoint shape mismatch: {name}')
        if legacy:
            warnings.warn(
                'Loading a legacy parameter-only checkpoint with original EpHod '
                'BatchNorm statistics. This uses supervised task state and is not '
                'PHOPT-only; the original EpHod checkpoint must be unchanged.',
                UserWarning, stacklevel=2,
            )
        with torch.no_grad():
            for name, value in destinations.items():
                value.copy_(canonical_state[name])

    def load_compatible_state_dict(self, state):
        canonical_state = {
            self._canonical_parameter_name(name): value
            for name, value in state.items()
        }
        compatible_state = {
            name: canonical_state[self._canonical_parameter_name(name)]
            for name in self.state_dict()
        }
        self.load_state_dict(compatible_state)

    @staticmethod
    def _canonical_parameter_name(name):
        return name.replace(
            'ephod_model.rlat_model.module.',
            'ephod_model.rlat_model.',
        )
    
    def print_trainable_parameters(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"trainable parameters: {trainable_params}")
        print(f"total parameters: {total_params}")
        print(f"trainable parameters ratio: {trainable_params / total_params:.2%}")

    def _normalize_sequence(self, sequence):
        return ephod_utils.replace_noncanonical(sequence, 'X')

    def _encode_sequence(self, accession, sequence):
        with torch.inference_mode():
            encoded = self.ephod_model.get_ESM1v_embeddings(
                [accession],
                [sequence],
            )[0].detach().to(dtype=torch.float32, device='cpu')
        return encoded.clone()

    def _get_sequence_embedding(self, accession, sequence):
        normalized_sequence = self._normalize_sequence(sequence)
        key = self.embedding_cache.make_key(
            ESM1V_MODEL_NAME,
            normalized_sequence,
        )
        embedding = self.embedding_cache.get(key)
        if embedding is None:
            embedding = self._encode_sequence(accession, normalized_sequence)
            embedding = self.embedding_cache.put(key, embedding)
        return embedding

    def _get_cached_embeddings(self, accs, sequences):
        if not sequences:
            raise ValueError('At least one protein sequence is required.')
        if len(accs) != len(sequences):
            raise ValueError('Accessions and sequences must have the same length.')

        embeddings = [
            self._get_sequence_embedding(accession, sequence)
            for accession, sequence in zip(accs, sequences)
        ]
        max_length = max(embedding.shape[-1] for embedding in embeddings)
        padded = [
            F.pad(embedding, (0, max_length - embedding.shape[-1]))
            for embedding in embeddings
        ]
        device = next(self.ephod_model.rlat_model.parameters()).device
        return torch.stack(padded).to(device=device, non_blocking=True)

    def forward(self, accs, sequences):
        if not self.embedding_cache.enabled:
            ephod_preds, _, _ = self.ephod_model.batch_predict(accs, sequences)
            return ephod_preds

        embeddings = self._get_cached_embeddings(accs, sequences)
        max_length = embeddings.shape[-1]
        masks = [
            [1] * len(sequence) + [0] * (max_length - len(sequence))
            for sequence in sequences
        ]
        masks = torch.tensor(
            masks,
            dtype=torch.int32,
            device=embeddings.device,
        )
        ephod_preds, _, _ = self.ephod_model.rlat_model(embeddings, masks)
        return ephod_preds
