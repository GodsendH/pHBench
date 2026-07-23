import sys

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(1, './baseline/EpHod')
from ephod import models, utils as ephod_utils

from .embedding_cache import EmbeddingCache


ESM1V_MODEL_NAME = 'esm1v_t33_650M_UR90S_1'


class pHPredictionModel(nn.Module):
    def __init__(self, pretrained=True, embedding_cache_dir=None,
                 embedding_memory_cache_size=256):
        super(pHPredictionModel, self).__init__()
        self.ephod_model = models.EpHodModel()
        self.embedding_cache = EmbeddingCache(
            embedding_cache_dir,
            max_memory_items=embedding_memory_cache_size,
        )
        
        self._set_parameter_requires_grad()
        
        if not pretrained:
            self._initialize_trainable_params()
        
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
        return {
            name: param.detach().to(device='cpu').clone()
            for name, param in self.named_parameters()
            if param.requires_grad
        }

    def load_trainable_state_dict(self, state):
        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.requires_grad:
                    param.copy_(state[name])
    
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
