import os
import torch
from torch import nn
from tqdm import tqdm
import time
from torch.utils.data import DataLoader
from .metrics import calculate_metrics

class BaseTrainer:
    def __init__(self, model, train_loader, valid_loader, test_loader,
                 writer, save_dir, patience=3, min_delta=0.001,
                 validate_every=10):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.writer = writer
        self.save_dir = save_dir
        
        # Early stopping parameters
        self.patience = patience
        self.min_delta = min_delta
        self.validate_every = validate_every
        self.best_loss = float('inf')
        self.counter = 0
        self.best_model = None
        self.best_model_trainable_only = False
        self.best_step = 0
        self.start_time = time.time()

    def save_checkpoint(self, state, filename):
        torch.save(state, filename)
        print(f"Checkpoint saved: {filename}")

    def load_checkpoint(self, filename):
        if os.path.isfile(filename):
            checkpoint = torch.load(filename, map_location='cpu')
            return checkpoint
        raise FileNotFoundError(f"No checkpoint found at {filename}")

    def _capture_model_state(self):
        if hasattr(self.model, 'trainable_state_dict'):
            return self.model.trainable_state_dict(), True

        state = {
            name: tensor.detach().to(device='cpu').clone()
            for name, tensor in self.model.state_dict().items()
        }
        return state, False

    def restore_model_state(self, state, trainable_only=False):
        if trainable_only:
            self.model.load_trainable_state_dict(state)
        else:
            self.model.load_state_dict(state)

    def load_model_checkpoint(self, filename):
        checkpoint = self.load_checkpoint(filename)
        if 'model_state_dict' in checkpoint:
            state = checkpoint['model_state_dict']
            trainable_only = checkpoint.get('trainable_only', False)
        else:
            state = checkpoint
            trainable_only = False
        self.restore_model_state(state, trainable_only=trainable_only)
        return checkpoint

    def restore_best_model(self):
        if self.best_model is None:
            raise RuntimeError('No best model has been recorded.')
        self.restore_model_state(
            self.best_model,
            trainable_only=self.best_model_trainable_only,
        )

    def early_stopping(self, valid_loss, step):
        if valid_loss < self.best_loss - self.min_delta:
            self.best_loss = valid_loss
            self.counter = 0
            self.best_model, self.best_model_trainable_only = self._capture_model_state()
            self.best_step = step
            self.save_checkpoint({
                'step': step,
                'model_state_dict': self.best_model,
                'trainable_only': self.best_model_trainable_only,
                'loss': valid_loss,
            }, os.path.join(self.save_dir, f'best_model.pth'))
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping triggered. Best model was at step {self.best_step}")
                return True
        return False

    def train_epoch(self):
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError
