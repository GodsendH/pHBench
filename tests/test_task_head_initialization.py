"""Fresh task heads must not inherit supervised state from EpHod checkpoints."""
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn

from models import base_model
from utils import BaseTrainer


class TaskHeadInitializationTests(unittest.TestCase):
    def make_model(self, *, pretrained, seed=42, mean=4.0, variance=0.2):
        def checkpoint_model(device=None, use_data_parallel=True):
            del device, use_data_parallel
            # The real RLAT constructor resets the global generator to its
            # checkpoint seed before pHPredictionModel initializes a fresh head.
            torch.manual_seed(10)
            model = nn.Module()
            model.esm1v_model = nn.BatchNorm1d(2)
            model.rlat_model = nn.Sequential(
                nn.Linear(2, 2), nn.BatchNorm1d(2), nn.Linear(2, 1),
            )
            with torch.no_grad():
                model.esm1v_model.running_mean.fill_(-7.0)
                model.esm1v_model.running_var.fill_(9.0)
                bn = model.rlat_model[1]
                bn.running_mean.fill_(mean)
                bn.running_var.fill_(variance)
                bn.num_batches_tracked.fill_(535536)
            return model

        torch.manual_seed(seed)
        with mock.patch.object(base_model.models, 'EpHodModel', checkpoint_model):
            return base_model.pHPredictionModel(
                pretrained=pretrained, device='cpu', use_data_parallel=False,
            )

    @staticmethod
    def predictions(model):
        model.eval()
        with torch.no_grad():
            return model.ephod_model.rlat_model(
                torch.tensor([[1.0, 2.0], [-3.0, 0.5], [0.0, -1.0]])
            )

    def test_fresh_head_does_not_depend_on_supervised_statistics(self):
        first = self.make_model(pretrained=False, mean=4.0, variance=0.2)
        second = self.make_model(pretrained=False, mean=-30.0, variance=8.0)
        for name, value in first.ephod_model.rlat_model.state_dict().items():
            torch.testing.assert_close(
                value, second.ephod_model.rlat_model.state_dict()[name],
                rtol=0, atol=0,
            )
        bn = first.ephod_model.rlat_model[1]
        torch.testing.assert_close(bn.running_mean, torch.zeros(2))
        torch.testing.assert_close(bn.running_var, torch.ones(2))
        self.assertEqual(bn.num_batches_tracked.item(), 0)
        torch.testing.assert_close(self.predictions(first), self.predictions(second))
        first.train()
        self.assertFalse(bn.training)
        torch.testing.assert_close(
            first.ephod_model.esm1v_model.running_mean, torch.full((2,), -7.0),
        )

    def test_fresh_initialization_obeys_caller_seed(self):
        first = self.make_model(pretrained=False, seed=0)
        repeat = self.make_model(pretrained=False, seed=0)
        different = self.make_model(pretrained=False, seed=1)
        one = first.ephod_model.rlat_model[0].weight
        torch.testing.assert_close(one, repeat.ephod_model.rlat_model[0].weight)
        self.assertFalse(torch.equal(one, different.ephod_model.rlat_model[0].weight))

    def test_pretrained_head_keeps_checkpoint_statistics(self):
        model = self.make_model(pretrained=True, mean=4.0, variance=0.2)
        bn = model.ephod_model.rlat_model[1]
        torch.testing.assert_close(bn.running_mean, torch.full((2,), 4.0))
        torch.testing.assert_close(bn.running_var, torch.full((2,), 0.2))
        self.assertEqual(bn.num_batches_tracked.item(), 535536)

    def test_trained_fresh_checkpoint_restores_its_own_buffers(self):
        with tempfile.TemporaryDirectory() as directory:
            fresh = self.make_model(pretrained=False, seed=0)
            optimizer = torch.optim.AdamW(fresh.get_trainable_params(), lr=0.01)
            loss = fresh.ephod_model.rlat_model(torch.ones(3, 2)).sub(7).square().mean()
            loss.backward()
            optimizer.step()
            expected = self.predictions(fresh)
            trainer = BaseTrainer(fresh, None, None, None, None, directory)
            trainer.early_stopping(1.0, 1)
            restored = self.make_model(pretrained=True, mean=90.0, variance=0.001)
            BaseTrainer(restored, None, None, None, None, directory).load_model_checkpoint(
                directory + '/best_model.pth'
            )
            torch.testing.assert_close(self.predictions(restored), expected, rtol=0, atol=0)
            self.assertFalse(any(
                'esm1v_model' in name for name in fresh.trainable_state_dict()
            ))

    def test_legacy_parameter_only_load_restores_legacy_statistics(self):
        legacy = self.make_model(pretrained=True)
        # This is the old on-disk contract, which omitted all task-head buffers.
        state = {
            name: value.detach().clone()
            for name, value in legacy.named_parameters() if value.requires_grad
        }
        target = self.make_model(pretrained=False)
        with self.assertWarnsRegex(UserWarning, 'legacy.*BatchNorm'):
            target.load_trainable_state_dict(state)
        torch.testing.assert_close(
            self.predictions(target), self.predictions(legacy), rtol=0, atol=0,
        )

    def test_partial_buffer_checkpoint_is_rejected_before_parameters_change(self):
        source = self.make_model(pretrained=False, seed=1)
        state = source.trainable_state_dict()
        state.pop('ephod_model.rlat_model.1.running_var')
        target = self.make_model(pretrained=False, seed=0)
        before = target.ephod_model.rlat_model[0].weight.detach().clone()
        with self.assertRaisesRegex(ValueError, 'Incomplete.*buffer'):
            target.load_trainable_state_dict(state)
        torch.testing.assert_close(target.ephod_model.rlat_model[0].weight, before)


if __name__ == '__main__':
    unittest.main()
