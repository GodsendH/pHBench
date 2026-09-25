"""Evaluation must be repeatable without changing later meta-training."""
import copy
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn
from torch.utils.data import DataLoader

import maml
import reptile
from utils.distributed import DistributedContext, NullSummaryWriter


class DropoutRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(1, 8)
        self.dropout = nn.Dropout(0.6)
        self.output = nn.Linear(8, 1)

    def forward(self, accs, sequences):
        del accs
        x = torch.tensor([[len(s) / 10] for s in sequences])
        return self.output(self.dropout(self.hidden(x).tanh())).flatten()

    def get_trainable_params(self):
        return list(self.parameters())


def tasks():
    return [
        {'env_ids': ['s1', 's2'], 'env_seqs': ['ACD', 'ACDEFG'],
         'env_pHs': torch.tensor([3.0, 8.0]), 'opt_id': 'q1',
         'opt_seq': 'ACDE', 'opt_pH': torch.tensor(6.0)},
        {'env_ids': ['s3', 's4'], 'env_seqs': ['AC', 'ACDEFGHIK'],
         'env_pHs': torch.tensor([4.0, 9.0]), 'opt_id': 'q2',
         'opt_seq': 'ACDEFGH', 'opt_pH': torch.tensor(7.0)},
    ]


def loader(rows, batch_size=2):
    return DataLoader(rows, batch_size=batch_size, collate_fn=list, shuffle=False)


class MetaEvaluationTests(unittest.TestCase):
    def make_trainer(self, kind, directory, batch_size=2):
        torch.manual_seed(42)
        model = DropoutRegressor()
        kwargs = dict(
            model=model, train_loader=None, valid_loader=loader(tasks(), batch_size),
            test_loader=loader(tasks(), batch_size), meta_lr=0.01, inner_lr=0.01,
            num_epochs=1, writer=NullSummaryWriter(), save_dir=directory,
            support_batch_size=1, inner_steps=2, seed=42,
        )
        if kind == 'maml':
            return maml.MAMLTrainer(**kwargs)
        context = DistributedContext(False, 0, 0, 1, torch.device('cpu'))
        return reptile.ReptileTrainer(
            **kwargs, distributed_context=context, train_batch_sampler=None,
        )

    def assert_state_unchanged(self, trainer, before, modes, rng, support_rng):
        for name, value in trainer.model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        self.assertEqual([m.training for m in trainer.model.modules()], modes)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
        self.assertTrue(torch.equal(trainer.support_generator.get_state(), support_rng))

    def test_evaluation_is_repeatable_and_restores_training_state(self):
        for kind in ['reptile', 'maml']:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                trainer = self.make_trainer(kind, directory)
                before = copy.deepcopy(trainer.model.state_dict())
                modes = [m.training for m in trainer.model.modules()]
                rng = torch.random.get_rng_state()
                support_rng = trainer.support_generator.get_state()
                loss = trainer.validate()
                first = trainer.test()
                self.assert_state_unchanged(trainer, before, modes, rng, support_rng)
                torch.manual_seed(987)
                trainer.support_generator.manual_seed(123)
                self.assertEqual(trainer.validate(), loss)
                self.assertEqual(trainer.test(), first)

    def test_query_labels_do_not_change_predictions_but_support_labels_do(self):
        for kind in ['reptile', 'maml']:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                trainer = self.make_trainer(kind, directory)
                first = [r['predicted_pH'] for r in trainer.test()]
                changed = tasks()
                for row in changed:
                    row['opt_pH'] = torch.tensor(100.0)
                trainer.test_loader = loader(changed)
                self.assertEqual(first, [r['predicted_pH'] for r in trainer.test()])
                for row in changed:
                    row['env_pHs'] += 3
                trainer.test_loader = loader(changed)
                self.assertNotEqual(first, [r['predicted_pH'] for r in trainer.test()])

    def test_evaluation_preserves_query_order_independence_and_module_modes(self):
        for kind in ['reptile', 'maml']:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                trainer = self.make_trainer(kind, directory)
                trainer.model.dropout.eval()
                modes = [m.training for m in trainer.model.modules()]
                first = {r['opt_id']: r['predicted_pH'] for r in trainer.test()}
                trainer.test_loader = loader(list(reversed(tasks())), batch_size=1)
                second = {r['opt_id']: r['predicted_pH'] for r in trainer.test()}
                self.assertEqual(first, second)
                self.assertEqual([m.training for m in trainer.model.modules()], modes)

    def test_failed_evaluation_restores_parameters_modes_and_rng(self):
        for kind in ['reptile', 'maml']:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                trainer = self.make_trainer(kind, directory)
                before = copy.deepcopy(trainer.model.state_dict())
                modes = [m.training for m in trainer.model.modules()]
                rng = torch.random.get_rng_state()
                support_rng = trainer.support_generator.get_state()
                forward = DropoutRegressor.forward

                def fail_query(model, accs, sequences):
                    if accs[0].startswith('q'):
                        raise RuntimeError('intentional query failure')
                    return forward(model, accs, sequences)

                with mock.patch.object(DropoutRegressor, 'forward', new=fail_query):
                    with self.assertRaisesRegex(RuntimeError, 'intentional query failure'):
                        trainer.validate()
                self.assert_state_unchanged(trainer, before, modes, rng, support_rng)

    def test_maml_validation_averages_samples_not_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer('maml', directory)
            with mock.patch.object(trainer, 'process_task', side_effect=[(2.0, 0, 0), (6.0, 0, 0)]):
                self.assertEqual(trainer.validate(), 4.0)
            trainer.valid_loader = loader(tasks(), batch_size=1)
            with mock.patch.object(trainer, 'process_task', side_effect=[(2.0, 0, 0), (6.0, 0, 0)]):
                self.assertEqual(trainer.validate(), 4.0)

    def test_maml_epoch_limit_restores_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer('maml', directory)
            trainer.train_loader = loader(tasks())
            trainer.num_epochs = 2
            trainer.validate_every = 1
            trainer.patience = 5
            with mock.patch.object(trainer, 'validate', side_effect=[1.0, 2.0]):
                trainer.train()
            checkpoint = torch.load(directory + '/best_model.pth', map_location='cpu')
            for name, value in trainer.model.state_dict().items():
                torch.testing.assert_close(value, checkpoint['model_state_dict'][name], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
