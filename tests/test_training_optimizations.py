import tempfile
import unittest
from unittest import mock

import torch
from torch import nn
import learn2learn as l2l

import reptile
from models import base_model
from utils import BaseTrainer
from utils.distributed import (
    DistributedContext,
    DistributedMetaBatchSampler,
    DistributedShardSampler,
    NullSummaryWriter,
)


class FakeESM(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))


class FakeRLAT(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))
        self.batchnorm = nn.BatchNorm1d(1)

    def forward(self, embeddings, masks):
        del masks
        features = embeddings.mean(dim=(1, 2)).reshape(-1, 1)
        features = self.batchnorm(features).reshape(-1)
        predictions = features * self.weight
        return predictions, embeddings, None


class FakeEpHodModel(nn.Module):
    def __init__(self, device=None, use_data_parallel=True):
        super().__init__()
        del device, use_data_parallel
        self.esm1v_model = FakeESM()
        self.rlat_model = FakeRLAT()
        self.encode_calls = 0

    def get_ESM1v_embeddings(self, accs, sequences):
        del accs
        self.encode_calls += len(sequences)
        length = len(sequences[0]) + 2
        values = torch.arange(length, dtype=torch.float32)
        return values.reshape(1, 1, length)

    def batch_predict(self, accs, sequences):
        embeddings = self.get_ESM1v_embeddings(accs, sequences)
        masks = torch.ones(embeddings.shape[0], embeddings.shape[-1])
        return self.rlat_model(embeddings, masks)


class TinyTrainableModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.trainable = nn.Parameter(torch.tensor(1.0))
        self.frozen = nn.Parameter(torch.tensor(2.0), requires_grad=False)


class TrainingOptimizationTests(unittest.TestCase):
    def test_distributed_meta_batches_cover_each_item_once(self):
        rank_zero = DistributedMetaBatchSampler(
            dataset_size=12,
            global_batch_size=5,
            rank=0,
            world_size=2,
            shuffle=False,
        )
        rank_one = DistributedMetaBatchSampler(
            dataset_size=12,
            global_batch_size=5,
            rank=1,
            world_size=2,
            shuffle=False,
        )

        rank_zero_batches = list(rank_zero)
        rank_one_batches = list(rank_one)
        self.assertEqual(len(rank_zero_batches), len(rank_one_batches))
        combined = []
        for left, right in zip(rank_zero_batches, rank_one_batches):
            combined.extend(left)
            combined.extend(right)
        self.assertEqual(sorted(combined), list(range(12)))
        self.assertEqual(len(combined), len(set(combined)))

    def test_distributed_validation_shards_do_not_overlap(self):
        shards = [
            set(DistributedShardSampler(11, rank=rank, world_size=3))
            for rank in range(3)
        ]
        self.assertFalse(shards[0] & shards[1])
        self.assertFalse(shards[0] & shards[2])
        self.assertFalse(shards[1] & shards[2])
        self.assertEqual(set.union(*shards), set(range(11)))

    def test_esm_stays_in_eval_and_cache_is_reused(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            with mock.patch.object(base_model.models, 'EpHodModel', FakeEpHodModel):
                model = base_model.pHPredictionModel(
                    pretrained=True,
                    embedding_cache_dir=cache_dir,
                    embedding_memory_cache_size=2,
                )

                self.assertTrue(model.training)
                self.assertTrue(model.ephod_model.rlat_model.training)
                self.assertFalse(model.ephod_model.esm1v_model.training)
                self.assertFalse(model.ephod_model.rlat_model.batchnorm.training)

                first = model(['id'], ['ACD'])
                second = model(['id'], ['ACD'])
                self.assertEqual(model.ephod_model.encode_calls, 1)
                torch.testing.assert_close(first, second)
                first.sum().backward()
                self.assertIsNotNone(model.ephod_model.rlat_model.weight.grad)

                model.train()
                self.assertFalse(model.ephod_model.esm1v_model.training)

                maml = l2l.algorithms.MAML(
                    model,
                    lr=0.01,
                    first_order=False,
                )
                learner = maml.clone()
                support_loss = learner(['id'], ['ACD']).pow(2).mean()
                learner.adapt(support_loss, allow_nograd=True)
                query_loss = learner(['id'], ['ACD']).pow(2).mean()
                query_loss.backward()

            with mock.patch.object(base_model.models, 'EpHodModel', FakeEpHodModel):
                reloaded = base_model.pHPredictionModel(
                    pretrained=True,
                    embedding_cache_dir=cache_dir,
                    embedding_memory_cache_size=0,
                )
                reloaded(['other-id'], ['ACD'])
                self.assertEqual(reloaded.ephod_model.encode_calls, 0)

    def test_checkpoint_contains_only_trainable_parameters(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            with mock.patch.object(base_model.models, 'EpHodModel', FakeEpHodModel):
                model = base_model.pHPredictionModel(
                    pretrained=True,
                    embedding_cache_dir=cache_dir,
                )
                trainer = BaseTrainer(
                    model,
                    train_loader=None,
                    valid_loader=None,
                    test_loader=None,
                    writer=None,
                    save_dir=cache_dir,
                )
                trainer.early_stopping(valid_loss=1.0, step=1)

                checkpoint = torch.load(
                    f'{cache_dir}/best_model.pth',
                    map_location='cpu',
                )
                self.assertTrue(checkpoint['trainable_only'])
                self.assertTrue(checkpoint['model_state_dict'])
                self.assertTrue(
                    all('rlat_model' in name for name in checkpoint['model_state_dict'])
                )

                trainable_state = model.trainable_state_dict()
                data_parallel_state = {
                    name.replace(
                        'ephod_model.rlat_model.',
                        'ephod_model.rlat_model.module.',
                    ): value
                    for name, value in trainable_state.items()
                }
                with torch.no_grad():
                    for parameter in model.get_trainable_params():
                        parameter.zero_()
                model.load_trainable_state_dict(data_parallel_state)
                restored_state = model.trainable_state_dict()
                for name, value in trainable_state.items():
                    torch.testing.assert_close(restored_state[name], value)

    def test_reptile_snapshot_only_restores_trainable_parameters(self):
        trainer = object.__new__(reptile.ReptileTrainer)
        trainer.model = TinyTrainableModel()

        snapshot = trainer._snapshot_trainable_state()
        self.assertEqual(set(snapshot), {'trainable'})

        with torch.no_grad():
            trainer.model.trainable.fill_(10.0)
            trainer.model.frozen.fill_(20.0)

        trainer._load_trainable_state(snapshot)
        self.assertEqual(trainer.model.trainable.item(), 1.0)
        self.assertEqual(trainer.model.frozen.item(), 20.0)

    def test_reptile_meta_batch_uses_mean_parameter_delta(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            with mock.patch.object(base_model.models, 'EpHodModel', FakeEpHodModel):
                model = base_model.pHPredictionModel(
                    pretrained=True,
                    embedding_cache_dir=cache_dir,
                    device='cpu',
                    use_data_parallel=False,
                )
                context = DistributedContext(
                    distributed=False,
                    rank=0,
                    local_rank=0,
                    world_size=1,
                    device=torch.device('cpu'),
                )
                trainer = reptile.ReptileTrainer(
                    model=model,
                    train_loader=None,
                    valid_loader=None,
                    test_loader=None,
                    meta_lr=1.0,
                    inner_lr=0.01,
                    num_epochs=1,
                    writer=NullSummaryWriter(),
                    save_dir=cache_dir,
                    distributed_context=context,
                    train_batch_sampler=None,
                    support_batch_size=1,
                    inner_steps=1,
                    seed=0,
                )
                tasks = [
                    {
                        'env_ids': ['support-1'],
                        'env_seqs': ['ACD'],
                        'env_pHs': torch.tensor([1.0]),
                        'opt_id': 'query-1',
                        'opt_seq': 'ACD',
                        'opt_pH': torch.tensor(1.0),
                    },
                    {
                        'env_ids': ['support-2'],
                        'env_seqs': ['ACDE'],
                        'env_pHs': torch.tensor([2.0]),
                        'opt_id': 'query-2',
                        'opt_seq': 'ACDE',
                        'opt_pH': torch.tensor(2.0),
                    },
                ]

                results = trainer._run_local_meta_batch(tasks)
                base_vector, delta_sum, _, task_count = results
                expected = base_vector + delta_sum / task_count
                trainer._apply_distributed_meta_update(*results)
                torch.testing.assert_close(
                    trainer._snapshot_trainable_vector(),
                    expected,
                )


if __name__ == '__main__':
    unittest.main()
