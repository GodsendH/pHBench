import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset_registry import DATASET_CHOICES, normalize_dataset_name, processed_dataset_directory
from models import pHPredictionModel, ProteinpHDataset, SupportDataset
from models.meta_evaluation import evaluation_state
from utils import (
    BaseTrainer,
    DistributedMetaBatchSampler,
    DistributedShardSampler,
    NullSummaryWriter,
    calculate_metrics,
    initialize_distributed,
    make_generator,
    print_metrics,
    seed_everything,
    seed_worker,
)


PROJECT_ROOT = Path(__file__).resolve().parent
LOGGING_INTERVAL = 10


def collate_tasks(batch):
    return batch


class ReptileTrainer(BaseTrainer):
    def __init__(self, model, train_loader, valid_loader, test_loader, meta_lr,
                 inner_lr, num_epochs, writer, save_dir, distributed_context,
                 train_batch_sampler, patience=3, min_delta=0.001,
                 validate_every=10, support_batch_size=1, inner_steps=5,
                 seed=0):
        super().__init__(
            model,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            save_dir,
            patience,
            min_delta,
            validate_every,
        )
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.num_epochs = num_epochs
        self.support_batch_size = support_batch_size
        self.inner_steps = inner_steps
        self.distributed_context = distributed_context
        self.train_batch_sampler = train_batch_sampler
        self.device = distributed_context.device
        self.support_generator = make_generator(seed + distributed_context.rank)
        self.trainable_params = self.model.get_trainable_params()
        self.inner_optimizer = torch.optim.SGD(
            self.trainable_params,
            lr=self.inner_lr,
        )
        self.synchronize_trainable_parameters()

    @property
    def is_main(self):
        return self.distributed_context.is_main

    def _snapshot_trainable_vector(self):
        detached = [param.detach() for param in self.trainable_params]
        return parameters_to_vector(detached).clone()

    def _load_trainable_vector(self, vector):
        pointer = 0
        with torch.no_grad():
            for param in self.trainable_params:
                numel = param.numel()
                values = vector[pointer:pointer + numel].view_as(param)
                param.copy_(values)
                pointer += numel

        if pointer != vector.numel():
            raise ValueError('Trainable parameter vector has an invalid size.')

    def _snapshot_trainable_state(self):
        return {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

    def _load_trainable_state(self, state):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.copy_(state[name])

    def synchronize_trainable_parameters(self):
        vector = self._snapshot_trainable_vector()
        self.distributed_context.broadcast(vector, source=0)
        self._load_trainable_vector(vector)

    def _adapt_current_model(self, task_data, shuffle=True):
        support_dataset = SupportDataset(
            task_data['env_ids'],
            task_data['env_seqs'],
            task_data['env_pHs'],
        )
        support_loader = DataLoader(
            support_dataset,
            batch_size=self.support_batch_size,
            shuffle=shuffle,
            generator=self.support_generator,
        )

        for _ in range(self.inner_steps):
            total_support_loss = None
            for support_batch in support_loader:
                support_accs, support_sequences, support_targets = support_batch
                support_targets = support_targets.to(
                    self.device,
                    non_blocking=True,
                )
                support_predictions = self.model(
                    support_accs,
                    support_sequences,
                )
                support_loss = nn.MSELoss()(
                    support_predictions.reshape(-1),
                    support_targets.reshape(-1),
                )
                if total_support_loss is None:
                    total_support_loss = support_loss
                else:
                    total_support_loss = total_support_loss + support_loss

            if total_support_loss is None:
                raise ValueError('A Reptile task must contain support examples.')

            average_support_loss = total_support_loss / len(support_loader)
            self.inner_optimizer.zero_grad(set_to_none=True)
            average_support_loss.backward()
            self.inner_optimizer.step()

    def _query_task(self, task_data):
        query_targets = task_data['opt_pH'].to(self.device).reshape(-1)
        with torch.no_grad():
            query_predictions = self.model(
                [task_data['opt_id']],
                [task_data['opt_seq']],
            ).reshape(-1)
            query_loss = nn.MSELoss()(query_predictions, query_targets)

        return (
            query_loss.item(),
            query_predictions.item(),
            query_targets.item(),
        )

    def _run_local_meta_batch(self, batch):
        base_vector = self._snapshot_trainable_vector()
        local_delta_sum = torch.zeros_like(base_vector)
        local_query_loss_sum = 0.0
        local_task_count = 0

        for task_data in batch:
            self._load_trainable_vector(base_vector)
            self._adapt_current_model(task_data)
            adapted_vector = self._snapshot_trainable_vector()
            query_loss, _, _ = self._query_task(task_data)

            local_delta_sum.add_(adapted_vector)
            local_delta_sum.sub_(base_vector)
            local_query_loss_sum += query_loss
            local_task_count += 1

        self._load_trainable_vector(base_vector)
        return (
            base_vector,
            local_delta_sum,
            local_query_loss_sum,
            local_task_count,
        )

    def _apply_distributed_meta_update(self, base_vector, local_delta_sum,
                                       local_query_loss_sum, local_task_count):
        self.distributed_context.all_reduce(local_delta_sum)

        statistics = torch.tensor(
            [local_query_loss_sum, float(local_task_count)],
            dtype=torch.float64,
            device=self.device,
        )
        self.distributed_context.all_reduce(statistics)
        global_task_count = statistics[1].item()
        if global_task_count == 0:
            raise RuntimeError('The global Reptile meta-batch is empty.')

        local_delta_sum.div_(global_task_count)
        local_delta_sum.mul_(self.meta_lr)
        local_delta_sum.add_(base_vector)
        self._load_trainable_vector(local_delta_sum)
        return statistics[0].item() / global_task_count

    def _broadcast_early_stopping(self, valid_loss, global_step):
        should_stop = False
        if self.is_main:
            print(f'Step {global_step}, Validation Loss: {valid_loss:.4f}')
            self.writer.add_scalar('Loss/validation', valid_loss, global_step)
            self.writer.flush()
            should_stop = self.early_stopping(valid_loss, global_step)

        stop_tensor = torch.tensor(
            int(should_stop),
            dtype=torch.int32,
            device=self.device,
        )
        self.distributed_context.broadcast(stop_tensor, source=0)
        should_stop = bool(stop_tensor.item())

        if should_stop:
            if self.is_main:
                self.restore_best_model()
            self.synchronize_trainable_parameters()

        return should_stop

    def train(self):
        global_step = 0
        for epoch in range(self.num_epochs):
            self.train_batch_sampler.set_epoch(epoch)
            epoch_loss = 0.0
            num_batches = 0
            progress = tqdm(
                self.train_loader,
                desc=f'Epoch {epoch + 1}/{self.num_epochs}',
                disable=not self.is_main,
            )

            for batch in progress:
                batch_results = self._run_local_meta_batch(batch)
                meta_batch_loss = self._apply_distributed_meta_update(
                    *batch_results,
                )
                epoch_loss += meta_batch_loss
                num_batches += 1

                average_loss = epoch_loss / num_batches
                if self.is_main and global_step % LOGGING_INTERVAL == 0:
                    self.writer.add_scalar(
                        'Loss/train_moving_avg',
                        average_loss,
                        global_step,
                    )
                    print(f'Step {global_step}, Train Loss: {average_loss:.4f}')
                global_step += 1

                if global_step % self.validate_every == 0:
                    valid_loss = self.validate()
                    if self._broadcast_early_stopping(valid_loss, global_step):
                        return

            if self.is_main:
                average_epoch_loss = epoch_loss / num_batches
                print(
                    f'Epoch {epoch + 1}, '
                    f'Average Train Loss: {average_epoch_loss:.4f}'
                )
                self.writer.add_scalar(
                    'Loss/train_epoch',
                    average_epoch_loss,
                    epoch,
                )
                self.writer.flush()

        if self.is_main and self.best_model is not None:
            self.restore_best_model()
        self.synchronize_trainable_parameters()

    def validate(self):
        local_loss_sum = 0.0
        local_task_count = 0
        progress = tqdm(
            self.valid_loader,
            desc='Validation',
            disable=not self.is_main,
        )

        with evaluation_state(
            self.model, self.support_generator, restore_parameters=True,
        ) as restore:
            for batch in progress:
                for task_data in batch:
                    restore()
                    self._adapt_current_model(task_data, shuffle=False)
                    query_loss, _, _ = self._query_task(task_data)
                    local_loss_sum += query_loss
                    local_task_count += 1

        statistics = torch.tensor(
            [local_loss_sum, float(local_task_count)],
            dtype=torch.float64,
            device=self.device,
        )
        self.distributed_context.all_reduce(statistics)
        if statistics[1].item() == 0:
            raise RuntimeError('The validation dataset is empty.')
        return statistics[0].item() / statistics[1].item()

    def test(self):
        if not self.is_main:
            return []

        predictions = []
        true_pHs = []
        predicted_pHs = []

        with evaluation_state(
            self.model, self.support_generator, restore_parameters=True,
        ) as restore:
            for batch in tqdm(self.test_loader, desc='Testing'):
                for task_data in batch:
                    restore()
                    self._adapt_current_model(task_data, shuffle=False)
                    _, predicted_pH, true_pH = self._query_task(task_data)
                    predictions.append({
                        'opt_id': task_data['opt_id'],
                        'true_pH': true_pH,
                        'predicted_pH': predicted_pH,
                    })
                    true_pHs.append(true_pH)
                    predicted_pHs.append(predicted_pH)

        metrics = calculate_metrics(true_pHs, predicted_pHs)
        print_metrics(metrics)
        return predictions


def get_run_name(args, world_size=1):
    pretrained_int = 1 if args.pretrained else 0
    dataset_suffix = '' if args.dataset == 'phopt' else f'_{args.dataset}'
    return (
        f'reptile_{args.retrieval_strategy}_topk{args.topk}'
        f'_mlr{args.meta_lr}_ilr{args.inner_lr}_ep{args.num_epochs}'
        f'_ve{args.validate_every}_pr{pretrained_int}_pt{int(args.patience)}'
        f'_is{args.inner_steps}_sbs{args.support_batch_size}'
        f'_gbs{args.meta_batch_size}_seed{args.seed}_ws{world_size}'
        f'{dataset_suffix}'
    )


def create_data_loader_kwargs(args, device):
    kwargs = {
        'num_workers': args.num_workers,
        'pin_memory': device.type == 'cuda',
        'worker_init_fn': seed_worker,
    }
    if args.num_workers > 0:
        kwargs['persistent_workers'] = True
    return kwargs


def main(args):
    args.dataset = normalize_dataset_name(args.dataset)
    distributed_context = initialize_distributed(args.distributed_backend)
    writer = None

    try:
        seed_everything(args.seed)
        run_name = get_run_name(args, distributed_context.world_size)
        save_dir = os.path.join(args.save_dir, run_name)
        log_dir = os.path.join(args.log_dir, run_name)

        if distributed_context.is_main:
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(args.predictions_dir, exist_ok=True)
        distributed_context.barrier()

        if distributed_context.is_main:
            writer = SummaryWriter(log_dir=log_dir)
            print(
                f'Distributed training: world_size={distributed_context.world_size}, '
                f'global_meta_batch={args.meta_batch_size}'
            )
        else:
            writer = NullSummaryWriter()

        data_dir = processed_dataset_directory(
            PROJECT_ROOT,
            args.dataset,
            args.topk,
            args.retrieval_strategy,
        )
        train_dataset = ProteinpHDataset(
            data_dir / 'retrieval_train.json',
            verbose=distributed_context.is_main,
        )
        valid_dataset = ProteinpHDataset(
            data_dir / 'retrieval_valid.json',
            verbose=distributed_context.is_main,
        )
        if args.random_test:
            test_path = processed_dataset_directory(
                PROJECT_ROOT, args.dataset, 5, 'opt_random'
            ) / 'retrieval_test.json'
        else:
            test_path = data_dir / 'retrieval_test.json'
        test_dataset = ProteinpHDataset(
            test_path,
            verbose=distributed_context.is_main,
        )

        train_batch_sampler = DistributedMetaBatchSampler(
            len(train_dataset),
            global_batch_size=args.meta_batch_size,
            rank=distributed_context.rank,
            world_size=distributed_context.world_size,
            shuffle=True,
            seed=args.seed,
        )
        valid_sampler = DistributedShardSampler(
            len(valid_dataset),
            rank=distributed_context.rank,
            world_size=distributed_context.world_size,
        )
        loader_kwargs = create_data_loader_kwargs(
            args,
            distributed_context.device,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            collate_fn=collate_tasks,
            **loader_kwargs,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=args.eval_batch_size,
            sampler=valid_sampler,
            collate_fn=collate_tasks,
            **loader_kwargs,
        )
        test_loader = None
        if distributed_context.is_main:
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.eval_batch_size,
                shuffle=False,
                collate_fn=collate_tasks,
                **loader_kwargs,
            )

        embedding_cache_dir = (
            None if args.disable_embedding_cache else args.embedding_cache_dir
        )
        model = pHPredictionModel(
            pretrained=args.pretrained,
            embedding_cache_dir=embedding_cache_dir,
            embedding_memory_cache_size=args.embedding_memory_cache_size,
            device=distributed_context.device,
            use_data_parallel=not distributed_context.distributed,
        ).to(distributed_context.device)
        seed_everything(args.seed + distributed_context.rank)

        trainer = ReptileTrainer(
            model,
            train_loader,
            valid_loader,
            test_loader,
            args.meta_lr,
            args.inner_lr,
            args.num_epochs,
            writer,
            save_dir,
            distributed_context,
            train_batch_sampler,
            patience=args.patience,
            min_delta=args.min_delta,
            validate_every=args.validate_every,
            support_batch_size=args.support_batch_size,
            inner_steps=args.inner_steps,
            seed=args.seed,
        )

        if args.mode == 'train':
            trainer.train()
        elif args.mode == 'test':
            model_path = args.checkpoint_path or os.path.join(
                save_dir,
                'best_model.pth',
            )
            if distributed_context.is_main:
                trainer.load_model_checkpoint(model_path)
            trainer.synchronize_trainable_parameters()
        else:
            raise ValueError('Invalid mode. Choose train or test.')

        distributed_context.barrier()
        predictions = trainer.test() if distributed_context.is_main else []
        distributed_context.barrier()

        if distributed_context.is_main:
            predictions_file = os.path.join(
                args.predictions_dir,
                f'predictions_{run_name}.csv',
            )
            pd.DataFrame(predictions).to_csv(predictions_file, index=False)
            print(f'Predictions saved to {predictions_file}')
    finally:
        if writer is not None:
            writer.close()
        distributed_context.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reptile pH Prediction Model')
    parser.add_argument('--dataset', default='phopt', choices=DATASET_CHOICES)
    parser.add_argument('--mode', default='train', choices=['train', 'test'])
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--meta_lr', type=float, default=1)
    parser.add_argument('--inner_lr', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--save_dir', default='./saved_models')
    parser.add_argument('--checkpoint_path')
    parser.add_argument('--predictions_dir', default='./predictions')
    parser.add_argument('--log_dir', default='./logs')
    parser.add_argument('--meta_batch_size', type=int, default=5)
    parser.add_argument('--support_batch_size', type=int, default=1)
    parser.add_argument('--eval_batch_size', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--inner_steps', type=int, default=5)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--min_delta', type=float, default=0.0001)
    parser.add_argument('--validate_every', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--distributed_backend', default='nccl')
    parser.add_argument(
        '--embedding_cache_dir',
        default='./data/features/esm1v_t33_650M_UR90S_1',
    )
    parser.add_argument('--embedding_memory_cache_size', type=int, default=256)
    parser.add_argument('--disable_embedding_cache', action='store_true')
    parser.add_argument(
        '--retrieval_strategy',
        default='opt_retrieval',
        choices=[
            'opt_retrieval',
            'opt_random',
            'opt_fixed_random',
            'opt_retrieval_scaled_0.2',
            'opt_retrieval_scaled_0.6',
        ],
    )
    parser.add_argument('--random_test', action='store_true')
    main(parser.parse_args())
