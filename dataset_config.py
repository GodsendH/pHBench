import argparse
import os
from dataclasses import dataclass
from typing import Dict


DATASET_CHOICES = ('phopt', 'dedup')


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    data_root: str
    fasta_files: Dict[str, str]

    @property
    def features_dir(self):
        return os.path.join(self.data_root, 'features')

    @property
    def processed_dir(self):
        return os.path.join(self.data_root, 'processed')

    def retrieval_dir(self, topk, strategy):
        return os.path.join(
            self.processed_dir,
            f'top{topk}',
            f'esm2_{strategy}',
        )


_DATASET_CONFIGS = {
    'phopt': DatasetConfig(
        name='phopt',
        data_root='data',
        fasta_files={
            'train': 'data/phopt_training.fasta',
            'valid': 'data/phopt_validation.fasta',
            'test': 'data/phopt_testing.fasta',
        },
    ),
    'dedup': DatasetConfig(
        name='dedup',
        data_root='dedup_data',
        fasta_files={
            'train': 'dedup_data/dedup/train.fasta',
            'valid': 'dedup_data/dedup/val.fasta',
            'test': 'dedup_data/dedup/test.fasta',
        },
    ),
}


def add_dataset_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        '--dataset',
        choices=DATASET_CHOICES,
        default='phopt',
        help='Dataset to prepare or train on (default: phopt)',
    )


def get_dataset_config(dataset: str):
    return _DATASET_CONFIGS[dataset]
