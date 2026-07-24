import argparse
import unittest

from dataset_config import add_dataset_argument, get_dataset_config


class DatasetConfigTests(unittest.TestCase):
    def test_phopt_defaults_preserve_existing_paths(self):
        config = get_dataset_config('phopt')

        self.assertEqual(config.fasta_files['train'], 'data/phopt_training.fasta')
        self.assertEqual(config.fasta_files['valid'], 'data/phopt_validation.fasta')
        self.assertEqual(config.fasta_files['test'], 'data/phopt_testing.fasta')
        self.assertEqual(config.features_dir, 'data/features')
        self.assertEqual(
            config.retrieval_dir(5, 'opt_retrieval'),
            'data/processed/top5/esm2_opt_retrieval',
        )

    def test_dedup_defaults_use_dedup_data_tree(self):
        config = get_dataset_config('dedup')

        self.assertEqual(config.fasta_files['train'], 'dedup_data/dedup/train.fasta')
        self.assertEqual(config.fasta_files['valid'], 'dedup_data/dedup/val.fasta')
        self.assertEqual(config.fasta_files['test'], 'dedup_data/dedup/test.fasta')
        self.assertEqual(config.features_dir, 'dedup_data/features')
        self.assertEqual(
            config.retrieval_dir(5, 'opt_retrieval'),
            'dedup_data/processed/top5/esm2_opt_retrieval',
        )

    def test_dataset_argument_defaults_to_phopt_and_accepts_dedup(self):
        parser = argparse.ArgumentParser()
        add_dataset_argument(parser)

        self.assertEqual(parser.parse_args([]).dataset, 'phopt')
        self.assertEqual(parser.parse_args(['--dataset', 'dedup']).dataset, 'dedup')


if __name__ == '__main__':
    unittest.main()
