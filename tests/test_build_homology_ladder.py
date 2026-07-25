import tempfile
import unittest
from pathlib import Path

from build_homology_ladder import (
    RawRecord,
    Record,
    build_components,
    clean_records,
    make_clusters,
    split_clusters,
    validate_thresholds,
)


def raw(identifier, sequence, ph, split='train', ec='1.1.1.1'):
    return RawRecord(
        source_split=split,
        source_id=identifier,
        header=f'{identifier} | organism | {ec} | {ph} | source',
        sequence=sequence,
        ph=float(ph),
        ec=ec,
    )


def cleaned(index, ph, ec, length):
    sequence = 'A' * (length - 1) + chr(ord('C') + index)
    identifier = f'id{index}'
    return Record(
        key=f'S{index:05d}',
        source_id=identifier,
        source_ids=(identifier,),
        source_splits=('train',),
        header=f'{identifier} | organism | {ec} | {ph} | source',
        sequence=sequence,
        ph=float(ph),
        ec=ec,
    )


class HomologyLadderTests(unittest.TestCase):
    def test_cleaning_drops_conflicting_labels_and_keeps_one_duplicate(self):
        records = [
            raw('unique', 'AAAA', 7),
            raw('same_1', 'CCCC', 6),
            raw('same_2', 'CCCC', 6, split='test'),
            raw('conflict_1', 'DDDD', 5),
            raw('conflict_2', 'DDDD', 6, split='valid'),
        ]

        result, removed, stats = clean_records(records)

        self.assertEqual(len(result), 2)
        self.assertEqual(len(removed), 1)
        self.assertEqual(removed[0]['reason'], 'conflicting_ph_labels')
        self.assertEqual(stats['conflicting_groups_removed'], 1)
        duplicate = next(record for record in result if record.sequence == 'CCCC')
        self.assertEqual(duplicate.source_splits, ('test', 'train'))

    def test_components_include_transitive_homology_edges(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'edges.tsv'
            path.write_text(
                'S00000\tS00001\t0.9\t1\t1\n'
                'S00001\tS00002\t0.9\t1\t1\n'
                'S00003\tS00003\t1\t1\t1\n',
                encoding='utf-8',
            )
            components, rows = build_components(
                ['S00000', 'S00001', 'S00002', 'S00003'],
                edge_path=path,
            )

        self.assertEqual(rows, 3)
        self.assertIn(['S00000', 'S00001', 'S00002'], components)
        self.assertIn(['S00003'], components)

    def test_cluster_split_preserves_every_component(self):
        records = {
            record.key: record
            for record in [
                cleaned(index, 5 + index % 5, f'{index % 3 + 1}.1.1.1', 200 + index)
                for index in range(30)
            ]
        }
        components = [
            [f'S{index:05d}', f'S{index + 1:05d}']
            for index in range(0, 30, 2)
        ]
        clusters = make_clusters(components, records, ph_bin_width=0.5)

        assignment, details = split_clusters(
            clusters,
            records,
            split_ratios={'train': 0.7, 'valid': 0.1, 'test': 0.2},
            ph_bin_width=0.5,
            attempts=8,
            seed=0,
        )

        self.assertEqual(set(assignment), {
            cluster['cluster_id'] for cluster in clusters
        })
        self.assertEqual(sum(details['actual_counts'].values()), 30)
        self.assertTrue(all(count > 0 for count in details['actual_counts'].values()))

    def test_threshold_validation_is_sorted_and_unique(self):
        self.assertEqual(
            validate_thresholds([30, 100, 70, 70]),
            (100, 70, 30),
        )
        with self.assertRaises(ValueError):
            validate_thresholds([0, 30])


if __name__ == '__main__':
    unittest.main()
