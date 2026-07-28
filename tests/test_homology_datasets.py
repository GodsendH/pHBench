from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dataset_registry import dataset_fasta_paths, processed_dataset_directory
from phgeofuse.datasets import apply_dataset
from scripts.build_homology_datasets import (
    CleanRecord,
    RawRecord,
    clean_records,
    normalize_fraction,
    split_clusters,
)


class HomologyDatasetTests(unittest.TestCase):
    def test_identity_and_coverage_accept_percent_or_fraction(self):
        self.assertEqual(normalize_fraction(20, "identity"), 0.2)
        self.assertEqual(normalize_fraction(0.8, "coverage"), 0.8)
        with self.assertRaises(ValueError):
            normalize_fraction(0, "identity")

    def test_duplicate_label_range_threshold_is_inclusive(self):
        records = [
            RawRecord("A", "org", "1.1.1.1", 6.0, 1.0, "ACD", "train", "a"),
            RawRecord("B", "org", "1.1.1.1", 6.25, 1.0, "ACD", "test", "b"),
            RawRecord("C", "org", "1.1.1.1", 7.0, 1.0, "EFG", "train", "a"),
            RawRecord("D", "org", "1.1.1.1", 7.3, 1.0, "EFG", "test", "b"),
        ]

        cleaned, conflicts = clean_records(records, 0.25)

        self.assertEqual([record.protein_id for record in cleaned], ["A"])
        self.assertEqual(cleaned[0].ph_opt, 6.125)
        self.assertEqual(cleaned[0].source_ids, ("A", "B"))
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["source_ids"], "C;D")

    def test_cluster_split_never_separates_members(self):
        records = [
            CleanRecord(
                protein_id=f"P{index:02d}",
                organism="org",
                ec=f"{index % 6 + 1}.1.1.1",
                ph_opt=4.0 + index * 0.2,
                sample_weight=1.0,
                sequence="A" * (50 + index) + "C",
                source_ids=(f"P{index:02d}",),
                source_splits=("train",),
                original_labels=(4.0 + index * 0.2,),
            )
            for index in range(30)
        ]
        memberships = {
            record.protein_id: f"cluster-{index // 2}"
            for index, record in enumerate(records)
        }

        assignment, score = split_clusters(
            records,
            memberships,
            {"train": 0.7, "validation": 0.1, "test": 0.2},
            seed=7,
            attempts=4,
            ph_bin_width=0.5,
        )

        self.assertGreaterEqual(score, 0.0)
        self.assertEqual(set(assignment.values()), {"train", "validation", "test"})
        self.assertEqual(len(assignment), 15)

    def test_dataset_paths_preserve_legacy_layout(self):
        root = Path("/project")
        self.assertEqual(
            dataset_fasta_paths(root, "phopt")["train"],
            root / "data" / "phopt_training.fasta",
        )
        self.assertEqual(
            dataset_fasta_paths(root, "identity20")["test"],
            root / "data" / "datasets" / "identity20" / "phopt_testing.fasta",
        )
        self.assertEqual(
            processed_dataset_directory(root, "identity20", 5, "opt_retrieval"),
            root / "data" / "processed" / "identity20" / "top5" / "esm2_opt_retrieval",
        )

    def test_phgeofuse_dataset_uses_isolated_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split_paths = dataset_fasta_paths(root, "identity20")
            for source in split_paths.values():
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text(">P | org | 1.1.1.1 | 7 | 1\nACD\n")
            config = {
                "_root": str(root),
                "data": {"splits": {}, "subsets": {"legacy": {}}},
                "paths": {"structures": "shared/structures"},
            }

            resolved = apply_dataset(config, "identity20")

        self.assertEqual(resolved["_dataset"], "identity20")
        self.assertNotIn("subsets", resolved["data"])
        self.assertEqual(
            resolved["paths"]["manifest"],
            "artifacts/phgeofuse/datasets/identity20/manifest.csv",
        )
        self.assertEqual(resolved["paths"]["structures"], "shared/structures")


if __name__ == "__main__":
    unittest.main()
