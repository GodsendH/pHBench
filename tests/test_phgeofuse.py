import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from phgeofuse.cache import atomic_torch_save, valid_torch_cache
from phgeofuse.chemistry import henderson_hasselbalch_charge
from phgeofuse.config import load_config
from phgeofuse.graph import build_edges, graph_feature_dim
from phgeofuse.io import ProteinRecord, parse_phopt_header, read_fasta, read_manifest
from phgeofuse.model import PHGeoFuse, compute_loss
from phgeofuse.retrieval import RetrievalStore, record_key
from phgeofuse.structures import acquire_structure, foldseek_three_di
from phgeofuse.engine import load_checkpoint, train_model
from utils.distributed import DistributedContext


class PHGeoFuseTests(unittest.TestCase):
    def test_phopt_fasta_parser(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sample.fasta"
            source.write_text(">P12345 | Organism | 1.2.3.4 | 7.5 | 0.8\nACDE\n")
            records = read_fasta(source, "train")
        self.assertEqual(records[0].protein_id, "P12345")
        self.assertEqual(records[0].sequence, "ACDE")
        self.assertEqual(records[0].ph_opt, 7.5)
        self.assertEqual(parse_phopt_header(">A | B | 1.1.1.1 | 6 | 1")[0], "A")

    def test_minimal_manifest_preserves_optional_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "manifest.csv"
            source.write_text("protein_id,sequence,ph_opt,split\nP1,ACD,7.0,train\n")
            record = read_manifest(source)[0]
        self.assertEqual(record.sample_weight, 1.0)
        self.assertEqual(record.status, "pending")

    def test_cache_metadata_invalidates_stale_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "cache.pt"
            atomic_torch_save(destination, {"value": 1, "metadata": {"sequence_sha256": "a"}})
            self.assertTrue(valid_torch_cache(destination, {"sequence_sha256": "a"}))
            self.assertFalse(valid_torch_cache(destination, {"sequence_sha256": "b"}))

    def test_offline_structure_requires_valid_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "offline structure cache"):
                acquire_structure("P12345", "ACD", Path(directory) / "missing.pdb", {}, True)

    def test_foldseek_masks_low_confidence_positions(self):
        with tempfile.TemporaryDirectory() as directory:
            structure = Path(directory) / "sample.pdb"
            structure.write_text("MODEL\nEND\n")

            def fake_run(command, **kwargs):
                Path(command[-1]).write_text("sample.pdb_A\tACD\tqwe\n")
                return mock.Mock(returncode=0, stdout="", stderr="")

            with mock.patch("phgeofuse.structures.shutil.which", return_value="foldseek"), mock.patch(
                "phgeofuse.structures.subprocess.run", side_effect=fake_run
            ):
                _, three_di = foldseek_three_di(
                    structure, "ACD", torch.tensor([90.0, 50.0, 80.0]),
                    {"structure": {"plddt_mask_threshold": 70}},
                )
        self.assertEqual(three_di, "q#e")

    def test_edges_are_translation_invariant(self):
        coords = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
        left_index, left_features = build_edges(coords, spatial_k=2, cutoff=8.0, rbf_bins=4)
        right_index, right_features = build_edges(coords + torch.tensor([10.0, -7.0, 3.0]), spatial_k=2, cutoff=8.0, rbf_bins=4)
        torch.testing.assert_close(left_index, right_index)
        torch.testing.assert_close(left_features, right_features)

    def test_henderson_hasselbalch_charge_has_expected_direction(self):
        types = torch.tensor([0, 5])
        pkas = torch.tensor([4.0, 10.5])
        charges = henderson_hasselbalch_charge(types, pkas, torch.tensor([3.0, 11.0]))
        self.assertLess(charges[0, 1], charges[0, 0])
        self.assertLess(charges[1, 1], charges[1, 0])
        self.assertLess(charges[0, 1], 0)
        self.assertGreater(charges[1, 0], 0)

    def test_retrieval_excludes_training_self(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for index, (split, label, vector) in enumerate(
                [("train", 5.0, [1.0, 0.0]), ("train", 9.0, [0.0, 1.0]), ("validation", 7.0, [1.0, 0.1])]
            ):
                embedding = root / f"embedding-{index}.pt"
                structure = root / f"structure-{index}.pdb"
                structure.write_text("END\n")
                atomic_torch_save(embedding, {"embedding": torch.tensor([vector]), "metadata": {}})
                records.append(
                    ProteinRecord(
                        protein_id=f"P{index}", sequence="A", split=split, ph_opt=label,
                        embedding_path=str(embedding), structure_path=str(structure),
                        structure_sha256=str(index), status="ready",
                    )
                )
            config = {"retrieval": {"top_k": 1, "require_foldseek": False, "mmseqs_binary": "missing"}, "structure": {"foldseek_binary": "missing"}}
            store = RetrievalStore.build(records, config, root / "retrieval.pt")
            first = store.rows[record_key(records[0])]
        self.assertAlmostEqual(first["saprot_value"], 9.0)

    def test_model_forward_and_backward(self):
        config = _tiny_config()
        model = PHGeoFuse(config, torch.device("cpu"))
        batch = _tiny_batch()
        outputs = model(batch)
        self.assertEqual(outputs["probabilities"].shape, (2, 5))
        torch.testing.assert_close(outputs["probabilities"].sum(dim=1), torch.ones(2))
        loss, parts = compute_loss(outputs, batch, config)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))
        self.assertIn("ec", parts)

    def test_tiny_training_writes_reloadable_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _tiny_config()
            config.update(
                {
                    "_root": str(root),
                    "paths": {
                        "retrieval": str(root / "retrieval.pt"),
                        "runs": str(root / "runs"),
                    },
                    "training": {
                        "seed": 3, "run_name": "smoke", "per_device_batch_size": 2,
                        "global_batch_size": 2, "num_workers": 0, "epochs": 1,
                        "learning_rate": 1e-3, "weight_decay": 0.0,
                        "warmup_fraction": 0.0, "early_stopping_patience": 2,
                    },
                }
            )
            config["retrieval"].update(
                {"top_k": 1, "require_foldseek": False, "mmseqs_binary": "missing"}
            )
            config["structure"] = {"foldseek_binary": "missing"}
            records = _tiny_records(root)
            context = DistributedContext(False, 0, 0, 1, torch.device("cpu"))
            checkpoint = train_model(records, config, context)
            self.assertTrue(checkpoint.is_file())
            restored = PHGeoFuse(config, torch.device("cpu"))
            payload = load_checkpoint(checkpoint, restored)
            self.assertEqual(payload["epoch"], 0)


def _tiny_config():
    return {
        "model": {
            "mode": "frozen", "embedding_dim": 4, "hidden_dim": 8,
            "egnn_layers": 2, "attention_heads": 2, "dropout": 0.0,
            "ph_min": 6.0, "ph_max": 8.0, "ph_step": 0.5,
        },
        "graph": {"rbf_bins": 4},
        "retrieval": {"dropout": 0.0},
        "loss": {"soft_label_sigma": 0.35, "regression_weight": 0.5, "ec_weight": 0.1},
    }


def _tiny_batch():
    lengths = torch.tensor([3, 2])
    coords = torch.tensor([[0., 0., 0.], [3., 0., 0.], [6., 0., 0.], [0., 0., 0.], [3., 0., 0.]])
    first_index, first_features = build_edges(coords[:3], spatial_k=2, cutoff=8.0, rbf_bins=4)
    second_index, second_features = build_edges(coords[3:], spatial_k=1, cutoff=8.0, rbf_bins=4)
    return {
        "keys": ["train::a", "train::b"], "saprot_texts": ["AaAaAa", "AaAa"],
        "lengths": lengths, "node_features": torch.randn(5, graph_feature_dim()),
        "coords": coords, "edge_index": torch.cat([first_index, second_index + 3], dim=1),
        "edge_features": torch.cat([first_features, second_features]),
        "ionizable_type": torch.tensor([0, 5, -1, 1, 4]),
        "pka": torch.tensor([4.0, 10.5, 7.0, 4.3, 6.0]),
        "graph_index": torch.tensor([0, 0, 0, 1, 1]),
        "embeddings": torch.randn(5, 4), "labels": torch.tensor([6.5, 7.5]),
        "weights": torch.ones(2), "ec_labels": torch.tensor([0, 1]),
        "retrieval": torch.tensor([[6.0, 7.0, .8, .7, .5, .2, .3, 1., 1.], [7.0, 8.0, .8, .7, .5, .2, .3, 1., 1.]]),
    }


def _tiny_records(root: Path):
    records = []
    specifications = [
        ("train", 5.5), ("train", 8.5),
        ("validation", 6.0), ("validation", 8.0),
        ("test", 7.0),
    ]
    for index, (split, label) in enumerate(specifications):
        graph_path = root / f"graph-{index}.pt"
        embedding_path = root / f"embedding-{index}.pt"
        structure_path = root / f"structure-{index}.pdb"
        structure_path.write_text("END\n")
        graph = _tiny_batch()
        node_slice = slice(0, 3)
        edge_index, edge_features = build_edges(
            graph["coords"][node_slice], spatial_k=2, cutoff=8.0, rbf_bins=4
        )
        atomic_torch_save(
            graph_path,
            {
                "coords": graph["coords"][node_slice],
                "node_features": graph["node_features"][node_slice],
                "edge_index": edge_index,
                "edge_features": edge_features,
                "ionizable_type": graph["ionizable_type"][node_slice],
                "pka": graph["pka"][node_slice],
                "metadata": {"length": 3},
            },
        )
        atomic_torch_save(
            embedding_path,
            {"embedding": torch.randn(3, 4), "metadata": {"length": 3}},
        )
        records.append(
            ProteinRecord(
                protein_id=f"P{index}", sequence="ADE", split=split, ph_opt=label,
                ec="1.1.1.1", graph_path=str(graph_path), embedding_path=str(embedding_path),
                structure_path=str(structure_path), structure_sha256=str(index),
                three_di="aaa", status="ready",
            )
        )
    return records


if __name__ == "__main__":
    unittest.main()
