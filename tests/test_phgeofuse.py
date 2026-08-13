import copy
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
from phgeofuse.dataset import ProteinGraphDataset
from phgeofuse.graph import build_edges, graph_feature_dim
from phgeofuse.io import ProteinRecord, parse_phopt_header, read_fasta, read_manifest
from phgeofuse.model import PHGeoFuse, compute_dual_view_loss, compute_loss
from phgeofuse.retrieval import (
    RETRIEVAL_FEATURE_DIM,
    RetrievalStore,
    SequenceHit,
    _aggregate_retrieval_view,
    _is_low_homology_pair,
    _safe_key,
    ensure_retrieval_store,
    record_key,
    retrieval_build_signature,
)
from phgeofuse.structures import (
    acquire_structure,
    foldseek_three_di,
    select_chain,
    sequence_matches,
)
from phgeofuse.engine import (
    _apply_homology_residual_scale,
    _configure_trainable_scope,
    _set_training_mode,
    _update_validation_tracking,
    calibrate_checkpoint,
    load_checkpoint,
    load_initial_checkpoint,
    resolve_evaluation_split,
    select_homology_residual_scale,
    train_model,
)
from utils.distributed import DistributedContext


class PHGeoFuseTests(unittest.TestCase):
    def test_checkpoint_improvement_is_independent_of_patience_delta(self):
        state = _update_validation_tracking(
            rmse=0.9995,
            best_rmse=1.0,
            patience_rmse=1.0,
            stale=2,
            min_delta=0.001,
        )

        best_rmse, patience_rmse, stale, checkpoint_improved = state
        self.assertEqual(best_rmse, 0.9995)
        self.assertEqual(patience_rmse, 1.0)
        self.assertEqual(stale, 3)
        self.assertTrue(checkpoint_improved)

    def test_phopt_fasta_parser(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sample.fasta"
            source.write_text(">P12345 | Organism | 1.2.3.4 | 7.5 | 0.8\nACDE\n")
            records = read_fasta(source, "train")
        self.assertEqual(records[0].protein_id, "P12345")
        self.assertEqual(records[0].sequence, "ACDE")
        self.assertEqual(records[0].ph_opt, 7.5)
        self.assertEqual(parse_phopt_header(">A | B | 1.1.1.1 | 6 | 1")[0], "A")

    def test_ephod_low_identity_subset_matches_test_fasta(self):
        root = Path(__file__).resolve().parents[1]
        full_test = {
            record.protein_id: record
            for record in read_fasta(root / "data" / "phopt_testing.fasta", "test")
        }
        subset = read_fasta(
            root / "data" / "phopt_testing_low_identity.fasta", "test"
        )

        self.assertEqual(len(subset), 999)
        self.assertEqual(len({record.protein_id for record in subset}), 999)
        for record in subset:
            self.assertIn(record.protein_id, full_test)
            self.assertEqual(record.sequence, full_test[record.protein_id].sequence)
            self.assertEqual(record.ph_opt, full_test[record.protein_id].ph_opt)

    def test_low_identity_evaluation_reports_incomplete_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subset = root / "subset.fasta"
            subset.write_text(
                ">P1 | Organism | 1.1.1.1 | 6.0 | 1.0\nACD\n"
                ">P2 | Organism | 1.1.1.1 | 8.0 | 1.0\nEFG\n"
            )
            records = [
                ProteinRecord(
                    protein_id="P1", sequence="ACD", split="test", ph_opt=6.0,
                    status="ready",
                ),
                ProteinRecord(
                    protein_id="P2", sequence="EFG", split="test", ph_opt=8.0,
                    status="failed",
                ),
            ]
            config = {
                "_root": str(root),
                "data": {
                    "subsets": {
                        "test_low_identity": {
                            "source_split": "test",
                            "fasta": "subset.fasta",
                            "expected_count": 2,
                        }
                    }
                },
            }

            source_split, allowed_ids, metadata = resolve_evaluation_split(
                records, config, "test_low_identity"
            )
            dataset = ProteinGraphDataset(
                records,
                source_split,
                retrieval=None,
                allowed_protein_ids=allowed_ids,
            )

        self.assertEqual(source_split, "test")
        self.assertEqual(allowed_ids, {"P1", "P2"})
        self.assertEqual([record.protein_id for record in dataset.records], ["P1"])
        self.assertEqual(metadata["subset_requested_count"], 2)
        self.assertEqual(metadata["subset_evaluated_count"], 1)
        self.assertEqual(metadata["subset_unavailable_ids"], ["P2"])
        self.assertFalse(metadata["subset_complete"])

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

    def test_sequence_matching_treats_target_x_as_wildcard(self):
        self.assertTrue(sequence_matches("ACDKFG", "ACDXFG"))
        self.assertFalse(sequence_matches("ACEKFG", "ACDXFG"))
        self.assertFalse(sequence_matches("ACDKF", "ACDXFG"))

    def test_select_chain_accepts_x_wildcard(self):
        with tempfile.TemporaryDirectory() as directory:
            structure = Path(directory) / "sample.pdb"
            structure.write_text(
                "".join(
                    [
                        _pdb_ca_line(1, "ALA", 1, 0.0),
                        _pdb_ca_line(2, "CYS", 2, 3.0),
                        _pdb_ca_line(3, "ASP", 3, 6.0),
                    ]
                )
                + "END\n"
            )
            chain, residues = select_chain(structure, "AXD")
            self.assertEqual(chain, "A")
            self.assertEqual("".join(residue.amino_acid for residue in residues), "ACD")
            with self.assertRaisesRegex(ValueError, "structure sequence mismatch"):
                select_chain(structure, "AXE")

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

    def test_foldseek_accepts_x_wildcard(self):
        with tempfile.TemporaryDirectory() as directory:
            structure = Path(directory) / "sample.pdb"
            structure.write_text("MODEL\nEND\n")

            def fake_run(command, **kwargs):
                Path(command[-1]).write_text("sample.pdb_A\tACD\tqwe\n")
                return mock.Mock(returncode=0, stdout="", stderr="")

            with mock.patch("phgeofuse.structures.shutil.which", return_value="foldseek"), mock.patch(
                "phgeofuse.structures.subprocess.run", side_effect=fake_run
            ):
                observed, three_di = foldseek_three_di(
                    structure, "AXD", torch.tensor([90.0, 90.0, 90.0]), {}
                )
        self.assertEqual(observed, "ACD")
        self.assertEqual(three_di, "qwe")

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

    def test_v1_retrieval_rows_expand_to_v2_features(self):
        row = {
            "saprot_value": 6.5,
            "foldseek_value": 7.0,
            "identity": 0.3,
            "saprot_available": True,
        }
        store = RetrievalStore({"schema_version": "1", "rows": {"test::P": row}})

        normal = store.features("test::P")
        low = store.features("test::P", view="low_homology")

        self.assertEqual(normal.shape, (RETRIEVAL_FEATURE_DIM,))
        torch.testing.assert_close(normal, low)
        self.assertEqual(float(normal[0]), 6.5)
        self.assertEqual(float(normal[9]), 0.0)

    def test_low_homology_filter_matches_identity20_boundary(self):
        query = ProteinRecord("Q", "A" * 10, "train")
        target = ProteinRecord("T", "C" * 10, "train")
        key = (_safe_key(query), _safe_key(target))
        config = {
            "retrieval": {
                "low_homology_identity": 0.2,
                "low_homology_coverage": 0.8,
                "low_homology_coverage_mode": 0,
            }
        }

        self.assertFalse(
            _is_low_homology_pair(
                query, target, {key: SequenceHit(0.2, 0.8, 0.8, 10.0)}, config
            )
        )
        self.assertTrue(
            _is_low_homology_pair(
                query, target, {key: SequenceHit(0.199, 0.8, 0.8, 10.0)}, config
            )
        )
        self.assertTrue(
            _is_low_homology_pair(
                query, target, {key: SequenceHit(0.2, 0.79, 0.8, 10.0)}, config
            )
        )
        self.assertTrue(_is_low_homology_pair(query, target, {}, config))

    def test_low_homology_view_filters_high_identity_candidates(self):
        query = ProteinRecord("Q", "A" * 10, "train")
        high = ProteinRecord("H", "A" * 10, "train", ph_opt=5.0)
        low = ProteinRecord("L", "C" * 10, "train", ph_opt=9.0)
        training = [high, low]
        hits = {
            (_safe_key(query), _safe_key(high)): SequenceHit(0.8, 1.0, 1.0, 50.0),
            (_safe_key(query), _safe_key(low)): SequenceHit(0.1, 1.0, 1.0, 20.0),
        }
        config = {
            "retrieval": {
                "low_homology_identity": 0.2,
                "low_homology_coverage": 0.8,
                "low_homology_coverage_mode": 0,
            }
        }

        normal = _aggregate_retrieval_view(
            query, training, torch.tensor([5.0, 9.0]),
            [(0, 0.9), (1, 0.8)], [], hits, 1, config,
            low_homology=False,
        )
        filtered = _aggregate_retrieval_view(
            query, training, torch.tensor([5.0, 9.0]),
            [(0, 0.9), (1, 0.8)], [], hits, 1, config,
            low_homology=True,
        )

        self.assertAlmostEqual(normal["saprot_value"], 5.0)
        self.assertAlmostEqual(filtered["saprot_value"], 9.0)
        self.assertAlmostEqual(filtered["identity"], 0.1)

    def test_homology_training_rebuilds_v1_retrieval_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "retrieval.pt"
            record = ProteinRecord("P", "ACD", "train", ph_opt=7.0, status="ready")
            atomic_torch_save(
                destination,
                {"schema_version": "1", "rows": {record_key(record): {}}},
            )
            config = {
                "_root": directory,
                "paths": {"retrieval": str(destination)},
                "homology_training": {"enabled": True},
                "retrieval": {"top_k": 1, "candidate_k": 2},
            }
            rebuilt = RetrievalStore({"schema_version": "2", "rows": {}})
            with mock.patch.object(
                RetrievalStore, "build", return_value=rebuilt
            ) as build:
                result = ensure_retrieval_store([record], config)

        self.assertIs(result, rebuilt)
        build.assert_called_once()

    def test_homology_retrieval_signature_controls_cache_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "retrieval.pt"
            record = ProteinRecord("P", "ACD", "train", ph_opt=7.0, status="ready")
            config = {
                "_root": directory,
                "paths": {"retrieval": str(destination)},
                "homology_training": {"enabled": True},
                "retrieval": {"top_k": 1, "candidate_k": 2},
            }
            row = {"low_homology": {}}
            atomic_torch_save(
                destination,
                {
                    "schema_version": "2",
                    "build_signature": retrieval_build_signature(config),
                    "rows": {record_key(record): row},
                },
            )
            with mock.patch.object(RetrievalStore, "build") as build:
                store = ensure_retrieval_store([record], config)
                build.assert_not_called()
            self.assertEqual(store.payload["schema_version"], "2")

            changed = copy.deepcopy(config)
            changed["retrieval"]["candidate_k"] = 3
            rebuilt = RetrievalStore({"schema_version": "2", "rows": {}})
            with mock.patch.object(
                RetrievalStore, "build", return_value=rebuilt
            ) as build:
                result = ensure_retrieval_store([record], changed)

        self.assertIs(result, rebuilt)
        build.assert_called_once()

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

    def test_fixed_fusion_renormalizes_unavailable_experts(self):
        config = _tiny_config()
        config["fusion"] = {
            "mode": "fixed",
            "fixed_weights": [0.25, 0.25, 0.5],
        }
        model = PHGeoFuse(config, torch.device("cpu"))
        batch = _tiny_batch()
        batch["retrieval"][0, 8] = 0.0

        outputs = model(batch)

        torch.testing.assert_close(
            outputs["gate_weights"],
            torch.tensor([[0.5, 0.5, 0.0], [0.25, 0.25, 0.5]]),
        )
        loss, _ = compute_loss(outputs, batch, config)
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.geometry.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in model.gate.parameters()))

    def test_reliability_gate_uses_supervision_and_masks_unavailable_experts(self):
        config = _tiny_config()
        config["fusion"] = {
            "mode": "reliability",
            "gate_hidden_dim": 16,
            "gate_dropout": 0.0,
            "gate_temperature": 1.5,
            "gate_prior": [0.25, 0.25, 0.5],
        }
        config["loss"].update(
            {
                "gate_supervision_weight": 1.0,
                "gate_prior_weight": 0.01,
                "gate_target_temperature": 0.3,
            }
        )
        model = PHGeoFuse(config, torch.device("cpu"))
        batch = _tiny_batch()
        batch["retrieval"][0, 8] = 0.0

        outputs = model(batch)
        loss, parts = compute_loss(outputs, batch, config)
        loss.backward()

        self.assertEqual(model.reliability_gate[0].in_features, 15)
        torch.testing.assert_close(outputs["gate_weights"].sum(dim=-1), torch.ones(2))
        torch.testing.assert_close(outputs["gate_weights"][0, 2], torch.tensor(0.0))
        self.assertIn("gate_supervision", parts)
        self.assertIn("gate_prior", parts)
        self.assertTrue(
            any(parameter.grad is not None for parameter in model.reliability_gate.parameters())
        )
        self.assertTrue(all(parameter.grad is None for parameter in model.gate.parameters()))

    def test_homology_gate_dual_view_encodes_once(self):
        config = _tiny_config()
        config["fusion"] = {
            "mode": "homology_reliability",
            "gate_hidden_dim": 16,
            "gate_dropout": 0.0,
            "gate_temperature": 1.0,
        }
        model = PHGeoFuse(config, torch.device("cpu"))
        batch = _tiny_batch()
        low = batch["retrieval"].clone()
        low[:, 4] = 0.1

        with mock.patch.object(
            model.geometry, "forward", wraps=model.geometry.forward
        ) as encode:
            outputs = model(
                batch,
                retrieval_views={"normal": batch["retrieval"], "low_homology": low},
            )

        self.assertEqual(encode.call_count, 1)
        self.assertEqual(set(outputs), {"normal", "low_homology"})
        self.assertEqual(model.homology_gate[0].in_features, 21)
        torch.testing.assert_close(
            outputs["normal"]["gate_weights"].sum(dim=-1), torch.ones(2)
        )

    def test_zero_initialized_homology_residual_preserves_learned_gate(self):
        source_config = _tiny_config()
        source_model = PHGeoFuse(source_config, torch.device("cpu"))
        target_config = copy.deepcopy(source_config)
        target_config["fusion"] = {
            "mode": "homology_residual",
            "gate_hidden_dim": 16,
            "gate_dropout": 0.0,
            "gate_temperature": 1.5,
        }
        target_model = PHGeoFuse(target_config, torch.device("cpu"))
        incompatible = target_model.load_state_dict(
            source_model.state_dict(), strict=False
        )
        self.assertTrue(incompatible.missing_keys)
        self.assertTrue(
            all(name.startswith("homology_gate.") for name in incompatible.missing_keys)
        )
        self.assertFalse(incompatible.unexpected_keys)
        source_model.eval()
        target_model.eval()
        batch = _tiny_batch()

        source = source_model(batch)
        target = target_model(batch)

        torch.testing.assert_close(target["mean"], source["mean"], rtol=0, atol=0)
        torch.testing.assert_close(
            target["gate_weights"], source["gate_weights"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            target["baseline_mean"], source["mean"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            target["homology_residual_logits"],
            torch.zeros_like(target["homology_residual_logits"]),
            rtol=0,
            atol=0,
        )

    def test_zero_residual_scale_restores_learned_gate_after_training(self):
        source_config = _tiny_config()
        source_model = PHGeoFuse(source_config, torch.device("cpu"))
        target_config = copy.deepcopy(source_config)
        target_config["fusion"] = {
            "mode": "homology_residual",
            "gate_hidden_dim": 16,
            "gate_dropout": 0.0,
            "gate_temperature": 1.5,
        }
        target_model = PHGeoFuse(target_config, torch.device("cpu"))
        target_model.load_state_dict(source_model.state_dict(), strict=False)
        with torch.no_grad():
            target_model.homology_gate[-1].bias.copy_(
                torch.tensor([-2.0, 1.0, 1.0])
            )
            target_model.homology_residual_scale.zero_()
        source_model.eval()
        target_model.eval()
        batch = _tiny_batch()

        source = source_model(batch)
        target = target_model(batch)

        torch.testing.assert_close(target["mean"], source["mean"], rtol=0, atol=0)
        torch.testing.assert_close(
            target["gate_weights"], source["gate_weights"], rtol=0, atol=0
        )

    def test_validation_scale_selection_falls_back_to_baseline(self):
        config = _tiny_config()
        config["fusion"] = {
            "mode": "homology_residual",
            "gate_temperature": 1.0,
        }
        config["homology_training"] = {
            "enabled": True,
            "residual_scale_grid": [0.0, 0.5, 1.0],
            "residual_scale_min_improvement": 0.0,
        }
        model = PHGeoFuse(config, torch.device("cpu"))
        context = DistributedContext(False, 0, 0, 1, torch.device("cpu"))
        rows = []
        for label, experts in ((6.0, [6.0, 8.0, 8.0]), (8.0, [8.0, 6.0, 6.0])):
            rows.append(
                {
                    "key": str(label),
                    "label": label,
                    "prediction": label,
                    "global_prediction": experts[0],
                    "saprot_prediction": experts[1],
                    "foldseek_prediction": experts[2],
                    "saprot_available": True,
                    "foldseek_available": True,
                    "uncertainty": 0.0,
                    "global_gate": 0.98,
                    "saprot_gate": 0.01,
                    "foldseek_gate": 0.01,
                    "sequence_identity": 0.1,
                    "query_coverage": 0.5,
                    "target_coverage": 0.5,
                    "_expert_means": experts,
                    "_expert_variances": [0.0, 0.0, 0.0],
                    "_expert_available": [True, True, True],
                    "_baseline_gate_weights": [0.98, 0.01, 0.01],
                    "_homology_residual_logits": [-10.0, 10.0, 10.0],
                }
            )
        baseline_rows = _apply_homology_residual_scale(rows, 0.0, config)
        evaluation = {"rows": rows, "metrics": {}}

        selected = select_homology_residual_scale(
            model, evaluation, config, context
        )

        self.assertEqual(selected["metrics"]["homology_residual_scale"], 0.0)
        self.assertEqual(float(model.homology_residual_scale), 0.0)
        self.assertEqual(
            [row["prediction"] for row in selected["rows"]],
            [row["prediction"] for row in baseline_rows],
        )
        self.assertGreater(
            selected["metrics"]["homology_residual_scale_rmse"]["1.0"],
            selected["metrics"]["rmse"],
        )

    def test_validation_scale_selection_prefers_simpler_near_optimum(self):
        config = _tiny_config()
        config["fusion"] = {
            "mode": "homology_residual",
            "gate_temperature": 1.0,
        }
        config["homology_training"] = {
            "enabled": True,
            "residual_scale_grid": [0.0, 0.5, 1.0],
            "residual_scale_min_improvement": 0.0,
        }
        model = PHGeoFuse(config, torch.device("cpu"))
        context = DistributedContext(False, 0, 0, 1, torch.device("cpu"))
        rows = []
        for label, experts in ((6.0, [8.0, 6.0, 6.0]), (8.0, [6.0, 8.0, 8.0])):
            rows.append(
                {
                    "key": str(label), "label": label,
                    "prediction": experts[0],
                    "global_prediction": experts[0],
                    "saprot_prediction": experts[1],
                    "foldseek_prediction": experts[2],
                    "saprot_available": True, "foldseek_available": True,
                    "uncertainty": 0.0,
                    "global_gate": 0.98, "saprot_gate": 0.01,
                    "foldseek_gate": 0.01,
                    "sequence_identity": 0.1,
                    "query_coverage": 0.5, "target_coverage": 0.5,
                    "_expert_means": experts,
                    "_expert_variances": [0.0, 0.0, 0.0],
                    "_expert_available": [True, True, True],
                    "_baseline_gate_weights": [0.98, 0.01, 0.01],
                    "_homology_residual_logits": [-4.0, 2.0, 2.0],
                }
            )
        rmses = {}
        for scale in (0.0, 0.5, 1.0):
            adjusted = _apply_homology_residual_scale(rows, scale, config)
            rmses[scale] = math.sqrt(
                sum((row["prediction"] - row["label"]) ** 2 for row in adjusted)
                / len(adjusted)
            )
        self.assertLess(rmses[1.0], rmses[0.5])
        self.assertLess(rmses[0.5], rmses[0.0])
        config["homology_training"]["residual_scale_simplicity_tolerance"] = (
            rmses[0.5] - rmses[1.0] + 1e-6
        )

        selected = select_homology_residual_scale(
            model, {"rows": rows, "metrics": {}}, config, context
        )

        self.assertEqual(selected["metrics"]["homology_residual_scale"], 0.5)
        self.assertEqual(float(model.homology_residual_scale), 0.5)

    def test_dual_view_loss_uses_configured_weights(self):
        config = _tiny_config()
        config["fusion"] = {"mode": "homology_reliability", "gate_dropout": 0.0}
        config["homology_training"] = {
            "normal_loss_weight": 1.0,
            "low_homology_loss_weight": 0.5,
            "consistency_weight": 0.1,
        }
        config["loss"].update(
            {
                "distribution_weight": 0.0,
                "ec_weight": 0.0,
                "gate_supervision_weight": 0.0,
                "gate_prior_weight": 0.0,
            }
        )
        model = PHGeoFuse(config, torch.device("cpu"))
        batch = _tiny_batch()
        outputs = model(
            batch,
            retrieval_views={
                "normal": batch["retrieval"],
                "low_homology": batch["retrieval"],
            },
        )

        total, parts = compute_dual_view_loss(outputs, batch, config)
        normal_mse = parts["normal_mse"]
        low_mse = parts["low_homology_mse"]
        expected = normal_mse + 0.5 * low_mse + 0.1 * parts["consistency"]

        torch.testing.assert_close(total.detach(), expected)

    def test_preservation_loss_uses_inclusive_identity_coverage_boundary(self):
        config = _tiny_config()
        config["fusion"] = {
            "mode": "homology_residual",
            "gate_dropout": 0.0,
            "gate_temperature": 1.0,
        }
        config["homology_training"] = {
            "normal_loss_weight": 0.0,
            "low_homology_loss_weight": 0.0,
            "consistency_weight": 0.0,
            "preservation_weight": 0.5,
            "preservation_scope": "high_homology",
            "preservation_identity": 20,
            "preservation_coverage": 80,
        }
        config["loss"].update(
            {
                "distribution_weight": 0.0,
                "ec_weight": 0.0,
                "gate_supervision_weight": 0.0,
                "gate_prior_weight": 0.0,
            }
        )
        model = PHGeoFuse(config, torch.device("cpu"))
        batch = _tiny_batch()
        batch["retrieval"][0, [4, 9, 10]] = torch.tensor([0.20, 0.80, 0.80])
        batch["retrieval"][1, [4, 9, 10]] = torch.tensor([0.20, 0.80, 0.79])
        outputs = model(
            batch,
            retrieval_views={
                "normal": batch["retrieval"],
                "low_homology": batch["retrieval"],
            },
        )
        outputs["normal"]["mean"] = outputs["normal"]["baseline_mean"] + torch.tensor(
            [1.0, 2.0]
        )

        total, parts = compute_dual_view_loss(outputs, batch, config)

        torch.testing.assert_close(parts["preservation"], torch.tensor(0.5))
        torch.testing.assert_close(total.detach(), torch.tensor(0.25))

    def test_gate_only_scope_freezes_all_other_parameters(self):
        config = _tiny_config()
        config["fusion"] = {"mode": "homology_residual", "gate_dropout": 0.0}
        config["training"] = {"trainable_scope": "homology_gate"}
        model = PHGeoFuse(config, torch.device("cpu"))

        _configure_trainable_scope(model, config, init_checkpoint="checkpoint.pt")
        _set_training_mode(model, config)

        trainable = {
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith("homology_gate.") for name in trainable))
        self.assertFalse(model.training)
        self.assertTrue(model.homology_gate.training)
        batch = _tiny_batch()
        outputs = model(
            batch,
            retrieval_views={
                "normal": batch["retrieval"],
                "low_homology": batch["retrieval"],
            },
        )
        loss, _ = compute_dual_view_loss(outputs, batch, config)
        loss.backward()
        self.assertTrue(
            all(
                parameter.grad is not None
                for parameter in model.homology_gate.parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.grad is None
                for name, parameter in model.named_parameters()
                if not name.startswith("homology_gate.")
            )
        )

    def test_initial_checkpoint_allows_only_new_homology_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            source_config = _tiny_config()
            source_model = PHGeoFuse(source_config, torch.device("cpu"))
            checkpoint = Path(directory) / "source.pt"
            atomic_torch_save(
                checkpoint,
                {
                    "model_state_dict": source_model.state_dict(),
                    "saprot_adapter_only": False,
                },
            )
            target_config = _tiny_config()
            target_config["fusion"] = {"mode": "homology_residual"}
            target_model = PHGeoFuse(target_config, torch.device("cpu"))

            load_initial_checkpoint(checkpoint, target_model)

            v1_config = _tiny_config()
            v1_config["fusion"] = {"mode": "homology_reliability"}
            v1_model = PHGeoFuse(v1_config, torch.device("cpu"))
            v1_checkpoint = Path(directory) / "homology-v1.pt"
            atomic_torch_save(
                v1_checkpoint,
                {
                    "model_state_dict": v1_model.state_dict(),
                    "saprot_adapter_only": False,
                },
            )
            with self.assertRaisesRegex(RuntimeError, "baseline checkpoint"):
                load_initial_checkpoint(v1_checkpoint, target_model)

            invalid = source_model.state_dict()
            invalid.pop("geometry.project.0.weight")
            invalid_checkpoint = Path(directory) / "invalid.pt"
            atomic_torch_save(
                invalid_checkpoint,
                {"model_state_dict": invalid, "saprot_adapter_only": False},
            )
            with self.assertRaisesRegex(RuntimeError, "invalid initialization"):
                load_initial_checkpoint(invalid_checkpoint, target_model)

    def test_supervised_gate_loss_pushes_probability_to_best_expert(self):
        config = _tiny_config()
        config["fusion"] = {"mode": "reliability"}
        config["loss"].update(
            {
                "mse_weight": 0.0,
                "distribution_weight": 0.0,
                "ec_weight": 0.0,
                "gate_supervision_weight": 1.0,
                "gate_prior_weight": 0.0,
                "gate_target_temperature": 0.1,
            }
        )
        gate_logits = torch.zeros(1, 3, requires_grad=True)
        outputs = {
            "mean": torch.tensor([7.0]),
            "logits": torch.zeros(1, 5),
            "ec_logits": torch.zeros(1, 7),
            "gate_weights": torch.softmax(gate_logits, dim=-1),
            "expert_means": torch.tensor([[7.0, 5.0, 9.0]]),
            "expert_available": torch.ones(1, 3, dtype=torch.bool),
        }
        batch = {
            "labels": torch.tensor([7.0]),
            "weights": torch.ones(1),
            "ec_labels": torch.tensor([-1]),
        }

        loss, _ = compute_loss(outputs, batch, config)
        loss.backward()

        self.assertLess(float(gate_logits.grad[0, 0]), 0.0)
        self.assertGreater(float(gate_logits.grad[0, 1]), 0.0)
        self.assertGreater(float(gate_logits.grad[0, 2]), 0.0)

    def test_loss_uses_final_fused_mse_as_primary_objective(self):
        config = _tiny_config()
        config["loss"].update(
            {"mse_weight": 1.0, "distribution_weight": 0.0, "ec_weight": 0.0}
        )
        batch = {
            "labels": torch.tensor([6.0, 8.0]),
            "weights": torch.tensor([0.5, 3.0]),
            "ec_labels": torch.tensor([-1, -1]),
        }
        outputs = {
            "mean": torch.tensor([7.0, 6.0]),
            "logits": torch.zeros(2, 5),
            "ec_logits": torch.zeros(2, 7),
        }

        loss, parts = compute_loss(outputs, batch, config)

        torch.testing.assert_close(loss, torch.tensor(2.5))
        torch.testing.assert_close(parts["mse"], torch.tensor(2.5))

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
                        "diagnostics": {"evaluate_train": True},
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
            metrics_path = root / "runs" / "smoke_frozen_seed3" / "metrics.jsonl"
            row = json.loads(metrics_path.read_text().splitlines()[0])
            self.assertIn("train_evaluation", row)
            self.assertIn("global_rmse", row["validation"])
            self.assertIn("bias_neutral", row["validation"])
            self.assertIn("mean_global_gate", row["validation"])
            self.assertIn("gate_best_expert_rate", row["validation"])
            self.assertIn("generalization_gap_rmse", row)
            self.assertEqual(
                set(row["validation_loss_components"]), {"mse", "distribution", "ec"}
            )

    def test_tiny_homology_gate_training_from_initial_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_config = _tiny_config()
            source_model = PHGeoFuse(source_config, torch.device("cpu"))
            initial = root / "initial.pt"
            atomic_torch_save(
                initial,
                {
                    "model_state_dict": source_model.state_dict(),
                    "saprot_adapter_only": False,
                },
            )
            config = _tiny_config()
            config.update(
                {
                    "_root": str(root),
                    "paths": {
                        "retrieval": str(root / "retrieval.pt"),
                        "runs": str(root / "runs"),
                    },
                    "fusion": {
                        "mode": "homology_residual",
                        "gate_hidden_dim": 8,
                        "gate_dropout": 0.0,
                        "gate_temperature": 1.0,
                        "gate_prior": [0.25, 0.25, 0.5],
                    },
                    "homology_training": {
                        "enabled": True,
                        "normal_loss_weight": 1.0,
                        "low_homology_loss_weight": 0.5,
                        "consistency_weight": 0.1,
                        "preservation_weight": 0.5,
                        "preservation_scope": "high_homology",
                        "preservation_identity": 0.2,
                        "preservation_coverage": 0.8,
                        "residual_scale_grid": [0.0, 0.5, 1.0],
                        "residual_scale_min_improvement": 0.0,
                    },
                    "training": {
                        "seed": 11,
                        "run_name": "homology-smoke",
                        "trainable_scope": "homology_gate",
                        "per_device_batch_size": 2,
                        "global_batch_size": 2,
                        "num_workers": 0,
                        "epochs": 1,
                        "learning_rate": 1e-3,
                        "weight_decay": 0.0,
                        "warmup_fraction": 0.0,
                        "early_stopping_patience": 2,
                    },
                }
            )
            config["loss"].update(
                {"gate_supervision_weight": 0.1, "gate_prior_weight": 0.01}
            )
            config["retrieval"].update(
                {
                    "top_k": 1,
                    "candidate_k": 2,
                    "require_foldseek": False,
                    "mmseqs_binary": "missing",
                }
            )
            config["structure"] = {"foldseek_binary": "missing"}
            records = _tiny_records(root)
            context = DistributedContext(False, 0, 0, 1, torch.device("cpu"))

            checkpoint = train_model(
                records, config, context, init_checkpoint=initial
            )

            self.assertTrue(checkpoint.is_file())
            metrics = json.loads(
                (root / "runs" / "homology-smoke_frozen_seed11" / "metrics.jsonl")
                .read_text()
                .splitlines()[0]
            )
            self.assertIn("normal_mse", metrics["train_loss_components"])
            self.assertIn("low_homology_mse", metrics["train_loss_components"])
            self.assertIn("preservation", metrics["train_loss_components"])
            self.assertIn("consistency", metrics["train_loss_components"])
            self.assertIn("homology_residual_scale", metrics["validation"])
            payload = torch.load(checkpoint, map_location="cpu")
            self.assertIn("homology_residual_scale", payload)
            restored = PHGeoFuse(config, torch.device("cpu"))
            load_checkpoint(checkpoint, restored)
            self.assertEqual(
                float(restored.homology_residual_scale),
                payload["homology_residual_scale"],
            )

            calibrated = root / "calibrated.pt"
            calibrate_checkpoint(
                records, config, checkpoint, context, output=calibrated
            )
            calibrated_payload = torch.load(calibrated, map_location="cpu")
            self.assertEqual(
                calibrated_payload["calibration"]["split"], "validation"
            )
            self.assertEqual(
                calibrated_payload["homology_residual_scale"],
                calibrated_payload["calibration"]["metrics"][
                    "homology_residual_scale"
                ],
            )
            self.assertTrue(
                calibrated.with_suffix(".calibration.metrics.json").is_file()
            )

    def test_train_diagnostics_do_not_change_parameter_updates(self):
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
                        "seed": 7, "run_name": "without-diagnostics",
                        "per_device_batch_size": 2, "global_batch_size": 2,
                        "num_workers": 0, "epochs": 2, "learning_rate": 1e-3,
                        "weight_decay": 0.0, "warmup_fraction": 0.0,
                        "early_stopping_patience": 3,
                    },
                }
            )
            config["retrieval"].update(
                {"top_k": 1, "require_foldseek": False, "mmseqs_binary": "missing"}
            )
            config["structure"] = {"foldseek_binary": "missing"}
            records = _tiny_records(root)
            context = DistributedContext(False, 0, 0, 1, torch.device("cpu"))

            train_model(records, config, context)
            diagnostic_config = copy.deepcopy(config)
            diagnostic_config["training"]["run_name"] = "with-diagnostics"
            diagnostic_config["training"]["diagnostics"] = {"evaluate_train": True}
            train_model(records, diagnostic_config, context)

            baseline = torch.load(
                root / "runs" / "without-diagnostics_frozen_seed7" / "last.pt",
                map_location="cpu",
            )["model_state_dict"]
            diagnostic = torch.load(
                root / "runs" / "with-diagnostics_frozen_seed7" / "last.pt",
                map_location="cpu",
            )["model_state_dict"]
            self.assertEqual(set(baseline), set(diagnostic))
            for name in baseline:
                torch.testing.assert_close(baseline[name], diagnostic[name], rtol=0, atol=0)


def _tiny_config():
    return {
        "model": {
            "mode": "frozen", "embedding_dim": 4, "hidden_dim": 8,
            "egnn_layers": 2, "attention_heads": 2, "dropout": 0.0,
            "ph_min": 6.0, "ph_max": 8.0, "ph_step": 0.5,
        },
        "graph": {"rbf_bins": 4},
        "retrieval": {"dropout": 0.0},
        "loss": {
            "soft_label_sigma": 0.35,
            "mse_weight": 1.0,
            "distribution_weight": 0.2,
            "ec_weight": 0.1,
        },
    }


def _pdb_ca_line(serial: int, residue: str, number: int, x: float) -> str:
    return (
        f"ATOM  {serial:5d}  CA  {residue:>3s} A{number:4d}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{90.0:6.2f}          C  \n"
    )


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
        "retrieval": torch.tensor(
            [
                [6.0, 7.0, .8, .7, .5, .2, .3, 1., 1., .9, .9, 1., 1., .1, .1],
                [7.0, 8.0, .8, .7, .5, .2, .3, 1., 1., .9, .9, 1., 1., .1, .1],
            ]
        ),
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
