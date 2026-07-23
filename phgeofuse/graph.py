from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .cache import atomic_torch_save, sha256_file, sha256_text, valid_torch_cache
from .chemistry import IONIZABLE_RESIDUES, residue_ionization
from .config import get
from .structures import Residue, run_propka, select_chain


GRAPH_SCHEMA_VERSION = "1"
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AA_TO_INDEX = {residue: index for index, residue in enumerate(AA_ALPHABET)}


def graph_feature_dim() -> int:
    return len(AA_ALPHABET) + 1 + 1 + 3 + len(IONIZABLE_RESIDUES) + 1 + 1


def build_graph_artifact(
    structure_path: str | Path,
    sequence: str,
    destination: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    structure_sha = sha256_file(structure_path)
    metadata = {
        "schema_version": GRAPH_SCHEMA_VERSION,
        "sequence_sha256": sha256_text(sequence),
        "structure_sha256": structure_sha,
    }
    if valid_torch_cache(destination, metadata):
        return torch.load(destination, map_location="cpu")
    chain, residues = select_chain(structure_path, sequence)
    coords = torch.tensor([residue.ca for residue in residues], dtype=torch.float32)
    plddt = torch.tensor([residue.plddt for residue in residues], dtype=torch.float32)
    rsa, secondary, annotation_source = _annotations(structure_path, residues, coords)
    predicted_pka, propka_error = run_propka(structure_path, residues, config)
    ionizable_type, pka, pka_missing = residue_ionization(sequence, predicted_pka)
    node_features = _node_features(sequence, plddt, rsa, secondary, ionizable_type, pka, pka_missing)
    edge_index, edge_features = build_edges(
        coords,
        spatial_k=int(get(config, "graph.spatial_k", 16)),
        cutoff=float(get(config, "graph.cutoff", 16.0)),
        rbf_bins=int(get(config, "graph.rbf_bins", 16)),
    )
    payload = {
        "coords": coords,
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_features": edge_features,
        "plddt": plddt,
        "rsa": rsa,
        "secondary": secondary,
        "ionizable_type": ionizable_type,
        "pka": pka,
        "pka_missing": pka_missing,
        "metadata": {
            **metadata,
            "chain": chain,
            "length": len(sequence),
            "annotation_source": annotation_source,
            "propka_error": propka_error,
        },
    }
    atomic_torch_save(destination, payload)
    return payload


def _annotations(
    structure_path: str | Path,
    residues: list[Residue],
    coords: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    try:
        import biotite.structure as struc
        from biotite.structure.io import load_structure

        atoms = load_structure(str(structure_path))
        if getattr(atoms, "stack_depth", lambda: 1)() > 1:
            atoms = atoms[0]
        atoms = atoms[atoms.chain_id == residues[0].chain]
        starts = struc.get_residue_starts(atoms, add_exclusive_stop=True)
        atom_sasa = struc.sasa(atoms)
        maximum = {
            "A": 129, "R": 274, "N": 195, "D": 193, "C": 167, "Q": 223,
            "E": 225, "G": 104, "H": 224, "I": 197, "L": 201, "K": 236,
            "M": 224, "F": 240, "P": 159, "S": 155, "T": 172, "W": 285,
            "Y": 263, "V": 174,
        }
        rsa = []
        for index, residue in enumerate(residues):
            area = float(atom_sasa[starts[index]:starts[index + 1]].sum())
            rsa.append(min(1.0, area / maximum.get(residue.amino_acid, 200.0)))
        sse_values = struc.annotate_sse(atoms, chain_id=residues[0].chain)
        mapping = {"a": 0, "b": 1, "c": 2}
        secondary = torch.tensor([mapping.get(str(value).lower(), 2) for value in sse_values], dtype=torch.long)
        if len(rsa) == len(residues) and secondary.numel() == len(residues):
            return torch.tensor(rsa), secondary, "biotite"
    except (ImportError, OSError, ValueError, IndexError, AttributeError, TypeError):
        pass
    distances = torch.cdist(coords, coords)
    neighbors = ((distances < 10.0) & (distances > 0)).sum(dim=1).float()
    rsa = torch.exp(-neighbors / 8.0).clamp(0.0, 1.0)
    secondary = _geometry_secondary(coords)
    return rsa, secondary, "geometry_fallback"


def _geometry_secondary(coords: torch.Tensor) -> torch.Tensor:
    result = torch.full((coords.shape[0],), 2, dtype=torch.long)
    if coords.shape[0] < 4:
        return result
    vectors = F.normalize(coords[1:] - coords[:-1], dim=-1)
    cosines = (vectors[:-1] * vectors[1:]).sum(dim=-1)
    result[1:-1] = torch.where(cosines > 0.25, 0, torch.where(cosines < -0.25, 1, 2))
    return result


def _node_features(sequence, plddt, rsa, secondary, ionizable_type, pka, pka_missing):
    aa_index = torch.tensor([AA_TO_INDEX.get(residue, AA_TO_INDEX["X"]) for residue in sequence])
    aa = F.one_hot(aa_index, num_classes=len(AA_ALPHABET)).float()
    sse = F.one_hot(secondary.clamp(0, 2), num_classes=3).float()
    ionizable = torch.zeros((len(sequence), len(IONIZABLE_RESIDUES)), dtype=torch.float32)
    valid = ionizable_type >= 0
    ionizable[valid, ionizable_type[valid]] = 1.0
    return torch.cat(
        [aa, (plddt / 100.0).unsqueeze(-1), rsa.unsqueeze(-1), sse, ionizable,
         (pka / 14.0).unsqueeze(-1), pka_missing.unsqueeze(-1)],
        dim=-1,
    )


def build_edges(coords: torch.Tensor, spatial_k: int = 16, cutoff: float = 16.0, rbf_bins: int = 16):
    length = coords.shape[0]
    distances = torch.cdist(coords, coords)
    pairs: set[tuple[int, int]] = set()
    for index in range(length - 1):
        pairs.add((index, index + 1))
        pairs.add((index + 1, index))
    for index in range(length):
        candidates = torch.argsort(distances[index])
        selected = 0
        for neighbor in candidates.tolist():
            if neighbor == index or float(distances[index, neighbor]) > cutoff:
                continue
            pairs.add((index, neighbor))
            selected += 1
            if selected >= spatial_k:
                break
    ordered = sorted(pairs)
    edge_index = torch.tensor(ordered, dtype=torch.long).t().contiguous()
    source, target = edge_index
    edge_distance = torch.linalg.vector_norm(coords[source] - coords[target], dim=-1)
    centers = torch.linspace(0.0, cutoff, rbf_bins)
    width = cutoff / max(1, rbf_bins - 1)
    rbf = torch.exp(-((edge_distance.unsqueeze(-1) - centers) / width) ** 2)
    offset = ((target - source).float() / max(1, length - 1)).unsqueeze(-1)
    sequential = ((target - source).abs() == 1).float().unsqueeze(-1)
    return edge_index, torch.cat([rbf, offset, sequential], dim=-1)
