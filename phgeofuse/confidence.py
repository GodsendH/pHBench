"""Explicit confidence units; never infer units from a small observed value."""
from __future__ import annotations

import math

CONFIDENCE_VERSION = "plddt_0_100_v1"


def confidence_scale(provenance: dict) -> str:
    explicit = provenance.get("plddt_scale")
    if explicit is not None:
        if explicit not in {"0_1", "0_100"}:
            raise ValueError(f"unsupported pLDDT scale: {explicit}")
        return explicit
    if provenance.get("source") == "alphafold_db":
        return "0_100"
    if (provenance.get("source") == "esmfold"
            and provenance.get("model") == "facebook/esmfold_v1"):
        # Historical Hugging Face output_to_pdb writes categorical_lddt (0..1).
        return "0_1"
    raise ValueError("structure confidence provenance does not establish its units")


def normalize_plddt(value: float, scale: str) -> float:
    limit = {"0_1": 1.0, "0_100": 100.0}.get(scale)
    if limit is None:
        raise ValueError(f"unsupported pLDDT scale: {scale}")
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= limit:
        raise ValueError(f"invalid pLDDT {value} for scale {scale}")
    return value * (100.0 if scale == "0_1" else 1.0)


def normalize_pdb_confidence(pdb: str, scale: str) -> str:
    lines = []
    for line in pdb.splitlines(keepends=True):
        if line.startswith(("ATOM  ", "HETATM")):
            value = normalize_plddt(float(line[60:66]), scale)
            line = line[:60] + f"{value:6.2f}" + line[66:]
        lines.append(line)
    return "".join(lines)
