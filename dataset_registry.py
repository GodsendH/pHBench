from __future__ import annotations

from pathlib import Path


DATASET_CHOICES = (
    "phopt",
    "identity100",
    "identity50",
    "identity30",
    "identity20",
)

_ALIASES = {
    "default": "phopt",
    "original": "phopt",
    "identity_100": "identity100",
    "identity_50": "identity50",
    "identity_30": "identity30",
    "identity_20": "identity20",
}


def normalize_dataset_name(dataset: str) -> str:
    name = _ALIASES.get(dataset.strip().lower(), dataset.strip().lower())
    if name not in DATASET_CHOICES:
        choices = ", ".join(DATASET_CHOICES)
        raise ValueError(f"unknown dataset {dataset!r}; choose one of: {choices}")
    return name


def dataset_name_for_identity(identity: float) -> str:
    percent = identity * 100.0 if identity <= 1.0 else identity
    rounded = round(percent)
    if abs(percent - rounded) > 1e-9:
        raise ValueError("identity dataset names require an integer percentage")
    return f"identity{rounded}"


def dataset_directory(project_root: str | Path, dataset: str) -> Path:
    name = normalize_dataset_name(dataset)
    root = Path(project_root)
    if name == "phopt":
        return root / "data"
    return root / "data" / "datasets" / name


def dataset_fasta_paths(project_root: str | Path, dataset: str) -> dict[str, Path]:
    directory = dataset_directory(project_root, dataset)
    return {
        "train": directory / "phopt_training.fasta",
        "validation": directory / "phopt_validation.fasta",
        "test": directory / "phopt_testing.fasta",
    }


def processed_dataset_directory(
    project_root: str | Path,
    dataset: str,
    top_k: int,
    retrieval_strategy: str,
) -> Path:
    name = normalize_dataset_name(dataset)
    root = Path(project_root) / "data" / "processed"
    if name != "phopt":
        root /= name
    return root / f"top{top_k}" / f"esm2_{retrieval_strategy}"
