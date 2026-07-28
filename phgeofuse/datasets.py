from __future__ import annotations

from pathlib import Path

from dataset_registry import dataset_fasta_paths, normalize_dataset_name


def apply_dataset(config: dict, dataset: str) -> dict:
    name = normalize_dataset_name(dataset)
    config["_dataset"] = name
    if name == "phopt":
        return config

    project_root = Path(config["_root"])
    split_paths = dataset_fasta_paths(project_root, name)
    missing = [str(source) for source in split_paths.values() if not source.is_file()]
    if missing:
        raise FileNotFoundError(
            f"dataset {name} has not been generated; missing files: {missing}"
        )

    data = config.setdefault("data", {})
    data["manifest"] = None
    data["splits"] = {
        split: str(source.relative_to(project_root))
        for split, source in split_paths.items()
    }
    data.pop("subsets", None)

    artifact_root = Path("artifacts") / "phgeofuse" / "datasets" / name
    paths = config.setdefault("paths", {})
    paths["manifest"] = str(artifact_root / "manifest.csv")
    paths["retrieval"] = str(artifact_root / "retrieval.pt")
    paths["runs"] = str(artifact_root / "runs")
    paths["predictions"] = str(artifact_root / "predictions")
    return config
