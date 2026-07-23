from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

from .cache import artifact_path, atomic_json, sha256_text
from .config import config_hash, get, load_config, path
from .graph import GRAPH_SCHEMA_VERSION, build_graph_artifact
from .io import ProteinRecord, records_from_config, write_manifest
from .saprot import embedding_cache_path
from .structures import acquire_structure, foldseek_three_di


def prepare_records(
    records: Iterable[ProteinRecord],
    config: dict,
    *,
    offline: bool,
    manifest_path: str | Path | None = None,
) -> tuple[list[ProteinRecord], list[dict[str, str]]]:
    structures_root = path(config, "paths.structures", "artifacts/phgeofuse/structures")
    graphs_root = path(config, "paths.graphs", "artifacts/phgeofuse/graphs")
    embeddings_root = path(config, "paths.embeddings", "artifacts/phgeofuse/embeddings")
    prepared: list[ProteinRecord] = []
    failures: list[dict[str, str]] = []
    for record in records:
        try:
            structure_file = artifact_path(structures_root, record.sequence_sha256, ".pdb")
            provenance = acquire_structure(
                record.protein_id, record.sequence, structure_file, config, offline
            )
            graph_key = sha256_text(
                "\0".join(
                    [
                        GRAPH_SCHEMA_VERSION,
                        record.sequence_sha256,
                        str(provenance["structure_sha256"]),
                        str(get(config, "graph", {})),
                    ]
                )
            )
            graph_file = artifact_path(graphs_root, graph_key, ".pt")
            graph = build_graph_artifact(structure_file, record.sequence, graph_file, config)
            _, three_di = foldseek_three_di(
                structure_file, record.sequence, graph["plddt"], config
            )
            record.structure_path = str(structure_file)
            record.structure_source = str(provenance["source"])
            record.structure_sha256 = str(provenance["structure_sha256"])
            record.mean_plddt = float(provenance["mean_plddt"])
            record.three_di = three_di
            record.graph_path = str(graph_file)
            record.embedding_path = str(
                embedding_cache_path(embeddings_root, record.sequence, three_di, config)
            )
            record.status = "ready"
            record.error = ""
        except Exception as exc:  # Preserve a complete, resumable failure manifest.
            record.status = "failed"
            record.error = f"{type(exc).__name__}: {exc}"
            failures.append(
                {"protein_id": record.protein_id, "split": record.split, "error": record.error}
            )
        prepared.append(record)
        if manifest_path is not None:
            write_manifest(manifest_path, prepared)
    return prepared, failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare pH-GeoFuse structures and graphs")
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--online", action="store_true")
    mode.add_argument("--offline", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--manifest")
    args = parser.parse_args()

    config = load_config(args.config)
    records = records_from_config(config)
    if args.limit is not None:
        records = records[: args.limit]
    manifest = (
        Path(args.manifest).expanduser().resolve()
        if args.manifest
        else path(config, "paths.manifest", "artifacts/phgeofuse/manifest.csv")
    )
    prepared, failures = prepare_records(
        records, config, offline=args.offline, manifest_path=manifest
    )
    write_manifest(manifest, prepared)
    failure_path = manifest.with_suffix(".failures.json")
    atomic_json(
        failure_path,
        {
            "config_hash": config_hash(config),
            "offline": args.offline,
            "total": len(prepared),
            "ready": sum(record.status == "ready" for record in prepared),
            "failures": failures,
        },
    )
    print(
        f"Prepared {len(prepared) - len(failures)}/{len(prepared)} proteins; "
        f"manifest={manifest}"
    )
    if failures:
        raise SystemExit(
            f"{len(failures)} proteins failed; details were written to {failure_path}"
        )


if __name__ == "__main__":
    main()
