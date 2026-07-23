from __future__ import annotations

import csv
import math
import os
import re
import uuid
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, Iterator

from .cache import sha256_text
from .config import get, path


VALID_AA = frozenset("ACDEFGHIKLMNPQRSTVWYBXZJUO")


@dataclass
class ProteinRecord:
    protein_id: str
    sequence: str
    split: str
    ph_opt: float = math.nan
    ec: str = ""
    organism: str = ""
    sample_weight: float = 1.0
    sequence_sha256: str = ""
    structure_path: str = ""
    structure_source: str = ""
    structure_sha256: str = ""
    mean_plddt: float = math.nan
    three_di: str = ""
    graph_path: str = ""
    embedding_path: str = ""
    status: str = "pending"
    error: str = ""

    def __post_init__(self) -> None:
        self.sequence = normalize_sequence(self.sequence)
        if not self.sequence_sha256:
            self.sequence_sha256 = sha256_text(self.sequence)


def normalize_sequence(sequence: str) -> str:
    value = re.sub(r"\s+", "", sequence).upper()
    if not value:
        raise ValueError("protein sequence must not be empty")
    invalid = sorted(set(value) - VALID_AA)
    if invalid:
        raise ValueError(f"unsupported amino-acid symbols: {''.join(invalid)}")
    return value


def parse_phopt_header(header: str) -> tuple[str, str, str, float, float]:
    columns = [column.strip() for column in header.lstrip(">").split("|")]
    if len(columns) != 5:
        raise ValueError(f"expected five pipe-separated PHOPT header fields: {header}")
    return columns[0], columns[1], columns[2], float(columns[3]), float(columns[4])


def read_fasta(path: str | Path, split: str, require_labels: bool = True) -> list[ProteinRecord]:
    records: list[ProteinRecord] = []
    for header, sequence in iter_fasta(path):
        if require_labels:
            protein_id, organism, ec, ph_opt, weight = parse_phopt_header(header)
        else:
            protein_id = header.lstrip(">").split()[0].split("|")[0]
            organism, ec, ph_opt, weight = "", "", math.nan, 1.0
        records.append(
            ProteinRecord(
                protein_id=protein_id,
                sequence=sequence,
                split=split,
                ph_opt=ph_opt,
                ec=ec,
                organism=organism,
                sample_weight=weight,
            )
        )
    _validate_unique(records)
    return records


def iter_fasta(path: str | Path) -> Iterator[tuple[str, str]]:
    source = Path(path)
    header: str | None = None
    chunks: list[str] = []
    with source.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header, chunks = line, []
            elif header is None:
                raise ValueError(f"FASTA sequence appears before a header in {source}")
            else:
                chunks.append(line)
    if header is not None:
        yield header, "".join(chunks)


def records_from_config(config: dict, require_labels: bool = True) -> list[ProteinRecord]:
    manifest = get(config, "data.manifest")
    if manifest:
        return read_manifest(path(config, "data.manifest"))
    records: list[ProteinRecord] = []
    split_paths = get(config, "data.splits", {})
    for split in ("train", "validation", "test"):
        source = split_paths.get(split)
        if source:
            source_path = Path(source)
            if not source_path.is_absolute():
                source_path = Path(config["_root"]) / source_path
            records.extend(read_fasta(source_path, split, require_labels=require_labels))
    _validate_unique(records)
    return records


def read_manifest(path: str | Path) -> list[ProteinRecord]:
    source = Path(path)
    with source.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    known = {field.name for field in fields(ProteinRecord)}
    records: list[ProteinRecord] = []
    for row in rows:
        unknown = set(row) - known
        if unknown:
            raise ValueError(f"unsupported manifest columns: {sorted(unknown)}")
        values = {key: value for key, value in row.items() if key in known}
        for name in ("ph_opt", "sample_weight", "mean_plddt"):
            if name not in values:
                continue
            raw = values[name]
            values[name] = float(raw) if raw not in (None, "") else math.nan
        records.append(ProteinRecord(**values))
    _validate_unique(records)
    return records


def write_manifest(path: str | Path, records: Iterable[ProteinRecord]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    names = [field.name for field in fields(ProteinRecord)]
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=names)
            writer.writeheader()
            for record in records:
                writer.writerow(asdict(record))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_unique(records: Iterable[ProteinRecord]) -> None:
    seen: set[tuple[str, str]] = set()
    for record in records:
        key = (record.split, record.protein_id)
        if key in seen:
            raise ValueError(f"duplicate protein ID within split: {key}")
        seen.add(key)
