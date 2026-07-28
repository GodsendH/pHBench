from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import tempfile
import urllib.request
from pathlib import Path


SOURCE_URL = (
    "https://zenodo.org/api/records/14252615/files/pHopt_data.csv/content"
)
SOURCE_MD5 = "d5d0887710e70f2aa5153c15178a6d08"
EXPECTED_COUNT = 999


def _metadata_bytes(source: str | None) -> bytes:
    if source:
        return Path(source).expanduser().read_bytes()
    with urllib.request.urlopen(SOURCE_URL, timeout=120) as response:
        return response.read()


def _official_sequences(payload: bytes) -> dict[str, str]:
    digest = hashlib.md5(payload).hexdigest()
    if digest != SOURCE_MD5:
        raise ValueError(
            f"unexpected EpHod metadata checksum: expected {SOURCE_MD5}, got {digest}"
        )
    rows = csv.DictReader(io.StringIO(payload.decode("utf-8")))
    selected = {
        row["Accession"].strip(): row["Sequence"].strip().upper()
        for row in rows
        if row["Split"].strip() == "Testing"
        and row["Test <20% to Train"].strip() == "True"
    }
    if len(selected) != EXPECTED_COUNT:
        raise ValueError(
            f"expected {EXPECTED_COUNT} official low-identity records, got {len(selected)}"
        )
    return selected


def _fasta_entries(source: Path):
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


def build_subset(metadata: bytes, test_fasta: Path, destination: Path) -> None:
    official = _official_sequences(metadata)
    selected: list[tuple[str, str]] = []
    for header, sequence in _fasta_entries(test_fasta):
        protein_id = header.lstrip(">").split("|", 1)[0].strip()
        expected = official.get(protein_id)
        if expected is None:
            continue
        if sequence.upper() != expected:
            raise ValueError(f"sequence mismatch for official EpHod record {protein_id}")
        selected.append((header, sequence))

    observed = {header.lstrip(">").split("|", 1)[0].strip() for header, _ in selected}
    missing = sorted(set(official) - observed)
    if missing or len(selected) != EXPECTED_COUNT:
        raise ValueError(
            f"local test FASTA does not match the official subset: "
            f"selected={len(selected)}, missing={missing[:10]}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for header, sequence in selected:
                handle.write(f"{header}\n{sequence}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Build the official EpHod <20% identity pHopt test subset"
    )
    parser.add_argument(
        "--metadata",
        help="Downloaded official pHopt_data.csv; omitted to download from Zenodo",
    )
    parser.add_argument(
        "--test-fasta",
        type=Path,
        default=project_root / "data" / "phopt_testing.fasta",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "data" / "phopt_testing_low_identity.fasta",
    )
    args = parser.parse_args()
    build_subset(_metadata_bytes(args.metadata), args.test_fasta, args.output)
    print(f"Wrote {EXPECTED_COUNT} records to {args.output}")


if __name__ == "__main__":
    main()
