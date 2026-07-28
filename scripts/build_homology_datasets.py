from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = (
    PROJECT_ROOT / "data" / "phopt_training.fasta",
    PROJECT_ROOT / "data" / "phopt_validation.fasta",
    PROJECT_ROOT / "data" / "phopt_testing.fasta",
)
SPLITS = ("train", "validation", "test")
SPLIT_FILENAMES = {
    "train": "phopt_training.fasta",
    "validation": "phopt_validation.fasta",
    "test": "phopt_testing.fasta",
}
DEFAULT_RATIOS = {"train": 0.723, "validation": 0.077, "test": 0.2}
FEATURE_WEIGHTS = {"total": 8.0, "ph": 3.0, "ec": 1.0, "length": 1.0}


@dataclass(frozen=True)
class RawRecord:
    protein_id: str
    organism: str
    ec: str
    ph_opt: float
    sample_weight: float
    sequence: str
    source_split: str
    source_path: str


@dataclass(frozen=True)
class CleanRecord:
    protein_id: str
    organism: str
    ec: str
    ph_opt: float
    sample_weight: float
    sequence: str
    source_ids: tuple[str, ...]
    source_splits: tuple[str, ...]
    original_labels: tuple[float, ...]

    @property
    def sequence_sha256(self) -> str:
        return hashlib.sha256(self.sequence.encode("ascii")).hexdigest()


def normalize_fraction(value: float, name: str) -> float:
    fraction = value / 100.0 if value > 1.0 else value
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"{name} must be in (0, 1] or (0, 100]")
    return fraction


def dataset_name(identity: float) -> str:
    percent = identity * 100.0
    rounded = round(percent)
    if abs(percent - rounded) > 1e-9:
        raise ValueError("identity values must resolve to integer percentages")
    return f"identity{rounded}"


def normalize_sequence(sequence: str) -> str:
    value = "".join(sequence.split()).upper().rstrip("*")
    if not value:
        raise ValueError("protein sequence must not be empty")
    allowed = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")
    invalid = sorted(set(value) - allowed)
    if invalid:
        raise ValueError(f"unsupported amino-acid symbols: {''.join(invalid)}")
    return value


def read_fasta(source: Path, source_split: str) -> list[RawRecord]:
    records: list[RawRecord] = []
    header: str | None = None
    chunks: list[str] = []

    def append_record() -> None:
        if header is None:
            return
        columns = [column.strip() for column in header.lstrip(">").split("|")]
        if len(columns) != 5:
            raise ValueError(f"expected five PHOPT header fields in {source}: {header}")
        records.append(
            RawRecord(
                protein_id=columns[0],
                organism=columns[1],
                ec=columns[2],
                ph_opt=float(columns[3]),
                sample_weight=float(columns[4]),
                sequence=normalize_sequence("".join(chunks)),
                source_split=source_split,
                source_path=str(source),
            )
        )

    with source.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                append_record()
                header, chunks = line, []
            elif header is None:
                raise ValueError(f"sequence appears before a FASTA header in {source}")
            else:
                chunks.append(line)
    append_record()
    return records


def clean_records(
    records: list[RawRecord],
    label_conflict_threshold: float,
) -> tuple[list[CleanRecord], list[dict[str, object]]]:
    by_sequence: dict[str, list[RawRecord]] = defaultdict(list)
    for record in records:
        by_sequence[record.sequence].append(record)

    cleaned: list[CleanRecord] = []
    conflicts: list[dict[str, object]] = []
    for sequence, group in sorted(by_sequence.items(), key=lambda item: item[0]):
        labels = tuple(sorted(record.ph_opt for record in group))
        label_range = max(labels) - min(labels)
        source_ids = tuple(sorted(record.protein_id for record in group))
        source_splits = tuple(sorted({record.source_split for record in group}))
        if label_range > label_conflict_threshold + 1e-12:
            conflicts.append(
                {
                    "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
                    "sequence_length": len(sequence),
                    "source_ids": ";".join(source_ids),
                    "source_splits": ";".join(source_splits),
                    "labels": ";".join(format(value, ".12g") for value in labels),
                    "label_range": label_range,
                    "reason": "label_range_exceeds_threshold",
                }
            )
            continue

        median_label = float(statistics.median(labels))
        representative = min(
            group,
            key=lambda record: (
                abs(record.ph_opt - median_label),
                record.protein_id,
            ),
        )
        cleaned.append(
            CleanRecord(
                protein_id=representative.protein_id,
                organism=representative.organism,
                ec=representative.ec,
                ph_opt=median_label,
                sample_weight=float(
                    statistics.median(record.sample_weight for record in group)
                ),
                sequence=sequence,
                source_ids=source_ids,
                source_splits=source_splits,
                original_labels=labels,
            )
        )

    cleaned.sort(key=lambda record: record.protein_id)
    protein_ids = [record.protein_id for record in cleaned]
    if len(protein_ids) != len(set(protein_ids)):
        raise ValueError("cleaned records contain duplicate representative protein IDs")
    return cleaned, conflicts


def write_fasta(destination: Path, records: list[CleanRecord]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                f">{record.protein_id} | {record.organism} | {record.ec} | "
                f"{format(record.ph_opt, '.12g')} | {format(record.sample_weight, '.12g')}\n"
            )
            handle.write(f"{record.sequence}\n")


def resolve_mmseqs(binary: str | None) -> str:
    candidate = binary or shutil.which("mmseqs")
    if candidate and (Path(candidate).is_file() or shutil.which(candidate)):
        return str(candidate)
    raise FileNotFoundError(
        "MMseqs2 was not found; activate the phbench environment or pass --mmseqs"
    )


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def mmseqs_version(binary: str) -> str:
    result = subprocess.run(
        [binary, "version"], check=True, text=True, capture_output=True
    )
    return (result.stdout or result.stderr).strip()


def build_clusters(
    records: list[CleanRecord],
    identity: float,
    coverage: float,
    coverage_mode: int,
    mmseqs: str,
    work_root: Path,
    threads: int,
) -> dict[str, str]:
    work_root.mkdir(parents=True, exist_ok=True)
    fasta = work_root / "all_unique.fasta"
    write_fasta(fasta, records)
    database = work_root / "sequences"
    alignment_db = work_root / "alignments"
    tmp = work_root / "tmp"
    edges_tsv = work_root / "homology_edges.tsv"
    run_command([mmseqs, "createdb", str(fasta), str(database), "-v", "1"])
    run_command(
        [
            mmseqs,
            "search",
            str(database),
            str(database),
            str(alignment_db),
            str(tmp),
            "--min-seq-id",
            format(identity, ".12g"),
            "-c",
            format(coverage, ".12g"),
            "--cov-mode",
            str(coverage_mode),
            "--alignment-mode",
            "3",
            "--max-seqs",
            str(len(records)),
            "-s",
            "7.5",
            "--threads",
            str(threads),
            "-v",
            "1",
        ]
    )
    run_command(
        [
            mmseqs,
            "convertalis",
            str(database),
            str(database),
            str(alignment_db),
            str(edges_tsv),
            "--format-output",
            "query,target",
            "-v",
            "1",
        ]
    )

    parent = {record.protein_id: record.protein_id for record in records}

    def find(protein_id: str) -> str:
        root = protein_id
        while parent[root] != root:
            root = parent[root]
        while parent[protein_id] != protein_id:
            next_id = parent[protein_id]
            parent[protein_id] = root
            protein_id = next_id
        return root

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        representative = min(left_root, right_root)
        other = right_root if representative == left_root else left_root
        parent[other] = representative

    with edges_tsv.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            query, target = raw_line.rstrip("\n").split("\t")[:2]
            if query not in parent or target not in parent:
                raise ValueError(f"MMseqs2 returned an unknown sequence: {query}, {target}")
            union(query, target)
    return {record.protein_id: find(record.protein_id) for record in records}


def _length_bin(length: int) -> str:
    if length <= 100:
        return "000-100"
    if length <= 200:
        return "101-200"
    if length <= 400:
        return "201-400"
    if length <= 800:
        return "401-800"
    return "801+"


def _record_features(record: CleanRecord, ph_bin_width: float) -> dict[str, str]:
    return {
        "ph": str(math.floor(record.ph_opt / ph_bin_width + 1e-12)),
        "ec": record.ec.split(".", 1)[0] if record.ec else "unknown",
        "length": _length_bin(len(record.sequence)),
    }


def _cluster_features(
    cluster_records: list[CleanRecord], ph_bin_width: float
) -> dict[str, Counter[str]]:
    values = {"total": Counter({"all": len(cluster_records)})}
    for feature in ("ph", "ec", "length"):
        values[feature] = Counter(
            _record_features(record, ph_bin_width)[feature]
            for record in cluster_records
        )
    return values


def split_clusters(
    records: list[CleanRecord],
    memberships: dict[str, str],
    ratios: dict[str, float],
    seed: int,
    attempts: int,
    ph_bin_width: float,
) -> tuple[dict[str, str], float]:
    clusters: dict[str, list[CleanRecord]] = defaultdict(list)
    for record in records:
        clusters[memberships[record.protein_id]].append(record)

    cluster_values = {
        cluster_id: _cluster_features(group, ph_bin_width)
        for cluster_id, group in clusters.items()
    }
    totals = {feature: Counter() for feature in FEATURE_WEIGHTS}
    for values in cluster_values.values():
        for feature, counts in values.items():
            totals[feature].update(counts)

    targets = {
        split: {
            feature: {
                category: count * ratios[split]
                for category, count in counts.items()
            }
            for feature, counts in totals.items()
        }
        for split in SPLITS
    }

    def objective(current: dict[str, dict[str, Counter[str]]]) -> float:
        score = 0.0
        for split in SPLITS:
            for feature, weight in FEATURE_WEIGHTS.items():
                for category, target in targets[split][feature].items():
                    denominator = max(target, 1.0)
                    error = (current[split][feature][category] - target) / denominator
                    score += weight * error * error
        return score

    best_assignment: dict[str, str] | None = None
    best_score = math.inf
    for attempt in range(max(1, attempts)):
        rng = random.Random(seed + attempt)
        ordered = sorted(
            clusters,
            key=lambda cluster_id: (
                -len(clusters[cluster_id]),
                rng.random(),
                cluster_id,
            ),
        )
        current = {
            split: {feature: Counter() for feature in FEATURE_WEIGHTS}
            for split in SPLITS
        }
        assignment: dict[str, str] = {}
        for cluster_id in ordered:
            values = cluster_values[cluster_id]
            candidates = list(SPLITS)
            rng.shuffle(candidates)
            best_split = candidates[0]
            best_delta = math.inf
            for split in candidates:
                delta = 0.0
                for feature, weight in FEATURE_WEIGHTS.items():
                    for category, increment in values[feature].items():
                        target = targets[split][feature][category]
                        denominator = max(target, 1.0)
                        before = (current[split][feature][category] - target) / denominator
                        after = (
                            current[split][feature][category] + increment - target
                        ) / denominator
                        delta += weight * (after * after - before * before)
                if delta < best_delta:
                    best_delta = delta
                    best_split = split
            assignment[cluster_id] = best_split
            for feature, counts in values.items():
                current[best_split][feature].update(counts)

        score = objective(current)
        if score < best_score:
            best_assignment, best_score = assignment, score

    if best_assignment is None:
        raise RuntimeError("failed to assign homology clusters")
    observed = Counter(best_assignment.values())
    if any(observed[split] == 0 for split in SPLITS):
        raise ValueError("cluster assignment produced an empty split")
    return best_assignment, best_score


def audit_splits(
    split_paths: dict[str, Path],
    identity: float,
    coverage: float,
    coverage_mode: int,
    mmseqs: str,
    work_root: Path,
    threads: int,
) -> dict[str, object]:
    work_root.mkdir(parents=True, exist_ok=True)
    violations: list[dict[str, str]] = []
    comparisons = (
        ("validation", "train"),
        ("test", "train"),
        ("test", "validation"),
    )
    for query_split, target_split in comparisons:
        output = work_root / f"audit_{query_split}_vs_{target_split}.tsv"
        tmp = work_root / f"audit_{query_split}_vs_{target_split}_tmp"
        run_command(
            [
                mmseqs,
                "easy-search",
                str(split_paths[query_split]),
                str(split_paths[target_split]),
                str(output),
                str(tmp),
                "--min-seq-id",
                format(identity, ".12g"),
                "-c",
                format(coverage, ".12g"),
                "--cov-mode",
                str(coverage_mode),
                "--alignment-mode",
                "3",
                "--max-seqs",
                "1000",
                "-s",
                "7.5",
                "--threads",
                str(threads),
                "--format-output",
                "query,target,pident,alnlen,qcov,tcov,evalue,bits",
                "-v",
                "1",
            ]
        )
        with output.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                columns = raw_line.rstrip("\n").split("\t")
                violations.append(
                    {
                        "comparison": f"{query_split}_vs_{target_split}",
                        "query": columns[0],
                        "target": columns[1],
                        "pident": columns[2],
                        "qcov": columns[4],
                        "tcov": columns[5],
                    }
                )
                if len(violations) >= 100:
                    break
        if len(violations) >= 100:
            break
    return {
        "passed": not violations,
        "violation_count_capped": len(violations),
        "violations": violations,
    }


def write_csv(destination: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        delimiter = "\t" if destination.suffix == ".tsv" else ","
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter=delimiter,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def build_dataset(
    records: list[CleanRecord],
    conflicts: list[dict[str, object]],
    identity: float,
    coverage: float,
    coverage_mode: int,
    output_root: Path,
    mmseqs: str,
    seed: int,
    attempts: int,
    ph_bin_width: float,
    threads: int,
    label_conflict_threshold: float,
    force: bool,
) -> dict[str, object]:
    name = dataset_name(identity)
    destination = output_root / name
    if destination.exists() and not force:
        raise FileExistsError(f"dataset already exists: {destination}; pass --force")

    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{name}.", dir=output_root) as temporary:
        build_root = Path(temporary) / name
        build_root.mkdir()
        work_root = build_root / "work"
        memberships = build_clusters(
            records,
            identity,
            coverage,
            coverage_mode,
            mmseqs,
            work_root / "cluster",
            threads,
        )
        cluster_assignment, split_score = split_clusters(
            records,
            memberships,
            DEFAULT_RATIOS,
            seed,
            attempts,
            ph_bin_width,
        )
        split_records = {split: [] for split in SPLITS}
        record_splits: dict[str, str] = {}
        for record in records:
            split = cluster_assignment[memberships[record.protein_id]]
            split_records[split].append(record)
            record_splits[record.protein_id] = split
        for split in SPLITS:
            split_records[split].sort(key=lambda record: record.protein_id)

        split_paths = {
            split: build_root / SPLIT_FILENAMES[split] for split in SPLITS
        }
        for split, output in split_paths.items():
            write_fasta(output, split_records[split])

        audit = audit_splits(
            split_paths,
            identity,
            coverage,
            coverage_mode,
            mmseqs,
            work_root / "audit",
            threads,
        )
        if not audit["passed"]:
            raise ValueError(
                f"cross-split homology audit failed for {name}: "
                f"{audit['violations'][:3]}"
            )

        cluster_rows = [
            {
                "cluster_id": cluster_id,
                "protein_id": record.protein_id,
                "split": record_splits[record.protein_id],
            }
            for record in records
            for cluster_id in [memberships[record.protein_id]]
        ]
        cluster_rows.sort(key=lambda row: (row["cluster_id"], row["protein_id"]))
        write_csv(
            build_root / "clusters.tsv",
            cluster_rows,
            ["cluster_id", "protein_id", "split"],
        )

        record_rows = []
        for record in records:
            record_rows.append(
                {
                    "dataset": name,
                    "split": record_splits[record.protein_id],
                    "cluster_id": memberships[record.protein_id],
                    "protein_id": record.protein_id,
                    "sequence_sha256": record.sequence_sha256,
                    "sequence_length": len(record.sequence),
                    "ph_opt": format(record.ph_opt, ".12g"),
                    "sample_weight": format(record.sample_weight, ".12g"),
                    "ec": record.ec,
                    "organism": record.organism,
                    "source_ids": ";".join(record.source_ids),
                    "source_splits": ";".join(record.source_splits),
                    "original_labels": ";".join(
                        format(value, ".12g") for value in record.original_labels
                    ),
                    "label_range": format(
                        max(record.original_labels) - min(record.original_labels), ".12g"
                    ),
                }
            )
        write_csv(
            build_root / "records.tsv",
            record_rows,
            list(record_rows[0]),
        )
        write_csv(
            build_root / "duplicate_conflicts.tsv",
            conflicts,
            [
                "sequence_sha256",
                "sequence_length",
                "source_ids",
                "source_splits",
                "labels",
                "label_range",
                "reason",
            ],
        )

        cluster_sizes = Counter(memberships.values())
        metadata = {
            "dataset": name,
            "identity": identity,
            "identity_percent": identity * 100.0,
            "coverage": coverage,
            "coverage_mode": coverage_mode,
            "cluster_mode": "all_vs_all_search_connected_component",
            "search_sensitivity": 7.5,
            "label_conflict_threshold": label_conflict_threshold,
            "seed": seed,
            "split_attempts": attempts,
            "target_ratios": DEFAULT_RATIOS,
            "split_objective": split_score,
            "counts": {split: len(split_records[split]) for split in SPLITS},
            "total_records": len(records),
            "removed_conflict_groups": len(conflicts),
            "cluster_count": len(cluster_sizes),
            "largest_cluster": max(cluster_sizes.values()),
            "mmseqs_version": mmseqs_version(mmseqs),
            "audit": audit,
        }
        (build_root / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        shutil.rmtree(work_root, ignore_errors=True)
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(build_root, destination)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build strict PHOPT homology-controlled train/validation/test splits"
    )
    parser.add_argument(
        "--identity",
        type=float,
        nargs="+",
        default=[100.0, 50.0, 30.0, 20.0],
        help="One or more identity thresholds as fractions or percentages",
    )
    parser.add_argument(
        "--coverage",
        type=float,
        default=80.0,
        help="Alignment coverage threshold as a fraction or percentage",
    )
    parser.add_argument("--coverage-mode", type=int, default=0)
    parser.add_argument("--label-conflict-threshold", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-attempts", type=int, default=32)
    parser.add_argument("--ph-bin-width", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--mmseqs")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "datasets",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    identities = [normalize_fraction(value, "identity") for value in args.identity]
    if len(identities) != len(set(identities)):
        raise ValueError("identity thresholds must be unique")
    coverage = normalize_fraction(args.coverage, "coverage")
    if args.label_conflict_threshold < 0:
        raise ValueError("label conflict threshold must be non-negative")
    if args.split_attempts < 1:
        raise ValueError("split attempts must be positive")
    if args.ph_bin_width <= 0:
        raise ValueError("pH bin width must be positive")
    mmseqs = resolve_mmseqs(args.mmseqs)

    raw_records: list[RawRecord] = []
    for source, split in zip(DEFAULT_INPUTS, SPLITS, strict=True):
        raw_records.extend(read_fasta(source, split))
    cleaned, conflicts = clean_records(raw_records, args.label_conflict_threshold)
    print(
        f"Loaded {len(raw_records)} records; retained {len(cleaned)} unique sequences; "
        f"removed {len(conflicts)} conflicting sequence groups"
    )

    for identity in identities:
        metadata = build_dataset(
            cleaned,
            conflicts,
            identity,
            coverage,
            args.coverage_mode,
            args.output_root.resolve(),
            mmseqs,
            args.seed,
            args.split_attempts,
            args.ph_bin_width,
            args.threads,
            args.label_conflict_threshold,
            args.force,
        )
        print(
            f"Built {metadata['dataset']}: counts={metadata['counts']}, "
            f"clusters={metadata['cluster_count']}, audit=passed"
        )


if __name__ == "__main__":
    main()
