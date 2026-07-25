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
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


DEFAULT_THRESHOLDS = (100, 90, 70, 50, 30)
SPLITS = ('train', 'valid', 'test')
FEATURE_WEIGHTS = {'ph': 1.0, 'ec': 0.5, 'length': 0.25}


@dataclass(frozen=True)
class RawRecord:
    source_split: str
    source_id: str
    header: str
    sequence: str
    ph: float
    ec: str


@dataclass(frozen=True)
class Record:
    key: str
    source_id: str
    source_ids: Tuple[str, ...]
    source_splits: Tuple[str, ...]
    header: str
    sequence: str
    ph: float
    ec: str


class UnionFind:
    def __init__(self, values: Iterable[str]):
        self.parent = {value: value for value in values}
        self.rank = {value: 0 for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def parse_fasta(path: Path, source_split: str) -> List[RawRecord]:
    records = []
    header = None
    chunks = []

    def emit() -> None:
        if header is None:
            return
        parts = [part.strip() for part in header.split(' | ')]
        if len(parts) != 5:
            raise ValueError(
                f'Expected a five-field FASTA header in {path}, got: {header}'
            )
        sequence = ''.join(chunks).upper()
        if not sequence:
            raise ValueError(f'Empty sequence in {path}: {header}')
        try:
            ph = float(parts[3])
        except ValueError as exc:
            raise ValueError(f'Invalid pH value in {path}: {header}') from exc
        records.append(
            RawRecord(
                source_split=source_split,
                source_id=parts[0],
                header=header,
                sequence=sequence,
                ph=ph,
                ec=parts[2],
            )
        )

    with path.open('r', encoding='utf-8') as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith('>'):
                emit()
                header = line[1:]
                chunks = []
            else:
                if header is None:
                    raise ValueError(f'Sequence before first header in {path}')
                chunks.append(line)
    emit()
    if not records:
        raise ValueError(f'No FASTA records found in {path}')
    return records


def sequence_digest(sequence: str) -> str:
    return hashlib.sha256(sequence.encode('ascii')).hexdigest()


def clean_records(
    records: Sequence[RawRecord],
    conflict_tolerance: float = 0.0,
) -> Tuple[List[Record], List[dict], dict]:
    grouped = defaultdict(list)
    for record in records:
        grouped[record.sequence].append(record)

    kept_groups = []
    removed = []
    duplicate_extra_records = 0
    same_label_duplicate_groups = 0
    for sequence, entries in grouped.items():
        ph_values = sorted({entry.ph for entry in entries})
        if ph_values[-1] - ph_values[0] > conflict_tolerance:
            removed.append(
                {
                    'sequence_sha256': sequence_digest(sequence),
                    'source_ids': ','.join(
                        sorted(entry.source_id for entry in entries)
                    ),
                    'source_splits': ','.join(
                        sorted({entry.source_split for entry in entries})
                    ),
                    'ph_values': ','.join(f'{value:g}' for value in ph_values),
                    'record_count': len(entries),
                    'reason': 'conflicting_ph_labels',
                }
            )
            continue

        representative = min(
            entries,
            key=lambda entry: (entry.source_id, entry.header),
        )
        if len(entries) > 1:
            same_label_duplicate_groups += 1
            duplicate_extra_records += len(entries) - 1
        kept_groups.append((sequence, entries, representative))

    kept_groups.sort(key=lambda item: sequence_digest(item[0]))
    cleaned = []
    for index, (sequence, entries, representative) in enumerate(kept_groups):
        cleaned.append(
            Record(
                key=f'S{index:05d}',
                source_id=representative.source_id,
                source_ids=tuple(sorted(entry.source_id for entry in entries)),
                source_splits=tuple(
                    sorted({entry.source_split for entry in entries})
                ),
                header=representative.header,
                sequence=sequence,
                ph=representative.ph,
                ec=representative.ec,
            )
        )

    stats = {
        'input_records': len(records),
        'input_unique_sequences': len(grouped),
        'cleaned_sequences': len(cleaned),
        'conflicting_groups_removed': len(removed),
        'conflicting_records_removed': sum(
            row['record_count'] for row in removed
        ),
        'same_label_duplicate_groups': same_label_duplicate_groups,
        'same_label_duplicate_extra_records_removed': duplicate_extra_records,
    }
    return cleaned, removed, stats


def write_fasta(path: Path, records: Iterable[Record], synthetic_ids=False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as handle:
        for record in records:
            header = record.key if synthetic_ids else record.header
            handle.write(f'>{header}\n')
            for start in range(0, len(record.sequence), 80):
                handle.write(f'{record.sequence[start:start + 80]}\n')


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def mmseqs_version(binary: str) -> str:
    try:
        result = subprocess.run(
            [binary, 'version'],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f'MMseqs2 executable not found: {binary}. Install MMseqs2 or pass '
            '--mmseqs with its full path.'
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or '').strip()
        raise RuntimeError(f'Failed to run MMseqs2: {detail}') from exc
    return (result.stdout or result.stderr).strip().splitlines()[0]


def run_all_vs_all_search(
    binary: str,
    fasta_path: Path,
    result_path: Path,
    temp_path: Path,
    threshold: int,
    coverage: float,
    sensitivity: float,
    threads: int,
) -> List[str]:
    command = [
        binary,
        'easy-search',
        str(fasta_path),
        str(fasta_path),
        str(result_path),
        str(temp_path),
        '--min-seq-id',
        f'{threshold / 100.0:.6f}',
        '-c',
        f'{coverage:.6f}',
        '--cov-mode',
        '0',
        '--alignment-mode',
        '3',
        '--max-seqs',
        '10000',
        '-s',
        f'{sensitivity:g}',
        '--format-output',
        'query,target,fident,qcov,tcov',
        '--threads',
        str(threads),
        '-v',
        '1',
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f'MMseqs2 search for identity{threshold} failed with exit code '
            f'{exc.returncode}'
        ) from exc
    return command


def build_components(
    record_keys: Sequence[str],
    edge_path: Path = None,
) -> Tuple[List[List[str]], int]:
    union_find = UnionFind(record_keys)
    alignment_rows = 0
    known_keys = set(record_keys)
    if edge_path is not None:
        with edge_path.open('r', encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, start=1):
                fields = line.rstrip('\n').split('\t')
                if len(fields) != 5:
                    raise ValueError(
                        f'Expected 5 MMseqs2 columns at {edge_path}:'
                        f'{line_number}, got {len(fields)}'
                    )
                query, target = fields[:2]
                if query not in known_keys or target not in known_keys:
                    raise ValueError(
                        f'Unknown sequence ID at {edge_path}:{line_number}'
                    )
                alignment_rows += 1
                union_find.union(query, target)

    grouped = defaultdict(list)
    for key in record_keys:
        grouped[union_find.find(key)].append(key)
    components = [sorted(members) for members in grouped.values()]
    components.sort(key=lambda members: (members[0], len(members)))
    return components, alignment_rows


def feature_counter(record: Record, ph_bin_width: float) -> Counter:
    length = len(record.sequence)
    if length < 250:
        length_bin = 'lt250'
    elif length < 500:
        length_bin = '250_499'
    elif length < 750:
        length_bin = '500_749'
    else:
        length_bin = 'ge750'
    return Counter(
        {
            f'ph:{math.floor(record.ph / ph_bin_width)}': 1,
            f'ec:{record.ec.split(".")[0]}': 1,
            f'length:{length_bin}': 1,
        }
    )


def feature_weight(feature: str) -> float:
    return FEATURE_WEIGHTS[feature.split(':', 1)[0]]


def make_clusters(
    components: Sequence[Sequence[str]],
    records_by_key: Mapping[str, Record],
    ph_bin_width: float,
) -> List[dict]:
    clusters = []
    for index, members in enumerate(components):
        features = Counter()
        for key in members:
            features.update(feature_counter(records_by_key[key], ph_bin_width))
        clusters.append(
            {
                'cluster_id': f'C{index:05d}',
                'members': tuple(members),
                'size': len(members),
                'features': features,
            }
        )
    return clusters


def assignment_score(
    counts: Mapping[str, int],
    feature_counts: Mapping[str, Counter],
    target_counts: Mapping[str, float],
    target_features: Mapping[str, Mapping[str, float]],
) -> float:
    score = 0.0
    for split in SPLITS:
        score += (
            (counts[split] - target_counts[split]) ** 2
            / max(target_counts[split], 1.0)
        )
        for feature, target in target_features[split].items():
            score += feature_weight(feature) * (
                (feature_counts[split].get(feature, 0) - target) ** 2
                / max(target, 1.0)
            )
    return score


def split_clusters(
    clusters: Sequence[dict],
    records_by_key: Mapping[str, Record],
    split_ratios: Mapping[str, float],
    ph_bin_width: float,
    attempts: int,
    seed: int,
) -> Tuple[Dict[str, str], dict]:
    total_records = len(records_by_key)
    global_features = Counter()
    for record in records_by_key.values():
        global_features.update(feature_counter(record, ph_bin_width))
    target_counts = {
        split: split_ratios[split] * total_records for split in SPLITS
    }
    target_features = {
        split: {
            feature: split_ratios[split] * count
            for feature, count in global_features.items()
        }
        for split in SPLITS
    }

    best = None
    for attempt in range(attempts):
        rng = random.Random(seed + attempt)
        ordered = list(clusters)
        rng.shuffle(ordered)
        ordered.sort(key=lambda cluster: cluster['size'], reverse=True)
        counts = {split: 0 for split in SPLITS}
        feature_counts = {split: Counter() for split in SPLITS}
        assignment = {}

        for cluster in ordered:
            choices = []
            for split in SPLITS:
                count = counts[split]
                target_count = target_counts[split]
                delta = (
                    ((count + cluster['size'] - target_count) ** 2)
                    - ((count - target_count) ** 2)
                ) / max(target_count, 1.0)
                for feature, amount in cluster['features'].items():
                    observed = feature_counts[split].get(feature, 0)
                    target = target_features[split][feature]
                    delta += feature_weight(feature) * (
                        ((observed + amount - target) ** 2)
                        - ((observed - target) ** 2)
                    ) / max(target, 1.0)
                overshoot = max(
                    0.0,
                    count + cluster['size'] - target_count,
                )
                choices.append((delta + 2.0 * overshoot, rng.random(), split))

            chosen_split = min(choices)[2]
            assignment[cluster['cluster_id']] = chosen_split
            counts[chosen_split] += cluster['size']
            feature_counts[chosen_split].update(cluster['features'])

        score = assignment_score(
            counts,
            feature_counts,
            target_counts,
            target_features,
        )
        candidate = (score, attempt, assignment, counts)
        if best is None or candidate[:2] < best[:2]:
            best = candidate

    score, attempt, assignment, counts = best
    details = {
        'score': score,
        'selected_attempt': attempt,
        'selected_seed': seed + attempt,
        'target_counts': target_counts,
        'actual_counts': counts,
    }
    return assignment, details


def label_stats(records: Sequence[Record]) -> dict:
    ph_values = [record.ph for record in records]
    lengths = [len(record.sequence) for record in records]
    ec_counts = Counter(record.ec.split('.')[0] for record in records)
    return {
        'count': len(records),
        'ph_mean': statistics.fmean(ph_values),
        'ph_population_sd': statistics.pstdev(ph_values),
        'ph_min': min(ph_values),
        'ph_max': max(ph_values),
        'length_mean': statistics.fmean(lengths),
        'length_median': statistics.median(lengths),
        'length_max': max(lengths),
        'ec_class_counts': dict(sorted(ec_counts.items())),
    }


def write_cleaning_outputs(
    root: Path,
    records: Sequence[Record],
    removed: Sequence[dict],
) -> None:
    with (root / 'cleaned_metadata.tsv').open(
        'w', encoding='utf-8', newline=''
    ) as handle:
        fieldnames = (
            'sequence_id',
            'representative_id',
            'source_ids',
            'source_splits',
            'sequence_sha256',
            'ph',
            'ec',
            'length',
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    'sequence_id': record.key,
                    'representative_id': record.source_id,
                    'source_ids': ','.join(record.source_ids),
                    'source_splits': ','.join(record.source_splits),
                    'sequence_sha256': sequence_digest(record.sequence),
                    'ph': f'{record.ph:g}',
                    'ec': record.ec,
                    'length': len(record.sequence),
                }
            )

    with (root / 'removed_conflicts.tsv').open(
        'w', encoding='utf-8', newline=''
    ) as handle:
        fieldnames = (
            'sequence_sha256',
            'source_ids',
            'source_splits',
            'ph_values',
            'record_count',
            'reason',
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerows(removed)


def write_level(
    level_dir: Path,
    threshold: int,
    clusters: Sequence[dict],
    assignment: Mapping[str, str],
    records_by_key: Mapping[str, Record],
    alignment_rows: int,
    split_details: dict,
    source_settings: dict,
) -> dict:
    level_dir.mkdir(parents=True)
    split_records = {split: [] for split in SPLITS}
    split_by_key = {}
    cluster_by_key = {}
    for cluster in clusters:
        split = assignment[cluster['cluster_id']]
        for key in cluster['members']:
            split_records[split].append(records_by_key[key])
            split_by_key[key] = split
            cluster_by_key[key] = cluster['cluster_id']
    for split in SPLITS:
        split_records[split].sort(key=lambda record: record.key)
        write_fasta(level_dir / f'{split}.fasta', split_records[split])

    with (level_dir / 'assignments.tsv').open(
        'w', encoding='utf-8', newline=''
    ) as handle:
        fieldnames = (
            'sequence_id',
            'source_id',
            'source_splits',
            'cluster_id',
            'split',
            'ph',
            'ec',
            'length',
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for key in sorted(records_by_key):
            record = records_by_key[key]
            writer.writerow(
                {
                    'sequence_id': key,
                    'source_id': record.source_id,
                    'source_splits': ','.join(record.source_splits),
                    'cluster_id': cluster_by_key[key],
                    'split': split_by_key[key],
                    'ph': f'{record.ph:g}',
                    'ec': record.ec,
                    'length': len(record.sequence),
                }
            )

    with (level_dir / 'clusters.tsv').open(
        'w', encoding='utf-8', newline=''
    ) as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(('cluster_id', 'sequence_id'))
        for cluster in clusters:
            for key in cluster['members']:
                writer.writerow((cluster['cluster_id'], key))

    cross_split_edges = 0
    edge_path = source_settings.get('edge_path')
    if edge_path is not None:
        with Path(edge_path).open('r', encoding='utf-8') as handle:
            for line in handle:
                query, target = line.split('\t', 2)[:2]
                if split_by_key[query] != split_by_key[target]:
                    cross_split_edges += 1
    if cross_split_edges:
        raise RuntimeError(
            f'identity{threshold} has {cross_split_edges} cross-split homology '
            'edges; the connected-component assignment is invalid.'
        )

    cluster_sizes = sorted(
        (cluster['size'] for cluster in clusters),
        reverse=True,
    )
    manifest = {
        'dataset': f'homology{threshold}',
        'identity_threshold_percent': threshold,
        'coverage': source_settings['coverage'],
        'coverage_mode': 0,
        'sensitivity': source_settings['sensitivity'],
        'alignment_mode': 3,
        'max_seqs': 10000,
        'method': 'MMseqs2 all-vs-all edges plus connected components',
        'alignment_rows': alignment_rows,
        'cross_split_edges_at_threshold': cross_split_edges,
        'cluster_statistics': {
            'count': len(clusters),
            'singletons': sum(size == 1 for size in cluster_sizes),
            'max_size': max(cluster_sizes),
            'median_size': statistics.median(cluster_sizes),
        },
        'split_optimization': split_details,
        'splits': {
            split: label_stats(split_records[split]) for split in SPLITS
        },
        'files': {},
    }
    for filename in (
        'train.fasta',
        'valid.fasta',
        'test.fasta',
        'assignments.tsv',
        'clusters.tsv',
    ):
        path = level_dir / filename
        manifest['files'][filename] = file_sha256(path)
    with (level_dir / 'manifest.json').open('w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write('\n')
    return manifest


def validate_thresholds(values: Sequence[int]) -> Tuple[int, ...]:
    thresholds = tuple(sorted(set(values), reverse=True))
    if not thresholds or any(value <= 0 or value > 100 for value in thresholds):
        raise ValueError('Identity thresholds must be integers in 1..100')
    return thresholds


def default_threads() -> int:
    for name in ('SLURM_CPUS_PER_TASK', 'OMP_NUM_THREADS'):
        value = os.environ.get(name)
        if value and value.isdigit() and int(value) > 0:
            return int(value)
    return 1


def build_ladder(args: argparse.Namespace) -> Path:
    thresholds = validate_thresholds(args.thresholds)
    if not 0.0 < args.coverage <= 1.0:
        raise ValueError('--coverage must be in (0, 1]')
    if args.ph_bin_width <= 0:
        raise ValueError('--ph-bin-width must be positive')
    if args.split_attempts < 1 or args.threads < 1:
        raise ValueError('--split-attempts and --threads must be positive')

    source_paths = {
        'train': Path(args.train),
        'valid': Path(args.valid),
        'test': Path(args.test),
    }
    raw_by_split = {
        split: parse_fasta(path, split) for split, path in source_paths.items()
    }
    raw_records = [
        record
        for split in SPLITS
        for record in raw_by_split[split]
    ]
    cleaned, removed, cleaning_stats = clean_records(
        raw_records,
        conflict_tolerance=args.conflict_tolerance,
    )
    records_by_key = {record.key: record for record in cleaned}
    original_total = sum(len(raw_by_split[split]) for split in SPLITS)
    split_ratios = {
        split: len(raw_by_split[split]) / original_total for split in SPLITS
    }

    output_root = Path(args.output_root)
    if output_root.exists():
        raise FileExistsError(
            f'Output path already exists: {output_root}. Choose a new path so '
            'an existing benchmark is not overwritten.'
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    build_root = Path(
        tempfile.mkdtemp(
            prefix=f'.{output_root.name}.building-',
            dir=str(output_root.parent),
        )
    )

    try:
        write_cleaning_outputs(build_root, cleaned, removed)
        work_dir = build_root / '.work'
        work_dir.mkdir()
        cleaned_fasta = work_dir / 'cleaned.fasta'
        write_fasta(cleaned_fasta, cleaned, synthetic_ids=True)

        needs_mmseqs = any(threshold < 100 for threshold in thresholds)
        version = mmseqs_version(args.mmseqs) if needs_mmseqs else None
        level_manifests = {}
        for threshold in thresholds:
            print(
                f'Building identity{threshold} from {len(cleaned)} cleaned '
                'sequences...',
                flush=True,
            )
            edge_path = None
            command = None
            if threshold < 100:
                edge_path = work_dir / f'identity{threshold}_edges.tsv'
                command = run_all_vs_all_search(
                    binary=args.mmseqs,
                    fasta_path=cleaned_fasta,
                    result_path=edge_path,
                    temp_path=work_dir / f'tmp_identity{threshold}',
                    threshold=threshold,
                    coverage=args.coverage,
                    sensitivity=args.sensitivity,
                    threads=args.threads,
                )
            components, alignment_rows = build_components(
                list(records_by_key),
                edge_path=edge_path,
            )
            clusters = make_clusters(
                components,
                records_by_key,
                ph_bin_width=args.ph_bin_width,
            )
            assignment, split_details = split_clusters(
                clusters,
                records_by_key,
                split_ratios=split_ratios,
                ph_bin_width=args.ph_bin_width,
                attempts=args.split_attempts,
                seed=args.seed,
            )
            level_manifest = write_level(
                build_root / f'identity{threshold}',
                threshold,
                clusters,
                assignment,
                records_by_key,
                alignment_rows,
                split_details,
                source_settings={
                    'coverage': args.coverage,
                    'sensitivity': args.sensitivity,
                    'edge_path': str(edge_path) if edge_path else None,
                },
            )
            level_manifest['mmseqs_command'] = command
            level_manifests[f'homology{threshold}'] = level_manifest
            counts = level_manifest['split_optimization']['actual_counts']
            print(
                f'identity{threshold}: clusters={len(clusters)}, '
                f'train={counts["train"]}, valid={counts["valid"]}, '
                f'test={counts["test"]}',
                flush=True,
            )

        shutil.rmtree(work_dir)
        root_manifest = {
            'schema_version': 1,
            'method': (
                'global exact-sequence cleaning followed by threshold-specific '
                'MMseqs2 connected-component group splitting'
            ),
            'thresholds_percent': list(thresholds),
            'coverage': args.coverage,
            'ph_bin_width': args.ph_bin_width,
            'split_attempts': args.split_attempts,
            'seed': args.seed,
            'split_ratios': split_ratios,
            'mmseqs_version': version,
            'cleaning': cleaning_stats,
            'source_files': {
                split: {
                    'path': str(path.resolve()),
                    'sha256': file_sha256(path),
                    'record_count': len(raw_by_split[split]),
                }
                for split, path in source_paths.items()
            },
            'levels': level_manifests,
            'training_warning': (
                'Do not use the EpHod supervised RLAT checkpoint pretrained on '
                'the original phopt split. Omit --pretrained for these datasets.'
            ),
            'command': sys.argv,
        }
        with (build_root / 'manifest.json').open('w', encoding='utf-8') as handle:
            json.dump(root_manifest, handle, indent=2, sort_keys=True)
            handle.write('\n')
        os.replace(build_root, output_root)
        return output_root
    finally:
        if build_root.exists():
            shutil.rmtree(build_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Build progressively lower-homology pH benchmark datasets.'
    )
    parser.add_argument('--train', default='data/phopt_training.fasta')
    parser.add_argument('--valid', default='data/phopt_validation.fasta')
    parser.add_argument('--test', default='data/phopt_testing.fasta')
    parser.add_argument('--output-root', default='homology_data')
    parser.add_argument(
        '--thresholds',
        nargs='+',
        type=int,
        default=list(DEFAULT_THRESHOLDS),
    )
    parser.add_argument('--coverage', type=float, default=0.8)
    parser.add_argument('--sensitivity', type=float, default=7.5)
    parser.add_argument('--ph-bin-width', type=float, default=0.5)
    parser.add_argument('--split-attempts', type=int, default=64)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--threads', type=int, default=default_threads())
    parser.add_argument('--mmseqs', default='mmseqs')
    parser.add_argument('--conflict-tolerance', type=float, default=0.0)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        output = build_ladder(args)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(f'Homology ladder written to {output}')


if __name__ == '__main__':
    main()
