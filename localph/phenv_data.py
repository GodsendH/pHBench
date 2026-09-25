"""Auditable pHenv staging; labels remain distinct from enzyme optimum pH.

This stage records every source row and prepares sequence-only search inputs.
It does NOT certify homology isolation or emit a training-ready split.
"""
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time

REQUIRED = {'Accession', 'Organism', 'Sample Weight', 'pHenv', 'Split', 'Sequence'}
ALPHABET = set('ACDEFGHIKLMNPQRSTVWYBXZJUO')
TRANSLATE = str.maketrans({c: 'X' for c in 'BJOUZ'})


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def sequence_hash(sequence):
    return hashlib.sha256(sequence.encode('utf-8')).hexdigest()


def parse_row(row):
    """Preserve sequence and label; flag unsupported rows instead of truncation."""
    issues = []
    seq = row.get('Sequence') or ''
    if not seq or set(seq) - ALPHABET:
        issues.append('invalid_sequence')
    if not 32 <= len(seq) <= 1022:
        issues.append('length_outside_32_1022')
    if row.get('Split') not in ('Training', 'Validation'):
        issues.append('invalid_split')
    if not (row.get('Organism') or '').strip():
        issues.append('missing_organism')
    if not (row.get('Accession') or '').strip():
        issues.append('missing_accession')
    values = []
    for key in ('pHenv', 'Sample Weight'):
        try:
            v = float(row[key])
            if not math.isfinite(v) or (key == 'pHenv' and not 0 <= v <= 14) or (key == 'Sample Weight' and v <= 0):
                raise ValueError()
        except (ValueError, TypeError, KeyError):
            issues.append('invalid_' + key.replace(' ', '_'))
            v = None
        values.append(v)
    return seq, seq.translate(TRANSLATE), values[0], values[1], issues


def audit(source, phopt_manifest, output):
    source, phopt_manifest, output = map(Path, (source, phopt_manifest, output))
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    source_stat = source.stat()
    source_sha, phopt_sha = sha_file(source), sha_file(phopt_manifest)
    db_path = output / 'manifest.sqlite'
    db = sqlite3.connect(db_path)
    db.execute('PRAGMA cache_size=-65536')
    db.executescript('''
      CREATE TABLE records (
        row_id INTEGER PRIMARY KEY, source_index TEXT, accession TEXT, organism TEXT,
        published_split TEXT, phenv REAL, published_weight REAL,
        raw_sha TEXT, normalized_sha TEXT, sequence TEXT, length INTEGER, issues TEXT);
      CREATE TABLE phopt (sample_key TEXT PRIMARY KEY, split TEXT, raw_sha TEXT, normalized_sha TEXT);
    ''')
    # Only these four sequence/identifier fields are used. No PHOPT label is parsed.
    phopt_rows = []
    with phopt_manifest.open(newline='', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            seq = r['sequence']
            phopt_rows.append((r['split'] + '::' + r['protein_id'], r['split'], sequence_hash(seq),
                               sequence_hash(seq.translate(TRANSLATE))))
    db.executemany('INSERT INTO phopt VALUES (?,?,?,?)', phopt_rows)
    issues_count, chars, count = Counter(), Counter(), 0
    pending = []
    with source.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not REQUIRED <= set(reader.fieldnames or []):
            raise ValueError('pHenv source lacks required columns')
        fields = reader.fieldnames
        for count, row in enumerate(reader, 1):
            if None in row:
                raise ValueError(f'extra CSV fields on source row {count}')
            seq, normalized, y, weight, issues = parse_row(row)
            issues_count.update(issues)
            chars.update(seq)
            pending.append((count, row.get(''), row.get('Accession'), row.get('Organism'), row.get('Split'),
                            y, weight, sequence_hash(seq), sequence_hash(normalized), seq, len(seq), '|'.join(issues)))
            if len(pending) == 10000:
                db.executemany('INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?,?,?)', pending)
                db.commit()
                pending.clear()
            if count % 200000 == 0:
                status = {'state': 'audit_ingest', 'rows': count, 'seconds': time.monotonic() - start}
                write_json(output / 'status.json', status)
                print(json.dumps(status), flush=True)
        if pending:
            db.executemany('INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?,?,?)', pending)
            db.commit()
    print(json.dumps({'state': 'indexing', 'rows': count}), flush=True)
    db.executescript('''
      CREATE INDEX accession_idx ON records(accession);
      CREATE INDEX norm_idx ON records(normalized_sha);
      CREATE INDEX organism_idx ON records(organism);
      CREATE INDEX phopt_norm_idx ON phopt(normalized_sha);
      CREATE TABLE sequence_groups AS SELECT normalized_sha, MIN(row_id) AS representative,
        COUNT(*) AS rows, COUNT(DISTINCT published_split) AS splits,
        COUNT(DISTINCT phenv) AS labels, MIN(phenv) AS min_phenv, MAX(phenv) AS max_phenv,
        MAX(issues <> '') AS has_issue FROM records GROUP BY normalized_sha;
      CREATE UNIQUE INDEX sequence_group_idx ON sequence_groups(normalized_sha);
      CREATE TABLE exact_phopt_overlap AS SELECT r.row_id, p.sample_key, p.split,
        (r.raw_sha = p.raw_sha) AS raw_exact FROM records r JOIN phopt p USING(normalized_sha);
      CREATE INDEX overlap_row_idx ON exact_phopt_overlap(row_id);
      CREATE TABLE search_candidates AS SELECT g.representative AS row_id FROM sequence_groups g
        WHERE g.labels=1 AND g.has_issue=0
        AND NOT EXISTS (SELECT 1 FROM phopt p WHERE p.normalized_sha=g.normalized_sha);
      CREATE UNIQUE INDEX candidate_row_idx ON search_candidates(row_id);
    ''')
    db.commit()
    def scalar(query):
        return db.execute(query).fetchone()[0]
    def export_sql(name, query):
        cur = db.execute(query)
        with (output / name).open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow([col[0] for col in cur.description])
            writer.writerows(cur)
    export_sql('organisms.tsv', '''SELECT organism, COUNT(*) AS rows,
        COUNT(DISTINCT normalized_sha) AS unique_sequences,
        COUNT(DISTINCT published_split) AS splits, COUNT(DISTINCT phenv) AS labels,
        MIN(phenv) AS min_phenv, MAX(phenv) AS max_phenv,
        SUM(published_split='Training') AS training_rows,
        SUM(published_split='Validation') AS validation_rows
        FROM records GROUP BY organism ORDER BY organism''')
    export_sql('exact_phopt_overlap.tsv', 'SELECT * FROM exact_phopt_overlap ORDER BY row_id,sample_key')
    export_sql('sequence_conflicts.tsv', 'SELECT * FROM sequence_groups WHERE labels>1 ORDER BY representative')
    export_sql('duplicate_accessions.tsv', '''SELECT accession, COUNT(*) AS rows,
        COUNT(DISTINCT normalized_sha) AS sequences, COUNT(DISTINCT phenv) AS labels
        FROM records GROUP BY accession HAVING COUNT(*)>1''')
    export_sql('invalid_rows.tsv', "SELECT row_id,accession,length,issues FROM records WHERE issues<>''")
    # Search candidates are not a selected training set. Organism/family isolation is pending.
    fasta = output / 'phenv_search.fasta'
    with fasta.open('w', encoding='ascii') as f:
        for row_id, seq in db.execute('SELECT r.row_id,r.sequence FROM records r JOIN search_candidates c USING(row_id) ORDER BY row_id'):
            f.write(f'>env_{row_id}\n{seq.translate(TRANSLATE)}\n')
    with (output / 'phopt_quarantine.fasta').open('w', encoding='ascii') as f:
        with phopt_manifest.open(newline='', encoding='utf-8-sig') as raw:
            for i, row in enumerate(csv.DictReader(raw), 1):
                f.write(f'>phopt_{i}\n{row["sequence"].translate(TRANSLATE)}\n')
    split_rows = []
    for values in db.execute('''SELECT published_split,COUNT(*),COUNT(DISTINCT organism),
        SUM(length),MIN(length),MAX(length),MIN(phenv),MAX(phenv),
        SUM(phenv<=4),SUM(phenv>=10),SUM(phenv<5),SUM(phenv>=9)
        FROM records GROUP BY published_split ORDER BY published_split'''):
        split_rows.append(dict(zip(('split','rows','organisms','residues','min_length','max_length',
                                     'min_phenv','max_phenv','acid_le4','alkaline_ge10','acid_lt5','alkaline_ge9'), values)))
    report = {
        'state': 'raw_audit_complete', 'training_ready': False,
        'source': str(source.resolve()), 'source_sha256': source_sha, 'source_bytes': source_stat.st_size,
        'header': fields, 'rows': count, 'issues': dict(issues_count), 'alphabet_counts': dict(chars),
        'splits': split_rows,
        'organisms_total': scalar('SELECT COUNT(DISTINCT organism) FROM records'),
        'organisms_shared_between_published_splits': scalar('SELECT COUNT(*) FROM (SELECT organism FROM records GROUP BY organism HAVING COUNT(DISTINCT published_split)>1)'),
        'organisms_with_multiple_labels': scalar('SELECT COUNT(*) FROM (SELECT organism FROM records GROUP BY organism HAVING COUNT(DISTINCT phenv)>1)'),
        'raw_unique_sequences': scalar('SELECT COUNT(DISTINCT raw_sha) FROM records'),
        'normalized_unique_sequences': scalar('SELECT COUNT(*) FROM sequence_groups'),
        'normalized_sequence_groups_cross_split': scalar('SELECT COUNT(*) FROM sequence_groups WHERE splits>1'),
        'normalized_sequence_groups_conflicting_labels': scalar('SELECT COUNT(*) FROM sequence_groups WHERE labels>1'),
        'exact_phopt_overlap_rows': scalar('SELECT COUNT(DISTINCT row_id) FROM exact_phopt_overlap'),
        'phopt': {'manifest_sha256': phopt_sha, 'sequence_rows': len(phopt_rows), 'labels_consumed': False,
                  'quarantine_scope': 'all PHOPT sequences, including train/validation/test'},
        'homology_search_candidates': scalar('SELECT COUNT(*) FROM search_candidates'),
        'candidate_residues': scalar('SELECT SUM(length) FROM records JOIN search_candidates USING(row_id)') or 0,
        'homology_audited': False, 'source_organism_split_certified': False,
        'sequence_policy': 'Raw sequence preserved; BJOUZ->X for ESM/search only; no truncation. Invalid, conflicting normalized sequence groups and exact PHOPT matches excluded from search candidates.',
        'label_policy': 'pHenv only; original sample weights and split retained for provenance, not approved as training weights or a new strict split.',
    }
    db.close()
    if source_sha != sha_file(source) or phopt_sha != sha_file(phopt_manifest):
        raise ValueError('source changed during audit')
    report['artifact_sha256'] = {p.name: sha_file(p) for p in output.iterdir() if p.is_file() and p.name != 'status.json'}
    report['source_code_sha256'] = sha_file(Path(__file__))
    report['seconds'] = time.monotonic() - start
    write_json(output / 'report.json', report)
    write_json(output / 'status.json', {'state': report['state'], 'training_ready': False, 'seconds': report['seconds']})
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report
