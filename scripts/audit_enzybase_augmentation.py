"""Audit public external labels without consuming PHOPT validation/test labels.

PHOPT accessions and sequences are used only to quarantine duplicates. A
subsequent MMseqs search can conservatively exclude every detected >=30%
identity, >=80% bidirectional coverage hit to ANY PHOPT sample, including
training samples. That pool is safe to reuse in the existing development folds
with respect to the observed direct homology rule; this is not proof that all
biological homology has been detected.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phgeofuse.cache import atomic_json, sha256_file


def normalized(sequence):
    return sequence.strip().upper().translate(str.maketrans({c: "X" for c in "BJOUZ"}))


def counts(rows):
    return {"total": len(rows), "acid": sum(float(r["label"]) <= 4 for r in rows),
            "alkaline": sum(float(r["label"]) >= 10 for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--search", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    raw = args.source.read_bytes()
    if hashlib.md5(raw).hexdigest() != "60648330138d236baea7710e81cc581f":
        raise ValueError("EnzyBase12k metadata differs from Zenodo record 18405148")
    with args.source.open() as f:
        external = list(csv.DictReader(f))
    with (ROOT / "artifacts/phgeofuse/manifest.csv").open() as f:
        # Labels intentionally not accessed.
        phopt = [{"key": r["split"] + "::" + r["protein_id"], "id": r["protein_id"],
                  "sequence": normalized(r["sequence"])} for r in csv.DictReader(f)]
    ids = {r["id"] for r in phopt}
    seqs = {r["sequence"] for r in phopt}
    kept, rejected, used_ids, used_seqs = [], [], set(), set()
    reasons = Counter()
    for i, row in enumerate(external):
        seq = normalized(row["uniprot_seq_cut"])
        y = float(row["ph_optimum"])
        reason = None
        if not 0 <= y <= 14:
            reason = "invalid_label"
        elif row["uniprot_id"] in ids:
            reason = "PHOPT_accession_overlap"
        elif seq in seqs:
            reason = "PHOPT_exact_sequence_overlap"
        elif row["cut_to_uniprot_domain"].strip() != "0":
            reason = "domain_cut_or_unspecified"
        elif not 1 <= len(seq) <= 1022:
            reason = "outside_initial_full_sequence_length_contract"
        elif not set(seq) <= set("ACDEFGHIKLMNPQRSTVWYX"):
            reason = "invalid_sequence"
        elif row["uniprot_id"] in used_ids or seq in used_seqs:
            reason = "external_duplicate"
        if reason:
            rejected.append({"id": row["uniprot_id"], "reason": reason, "label": y})
            reasons[reason] += 1
        else:
            kept.append({"id": row["uniprot_id"], "label": y, "sequence": seq,
                         "ec": row["ec_id"], "source_row": i + 2})
            used_ids.add(row["uniprot_id"])
            used_seqs.add(seq)
    with (out / "external_candidates.fasta").open("w") as f:
        for i, row in enumerate(kept):
            f.write(f">ext{i}\n{row['sequence']}\n")
    with (out / "phopt_sequences_only.fasta").open("w") as f:
        for i, row in enumerate(phopt):
            f.write(f">ph{i}\n{row['sequence']}\n")
    summary = {"source": "https://doi.org/10.5281/zenodo.18405148", "license": "CC-BY-4.0",
               "source_sha256": sha256_file(args.source), "external_rows": len(external),
               "phopt_sequence_rows": len(phopt), "PHOPT_labels_read": False,
               "rejections": dict(reasons), "before_homology_filter": counts(kept),
               "ready_for_training": False}
    if args.search and kept:
        exe = shutil.which("mmseqs")
        if not exe:
            raise RuntimeError("mmseqs is required; activate phbench, no singleton fallback")
        destination = out / "all_hits.tsv"
        command = [exe, "easy-search", str(out / "external_candidates.fasta"),
                   str(out / "phopt_sequences_only.fasta"), str(destination), str(out / "mmseqs_tmp"),
                   "--min-seq-id", "0.3", "-c", "0.8", "--cov-mode", "0", "-s", "7.5",
                   "--max-seqs", "100000", "--threads", "4", "--format-output",
                   "query,target,fident,qcov,tcov,evalue,bits"]
        with (out / "mmseqs.log").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=600)
        excluded = set()
        nearest = {}
        with destination.open() as f:
            for values in csv.reader(f, delimiter="\t"):
                query, target = values[:2]
                ident, qcov, tcov = map(float, values[2:5])
                if ident >= .3 and qcov >= .8 and tcov >= .8:
                    idx = int(query[3:])
                    excluded.add(idx)
                    if idx not in nearest or ident > nearest[idx]["identity"]:
                        nearest[idx] = {"external_id": kept[idx]["id"], "PHOPT_key": phopt[int(target[2:])]["key"],
                                        "identity": ident, "query_coverage": qcov, "target_coverage": tcov}
        final = [r for i, r in enumerate(kept) if i not in excluded]
        summary.update(homology_rejected=len(excluded), after_homology_filter=counts(final),
                       mmseqs_command=command, mmseqs_version=subprocess.check_output([exe, "version"], text=True).strip(),
                       homology_rule="observed >=30% identity and >=80% both coverages to ANY PHOPT sequence",
                       ready_for_training=True)
        atomic_json(out / "excluded_homologs.json", list(nearest.values()))
    else:
        final = kept
    filename = "eligible_external.csv" if args.search else "candidates_not_training_ready.csv"
    with (out / filename).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "label", "sequence", "ec", "source_row"])
        w.writeheader()
        w.writerows(final)
    atomic_json(out / "rejected.json", rejected)
    atomic_json(out / "audit.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
