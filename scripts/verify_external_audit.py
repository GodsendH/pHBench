"""Independent row and exclusion verification for the EnzyBase training pool."""
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phgeofuse.cache import sha256_file, atomic_json


def normalized(s):
    return s.strip().upper().translate(str.maketrans({c: "X" for c in "BJOUZ"}))


def main():
    root = ROOT / "experiments/ion_context_phopt_20260917/external_data_audit"
    screened = root / "screened_v1"
    with (root / "metadata.csv").open() as f:
        source = list(csv.DictReader(f))
    with (screened / "eligible_external.csv").open() as f:
        eligible = list(csv.DictReader(f))
    with (ROOT / "artifacts/phgeofuse/manifest.csv").open() as f:
        phopt = [(r["protein_id"], normalized(r["sequence"])) for r in csv.DictReader(f)]
    ids, seqs = {r[0] for r in phopt}, {r[1] for r in phopt}
    if len({r["id"] for r in eligible}) != len(eligible) or len({r["sequence"] for r in eligible}) != len(eligible):
        raise ValueError("duplicate accession/sequence in eligible pool")
    for r in eligible:
        original = source[int(r["source_row"]) - 2]
        if (r["id"] != original["uniprot_id"] or float(r["label"]) != float(original["ph_optimum"])
                or r["sequence"] != normalized(original["uniprot_seq_cut"])
                or original["cut_to_uniprot_domain"] != "0"
                or not 1 <= len(r["sequence"]) <= 1022
                or r["id"] in ids or r["sequence"] in seqs):
            raise ValueError("row provenance/PHOPT exclusion differs")
    candidates = (screened / "external_candidates.fasta").read_text().splitlines()
    mapping = {candidates[i][1:]: candidates[i + 1] for i in range(0, len(candidates), 2)}
    eligible_sequences = {r["sequence"] for r in eligible}
    qualifying_hits = 0
    with (screened / "all_hits.tsv").open() as f:
        for r in csv.reader(f, delimiter="\t"):
            if float(r[2]) >= .3 and float(r[3]) >= .8 and float(r[4]) >= .8:
                qualifying_hits += 1
                if mapping[r[0]] in eligible_sequences:
                    raise ValueError("eligible sequence has a prohibited observed PHOPT hit")
    result = {"verified": True, "eligible": len(eligible), "acid": sum(float(r["label"]) <= 4 for r in eligible),
              "alkaline": sum(float(r["label"]) >= 10 for r in eligible), "qualifying_hits_checked": qualifying_hits,
              "labels_match_published_source": True, "PHOPT_labels_read": False,
              "no_PHOPT_accession_or_exact_sequence_overlap": True, "no_observed_prohibited_homology": True,
              "search_limitation": "MMseqs is heuristic; absence of a detected hit is not absence of biological homology",
              "hashes": {str(p.relative_to(ROOT)): sha256_file(p) for p in
                         [root / "metadata.csv", screened / "eligible_external.csv", screened / "all_hits.tsv", Path(__file__)]}}
    atomic_json(screened / "verification.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "hashes"}, indent=2))


if __name__ == "__main__":
    main()
