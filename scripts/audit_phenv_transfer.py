"""Stage pHenv provenance and search inputs without using PHOPT labels."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from localph.phenv_data import audit

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--phopt-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    audit(a.source, a.phopt_manifest, a.output)
