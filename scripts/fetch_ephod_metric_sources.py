"""Fetch primary sources for the paused-training metric audit; no model fitting."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/metric_definition_audit_20260919/primary_sources'
OUT.mkdir(parents=True, exist_ok=True)
COMMIT = 'e823cd2f1172258dc1e81cc00326e6975f22d10a'
URLS = {
    'nature.html': 'https://www.nature.com/articles/s42256-025-01026-6',
    'europepmc.json': 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1038/s42256-025-01026-6&format=json&resultType=core',
    'prediction.csv': f'https://raw.githubusercontent.com/jafetgado/EpHod/{COMMIT}/example/prediction.csv',
    'trainutils.py': f'https://raw.githubusercontent.com/jafetgado/EpHod/{COMMIT}/ephod/training/trainutils.py',
}


def fetch(item):
    name, url = item
    try:
        r = requests.get(url, timeout=25)
        record = dict(url=url, status=r.status_code, length=len(r.content),
                      sha256=hashlib.sha256(r.content).hexdigest())
        if r.ok:
            (OUT/name).write_bytes(r.content)
        return name, record
    except requests.RequestException as error:
        return name, dict(url=url, error=str(error))


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=4) as pool:
        result = dict(pool.map(fetch, URLS.items()))
    (OUT/'fetch.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    p = OUT/'europepmc.json'
    if p.exists():
        for r in json.loads(p.read_text()).get('resultList', {}).get('result', []):
            print(json.dumps({k: r.get(k) for k in ['title', 'doi', 'pmcid', 'abstractText', 'fullTextUrlList']}, indent=2))
