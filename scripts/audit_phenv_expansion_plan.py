"""Read-only source/count audit of an expansion plan before searches finish."""
import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from localph.phenv_data import sha_file,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('audit','plan','pilot','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    ac=json.loads((a.audit/'report.json').read_text())
    pc=json.loads((a.plan/'plan.json').read_text())
    pilot=json.loads((a.pilot/'complete.json').read_text())
    if (ac['artifact_sha256']['manifest.sqlite']!=sha_file(a.audit/'manifest.sqlite')
        or pc['selection_sha256']!=sha_file(a.plan/'selection.sqlite')
        or pc['audit_report_sha256']!=sha_file(a.audit/'report.json')
        or pc['pilot_complete_sha256']!=sha_file(a.pilot/'complete.json')
        or pilot['records_sha256']!=sha_file(a.pilot/'records.csv')):
        raise ValueError('plan/audit source differs')
    db=sqlite3.connect(f'file:{(a.audit/"manifest.sqlite").resolve()}?mode=ro',uri=True)
    db.execute('ATTACH DATABASE ? AS staging',(f'file:{(a.plan/"selection.sqlite").resolve()}?mode=ro',))
    # search_candidates.row_id has no declared affinity (from CREATE AS).
    # A LEFT JOIN to an INTEGER column can induce a quadratic scan in SQLite.
    # Integer sets and primary-key source lookup have bounded linear coverage.
    eligible={int(r[0]) for r in db.execute('SELECT row_id FROM search_candidates')}
    counts={s:Counter() for s in ('train','validation')}
    organisms={s:defaultdict(set) for s in counts}
    selected=set()
    val=set()
    validation_organisms=set(pc['validation_organisms'])
    query=('SELECT s.row_id,s.split,r.organism,r.phenv,r.length FROM staging.selected s '
           'CROSS JOIN records r WHERE r.row_id=s.row_id')
    for rid,split,org,y,length in db.execute(query):
        if rid not in eligible or rid in selected or split not in counts:
            raise ValueError('ineligible or duplicate planned row')
        selected.add(rid)
        c=counts[split]
        c['sequences']+=1
        c['residues']+=length
        region='acid' if y<=4 else 'alkaline' if y>=10 else 'core'
        c[region+'_sequences']+=1
        organisms[split]['all'].add(org)
        organisms[split][region].add(org)
        if split=='validation':
            val.add(rid)
        if (org in validation_organisms)!=(split=='validation'):
            raise ValueError('source organism assigned to the wrong side')
    db.close()
    with (a.pilot/'records.csv').open(newline='') as f:
        expected_val={int(r['source_row']) for r in csv.DictReader(f) if r['split']=='validation'}
    if val!=expected_val or organisms['train']['all']&organisms['validation']['all']:
        raise ValueError('fixed validation differs or organism overlap exists')
    for split in counts:
        if counts[split]['sequences']!=pc['counts_before_homology'][split]:
            raise ValueError('planned count differs')
        counts[split]['organisms']=len(organisms[split]['all'])
        for region in ('acid','core','alkaline'):
            counts[split][region+'_organisms']=len(organisms[split][region])
    result={'verified_plan_source_membership':True,'training_ready':False,
        'counts_before_homology':counts,'shared_organisms':0,'pilot_validation_keys_unchanged':True,
        'fp16_cache_bytes_before_homology':sum(c['residues'] for c in counts.values())*1280*2,
        'plan_sha256':sha_file(a.plan/'plan.json'),'source_sha256':sha_file(Path(__file__)),
        'scope':'Plan/source membership only; full homology exclusion remains incomplete.'}
    write_json(a.output,result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
