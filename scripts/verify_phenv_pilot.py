"""Independently recompute pilot lineage, weights and observed-hit isolation."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from localph.phenv_data import sha_file,write_json,TRANSLATE,sequence_hash


def verify(audit,data):
    cert=json.loads((data/'complete.json').read_text())
    ac=json.loads((audit/'report.json').read_text())
    if cert['audit_report_sha256']!=sha_file(audit/'report.json') or cert['records_sha256']!=sha_file(data/'records.csv'):
        raise ValueError('source certificate mismatch')
    if sha_file(audit/'manifest.sqlite')!=ac['artifact_sha256']['manifest.sqlite']:
        raise ValueError('source database mismatch')
    with (data/'records.csv').open(newline='') as f: rows=list(csv.DictReader(f))
    if len({r['key'] for r in rows})!=len(rows): raise ValueError('duplicate records')
    train={r['key'] for r in rows if r['split']=='train'}
    val={r['key'] for r in rows if r['split']=='validation'}
    if not train or not val or train&val or len(train|val)!=len(rows): raise ValueError('invalid split')
    orgs={s:{r['organism'] for r in rows if r['split']==s} for s in ('train','validation')}
    if orgs['train']&orgs['validation']: raise ValueError('organism leakage')
    if len({r['normalized_sha'] for r in rows})!=len(rows): raise ValueError('sequence duplicate')
    db=sqlite3.connect(f'file:{(audit/"manifest.sqlite").resolve()}?mode=ro',uri=True)
    totals=defaultdict(float)
    for r in rows:
        raw=db.execute('SELECT accession,organism,phenv,published_split,sequence,normalized_sha,issues FROM records WHERE row_id=?',(int(r['source_row']),)).fetchone()
        expected=(r['accession'],r['organism'],float(r['phenv']),r['published_split'],r['sequence'],r['normalized_sha'],'')
        if raw is None or (*raw[:4],raw[4].translate(TRANSLATE),*raw[5:])!=expected:
            raise ValueError('source row lineage differs')
        if r['key']!='env_'+r['source_row'] or sequence_hash(r['sequence'])!=r['normalized_sha']:
            raise ValueError('sequence hash/key differs')
        if db.execute('SELECT 1 FROM exact_phopt_overlap WHERE row_id=? LIMIT 1',(int(r['source_row']),)).fetchone():
            raise ValueError('exact PHOPT overlap')
        if r['split']=='train':
            w=float(r['train_weight'])
            if not math.isfinite(w) or w<=0: raise ValueError('invalid weight')
            totals[r['organism']]+=w
        elif r['train_weight']!='': raise ValueError('validation used as training weight fit')
    db.close()
    expected_mass=len(train)/len(orgs['train'])
    if any(not math.isclose(x,expected_mass,rel_tol=1e-7) for x in totals.values()):
        raise ValueError('weights are not balanced by training organism')
    if {s['name'] for s in cert['searches']}!={'env_to_phopt','phopt_to_env','train_to_validation','validation_to_train'}:
        raise ValueError('required search directions missing')
    checked=0
    for search in cert['searches']:
        path=data/(search['name']+'.tsv')
        if sha_file(path)!=search['hits_sha256']: raise ValueError('search results changed')
        with path.open() as f:
            for line in f:
                q,t,identity,qc,tc,_,_=line.rstrip().split('\t')
                if float(identity)<.2 or min(float(qc),float(tc))<.8: continue
                checked+=1
                if search['name']=='env_to_phopt' and q in train|val: raise ValueError('PHOPT hit retained')
                if search['name']=='phopt_to_env' and t in train|val: raise ValueError('reverse PHOPT hit retained')
                if (q in train and t in val) or (q in val and t in train): raise ValueError('cross split homology hit retained')
    result={'verified':True,'rows':len(rows),'train':len(train),'validation':len(val),
      'source_lineage_rows_checked':len(rows),'qualified_pairs_rechecked':checked,'organism_disjoint':True,
      'exact_phopt_overlap':0,'retained_observed_cross_split_hits':0,'phopt_labels_consumed':False,
      'complete_sha256':sha_file(data/'complete.json'),'verifier_sha256':sha_file(Path(__file__)),
      'limitation':'This checks every reported qualifying hit; heuristic search may miss remote homologs.'}
    write_json(data/'verification.json',result); print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',required=True,type=Path); p.add_argument('--data',required=True,type=Path)
    a=p.parse_args(); verify(a.audit,a.data)
