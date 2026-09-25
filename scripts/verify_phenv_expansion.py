"""Independently verify expanded pHenv lineage and every reported search hit."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from localph.phenv_data import sha_file,sequence_hash,TRANSLATE,write_json


def qualifying_hits(path):
    with path.open() as f:
        for line in f:
            q,t,identity,qc,tc,_,_=line.rstrip().split('\t')
            values=[float(v) for v in (identity,qc,tc)]
            if not all(math.isfinite(v) and 0<=v<=1 for v in values):
                raise ValueError('invalid identity/coverage in search output')
            if values[0]>=.2 and min(values[1:])>=.8:
                yield q,t


def verify(audit,plan,forward,data):
    ac=json.loads((audit/'report.json').read_text())
    pc=json.loads((plan/'plan.json').read_text())
    dc=json.loads((data/'complete.json').read_text())
    if (pc['audit_report_sha256']!=sha_file(audit/'report.json')
        or dc['audit_report_sha256']!=sha_file(audit/'report.json')
        or dc['plan_sha256']!=sha_file(plan/'plan.json')
        or dc['records_sha256']!=sha_file(data/'records.csv')
        or dc['exclusions_sha256']!=sha_file(data/'exclusions.tsv')
        or ac['artifact_sha256']['manifest.sqlite']!=sha_file(audit/'manifest.sqlite')
        or pc['selection_sha256']!=sha_file(plan/'selection.sqlite')):
        raise ValueError('dataset lineage hash differs')
    with sqlite3.connect(f'file:{(plan/"selection.sqlite").resolve()}?mode=ro',uri=True) as db:
        selected={rid:(split,shard) for rid,split,shard in db.execute('SELECT * FROM selected')}
    train={rid for rid,v in selected.items() if v[0]=='train'}
    val={rid for rid,v in selected.items() if v==('validation',-1)}
    if len(train|val)!=len(selected) or not train or not val:
        raise ValueError('invalid selection split')
    with sqlite3.connect(f'file:{(audit/"manifest.sqlite").resolve()}?mode=ro',uri=True) as db:
        rows=db.execute('SELECT row_id,organism,phenv,normalized_sha FROM records JOIN search_candidates USING(row_id)')
        raw={rid:(org,y,key) for rid,org,y,key in rows}
    strata=defaultdict(list)
    for org,y in {(v[0],v[1]) for v in raw.values()}:
        strata['acid' if y<=4 else 'alkaline' if y>=10 else 'core'].append(org)
    expected_val_org=set()
    for names in strata.values():
        names.sort(key=lambda s:(int.from_bytes(hashlib.sha256(f'42|organism:{s}'.encode()).digest()[:8],'big'),s))
        count=min(len(names)-1,max(1,round(.2*len(names))))
        expected_val_org.update(names[:count])
    if expected_val_org!=set(pc['validation_organisms']):
        raise ValueError('organism assignment is not the declared deterministic split')
    expected_train={rid for rid,(org,y,key) in raw.items() if org not in expected_val_org}
    if train!=expected_train or not val<=set(raw) or any(raw[rid][0] not in expected_val_org for rid in val):
        raise ValueError('expanded training coverage or organism isolation differs')
    # Every search shard must cover precisely the selection database's rows.
    def check_fasta(path):
        ids=set()
        with path.open() as f:
            for header in f:
                if not header.startswith('>env_'):
                    raise ValueError('unexpected sequence header')
                rid=int(header.strip()[5:])
                sequence=next(f).strip()
                if rid in ids or rid not in raw or sequence_hash(sequence)!=raw[rid][2]:
                    raise ValueError('search sequence differs from raw source or repeats')
                ids.add(rid)
        return ids

    covered=set()
    for shard in pc['shards']:
        path=plan/shard['fasta']
        if sha_file(path)!=shard['sha256']:
            raise ValueError('search shard changed')
        ids=check_fasta(path)
        expected={rid for rid,v in selected.items() if v==('train',shard['index'])}
        if ids!=expected or covered&ids or len(ids)!=shard['rows']:
            raise ValueError('search shard coverage differs')
        covered.update(ids)
    if covered!=train:
        raise ValueError('search shards do not cover all expanded training rows')
    vf=plan/pc['validation_fasta']
    if sha_file(vf)!=pc['validation_fasta_sha256']:
        raise ValueError('validation search input changed')
    if check_fasta(vf)!=val:
        raise ValueError('validation search coverage differs')
    fc=json.loads((forward/'complete.json').read_text())
    fp=json.loads((forward/'protocol.json').read_text())
    if (dc['full_forward_complete_sha256']!=sha_file(forward/'complete.json')
        or fc['protocol_sha256']!=sha_file(forward/'protocol.json')
        or fc['hits_sha256']!=sha_file(forward/'hits.tsv')
        or fc['exclusions_sha256']!=sha_file(forward/'excluded_row_ids.json')
        or fp['audit_report_sha256']!=sha_file(audit/'report.json')
        or fp['candidate_count']!=ac['homology_search_candidates']
        or (fp['identity_min'],fp['query_coverage_min'],fp['target_coverage_min'])!=(.2,.8,.8)):
        raise ValueError('full PHOPT forward certificate differs')
    blocked=defaultdict(set)
    seen_forward=set()
    checked=0
    for q,t in qualifying_hits(forward/'hits.tsv'):
        checked+=1
        if not q.startswith('env_') or not t.startswith('phopt_'):
            raise ValueError('forward identifier differs')
        rid=int(q[4:])
        if rid not in raw:
            raise ValueError('forward hit outside eligible source data')
        seen_forward.add(rid)
        if rid in val:
            raise ValueError('PHOPT hit retained in fixed validation')
        if rid in train:
            blocked[rid].add('PHOPT_forward_homology')
    if seen_forward!=set(json.loads((forward/'excluded_row_ids.json').read_text())):
        raise ValueError('forward exclusion list does not reproduce')
    tasks={(s['index'],direction) for s in pc['shards']
           for direction in ('phopt_to_env','train_to_validation','validation_to_train')}
    if ({(t['shard'],t['direction']) for t in pc['tasks']}!=tasks or len(pc['tasks'])!=len(tasks)
        or len(dc['searches'])!=len(tasks)):
        raise ValueError('required bidirectional searches missing or duplicated')
    for task, evidence in zip(pc['tasks'],dc['searches']):
        run=plan/'searches'/f'task_{task["index"]:05d}'
        done=json.loads((run/'complete.json').read_text())
        protocol=json.loads((run/'protocol.json').read_text())
        if (evidence['task']!=task or evidence['complete_sha256']!=sha_file(run/'complete.json')
            or done['task']!=task or done['plan_sha256']!=sha_file(plan/'plan.json')
            or done['hits_sha256']!=sha_file(run/'hits.tsv')
            or done['protocol_sha256']!=sha_file(run/'protocol.json')
            or protocol['task']!=task or protocol['plan_sha256']!=sha_file(plan/'plan.json')):
            raise ValueError('search task provenance differs')
        command=protocol['command']
        shard=pc['shards'][task['shard']]
        pf=(audit/'phopt_quarantine.fasta').name
        sf=Path(shard['fasta']).name
        vh=pc['validation_fasta_sha256']
        ph=ac['artifact_sha256']['phopt_quarantine.fasta']
        expected_q,expected_t,qh,th={
            'phopt_to_env':(pf,sf,ph,shard['sha256']),
            'train_to_validation':(sf,vf.name,shard['sha256'],vh),
            'validation_to_train':(vf.name,sf,vh,shard['sha256'])}[task['direction']]
        if (command[1]!='easy-search' or Path(command[2]).name!=expected_q or Path(command[3]).name!=expected_t
            or protocol['query_sha256']!=qh or protocol['target_sha256']!=th):
            raise ValueError('search query/target provenance differs')
        for option, expected in {'-s':'7.5','--min-seq-id':'0.2','-c':'0.8','--cov-mode':'0',
            '--alignment-mode':'3','--seq-id-mode':'0','-e':'100',
            '--max-accept':'2147483647','--max-rejected':'2147483647'}.items():
            if command[command.index(option)+1]!=expected:
                raise ValueError('search settings differ')
        target_count=len(val) if task['direction']=='train_to_validation' else pc['shards'][task['shard']]['rows']
        if command[command.index('--max-seqs')+1]!=str(target_count):
            raise ValueError('candidate cap is smaller than the target shard')
        for q,t in qualifying_hits(run/'hits.tsv'):
            checked+=1
            tk=q if task['direction']=='train_to_validation' else t
            if not tk.startswith('env_'):
                raise ValueError('training hit identifier differs')
            rid=int(tk[4:])
            if selected.get(rid)!=('train',task['shard']):
                raise ValueError('hit outside declared shard')
            if task['direction']=='phopt_to_env':
                if not q.startswith('phopt_'):
                    raise ValueError('reverse PHOPT identifier differs')
                reason='PHOPT_reverse_homology'
            else:
                vk=t if task['direction']=='train_to_validation' else q
                if not vk.startswith('env_') or int(vk[4:]) not in val:
                    raise ValueError('hit outside fixed auxiliary validation')
                reason='pHenv_validation_homology'
            blocked[rid].add(reason)
    with (data/'exclusions.tsv').open(newline='') as f:
        excluded={int(r['key'][4:]):set(r['reason'].split('|')) for r in csv.DictReader(f,delimiter='\t')}
    if excluded!=dict(blocked):
        raise ValueError('exclusion reasons do not reconstruct from all hits')
    expected=(train-set(blocked))|val
    seen=set()
    weights=defaultdict(float)
    split_org={s:set() for s in ('train','validation')}
    db=sqlite3.connect(f'file:{(audit/"manifest.sqlite").resolve()}?mode=ro',uri=True)
    with (data/'records.csv').open(newline='') as f:
        for r in csv.DictReader(f):
            rid=int(r['source_row'])
            if rid in seen or rid not in expected or r['key']!=f'env_{rid}' or r['split']!=selected[rid][0]:
                raise ValueError('retained coverage/split differs')
            seen.add(rid)
            source=db.execute('SELECT accession,organism,phenv,published_split,sequence,normalized_sha FROM records WHERE row_id=?',(rid,)).fetchone()
            actual=(r['accession'],r['organism'],float(r['phenv']),r['published_split'],r['sequence'],r['normalized_sha'])
            if (*source[:4],source[4].translate(TRANSLATE),source[5])!=actual or sequence_hash(r['sequence'])!=r['normalized_sha']:
                raise ValueError('retained raw source lineage differs')
            split_org[r['split']].add(r['organism'])
            if r['split']=='train':
                w=float(r['train_weight'])
                if not math.isfinite(w) or w<=0:
                    raise ValueError('invalid training weight')
                weights[r['organism']]+=w
            elif r['train_weight']!='':
                raise ValueError('validation weight used for training')
    db.close()
    if seen!=expected or split_org['train']&split_org['validation']:
        raise ValueError('retained coverage or organism split differs')
    ntrain=len(train-set(blocked))
    mass=ntrain/len(weights)
    if any(not math.isclose(v,mass,rel_tol=1e-7) for v in weights.values()):
        raise ValueError('organism training weights do not normalize equally')
    result={'verified':True,'rows':len(seen),'train':ntrain,'validation':len(val),
        'source_lineage_rows_checked':len(seen),'qualified_pairs_rechecked':checked,'organism_disjoint':True,
        'retained_observed_cross_split_hits':0,'phopt_labels_consumed':False,
        'complete_sha256':sha_file(data/'complete.json'),'verifier_sha256':sha_file(Path(__file__)),
        'limitation':'Observed hits only; fixed capped validation and heuristic searches do not prove absolute isolation.'}
    write_json(data/'verification.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('audit','plan','forward','data'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(verify(a.audit,a.plan,a.forward,a.data)),flush=True)
