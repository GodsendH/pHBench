"""Stage all eligible training organisms against a fixed pilot validation set.

plan writes sequence-only search shards; it does not execute searches or make
data training-ready. search runs one CPU task inside an allocation. finalize
requires the full forward PHOPT screen plus all reverse/cross-split tasks.
The validation organisms and validation proteins stay fixed from the pilot.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from localph.phenv_data import TRANSLATE, sequence_hash, sha_file, write_json

FIELDS = ['key','source_row','accession','organism','phenv','published_split',
          'split','sequence','normalized_sha','train_weight']


def connect(audit):
    return sqlite3.connect(f'file:{(Path(audit)/"manifest.sqlite").resolve()}?mode=ro', uri=True)


def rank(organism):
    return int.from_bytes(hashlib.sha256(f'42|organism:{organism}'.encode()).digest()[:8], 'big')


def checked_audit(audit):
    report = json.loads((audit/'report.json').read_text())
    for name in ('manifest.sqlite','phopt_quarantine.fasta'):
        if sha_file(audit/name) != report['artifact_sha256'][name]:
            raise ValueError('raw audit artifact differs')
    if report['organisms_with_multiple_labels']:
        raise ValueError('resolve conflicting organism labels before expansion')
    return report


def make_plan(audit, pilot, output, shard_size=50000):
    if shard_size < 1:
        raise ValueError('shard size must be positive')
    ac = checked_audit(audit)
    pc = json.loads((pilot/'complete.json').read_text())
    pv = json.loads((pilot/'verification.json').read_text())
    if (not pv.get('verified') or pv['complete_sha256']!=sha_file(pilot/'complete.json')
        or pc['records_sha256']!=sha_file(pilot/'records.csv')
        or pc['audit_report_sha256']!=sha_file(audit/'report.json')
        or pc['seed']!=42 or pc['organism_validation_fraction']!=.2):
        raise ValueError('pilot lineage or fixed organism split differs')
    with (pilot/'records.csv').open(newline='') as f:
        validation = {int(r['source_row']):r for r in csv.DictReader(f) if r['split']=='validation'}
    if not validation:
        raise ValueError('pilot validation is empty')
    db = connect(audit)
    org_labels = dict(db.execute('SELECT organism,MIN(phenv) FROM records JOIN search_candidates USING(row_id) GROUP BY organism'))
    strata = defaultdict(list)
    for org, y in org_labels.items():
        strata['acid' if y<=4 else 'alkaline' if y>=10 else 'core'].append(org)
    validation_organisms = set()
    for orgs in strata.values():
        orgs.sort(key=lambda org:(rank(org),org))
        count = min(len(orgs)-1, max(1,round(.2*len(orgs))))
        if count<1:
            raise ValueError('each pHenv stratum needs separate training/validation organisms')
        validation_organisms.update(orgs[:count])
    if {r['organism'] for r in validation.values()} - validation_organisms:
        raise ValueError('pilot validation organisms differ from the fixed seed42 partition')
    output.mkdir(parents=True, exist_ok=False)
    (output/'inputs').mkdir()
    selection = sqlite3.connect(output/'selection.sqlite')
    selection.execute('CREATE TABLE selected (row_id INTEGER PRIMARY KEY, split TEXT, shard INTEGER)')
    counts, residues = Counter(), Counter()
    shards, pending = [], []
    handle = None
    query = ('SELECT row_id,organism,phenv,sequence,normalized_sha FROM records '
             'JOIN search_candidates USING(row_id) ORDER BY row_id')
    seen_validation = set()
    with (output/'inputs/validation.fasta').open('w') as vf:
        for rid, org, y, raw, key in db.execute(query):
            seq = raw.translate(TRANSLATE)
            if org in validation_organisms:
                if rid not in validation:
                    counts['unused_validation_organism_sequences'] += 1
                    continue
                row = validation[rid]
                if (row['organism']!=org or float(row['phenv'])!=y or row['normalized_sha']!=key
                    or row['sequence']!=seq or sequence_hash(seq)!=key):
                    raise ValueError('pilot validation lineage differs from raw source')
                vf.write(f'>env_{rid}\n{seq}\n')
                seen_validation.add(rid)
                pending.append((rid,'validation',-1))
                counts['validation'] += 1
                residues['validation'] += len(seq)
            else:
                if counts['train'] % shard_size == 0:
                    if handle is not None:
                        handle.close()
                    number = len(shards)
                    relative = f'inputs/train_{number:05d}.fasta'
                    handle = (output/relative).open('w')
                    shards.append({'index':number,'fasta':relative,'rows':0,'residues':0})
                handle.write(f'>env_{rid}\n{seq}\n')
                shards[-1]['rows'] += 1
                shards[-1]['residues'] += len(seq)
                pending.append((rid,'train',shards[-1]['index']))
                counts['train'] += 1
                residues['train'] += len(seq)
            if len(pending)>=10000:
                selection.executemany('INSERT INTO selected VALUES (?,?,?)',pending)
                selection.commit()
                pending.clear()
    if handle is not None:
        handle.close()
    selection.executemany('INSERT INTO selected VALUES (?,?,?)',pending)
    selection.commit()
    selection.close()
    db.close()
    if seen_validation!=set(validation) or not counts['train']:
        raise ValueError('expansion failed to cover the fixed validation or training set')
    tasks = []
    for shard in shards:
        shard['sha256'] = sha_file(output/shard['fasta'])
        for direction in ('phopt_to_env','train_to_validation','validation_to_train'):
            tasks.append({'index':len(tasks),'direction':direction,'shard':shard['index']})
    result = {'state':'planned','training_ready':False,
        'scope':'All eligible non-validation-organism training sequences; fixed capped pilot validation proteins',
        'seed':42,'validation_organisms':sorted(validation_organisms),'counts_before_homology':dict(counts),
        'residues_before_homology':dict(residues),'shard_size':shard_size,'shards':shards,'tasks':tasks,
        'validation_fasta':'inputs/validation.fasta','validation_fasta_sha256':sha_file(output/'inputs/validation.fasta'),
        'selection_sha256':sha_file(output/'selection.sqlite'),'audit_report_sha256':sha_file(audit/'report.json'),
        'pilot_complete_sha256':sha_file(pilot/'complete.json'),'pilot_verification_sha256':sha_file(pilot/'verification.json'),
        'source_sha256':sha_file(Path(__file__)),'phopt_sequence_count':ac['phopt']['sequence_rows'],
        'phopt_labels_consumed':False,'model_fitted':False,
        'requires':'Full pHenv-to-PHOPT forward certificate, every planned CPU search, then finalize and independent verification',
        'search_rule':'identity>=0.2 and query/target coverage>=0.8; heuristic bidirectional search, not exhaustive proof'}
    write_json(output/'plan.json', result)
    return result


def load_plan(plan, audit):
    cert = json.loads((plan/'plan.json').read_text())
    if cert['audit_report_sha256']!=sha_file(audit/'report.json'):
        raise ValueError('plan belongs to another audit')
    if cert['selection_sha256']!=sha_file(plan/'selection.sqlite') or cert['source_sha256']!=sha_file(Path(__file__)):
        raise ValueError('plan selection or preparation code changed')
    return cert


def run_search(plan, audit, task_index, mmseqs, threads=8, memory='8G', dry_run=False):
    cert = load_plan(plan, audit)
    if not 0<=task_index<len(cert['tasks']) or threads<1:
        raise ValueError('invalid task index or thread count')
    task = cert['tasks'][task_index]
    shard = cert['shards'][task['shard']]
    sf, vf, pf = plan/shard['fasta'], plan/cert['validation_fasta'], audit/'phopt_quarantine.fasta'
    ac = json.loads((audit/'report.json').read_text())
    for file, digest in ((sf,shard['sha256']), (vf,cert['validation_fasta_sha256']),
                         (pf,ac['artifact_sha256']['phopt_quarantine.fasta'])):
        if sha_file(file)!=digest:
            raise ValueError('search input differs')
    direction = task['direction']
    query, target, target_count = {
        'phopt_to_env':(pf,sf,shard['rows']),
        'train_to_validation':(sf,vf,cert['counts_before_homology']['validation']),
        'validation_to_train':(vf,sf,shard['rows'])}[direction]
    run = plan/'searches'/f'task_{task_index:05d}'
    command = [str(mmseqs.resolve()),'easy-search',str(query.resolve()),str(target.resolve()),
        str((run/'hits.tsv').resolve()),str((run/'tmp').resolve()),'-s','7.5','--min-seq-id','0.2',
        '-c','0.8','--cov-mode','0','--alignment-mode','3','--seq-id-mode','0','-e','100',
        '--max-seqs',str(target_count),'--max-accept','2147483647','--max-rejected','2147483647',
        '--threads',str(threads),'--split-memory-limit',memory,
        '--format-output','query,target,fident,qcov,tcov,evalue,bits','--remove-tmp-files','1']
    if dry_run:
        return {'dry_run':True,'task':task,'command':command,'submits_job':False}
    run.mkdir(parents=True, exist_ok=False)
    version = subprocess.check_output([str(mmseqs),'version'],text=True).strip()
    start = time.monotonic()
    protocol = {'task':task,'plan_sha256':sha_file(plan/'plan.json'),'command':command,'mmseqs_version':version,
        'query_sha256':sha_file(query),'target_sha256':sha_file(target)}
    write_json(run/'protocol.json',protocol)
    with (run/'mmseqs.log').open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    result = {'state':'complete','task':task,'seconds':time.monotonic()-start,
        'plan_sha256':protocol['plan_sha256'],'protocol_sha256':sha_file(run/'protocol.json'),
        'hits_sha256':sha_file(run/'hits.tsv')}
    write_json(run/'complete.json',result)
    return result


def read_selection(plan):
    with sqlite3.connect(f'file:{(plan/"selection.sqlite").resolve()}?mode=ro',uri=True) as db:
        return {rid:(split,shard) for rid,split,shard in db.execute('SELECT * FROM selected')}


def exclusions(plan, audit, forward):
    cert = load_plan(plan,audit)
    selection = read_selection(plan)
    fc = json.loads((forward/'complete.json').read_text())
    fp = json.loads((forward/'protocol.json').read_text())
    if (fc['state']!='complete' or fc['protocol_sha256']!=sha_file(forward/'protocol.json')
        or fp['audit_report_sha256']!=sha_file(audit/'report.json')
        or fc['hits_sha256']!=sha_file(forward/'hits.tsv')
        or fc['exclusions_sha256']!=sha_file(forward/'excluded_row_ids.json')):
        raise ValueError('full forward PHOPT screen is incomplete or changed')
    removed = defaultdict(set)
    forward_ids = set()
    with (forward/'hits.tsv').open() as f:
        for line in f:
            q,t,ident,qc,tc,_,_ = line.rstrip().split('\t')
            if float(ident)>=.2 and min(float(qc),float(tc))>=.8:
                if not q.startswith('env_') or not t.startswith('phopt_'):
                    raise ValueError('unexpected full forward identifiers')
                forward_ids.add(int(q[4:]))
    if forward_ids!=set(json.loads((forward/'excluded_row_ids.json').read_text())):
        raise ValueError('full forward exclusions do not reproduce from hits')
    for rid in forward_ids:
        if rid in selection:
            if selection[rid][0]=='validation':
                raise ValueError('new PHOPT hit in fixed pilot validation; explicitly revise protocol before proceeding')
            removed[rid].add('PHOPT_forward_homology')
    evidence = []
    plan_hash = sha_file(plan/'plan.json')
    for task in cert['tasks']:
        run = plan/'searches'/f'task_{task["index"]:05d}'
        done = json.loads((run/'complete.json').read_text())
        if (done['state']!='complete' or done['task']!=task or done['plan_sha256']!=plan_hash
            or done['hits_sha256']!=sha_file(run/'hits.tsv')
            or done['protocol_sha256']!=sha_file(run/'protocol.json')):
            raise ValueError('search task certificate differs')
        protocol = json.loads((run/'protocol.json').read_text())
        if protocol['task']!=task or protocol['plan_sha256']!=plan_hash:
            raise ValueError('search protocol differs')
        with (run/'hits.tsv').open() as f:
            for line in f:
                q,t,ident,qc,tc,_,_ = line.rstrip().split('\t')
                if float(ident)<.2 or min(float(qc),float(tc))<.8:
                    continue
                train_key = q if task['direction']=='train_to_validation' else t
                if not train_key.startswith('env_'):
                    raise ValueError('invalid training search identifier')
                rid = int(train_key[4:])
                if selection.get(rid)!=('train',task['shard']):
                    raise ValueError('search hit is outside its training shard')
                if task['direction']=='phopt_to_env':
                    if not q.startswith('phopt_'):
                        raise ValueError('reverse PHOPT query differs')
                    reason = 'PHOPT_reverse_homology'
                else:
                    val_key = t if task['direction']=='train_to_validation' else q
                    if not val_key.startswith('env_') or selection.get(int(val_key[4:]))!=('validation',-1):
                        raise ValueError('cross-split hit is outside fixed validation')
                    reason = 'pHenv_validation_homology'
                removed[rid].add(reason)
        evidence.append({'task':task,'complete_sha256':sha_file(run/'complete.json')})
    return cert,selection,removed,evidence


def finalize(plan,audit,forward,output):
    checked_audit(audit)
    cert,selection,removed,evidence = exclusions(plan,audit,forward)
    output.mkdir(parents=True,exist_ok=False)
    db = connect(audit)
    org_counts = Counter()
    query = ('SELECT row_id,accession,organism,phenv,published_split,sequence,normalized_sha FROM records '
             'JOIN search_candidates USING(row_id) ORDER BY row_id')
    for rid,_,org,_,_,_,_ in db.execute(query):
        if selection.get(rid,('',))[0]=='train' and rid not in removed:
            org_counts[org] += 1
    ntrain, norg = sum(org_counts.values()), len(org_counts)
    if not ntrain:
        raise ValueError('homology purge emptied training')
    diagnostics = {s:Counter() for s in ('train','validation')}
    sources = {s:defaultdict(set) for s in diagnostics}
    with (output/'records.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=FIELDS)
        writer.writeheader()
        for rid,acc,org,y,published,raw,key in db.execute(query):
            if rid not in selection or rid in removed:
                continue
            split = selection[rid][0]
            weight = ntrain/(norg*org_counts[org]) if split=='train' else ''
            seq = raw.translate(TRANSLATE)
            writer.writerow(dict(zip(FIELDS,[f'env_{rid}',rid,acc,org,y,published,split,seq,key,weight])))
            region = 'acid' if y<=4 else 'alkaline' if y>=10 else 'core'
            diagnostics[split]['sequences'] += 1
            diagnostics[split]['residues'] += len(seq)
            diagnostics[split][region+'_sequences'] += 1
            sources[split]['all'].add(org)
            sources[split][region].add(org)
    db.close()
    if sources['train']['all'] & sources['validation']['all']:
        raise ValueError('organism leakage')
    for split in diagnostics:
        diagnostics[split]['organisms'] = len(sources[split]['all'])
        for region in ('acid','alkaline','core'):
            diagnostics[split][region+'_organisms'] = len(sources[split][region])
        if any(not diagnostics[split][r+'_sequences'] for r in ('acid','alkaline','core')):
            raise ValueError('a retained split has an empty pHenv stratum')
    with (output/'exclusions.tsv').open('w',newline='') as f:
        writer = csv.writer(f,delimiter='\t')
        writer.writerow(['key','reason'])
        writer.writerows((f'env_{rid}','|'.join(sorted(reasons))) for rid,reasons in sorted(removed.items()))
    result = {'state':'complete','scope':cert['scope'],'diagnostics':diagnostics,
        'seed':42,'records_sha256':sha_file(output/'records.csv'),'exclusions_sha256':sha_file(output/'exclusions.tsv'),
        'plan_sha256':sha_file(plan/'plan.json'),'audit_report_sha256':sha_file(audit/'report.json'),
        'pilot_complete_sha256':cert['pilot_complete_sha256'],'pilot_verification_sha256':cert['pilot_verification_sha256'],
        'full_forward_complete_sha256':sha_file(forward/'complete.json'),'searches':evidence,
        'source_sha256':sha_file(Path(__file__)),'phopt_labels_consumed':False,
        'train_weight_policy':'equal total weight per retained training organism, mean training weight 1',
        'removed_training_sequences':len(removed),'independent_verification_required':True,
        'limitations':['Fixed pilot validation is capped; not all heldout-organism proteins are used.',
            'One growth pH per organism name is not one independent label per protein.',
            'Sharded heuristic searches are not proof of absence of remote homology.']}
    write_json(output/'complete.json',result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['plan','search','finalize'])
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--pilot',type=Path)
    p.add_argument('--plan',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--forward',type=Path)
    p.add_argument('--shard-size',type=int,default=50000)
    p.add_argument('--task-index',type=int)
    p.add_argument('--mmseqs',type=Path)
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--memory',default='8G')
    p.add_argument('--dry-run',action='store_true')
    a = p.parse_args()
    if a.stage=='plan':
        if a.pilot is None or a.output is None:
            p.error('plan requires --pilot and --output')
        result = make_plan(a.audit,a.pilot,a.output,a.shard_size)
    elif a.stage=='search':
        if a.plan is None or a.task_index is None or a.mmseqs is None:
            p.error('search requires --plan, --task-index and --mmseqs')
        result = run_search(a.plan,a.audit,a.task_index,a.mmseqs,a.threads,a.memory,a.dry_run)
    else:
        if a.plan is None or a.forward is None or a.output is None:
            p.error('finalize requires --plan, --forward and --output')
        result = finalize(a.plan,a.audit,a.forward,a.output)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
