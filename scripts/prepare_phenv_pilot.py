"""Prepare a bounded organism-disjoint pilot with two-way homology purging.

Selection uses pHenv labels only, never PHOPT labels. This is an explicit
subsample for feasibility/transfer experiments, not the full-data endpoint.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import heapq
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from localph.phenv_data import sha_file, write_json, TRANSLATE


def rank(text,seed=42):
    return int.from_bytes(hashlib.sha256(f'{seed}|{text}'.encode()).digest()[:8],'big')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--mmseqs',required=True,type=Path)
    p.add_argument('--per-organism',type=int,default=8)
    p.add_argument('--per-extreme-organism',type=int,default=32)
    p.add_argument('--threads',type=int,default=4)
    a=p.parse_args()
    if min(a.per_organism,a.per_extreme_organism)<=0:
        raise ValueError('sampling caps must be positive')
    a.output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    report=json.loads((a.audit/'report.json').read_text())
    for name in ('manifest.sqlite','phopt_quarantine.fasta'):
        if sha_file(a.audit/name)!=report['artifact_sha256'][name]:
            raise ValueError('audit changed')
    if report['organisms_with_multiple_labels']:
        raise ValueError('pilot requires resolving organism labels first')
    db=sqlite3.connect(f'file:{(a.audit/"manifest.sqlite").resolve()}?mode=ro',uri=True)
    chosen=defaultdict(list); org_label={}
    query='SELECT row_id,organism,phenv,normalized_sha FROM records JOIN search_candidates USING(row_id) ORDER BY row_id'
    for rid,org,y,key in db.execute(query):
        org_label[org]=y
        cap=a.per_extreme_organism if y<=4 or y>=10 else a.per_organism
        item=(-rank(key),rid)
        if len(chosen[org])<cap: heapq.heappush(chosen[org],item)
        elif item>chosen[org][0]: heapq.heapreplace(chosen[org],item)
    strata=defaultdict(list)
    for org,y in org_label.items():
        strata['acid' if y<=4 else 'alkaline' if y>=10 else 'core'].append(org)
    val_org=set()
    for name,orgs in strata.items():
        orgs.sort(key=lambda x:(rank('organism:'+x),x))
        n=min(len(orgs)-1,max(1,round(.2*len(orgs))))
        if n<1: raise ValueError(f'insufficient organisms for {name}')
        val_org.update(orgs[:n])
    ids=sorted(rid for items in chosen.values() for _,rid in items)
    rows={}
    for i in range(0,len(ids),500):
        batch=ids[i:i+500]
        query='SELECT row_id,accession,organism,phenv,published_split,sequence,normalized_sha FROM records WHERE row_id IN ('+','.join('?' for _ in batch)+')'
        for rid,acc,org,y,split,seq,key in db.execute(query,batch):
            rows[f'env_{rid}']={'key':f'env_{rid}','source_row':rid,'accession':acc,'organism':org,'phenv':y,
                'published_split':split,'split':'validation' if org in val_org else 'train',
                'sequence':seq.translate(TRANSLATE),'normalized_sha':key}
    db.close()
    def fasta(name,keys):
        path=a.output/name
        path.write_text(''.join(f'>{key}\n{rows[key]["sequence"]}\n' for key in keys))
        return path.resolve()
    keys=sorted(rows,key=lambda k:rows[k]['source_row'])
    all_fasta=fasta('selected_before_search.fasta',keys)
    searches=[]
    def search(name,q,t,ntarget):
        dest=(a.output/(name+'.tsv')).resolve()
        command=[str(a.mmseqs.resolve()),'easy-search',str(q),str(t),str(dest),str((a.output/('tmp_'+name)).resolve()),
          '-s','7.5','--min-seq-id','0.2','-c','0.8','--cov-mode','0','--alignment-mode','3','--seq-id-mode','0',
          '-e','100','--max-seqs',str(ntarget),'--max-accept','2147483647','--max-rejected','2147483647',
          '--threads',str(a.threads),'--split-memory-limit','8G',
          '--format-output','query,target,fident,qcov,tcov,evalue,bits','--remove-tmp-files','1']
        t0=time.monotonic()
        with (a.output/(name+'.log')).open('w') as f: subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
        searches.append({'name':name,'command':command,'seconds':time.monotonic()-t0,'hits_sha256':sha_file(dest)})
        hits=[]
        with dest.open() as f:
            for line in f:
                qid,tid,ident,qc,tc,_,_=line.rstrip().split('\t')
                if float(ident)>=.2 and min(float(qc),float(tc))>=.8: hits.append((qid,tid))
        print(json.dumps({'search':name,'pairs':len(hits),'seconds':searches[-1]['seconds']}),flush=True)
        return hits
    target=(a.audit/'phopt_quarantine.fasta').resolve()
    blocked=set(q for q,t in search('env_to_phopt',all_fasta,target,report['phopt']['sequence_rows']))
    blocked.update(t for q,t in search('phopt_to_env',target,all_fasta,len(keys)))
    if not blocked<=set(rows): raise ValueError('unknown homology hit')
    remaining=[k for k in keys if k not in blocked]
    train=[k for k in remaining if rows[k]['split']=='train']
    val=[k for k in remaining if rows[k]['split']=='validation']
    if not train or not val: raise ValueError('PHOPT quarantine emptied a split')
    train_fa=fasta('train_before_purge.fasta',train); val_fa=fasta('validation.fasta',val)
    purged=set(q for q,t in search('train_to_validation',train_fa,val_fa,len(val)))
    purged.update(t for q,t in search('validation_to_train',val_fa,train_fa,len(train)))
    if not purged<=set(train): raise ValueError('purge must only remove training rows')
    train=[k for k in train if k not in purged]
    if not train: raise ValueError('homology purge emptied training')
    org_counts=Counter(rows[k]['organism'] for k in train)
    # Equal total loss mass per retained training organism, not per protein.
    norg=len(org_counts)
    for k in train:
        rows[k]['train_weight']=len(train)/(norg*org_counts[rows[k]['organism']])
    for k in val: rows[k]['train_weight']=''
    final=train+val
    fields=list(rows[final[0]])
    with (a.output/'records.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows[k] for k in final)
    with (a.output/'exclusions.tsv').open('w',newline='') as f:
        w=csv.writer(f,delimiter='\t'); w.writerow(['key','reason'])
        w.writerows((k,'PHOPT_homology') for k in sorted(blocked))
        w.writerows((k,'pHenv_validation_homology') for k in sorted(purged))
    fasta('train.fasta',train)
    diagnostics={}
    for split,ks in [('train',train),('validation',val)]:
        diagnostics[split]={'sequences':len(ks),'organisms':len(set(rows[k]['organism'] for k in ks)),
            'residues':sum(len(rows[k]['sequence']) for k in ks),
            'acid_sequences':sum(rows[k]['phenv']<=4 for k in ks),'alkaline_sequences':sum(rows[k]['phenv']>=10 for k in ks),
            'acid_organisms':len(set(rows[k]['organism'] for k in ks if rows[k]['phenv']<=4)),
            'alkaline_organisms':len(set(rows[k]['organism'] for k in ks if rows[k]['phenv']>=10))}
    if set(rows[k]['organism'] for k in train)&set(rows[k]['organism'] for k in val):
        raise ValueError('organism isolation failed')
    result={'state':'complete','scope':'organism-capped pilot; not the full pHenv dataset',
       'seed':42,'per_organism':a.per_organism,'per_extreme_organism':a.per_extreme_organism,
       'organism_validation_fraction':.2,'organism_strata':'pHenv<=4, 4<pHenv<10, pHenv>=10',
       'selected_before_search':len(keys),'removed_phopt_homology':len(blocked),'removed_validation_homology':len(purged),
       'diagnostics':diagnostics,'train_weight_policy':'equal total weight per retained training organism, mean training weight 1',
       'audit_report_sha256':sha_file(a.audit/'report.json'),'source_code_sha256':sha_file(Path(__file__)),
       'records_sha256':sha_file(a.output/'records.csv'),'phopt_labels_consumed':False,
       'mmseqs_version':subprocess.check_output([str(a.mmseqs),'version'],text=True).strip(),'searches':searches,
       'limitations':['No observed >=20% identity / >=80% coverage in either search direction across retained splits; heuristic search can miss remote homologs.',
          'Organism names, not strain-resolved taxonomy IDs, are the grouping key.',
          'Few independent extreme organisms; a pilot cannot establish full-data transfer or enzyme pHopt improvement.'],
       'seconds':time.monotonic()-start}
    write_json(a.output/'complete.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='searches'}),flush=True)


if __name__=='__main__': main()
