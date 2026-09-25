"""Quarantine observed pHenv homologs of all PHOPT sequences.

Run under run_bounded_local.py locally. A missing observed match is not proof
of absence of homology; the exact search recipe and every hit are retained.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from localph.phenv_data import sha_file, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mmseqs',type=Path,required=True)
    p.add_argument('--threads',type=int,default=8)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    cert=json.loads((a.audit/'report.json').read_text())
    for name in ('phenv_search.fasta','phopt_quarantine.fasta'):
        if sha_file(a.audit/name)!=cert['artifact_sha256'][name]:
            raise ValueError('audit search input changed')
    query=(a.audit/'phenv_search.fasta').resolve()
    target=(a.audit/'phopt_quarantine.fasta').resolve()
    hits=(a.output/'hits.tsv').resolve()
    command=[str(a.mmseqs.resolve()),'easy-search',str(query),str(target),str(hits),str((a.output/'tmp').resolve()),
        '-s','7.5','--min-seq-id','0.2','-c','0.8','--cov-mode','0','--alignment-mode','3','--seq-id-mode','0',
        '-e','100','--max-seqs',str(cert['phopt']['sequence_rows']),'--max-accept','2147483647',
        '--max-rejected','2147483647','--threads',str(a.threads),'--split-memory-limit','24G',
        '--format-output','query,target,fident,qcov,tcov,evalue,bits','--remove-tmp-files','1']
    protocol={'audit_report_sha256':sha_file(a.audit/'report.json'),'code_sha256':sha_file(Path(__file__)),
        'mmseqs_version':subprocess.check_output([str(a.mmseqs),'version'],text=True).strip(),
        'command':command,'candidate_count':cert['homology_search_candidates'],
        'identity_min':.2,'query_coverage_min':.8,'target_coverage_min':.8,'phopt_labels_consumed':False,
        'search_direction':'pHenv queries against every PHOPT sequence',
        'limitation':'Sensitive heuristic search, not an exhaustive homology proof. Reverse search has not been run.'}
    write_json(a.output/'protocol.json',protocol)
    start=time.monotonic()
    with (a.output/'mmseqs.log').open('w') as f:
        subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
    excluded=set(); lines=0
    with hits.open() as f:
        for line in f:
            q,t,ident,qcov,tcov,_,_=line.rstrip().split('\t')
            if not q.startswith('env_') or not t.startswith('phopt_'):
                raise ValueError('unexpected search identifiers')
            if float(ident)>=.2 and float(qcov)>=.8 and float(tcov)>=.8:
                excluded.add(int(q[4:])); lines+=1
    exclusions=a.output/'excluded_row_ids.json'
    write_json(exclusions,sorted(excluded))
    result={'state':'complete','seconds':time.monotonic()-start,'query_count':cert['homology_search_candidates'],
        'matching_pairs':lines,'excluded_query_count':len(excluded),
        'remaining_after_observed_homology':cert['homology_search_candidates']-len(excluded),
        'training_ready':False,'reason':'pHenv organism/family development split still required',
        'protocol_sha256':sha_file(a.output/'protocol.json'),'hits_sha256':sha_file(hits),
        'exclusions_sha256':sha_file(exclusions)}
    write_json(a.output/'complete.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
