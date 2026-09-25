"""Profile real pHenv homology search and frozen ESM encoding before scaling."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from localph.phenv_data import sha_file, write_json, TRANSLATE


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mmseqs',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,default=Path.home()/'.cache/torch/hub/checkpoints/esm1v_t33_650M_UR90S_1.pt')
    p.add_argument('--search-size',type=int,default=8192)
    p.add_argument('--threads',type=int,default=8)
    a=p.parse_args()
    if not 1<=a.search_size<=16384:
        raise ValueError('profiling sample must be 1..16384')
    a.output.mkdir(parents=True,exist_ok=False)
    report=json.loads((a.audit/'report.json').read_text())
    for filename in ('manifest.sqlite','phopt_quarantine.fasta'):
        if sha_file(a.audit/filename)!=report['artifact_sha256'][filename]:
            raise ValueError('audit artifact differs')
    db=sqlite3.connect(f'file:{(a.audit/"manifest.sqlite").resolve()}?mode=ro',uri=True)
    ids=np.array([r[0] for r in db.execute('SELECT row_id FROM search_candidates ORDER BY row_id')])
    rng=np.random.default_rng(42)
    chosen=np.sort(rng.choice(ids,size=min(a.search_size,len(ids)),replace=False))
    rows=[]
    for i in range(0,len(chosen),500):
        batch=chosen[i:i+500].tolist()
        rows.extend(db.execute('SELECT row_id,sequence,length FROM records WHERE row_id IN ('+','.join('?' for _ in batch)+') ORDER BY row_id',batch).fetchall())
    query=a.output/'profile_queries.fasta'
    query.write_text(''.join(f'>env_{i}\n{seq.translate(TRANSLATE)}\n' for i,seq,_ in rows))
    count=int(report['phopt']['sequence_rows'])
    command=[str(a.mmseqs.resolve()),'easy-search',str(query.resolve()),str((a.audit/'phopt_quarantine.fasta').resolve()),
        str((a.output/'hits.tsv').resolve()),str((a.output/'tmp_search').resolve()),
        '-s','7.5','--min-seq-id','0.2','-c','0.8','--cov-mode','0','--alignment-mode','3',
        '--seq-id-mode','0','-e','100','--max-seqs',str(count),'--max-accept','2147483647',
        '--max-rejected','2147483647','--threads',str(a.threads),
        '--format-output','query,target,fident,qcov,tcov,evalue,bits','--remove-tmp-files','1']
    protocol={'seed':42,'source_code_sha256':sha_file(Path(__file__)),
        'audit_report_sha256':sha_file(a.audit/'report.json'),'query_rows':len(rows),'query_ids':chosen.tolist(),
        'query_sha256':sha_file(query),'mmseqs_version':subprocess.check_output([str(a.mmseqs),'version'],text=True).strip(),
        'command':command,'phopt_labels_consumed':False,'scope':'resource profile only; not a complete quarantine or training split'}
    write_json(a.output/'protocol.json',protocol)
    start=time.monotonic()
    with (a.output/'search.log').open('w') as f:
        subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
    search_seconds=time.monotonic()-start
    hits=set()
    with (a.output/'hits.tsv').open() as f:
        for line in f:
            fields=line.rstrip().split('\t')
            # Require actual reported bidirectional coverage and identity as well.
            if float(fields[2])>=.2 and min(map(float,fields[3:5]))>=.8:
                hits.add(fields[0])
    result={'search_seconds':search_seconds,'search_queries':len(rows),'matched_queries':len(hits),
        'full_search_linear_hours':search_seconds*len(ids)/len(rows)/3600,
        'estimate_limit':'Linear extrapolation includes fixed startup; full query I/O, hit density and parallel efficiency may differ.'}
    print(json.dumps(result),flush=True)
    write_json(a.output/'search_profile.json',result)
    import torch
    import esm
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    t=time.monotonic()
    model,alphabet=esm.pretrained.load_model_and_alphabet_local(str(a.checkpoint))
    model=model.float().cuda().eval().requires_grad_(False)
    convert=alphabet.get_batch_converter()
    load_seconds=time.monotonic()-t
    edges=[0,128,256,384,512,768,1022]
    pop=Counter()
    for length,n in db.execute('SELECT length,COUNT(*) FROM records JOIN search_candidates USING(row_id) GROUP BY length'):
        pop[int(np.searchsorted(edges[1:],length,side='left'))]+=n
    strata=[]
    for b,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        sample=[r for r in rows if lo<r[2]<=hi][:6]
        if pop[b] and not sample:
            raise ValueError('profile sample has an empty population stratum')
        times=[]
        for i,seq,length in sample:
            _,_,tokens=convert([(str(i),seq.translate(TRANSLATE))])
            tokens=tokens.cuda()
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            begin=time.monotonic()
            with torch.inference_mode():
                h=model(tokens,repr_layers=[33],return_contacts=False)['representations'][33][0,1:-1]
                packed=h.half().cpu().numpy()
            torch.cuda.synchronize()
            elapsed=time.monotonic()-begin
            if packed.shape!=(length,1280) or not np.isfinite(packed).all():
                raise ValueError('invalid ESM token coverage')
            times.append({'row_id':i,'length':length,'seconds':elapsed,'peak_cuda_bytes':torch.cuda.max_memory_allocated()})
        row={'length_lower_exclusive':lo,'length_upper_inclusive':hi,'population':pop[b],'measurements':times,
             'mean_seconds':float(np.mean([r['seconds'] for r in times])) if times else 0.}
        strata.append(row)
        print(json.dumps({'event':'encoding_profile',**row}),flush=True)
    db.close()
    result.update({'encoder':'esm1v_t33_650M_UR90S_1','encoder_checkpoint_sha256':sha_file(a.checkpoint),
        'torch':torch.__version__,'cuda_gpu':torch.cuda.get_device_name(0),'batch_size':1,'encoder_precision':'float32',
        'stored_token_precision':'float16','load_seconds':load_seconds,'length_strata':strata,
        'full_encoding_estimated_gpu_hours':sum(r['population']*r['mean_seconds'] for r in strata)/3600,
        'full_token_cache_bytes':report['candidate_residues']*1280*2,
        'requires_cluster_for_full_encoding':True,'trained_model':False,'homology_full_pass_complete':False})
    write_json(a.output/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='length_strata'}),flush=True)


if __name__=='__main__':
    main()
