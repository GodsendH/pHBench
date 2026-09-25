"""Versioned fixed-test PHOPT removal experiment; never edits official FASTAs."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phgeofuse.io import read_fasta

FILES = dict(train='phopt_training.fasta', validation='phopt_validation.fasta', test='phopt_testing.fasta')
THRESHOLDS = (100, 50, 30, 20)
PROTOCOL = 'fixed_test_removal_v3'
FORMAT = 'query,target,nident,alnlen,qstart,qend,tstart,tend,evalue,bits,qlen,tlen'

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def dump(p, value):
    Path(p).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')

def tsv(p, rows, fields):
    with Path(p).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter='\t', extrasaction='ignore', lineterminator='\n')
        w.writeheader(); w.writerows(rows)

def fasta(p, records):
    with Path(p).open('w') as f:
        for r in records:
            f.write(f'>{r.protein_id} | {r.organism} | {r.ec} | {r.ph_opt:.12g} | {r.sample_weight:.12g}\n{r.sequence}\n')

def audit_inputs(records, audit):
    """Validate and describe original records without deduplication or label filtering."""
    groups = defaultdict(list)
    for r in records:
        if not math.isfinite(r.ph_opt) or not math.isfinite(r.sample_weight):
            raise ValueError(f'Nonfinite label/weight: {r.protein_id}')
        groups[r.sequence].append(r)
    for group in groups.values():
        labels = {r.ph_opt for r in group}
        for r in group:
            audit.append(dict(split=r.split, protein_id=r.protein_id,
                sequence_sha256=r.sequence_sha256, sequence_group_size=len(group),
                distinct_labels=len(labels), label_span=max(labels)-min(labels),
                label=r.ph_opt, action='kept'))
    return list(records)

def bin_key(r):
    return (math.floor(r.ph_opt), min(len(r.sequence) // 250, 4))

def near(hit, percent):
    return (hit['identity'] >= percent/100 and hit['query_coverage'] >= .8 and hit['target_coverage'] >= .8)

def matched_random(pool, reference, seed):
    """Sample within each original split, preserving the removal group's strata counts."""
    pools=defaultdict(list)
    for r in pool:pools[bin_key(r)].append(r)
    quotas=Counter(map(bin_key,reference));rng=random.Random(seed);ids=set()
    for key,n in sorted(quotas.items()):ids.update(r.protein_id for r in rng.sample(pools[key],n))
    selected=[r for r in pool if r.protein_id in ids]
    assert Counter(map(bin_key,selected))==quotas
    return selected

def build_conditions(train, valid, blocked):
    conditions={'fixedtest_control':{'train':train,'validation':valid}}
    for p in THRESHOLDS:
        filtered={s:[r for r in pool if not blocked(r,p)] for s,pool in [('train',train),('validation',valid)]}
        conditions[f'identity{p}']=filtered
        for seed in range(5):
            conditions[f'fixedtest_random{p}_seed{seed}']={
                'train':matched_random(train,filtered['train'],seed),
                'validation':matched_random(valid,filtered['validation'],100000+seed)}
    return conditions

def stats(rs):
    return dict(count=len(rs), unique_sequences=len({r.sequence for r in rs}),
        ph_bins=dict(sorted(Counter(str(math.floor(r.ph_opt)) for r in rs).items())),
        ec_classes=dict(Counter(r.ec.split('.')[0] for r in rs)),
        min_length=min(map(lambda r:len(r.sequence),rs)), max_length=max(map(lambda r:len(r.sequence),rs)))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--work', required=True)
    ap.add_argument('--mmseqs', default='mmseqs'); ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--install', action='store_true'); args=ap.parse_args()
    work=Path(args.work).resolve(); work.mkdir(parents=True,exist_ok=True)
    source={s:ROOT/'data'/f for s,f in FILES.items()}
    raw={s:read_fasta(p,s) for s,p in source.items()}
    audit=[]; train=audit_inputs(raw['train'],audit); valid=audit_inputs(raw['validation'],audit); test=raw['test']
    tsv(work/'input_audit.tsv',audit,['split','protein_id','sequence_sha256','sequence_group_size','distinct_labels','label_span','label','action'])
    (work/'cleaning.tsv').unlink(missing_ok=True)
    queries=train+valid
    # Synthetic search IDs avoid collisions between official train/validation IDs.
    qmap={f'q{i}':r for i,r in enumerate(queries)}; tmap={f't{i}':r for i,r in enumerate(test)}
    for name,mapping in [('queries',qmap),('test_search',tmap)]:
        (work/f'{name}.fasta').write_text(''.join(f'>{k}\n{r.sequence}\n' for k,r in mapping.items()))
    provenance=dict(protocol=PROTOCOL, source_hashes={s:digest(p) for s,p in source.items()},
        input_policy='all official records retained before similarity removal; no deduplication, label conflict removal, aggregation or reweighting',
        thresholds=list(THRESHOLDS),coverage=0.8,validation_policy='same_threshold_as_training',random_seeds=list(range(5)),
        random_policy='separate train and validation pH-by-length matched sampling; validation RNG seed=100000+seed',
        mmseqs_version=subprocess.check_output([args.mmseqs,'version'],text=True).strip(),commands=[])
    # Full target count, no hit/rejection truncation; search both orientations.
    hits=[]
    for name in ('forward','reverse','nofilter_audit'):
        reverse=name=='reverse'; result=work/f'{name}.tsv'
        q=work/('test_search.fasta' if reverse else 'queries.fasta')
        t=work/('queries.fasta' if reverse else 'test_search.fasta')
        cmd=[args.mmseqs,'easy-search',str(q),str(t),str(result),str(work/f'tmp_{name}'),
            '-s','7.5','-a','1','--alignment-mode','3','--seq-id-mode','0','--min-seq-id','0',
            '-c','0','-e','100','--max-seqs',str(max(len(queries),len(test))),
            '--max-accept','2147483647','--max-rejected','2147483647','--threads',str(args.threads),
            '--format-output',FORMAT,'--remove-tmp-files','1']
        if name=='nofilter_audit': cmd += ['--prefilter-mode','2']
        provenance['commands'].append(cmd)
        signature=hashlib.sha256(json.dumps([cmd,provenance['source_hashes'],digest(q),digest(t)]).encode()).hexdigest()
        marker=work/f'{name}.complete'
        if not(result.exists() and marker.exists() and marker.read_text()==signature):
            print(f'Running {name} MMseqs search',flush=True)
            with (work/f'{name}.log').open('w') as log: subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
            marker.write_text(signature)
        with result.open() as f:
            for line in f:
                a,b,ni,alen,qs,qe,ts,te,ev,bs,ql,tl=line.rstrip().split('\t')
                ident=int(ni)/int(alen)
                if int(ni)==0 and float(ev)<1e-20:
                    raise ValueError('MMseqs nident unavailable for a strong alignment; backtrace output is required')
                qc=(abs(int(qe)-int(qs))+1)/int(ql)
                tc=(abs(int(te)-int(ts))+1)/int(tl)
                if reverse: a,b,qc,tc,ql,tl=b,a,tc,qc,tl,ql
                qr,tr=qmap[a],tmap[b]
                hits.append(dict(split=qr.split,query_id=qr.protein_id,target_id=tr.protein_id,
                    identity=float(ident),query_coverage=float(qc),target_coverage=float(tc),alignment_length=int(alen),
                    evalue=float(ev),bitscore=float(bs),query_length=int(ql),target_length=int(tl),direction=name))
    # Exact sequence equality is checked independently of search sensitivity.
    exact=defaultdict(list)
    for r in test: exact[r.sequence].append(r)
    for r in queries:
        for t in exact[r.sequence]:
            hits.append(dict(split=r.split,query_id=r.protein_id,target_id=t.protein_id,identity=1.,query_coverage=1.,
                target_coverage=1.,alignment_length=len(r.sequence),evalue=0.,bitscore='',query_length=len(r.sequence),target_length=len(t.sequence),direction='exact_hash'))
    tsv(work/'alignments.tsv',hits,['split','query_id','target_id','identity','query_coverage','target_coverage','alignment_length','evalue','bitscore','query_length','target_length','direction'])
    full=[h for h in hits if near(h,0)]
    best={}
    for h in full:
        k=(h['split'],h['query_id'])
        if k not in best or h['identity']>best[k]['identity']: best[k]=h
    def blocked(r,p): return best.get((r.split,r.protein_id),{}).get('identity',-1)>=p/100
    conditions=build_conditions(train,valid,blocked)
    removals=[]
    for p in THRESHOLDS:
        if len(conditions[f'identity{p}']['train'])<32 or len(conditions[f'identity{p}']['validation'])<30:
            raise RuntimeError(f'identity{p}: insufficient training or validation records')
        for r in train+valid:
            if blocked(r,p): removals.append(dict(dataset=f'identity{p}',**best[(r.split,r.protein_id)]))
    tsv(work/'removal_log.tsv',removals,['dataset','split','query_id','target_id','identity','query_coverage','target_coverage','direction'])
    for lo,hi in [(20,30),(30,50),(50,100)]:
        for s in ('train','validation'):
            assert {r.protein_id for r in conditions[f'identity{lo}'][s]} <= {r.protein_id for r in conditions[f'identity{hi}'][s]}
    stage=work/'datasets'; stage.mkdir(exist_ok=True); summaries={}
    for name,splits in conditions.items():
        rs=splits['train'];validation=splits['validation']
        folder=stage/name; folder.mkdir(exist_ok=True)
        fasta(folder/FILES['train'],rs); fasta(folder/FILES['validation'],validation)
        shutil.copy2(source['test'],folder/FILES['test'])
        if name=='fixedtest_control':
            for split in ('train','validation'):shutil.copy2(source[split],folder/FILES[split])
        records=rs+validation+test
        reference_name=f'identity{name.split("random")[1].split("_")[0]}' if name.startswith('fixedtest_random') else name
        reference=conditions[reference_name]
        selected_keys={(r.split,r.protein_id) for r in rs+validation}
        ref_keys={(r.split,r.protein_id) for s in reference.values() for r in s}
        tsv(folder/'selection.tsv',[dict(split=r.split,protein_id=r.protein_id,
            retained=(r.split,r.protein_id) in selected_keys,
            reference_retained=(r.split,r.protein_id) in ref_keys,
            ph_bin=bin_key(r)[0],length_bin=bin_key(r)[1]) for r in train+valid],
            ['split','protein_id','retained','reference_retained','ph_bin','length_bin'])
        if name.startswith('identity'):
            assert not any(blocked(r,int(name[8:])) for r in validation)
        tsv(folder/'records.tsv',[dict(protein_id=r.protein_id,split=r.split,sequence_sha256=r.sequence_sha256,
            ph_opt=r.ph_opt,length=len(r.sequence),ec=r.ec) for r in records],['protein_id','split','sequence_sha256','ph_opt','length','ec'])
        ids={r.protein_id for r in rs}; nearest={}; local_nearest={}
        for h in hits:
            if h['split']!='train' or h['query_id'] not in ids:continue
            key=h['target_id']
            if key not in local_nearest or h['identity']>local_nearest[key]['identity']:local_nearest[key]=h
            if near(h,0) and (key not in nearest or h['identity']>nearest[key]['identity']):nearest[key]=h
        if name.startswith('identity'):
            assert not any(near(h,int(name[8:])) for h in nearest.values())
        tsv(folder/'test_nearest_neighbors.tsv',[
            dict(test_id=r.protein_id,full_coverage_train_id=nearest.get(r.protein_id,{}).get('query_id',''),
                 full_coverage_identity=nearest.get(r.protein_id,{}).get('identity',''),
                 local_train_id=local_nearest.get(r.protein_id,{}).get('query_id',''),
                 local_identity=local_nearest.get(r.protein_id,{}).get('identity',''),
                 local_query_coverage=local_nearest.get(r.protein_id,{}).get('query_coverage',''),
                 local_test_coverage=local_nearest.get(r.protein_id,{}).get('target_coverage','')) for r in test],
            ['test_id','full_coverage_train_id','full_coverage_identity','local_train_id','local_identity','local_query_coverage','local_test_coverage'])
        info=dict(protocol=provenance['protocol'],name=name,counts={s:stats(v) for s,v in [('train',rs),('validation',validation),('test',test)]},
            file_hashes={s:digest(folder/f) for s,f in FILES.items()},audit_directory=str(work),coverage=.8,
            identity_threshold=int(name[8:])/100 if name.startswith('identity') else None,
            validation_threshold=int(name[8:])/100 if name.startswith('identity') else None,
            random_reference=reference_name if name.startswith('fixedtest_random') else None,
            removal_counts={'train':len(train)-len(rs),'validation':len(valid)-len(validation)},
            source_hashes=provenance['source_hashes'],
            input_policy=provenance['input_policy'],
            sample_weight_policy='official weights preserved; provenance not reestimated',
            search_limitations='Union of two-direction sensitive and forward no-prefilter local alignments, E-value<=100. Not proof of absence of remote homology.',
            nearest_neighbor_policy='Maximum identity among emitted alignments, not maximum over all possible alignments; blank means no detected qualifying hit.')
        assert info['file_hashes']['test']==provenance['source_hashes']['test']
        for s,f in FILES.items():assert len(read_fasta(folder/f,s))==info['counts'][s]['count']
        dump(folder/'metadata.json',info);summaries[name]=info['counts']
    dump(work/'protocol.json',provenance); dump(work/'summary.json',summaries)
    print(json.dumps({n:{s:d['count'] for s,d in v.items()} for n,v in summaries.items()},indent=2),flush=True)
    if args.install:
        # Overwrite generated inputs in place. The user explicitly requested no new backup.
        for name in conditions:
            for rel in [Path('data/datasets')/name,Path('data/processed')/name,Path('data/features')/name]:
                old=ROOT/rel
                if not old.resolve().is_relative_to(ROOT.resolve()):raise ValueError(f'Unsafe overwrite path: {old}')
                if old.exists():
                    shutil.rmtree(old)
            artifact=ROOT/'artifacts/phgeofuse/datasets'/name
            if not artifact.resolve().is_relative_to(ROOT.resolve()):raise ValueError(f'Unsafe cache path: {artifact}')
            for filename in ('manifest.csv','manifest.failures.json','retrieval.pt'):
                (artifact/filename).unlink(missing_ok=True)
            shutil.copytree(stage/name,ROOT/'data/datasets'/name)
        dump(work/'installed.json',dict(protocol=PROTOCOL,backup=None,datasets=list(conditions)))

if __name__=='__main__': main()
