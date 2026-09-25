"""Independent post-install checks of exact thresholds and training support membership."""
import csv
import argparse
import hashlib
import json
import sys
from collections import Counter,defaultdict
from fractions import Fraction
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dataset_registry import dataset_fasta_paths
from phgeofuse.io import read_fasta,read_manifest
from phgeofuse.config import load_config
from phgeofuse.datasets import apply_dataset,validate_fixed_test_inputs

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--work',default=str(ROOT/'data/dataset_audits/fixed_test_removal_v3'))
    work=Path(parser.parse_args().work).resolve()
    summary=json.loads((work/'summary.json').read_text());protocol=json.loads((work/'protocol.json').read_text())
    official={s:read_fasta(p,s) for s,p in dataset_fasta_paths(ROOT,'phopt').items()}
    for s,p in dataset_fasta_paths(ROOT,'phopt').items():assert sha(p)==protocol['source_hashes'][s]
    raw_by_id={s:{r.protein_id:r for r in rs} for s,rs in official.items()}
    def blocks(p):
        with p.open() as f:
            return sum(1 for line in f if line.startswith('>'))
    # Independently reconstruct the full unfiltered official input pool.
    qs=[r for s in ('train','validation') for r in official[s]]
    kept={(r.split,r.protein_id) for r in qs}
    with (work/'input_audit.tsv').open() as f:input_audit=list(csv.DictReader(f,delimiter='\t'))
    assert len(input_audit)==len(qs)
    assert {(r['split'],r['protein_id']) for r in input_audit}==kept
    assert all(r['action']=='kept' for r in input_audit)
    assert not (work/'cleaning.tsv').exists()
    assert (work/'queries.fasta').read_text()==''.join(f'>q{i}\n{r.sequence}\n' for i,r in enumerate(qs))
    assert blocks(work/'queries.fasta')==len(qs)
    forbidden={p:set() for p in (100,50,30,20)}
    per_direction=defaultdict(set)
    for name in ('forward','reverse','nofilter_audit'):
        with (work/f'{name}.tsv').open() as f:
            for row in csv.reader(f,delimiter='\t'):
                a,b,ni,al,qs0,qe,ts,te,ev,bs,ql,tl=row
                if Fraction(abs(int(qe)-int(qs0))+1,int(ql))<Fraction(4,5):continue
                if Fraction(abs(int(te)-int(ts))+1,int(tl))<Fraction(4,5):continue
                index=int((b if name=='reverse' else a)[1:]);r=qs[index];key=(r.split,r.protein_id)
                identity=Fraction(int(ni),int(al))
                for p in forbidden:
                    if identity>=Fraction(p,100):forbidden[p].add(key)
                if identity>=Fraction(1,5):per_direction[name].add(key)
    test_sequences={r.sequence for r in official['test']}
    for r in qs:
        if r.sequence in test_sequences:
            for f in forbidden.values():f.add((r.split,r.protein_id))
    assert protocol['protocol']=='fixed_test_removal_v3'
    fingerprints=set();report={};val_hashes={}
    for name in summary:
        paths=dataset_fasta_paths(ROOT,name);splits={s:read_fasta(p,s) for s,p in paths.items()}
        assert sha(paths['test'])==protocol['source_hashes']['test']
        val_hashes[name]=sha(paths['validation'])
        if name.startswith('identity'):
            threshold=int(name[8:])
            for split in ('train','validation'):
                actual={(r.split,r.protein_id) for r in splits[split]}
                expected={key for key in kept if key[0]==split}-forbidden[threshold]
                assert actual==expected,(name,split)
        elif name=='fixedtest_control':
            for split in ('train','validation'):
                assert {(r.split,r.protein_id) for r in splits[split]}=={key for key in kept if key[0]==split}
                assert sha(paths[split])==protocol['source_hashes'][split]
        for s,rs in splits.items():
            assert len(rs)==summary[name][s]['count']
            for r in rs:
                old=raw_by_id[s][r.protein_id]
                assert (r.sequence,r.ph_opt,r.sample_weight)==(old.sequence,old.ph_opt,old.sample_weight)
        if name.startswith('fixedtest_random'):
            threshold=int(name.split('random')[1].split('_')[0])
            def distribution(rs):return Counter((int(r.ph_opt//1),min(len(r.sequence)//250,4)) for r in rs)
            for split in ('train','validation'):
                reference=read_fasta(dataset_fasta_paths(ROOT,f'identity{threshold}')[split],split)
                assert distribution(reference)==distribution(splits[split]),(name,split)
                assert {(r.split,r.protein_id) for r in splits[split]}<=kept
        config=apply_dataset(load_config(ROOT/'configs/phgeofuse_fixedtest_base_seed0.yaml'),name)
        manifest=read_manifest(ROOT/'artifacts/phgeofuse/datasets'/name/'manifest.csv')
        validate_fixed_test_inputs(manifest,config)
        assert all(r.status=='ready' for r in manifest)
        fingerprints.add(config['data']['dataset_fingerprint'])
        train={r.protein_id:r for r in splits['train']}
        for strategy in ('opt_retrieval','opt_random'):
            for split,rs in splits.items():
                suffix='valid' if split=='validation' else split
                p=ROOT/'data/processed'/name/'top5'/f'esm2_{strategy}'/f'retrieval_{suffix}.json'
                entries=json.loads(p.read_text());assert len(entries)==len(rs)
                for row,r in zip(entries,rs):
                    assert (row['opt_id'],row['opt_sequence'],float(row['opt_pH']))==(r.protein_id,r.sequence,r.ph_opt)
                    assert len(row['env_ids'])==len(set(row['env_ids']))==5
                    for pid,seq,label in zip(row['env_ids'],row['env_sequences'],row['env_pHs']):
                        assert pid in train and (seq,float(label))==(train[pid].sequence,train[pid].ph_opt)
                        assert split!='train' or pid!=r.protein_id
        report[name]={'counts':{s:len(rs) for s,rs in splits.items()},'official_test_identical':True,'support_membership_checked':True}
    assert len(fingerprints)==len(summary)
    for split in ('train','validation'):
        pools=[{r.protein_id for r in read_fasta(dataset_fasta_paths(ROOT,f'identity{p}')[split],split)} for p in (20,30,50,100)]
        assert all(a<=b for a,b in zip(pools,pools[1:]))
    assert len({val_hashes[f'identity{p}'] for p in (100,50,30,20)})==4
    info=dict(protocol=protocol['protocol'],datasets=report,validation_hashes=val_hashes,independent_rational_threshold_audit=True,
        nofilter_additional_exclusions_at20=len(per_direction['nofilter_audit']-(per_direction['forward']|per_direction['reverse'])),
        official_files_unchanged=True,all_conditions_have_unique_fingerprints=True,
        no_basic_cleaning=True,control_all_splits_byte_identical=True)
    (work/'final_audit.json').write_text(json.dumps(info,indent=2));print(json.dumps({k:v for k,v in info.items() if k not in ('datasets','validation_hashes')}))
    # TSV CRLF is unnecessary in a Linux repository. Leave FASTAs (especially test bytes) intact.
    for root in (ROOT/'data/datasets',work/'datasets'):
        for p in root.glob('*/*.tsv'):p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))

if __name__=='__main__':main()
