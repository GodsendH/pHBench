"""Verify migration against historical inputs, independent of migration counters."""
import argparse
import csv
from pathlib import Path
import sys
import torch
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from phgeofuse.cache import atomic_json,sha256_file
from phgeofuse.io import read_manifest
from phgeofuse.retrieval import record_key,RetrievalStore
from phgeofuse.saprot import embedding_key
from phgeofuse.config import load_config
from phgeofuse.structures import select_chain


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--experiment',type=Path,required=True)
    args=ap.parse_args();out=args.experiment.resolve();torch.set_num_threads(2)
    old=read_manifest(out/'historical_manifest.csv');new=read_manifest(out/'manifest.csv')
    qualities={r['key']:r for r in csv.DictReader((out/'quality.csv').open())}
    assert len(old)==len(new)==len(qualities)==9855
    config=load_config(out/'baseline.yaml');structures=set();embeddings=set();graphs=set();changed=0
    for a,b in zip(old,new):
        assert (a.split,a.protein_id,a.sequence,a.ph_opt,a.sample_weight)==(b.split,b.protein_id,b.sequence,b.ph_opt,b.sample_weight)
        q=qualities[record_key(b)]
        assert 0<=b.mean_plddt<=100 and abs(float(q['mean_plddt'])-b.mean_plddt)<1e-8
        assert Path(b.graph_path).exists() and Path(b.embedding_path).exists()
        if a.structure_sha256!=b.structure_sha256:
            changed+=1
            if b.structure_sha256 not in structures:
                assert a.structure_source=='esmfold'
                assert sha256_file(a.structure_path)==a.structure_sha256
                assert sha256_file(b.structure_path)==b.structure_sha256
                before=Path(a.structure_path).read_text().splitlines();after=Path(b.structure_path).read_text().splitlines()
                assert len(before)==len(after)
                for x,y in zip(before,after):
                    if x.startswith(('ATOM  ','HETATM')):
                        assert x[:60]==y[:60] and x[66:]==y[66:]
                        assert abs(float(y[60:66])-float(x[60:66])*100)<1e-6
                    else:assert x==y
                _,residues=select_chain(b.structure_path,b.sequence)
                assert abs(np.mean([r.plddt for r in residues])-b.mean_plddt)<1e-6
                structures.add(b.structure_sha256)
            if b.graph_path not in graphs:
                ga=torch.load(a.graph_path,map_location='cpu');gb=torch.load(b.graph_path,map_location='cpu')
                for k in ga:
                    if k in ('metadata','plddt','node_features'):continue
                    assert torch.equal(ga[k],gb[k]),k
                torch.testing.assert_close(gb['plddt'],100*ga['plddt'])
                columns=[i for i in range(ga['node_features'].shape[1]) if i!=21]
                assert torch.equal(ga['node_features'][:,columns],gb['node_features'][:,columns])
                torch.testing.assert_close(gb['node_features'][:,21],gb['plddt']/100)
                assert gb['metadata']['structure_sha256']==b.structure_sha256
                graphs.add(b.graph_path)
        if a.embedding_path!=b.embedding_path and b.embedding_path not in embeddings:
            cache=torch.load(b.embedding_path,map_location='cpu')
            assert cache['metadata']['embedding_key']==embedding_key(b.sequence,b.three_di,config)
            assert cache['metadata']['sequence_sha256']==b.sequence_sha256
            assert cache['embedding'].shape==(len(b.sequence),1280)
            assert torch.isfinite(cache['embedding']).all()
            embeddings.add(b.embedding_path)
    store=RetrievalStore.load(out/'retrieval.pt')
    training=[r for r in new if r.split=='train']
    assert store.payload['training_keys']==[record_key(r) for r in training]
    assert store.payload['training_sequences']==[r.sequence for r in training]
    assert store.payload['dataset_fingerprint']==config['data']['dataset_fingerprint']
    assert set(store.rows)=={record_key(r) for r in new}
    assert store.payload['training_structure_paths']==[r.structure_path for r in training]
    result=dict(status='passed',samples=len(new),corrected_records=changed,checked_structures=len(structures),
                checked_graphs=len(graphs),checked_new_embeddings=len(embeddings),
                historical_inputs_unchanged=True,coordinates_unchanged=True,labels_and_splits_unchanged=True,
                retrieval_rebuilt=True,manifest_sha256=sha256_file(out/'manifest.csv'),retrieval_sha256=sha256_file(out/'retrieval.pt'))
    atomic_json(out/'confidence_verification.json',result);print(result)


if __name__=='__main__':main()
