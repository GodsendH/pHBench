import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import torch
from scripts.build_fixed_test_datasets import audit_inputs,near,build_conditions,bin_key
from collections import Counter
from phgeofuse.io import ProteinRecord,read_fasta
from phgeofuse.datasets import apply_dataset,validate_fixed_test_inputs
from phgeofuse.retrieval import RetrievalStore,ensure_retrieval_store,record_key

class FixedTestTests(unittest.TestCase):
    def test_validation_changes_with_threshold_and_both_random_splits_match(self):
        train=[ProteinRecord(f'T{i}','ACD','train',7) for i in range(20)]
        valid=[ProteinRecord(f'V{i}','EFG','validation',7) for i in range(10)]
        scores={r.protein_id:(i+1)*5 for rs in (train,valid) for i,r in enumerate(rs)}
        conditions=build_conditions(train,valid,lambda r,p:scores[r.protein_id]>=p)
        self.assertEqual(len(conditions),25)
        self.assertEqual(len(conditions['fixedtest_control']['validation']),10)
        self.assertEqual([len(conditions[f'identity{p}']['validation']) for p in (100,50,30,20)],[10,9,5,3])
        repeated=build_conditions(train,valid,lambda r,p:scores[r.protein_id]>=p)
        for p in (100,50,30,20):
            for seed in range(5):
                name=f'fixedtest_random{p}_seed{seed}'
                for split in ('train','validation'):
                    selected=conditions[name][split]
                    self.assertEqual(Counter(map(bin_key,selected)),Counter(map(bin_key,conditions[f'identity{p}'][split])))
                    self.assertEqual([r.protein_id for r in selected],[r.protein_id for r in repeated[name][split]])
                    self.assertTrue(all(r.split==split for r in selected))
        # Random deletion must sample from the original validation pool, not the homology-filtered pool.
        strict={r.protein_id for r in conditions['identity20']['validation']}
        self.assertTrue(any({r.protein_id for r in conditions[f'fixedtest_random20_seed{s}']['validation']}-strict for s in range(5)))

    def test_inclusive_threshold_and_bilateral_coverage(self):
        h=dict(identity=.3,query_coverage=.8,target_coverage=.8)
        self.assertTrue(near(h,30))
        self.assertFalse(near({**h,'identity':.299999},30))
        self.assertFalse(near({**h,'target_coverage':.79999},30))
        self.assertFalse(near({**h,'query_coverage':.79999},30))

    def test_original_duplicates_and_conflicts_are_retained_without_reweighting(self):
        rs=[ProteinRecord('A','ACD','train',7),ProteinRecord('B','ACD','train',8),
            ProteinRecord('C','EFG','train',6),ProteinRecord('D','EFG','train',6)]
        rs[1].sample_weight=3
        audit=[];out=audit_inputs(rs,audit)
        self.assertEqual(out,rs)
        self.assertEqual([r.ph_opt for r in out],[7,8,6,6])
        self.assertEqual([r.sample_weight for r in out],[1,3,1,1])
        self.assertTrue(all(r['action']=='kept' for r in audit))
        self.assertEqual([r['distinct_labels'] for r in audit],[2,2,1,1])
        conditions=build_conditions(out,[],lambda r,p:r.sequence=='ACD')
        self.assertEqual(conditions['fixedtest_control']['train'],rs)
        self.assertEqual([r.protein_id for r in conditions['identity20']['train']],['C','D'])

    def test_invalid_numeric_input_stops_instead_of_silent_filtering(self):
        for field in ('ph_opt','sample_weight'):
            r=ProteinRecord('A','ACD','train',7)
            setattr(r,field,float('nan'))
            with self.assertRaisesRegex(ValueError,'Nonfinite'):audit_inputs([r],[])

    def test_replaced_data_and_old_checkpoint_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)/'data/datasets/identity20';folder.mkdir(parents=True)
            hashes={};records=[]
            for s,filename in [('train','training'),('validation','validation'),('test','testing')]:
                p=folder/f'phopt_{filename}.fasta';p.write_text(f'>P_{s} | org | 1.1.1.1 | 7 | 1\nACD\n')
                hashes[s]=hashlib.sha256(p.read_bytes()).hexdigest();records.extend(read_fasta(p,s))
            (folder/'metadata.json').write_text(json.dumps(dict(protocol='fixed_test_removal_v3',file_hashes=hashes)))
            config=apply_dataset({'_root':d},'identity20')
            validate_fixed_test_inputs(records,config)
            bad=copy.deepcopy(records);bad[0].ph_opt=8
            with self.assertRaisesRegex(ValueError,'Manifest'):validate_fixed_test_inputs(bad,config)
            checkpoint=Path(d)/'old.pt';torch.save({'config':{}},checkpoint)
            with self.assertRaisesRegex(ValueError,'checkpoint'):validate_fixed_test_inputs(records,config,checkpoint)
            v2_fingerprint=hashlib.sha256(json.dumps({'protocol':'fixed_test_removal_v2','hashes':hashes},sort_keys=True).encode()).hexdigest()
            torch.save({'config':{'data':{'dataset_fingerprint':v2_fingerprint}}},checkpoint)
            with self.assertRaisesRegex(ValueError,'checkpoint'):validate_fixed_test_inputs(records,config,checkpoint)
            old_fingerprint=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
            torch.save({'config':{'data':{'dataset_fingerprint':old_fingerprint}}},checkpoint)
            with self.assertRaisesRegex(ValueError,'checkpoint'):validate_fixed_test_inputs(records,config,checkpoint)
            (folder/'phopt_training.fasta').write_text('changed')
            with self.assertRaisesRegex(ValueError,'FASTAs'):apply_dataset({'_root':d},'identity20')

    def test_old_superset_cache_is_rebuilt(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'retrieval.pt';r=ProteinRecord('P','ACD','train',7,status='ready')
            torch.save({'rows':{record_key(r):{}},'training_keys':[record_key(r),'train::REMOVED'],
                'dataset_fingerprint':'new'},p)
            config={'_root':d,'paths':{'retrieval':str(p)},'data':{'dataset_fingerprint':'new'}}
            replacement=RetrievalStore({})
            with mock.patch.object(RetrievalStore,'build',return_value=replacement) as build:
                self.assertIs(ensure_retrieval_store([r],config),replacement)
                build.assert_called_once()

if __name__=='__main__':unittest.main()
