import csv
import json
from pathlib import Path
import sqlite3
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from localph.phenv_data import audit,sha_file,write_json
from scripts.prepare_phenv_expansion import make_plan,run_search,finalize,rank,FIELDS
from scripts.verify_phenv_expansion import verify


class ExpansionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.audit=self.root/'audit'
        self.pilot=self.root/'pilot'
        self.plan=self.root/'plan'
        self.forward=self.root/'forward'
        self.pilot.mkdir()
        self.forward.mkdir()
        source=self.root/'env.csv'
        phopt=self.root/'phopt.csv'
        alphabet='ACDEFGHIKLMNPQRSTVWY'
        rows=[]
        for i,y in enumerate([3.,3.,7.,7.,11.,11.]):
            for j in range(2):
                number=2*i+j
                rows.append([f'a{number}',f'org{i}',1.,y,'Training',alphabet[number]+'Q'*31])
        with source.open('w',newline='') as f:
            w=csv.writer(f)
            w.writerow(['Accession','Organism','Sample Weight','pHenv','Split','Sequence'])
            w.writerows(rows)
        with phopt.open('w',newline='') as f:
            w=csv.writer(f)
            w.writerow(['protein_id','split','sequence'])
            w.writerow(['p','test','W'*32])
        self.ac=audit(source,phopt,self.audit)
        val_org={min([f'org{i}',f'org{i+1}'],key=lambda v:(rank(v),v)) for i in (0,2,4)}
        with sqlite3.connect(self.audit/'manifest.sqlite') as db:
            raw=list(db.execute('SELECT row_id,accession,organism,phenv,published_split,sequence,normalized_sha FROM records'))
        val_rows=[]
        for rid,acc,org,y,split,seq,key in raw:
            if org in val_org and rid%2==1:
                val_rows.append(dict(zip(FIELDS,[f'env_{rid}',rid,acc,org,y,split,'validation',seq,key,''])))
        with (self.pilot/'records.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=FIELDS)
            w.writeheader()
            w.writerows(val_rows)
        pc={'records_sha256':sha_file(self.pilot/'records.csv'),'seed':42,'organism_validation_fraction':.2,
            'audit_report_sha256':sha_file(self.audit/'report.json')}
        write_json(self.pilot/'complete.json',pc)
        write_json(self.pilot/'verification.json',{'verified':True,'complete_sha256':sha_file(self.pilot/'complete.json')})
        self.cert=make_plan(self.audit,self.pilot,self.plan,shard_size=2)
        with sqlite3.connect(self.plan/'selection.sqlite') as db:
            self.train=[r[0] for r in db.execute("SELECT row_id FROM selected WHERE split='train' ORDER BY row_id")]
            self.val=[r[0] for r in db.execute("SELECT row_id FROM selected WHERE split='validation' ORDER BY row_id")]
        fp={'audit_report_sha256':sha_file(self.audit/'report.json'),
            'candidate_count':self.ac['homology_search_candidates'],
            'identity_min':.2,'query_coverage_min':.8,'target_coverage_min':.8}
        write_json(self.forward/'protocol.json',fp)
        (self.forward/'hits.tsv').write_text(f'env_{self.train[4]}\tphopt_0\t0.2\t0.8\t0.8\t1\t20\n')
        write_json(self.forward/'excluded_row_ids.json',[self.train[4]])
        fc={'state':'complete','protocol_sha256':sha_file(self.forward/'protocol.json'),
            'hits_sha256':sha_file(self.forward/'hits.tsv'),
            'exclusions_sha256':sha_file(self.forward/'excluded_row_ids.json')}
        write_json(self.forward/'complete.json',fc)

    def complete_searches(self):
        def fake_run(command,**kwargs):
            q,t=Path(command[2]),Path(command[3])
            ids=lambda path:{int(line[5:]) for line in path.read_text().splitlines() if line.startswith('>env_')}
            hits=[]
            if q.name=='phopt_quarantine.fasta' and self.train[0] in ids(t):
                hits.append(f'phopt_0\tenv_{self.train[0]}\t0.5\t1\t1\t1\t20')
            if q.name.startswith('train_') and self.train[2] in ids(q):
                hits.append(f'env_{self.train[2]}\tenv_{self.val[0]}\t0.5\t1\t1\t1\t20')
            Path(command[4]).write_text(''.join(line+'\n' for line in hits))
            return subprocess.CompletedProcess(command,0)
        with patch('scripts.prepare_phenv_expansion.subprocess.run',side_effect=fake_run), \
             patch('scripts.prepare_phenv_expansion.subprocess.check_output',return_value='18.8cc5c\n'):
            for task in self.cert['tasks']:
                run_search(self.plan,self.audit,task['index'],Path('/mock/mmseqs'))

    def test_full_expansion_retains_fixed_validation_and_reconstructs_exclusions(self):
        self.assertEqual(self.cert['counts_before_homology'],
            {'train':6,'validation':3,'unused_validation_organism_sequences':3})
        self.assertEqual(len(self.cert['tasks']),9)
        self.complete_searches()
        data=self.root/'ready'
        result=finalize(self.plan,self.audit,self.forward,data)
        self.assertEqual(result['removed_training_sequences'],3)
        checked=verify(self.audit,self.plan,self.forward,data)
        self.assertTrue(checked['verified'])
        self.assertEqual((checked['train'],checked['validation']),(3,3))
        self.assertEqual(checked['qualified_pairs_rechecked'],3)
        with (data/'records.csv').open(newline='') as f:
            rows=list(csv.DictReader(f))
        retained={int(r['source_row']) for r in rows}
        self.assertTrue(set(self.val)<=retained)
        self.assertTrue(set(self.train[::2]).isdisjoint(retained))
        self.assertTrue(all(float(r['train_weight'])==1 for r in rows if r['split']=='train'))

    def test_incomplete_searches_do_not_produce_training_ready_artifacts(self):
        with self.assertRaises(FileNotFoundError):
            finalize(self.plan,self.audit,self.forward,self.root/'not_ready')
        self.assertFalse((self.root/'not_ready').exists())
        self.assertFalse(self.cert['training_ready'])

    def test_search_dry_run_does_not_create_outputs_or_execute_a_binary(self):
        result=run_search(self.plan,self.audit,0,Path('/not-installed/mmseqs'),dry_run=True)
        self.assertTrue(result['dry_run'])
        self.assertFalse(result['submits_job'])
        self.assertFalse((self.plan/'searches').exists())
        self.assertEqual(result['command'][result['command'].index('--max-seqs')+1],'2')

    def test_new_phopt_hit_does_not_silently_change_fixed_validation(self):
        path=self.forward/'hits.tsv'
        path.write_text(path.read_text()+f'env_{self.val[0]}\tphopt_0\t0.3\t1\t1\t1\t20\n')
        write_json(self.forward/'excluded_row_ids.json',[self.train[4],self.val[0]])
        fc=json.loads((self.forward/'complete.json').read_text())
        fc['hits_sha256']=sha_file(path)
        fc['exclusions_sha256']=sha_file(self.forward/'excluded_row_ids.json')
        write_json(self.forward/'complete.json',fc)
        with self.assertRaisesRegex(ValueError,'fixed pilot validation'):
            finalize(self.plan,self.audit,self.forward,self.root/'not_ready')
        self.assertFalse((self.root/'not_ready').exists())

    def test_independent_verifier_rejects_relabeling_even_with_updated_output_hash(self):
        self.complete_searches()
        data=self.root/'ready'
        finalize(self.plan,self.audit,self.forward,data)
        with (data/'records.csv').open(newline='') as f:
            rows=list(csv.DictReader(f))
        rows[0]['phenv']=6.5
        with (data/'records.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        cert=json.loads((data/'complete.json').read_text())
        cert['records_sha256']=sha_file(data/'records.csv')
        write_json(data/'complete.json',cert)
        with self.assertRaisesRegex(ValueError,'raw source lineage'):
            verify(self.audit,self.plan,self.forward,data)


if __name__=='__main__':
    unittest.main()
