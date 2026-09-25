"""Exercise the artifact contract with a tiny CPU pretraining run."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import torch
from localph.phenv_data import sha_file,write_json
from localph.environment_training import load_cache,ShardedTokens
from localph.environment_transfer import EnvironmentTransfer

ROOT=Path(__file__).resolve().parents[1]


class EnvironmentPipelineTests(unittest.TestCase):
    def test_shard_boundary_slices(self):
        cache=ShardedTokens([np.ones((5,3)),np.full((7,3),2)])
        np.testing.assert_array_equal(cache[1:5],np.ones((4,3)))
        np.testing.assert_array_equal(cache[5:12],np.full((7,3),2))
        with self.assertRaises(ValueError): cache[4:6]

    def test_cpu_pretraining_checkpoint_and_corruption_detection(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); data=root/'data'; cache=root/'cache'
            data.mkdir(); cache.mkdir()
            keys=np.array([f'k{i}' for i in range(6)])
            with (data/'records.csv').open('w',newline='') as f:
                w=csv.writer(f); w.writerow(['key','phenv','organism','split','train_weight'])
                for i,key in enumerate(keys): w.writerow([key,[3,7,11][i%3],f'org{i}','train' if i<3 else 'validation',1 if i<3 else ''])
            write_json(data/'complete.json',{'records_sha256':sha_file(data/'records.csv')})
            np.save(cache/'tokens.npy',np.random.default_rng(1).normal(size=(6*32,1280)).astype(np.float16))
            np.savez(cache/'index.npz',keys=keys,offsets=np.arange(7)*32)
            write_json(cache/'protocol.json',{'certificate_sha256':sha_file(data/'complete.json')})
            write_json(cache/'complete.json',{'state':'complete','hashes':{n:sha_file(cache/n) for n in ('tokens.npy','index.npz','protocol.json')}})
            result=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/pretrain_environment_encoder.py'),
                '--data',str(data),'--cache',str(cache),'--output',str(root/'fit'),'--max-epochs','1','--device','cpu'],
                capture_output=True,text=True,timeout=45)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            cert=json.loads((root/'fit/complete.json').read_text())
            self.assertEqual(cert['task'],'pHenv'); self.assertEqual(cert['selected_epoch'],1)
            payload=torch.load(root/'fit/weights.pt',map_location='cpu',weights_only=True)
            model=EnvironmentTransfer(); model.load_state_dict(payload['state_dict'])
            self.assertTrue(torch.equal(model.enzyme_head.weight,torch.zeros_like(model.enzyme_head.weight)))
            load_cache(cache)
            with (cache/'tokens.npy').open('ab') as f: f.write(b'corruption')
            with self.assertRaises(ValueError): load_cache(cache)


if __name__=='__main__': unittest.main()
