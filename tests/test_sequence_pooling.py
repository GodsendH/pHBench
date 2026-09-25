import unittest
from types import SimpleNamespace
import numpy as np
import torch
from phgeofuse.sequence_pooling import encode_residue_pools
from phgeofuse.robust_fusion import pool_features

class SequencePoolingTests(unittest.TestCase):
    def test_all_residues_and_no_special_tokens_across_chunks(self):
        def tokenize(sequence,**kwargs):
            return {'input_ids':torch.tensor([[1000,*[ord(x) for x in sequence],2000]])}
        def model(input_ids):return SimpleNamespace(last_hidden_state=input_ids[...,None].float())
        sequence='ACDEXG'
        mean,std=encode_residue_pools(sequence,tokenize,model,'cpu',chunk_length=4)
        values=np.array([ord(x) for x in sequence],dtype=np.float32)
        np.testing.assert_allclose(mean,[values.mean()],atol=1e-6)
        np.testing.assert_allclose(std,[values.std()],atol=1e-6)

    def test_normalization_is_per_sample(self):
        mean=np.array([[3.,4.],[6.,8.]])
        std=np.array([[0.,2.],[0.,4.]])
        np.testing.assert_allclose(pool_features(mean,std,'mean_std'),[[.6,.8,0,1],[.6,.8,0,1]])
        with self.assertRaises(ValueError):pool_features(mean,np.zeros_like(std),'mean_std')

if __name__=='__main__':unittest.main()
