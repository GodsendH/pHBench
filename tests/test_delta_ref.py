import copy
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from phgeofuse.delta_ref.model import (DeltaNetwork, Standardizer, ReferencePredictor,
    select_panel, huber_location, weighted_median, blend_predictions)
from phgeofuse.delta_ref.metrics import metrics, acceptance, select_strength, seed_summary, paired_family_bootstrap
from phgeofuse.delta_ref.training import sample_pairs, save_bundle, load_bundle, fit_subset
from phgeofuse.delta_ref.data import assert_disjoint, validate_certificate, DevelopmentData


class DeltaRefTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(4)
        self.rng = np.random.default_rng(4)

    def test_antisymmetry_and_self_including_training_dropout(self):
        x, y = torch.randn(12, 9), torch.randn(12, 9)
        for kind in ("pair", "additive"):
            network = DeltaNetwork(9, 8, .2, kind).eval()
            torch.testing.assert_close(network(x,y), -network(y,x), atol=1e-7, rtol=0)
            self.assertTrue(torch.equal(network(x,x), torch.zeros(12)))
            network.train()
            self.assertTrue(torch.equal(network(x,x), torch.zeros(12)))

    def test_panel_is_key_stable_balanced_and_family_distinct(self):
        x = self.rng.normal(size=(30, 10))
        y = np.repeat([3., 7., 11.], [6,18,6])
        groups = np.array([f'g{i//2}' for i in range(30)])
        keys = np.array([f'k{i:03}' for i in range(30)])
        selected, weights = select_panel(x, y, groups, keys, 4)
        for b in [3.,7.,11.]:
            m = y[selected] == b
            self.assertAlmostEqual(weights[m].sum(), 1/3)
            self.assertEqual(len(set(groups[selected[m]])), m.sum())
        order = self.rng.permutation(len(x))
        sel2, w2 = select_panel(x[order], y[order], groups[order], keys[order], 4)
        np.testing.assert_array_equal(keys[selected], keys[order][sel2])
        np.testing.assert_allclose(weights, w2)

    def test_natural_panel_has_same_budget_but_different_bin_allocation(self):
        x=self.rng.normal(size=(100,8));y=np.r_[np.full(8,3.),np.full(84,7.),np.full(8,11.)]
        keys=np.array([f'k{i:03}' for i in range(100)])
        a,w=select_panel(x,y,keys,keys,4,balanced=True)
        b,v=select_panel(x,y,keys,keys,4,balanced=False)
        self.assertEqual(len(a),len(b))
        self.assertGreater((y[b]==7).sum(),(y[a]==7).sum())
        np.testing.assert_allclose(v,np.full(len(b),1/len(b)))

    def test_lora_prefix_matches_full_encoder_and_preserves_long_sequences(self):
        from unittest.mock import patch
        from transformers import EsmConfig,EsmModel
        from phgeofuse.delta_ref.lora import PrefixEncoder,pool_chunk_residues
        cfg=EsmConfig(vocab_size=33,hidden_size=16,num_hidden_layers=3,num_attention_heads=2,
                      intermediate_size=32,pad_token_id=1,mask_token_id=32,
                      hidden_dropout_prob=0.,attention_probs_dropout_prob=0.,token_dropout=False)
        cfg._commit_hash='fixture-pinned-revision'
        net=EsmModel(cfg).eval()
        class Tokenizer:
            def __call__(self,seq,**kwargs):
                ids=torch.tensor([[0]+[5+ord(a)%10 for a in seq]+[2]])
                return {'input_ids':ids,'attention_mask':torch.ones_like(ids)}
        tokenizer=Tokenizer();sequence='ACDXEFGHIKLMN'
        expected=[];sizes=[]
        with torch.no_grad():
            for start in range(0,len(sequence),8):
                chunk=sequence[start:start+8];expected.append(net(**tokenizer(chunk)).last_hidden_state[0]);sizes.append(len(chunk))
        mean,std=pool_chunk_residues(expected,sizes)
        expected=torch.cat([mean/mean.norm(),std/std.norm()])
        with tempfile.TemporaryDirectory() as tmp,patch('transformers.AutoModel.from_pretrained',return_value=net),patch('transformers.AutoTokenizer.from_pretrained',return_value=tokenizer),patch('transformers.utils.hub.cached_file',return_value='/fixture/config.json'):
            encoder=PrefixEncoder({'model':'fixture','last_layers':2,'rank':2,'alpha':4,'dropout':0.,'chunk_length':8},tmp,'cpu')
            torch.testing.assert_close(encoder.encode(sequence),expected,rtol=1e-5,atol=1e-5)
            cached=encoder._load_prefix(sequence)
            self.assertEqual(cached['lengths'],[8,5])
            encoder.set_adapter_training(True)
            value=encoder.encode(sequence,gradient=True)
            value[:5].sum().backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in encoder.parameters() if p.requires_grad))
            self.assertTrue(all(p.grad is None for p in encoder.parameters() if not p.requires_grad))
            encoder._load_prefix.cache_clear()

    def test_robust_location_and_order_invariance(self):
        values = np.array([[2.,2.,2.,12.], [1.,3.,5.,7.]])
        weights = np.ones(4)/4
        p = huber_location(values, weights)
        self.assertAlmostEqual(p[0], 2+1/3)
        self.assertAlmostEqual(p[1], 4)
        np.testing.assert_allclose(p, huber_location(values[:,[2,0,3,1]],weights))
        np.testing.assert_allclose(weighted_median(values,weights),[2,3])

    def make_predictor(self):
        x = self.rng.normal(size=(9,8))
        y = np.linspace(2,12,9)
        model = DeltaNetwork(8,8,0).eval()
        return ReferencePredictor(model,Standardizer.fit(x),x,y,[f'g{i}' for i in range(9)],
                                  [f'r{i}' for i in range(9)],np.ones(9)/9)

    def test_predictor_reference_permutation_bundle_and_no_query_labels(self):
        p = self.make_predictor()
        x = self.rng.normal(size=(5,8))
        original = p.predict(x,np.full(5,7.),.5)
        order = self.rng.permutation(9)
        reordered = ReferencePredictor(p.network,p.scaler,p.features[order],p.labels[order],p.groups[order],p.keys[order],p.weights[order])
        for a,b in zip(p.transfer(x),reordered.transfer(x)):
            np.testing.assert_allclose(a,b,atol=1e-7,rtol=0)
        self.assertNotIn('labels', inspect.signature(p.predict).parameters)
        with tempfile.TemporaryDirectory() as tmp:
            save_bundle(tmp,p,{'name':'test'},.5,{'fit_hash':'test'})
            restored,meta=load_bundle(tmp)
            np.testing.assert_allclose(restored.predict(x,np.full(5,7.),meta['strength'])['prediction'],original['prediction'],atol=1e-7)
            with (Path(tmp)/'weights.pt').open('ab') as f:f.write(b'corrupted')
            with self.assertRaises(ValueError):load_bundle(tmp)

    def test_fallback_exact_zero_and_nonfinite_errors(self):
        b=np.array([1.,7.,13.])
        np.testing.assert_array_equal(blend_predictions(b,[12,1,4],[0,2,3],0)['prediction'],b)
        np.testing.assert_array_equal(blend_predictions(b,[12,1,4],[0,2,3],1,[False]*3)['prediction'],b)
        p=self.make_predictor()
        p.groups[:]='one_family'
        output=p.predict(np.ones((3,8)),b,1.)
        np.testing.assert_array_equal(output['prediction'],b)
        self.assertTrue(output['fallback'].all())
        with self.assertRaises(ValueError):p.transfer(np.full((2,8),np.nan))
        with self.assertRaises(ValueError):blend_predictions(b,[1,np.inf,3],[0]*3,0)
        with self.assertRaises(ValueError):Standardizer.fit(np.array([[np.nan]]))

    def test_panel_estimates_can_exceed_reference_label_range(self):
        # Exact additive potential: reference-label transfer should recover an
        # unseen query outside the panel's label range, unlike label averaging.
        class LinearDifference(DeltaNetwork):
            def encode(self,x):return x
            def difference(self,q,r):return q[...,0]-r[...,0]
        model=LinearDifference(1,2,0,'additive')
        p=ReferencePredictor(model,Standardizer(np.zeros(1),np.ones(1)),np.array([[5.],[6.],[7.]]),
                             [5,6,7],['a','b','c'],['a','b','c'],np.ones(3)/3)
        t,d,ok=p.transfer(np.array([[11.]]))
        np.testing.assert_allclose(t,[11.])
        self.assertTrue(ok[0])

    def test_sampling_foreign_distinct_families_and_reproducibility(self):
        labels=np.tile([3.,7.,11.],20)
        groups=np.array([f'g{i//2}' for i in range(len(labels))])
        q,r=sample_pairs(labels,groups,np.random.default_rng(42))
        self.assertEqual(len(q),len(labels)*8)
        for i in range(len(labels)):
            partners=r[q==i]
            self.assertEqual(len(partners),8)
            self.assertEqual(len(set(groups[partners])),8)
            self.assertNotIn(groups[i],groups[partners])
        q2,r2=sample_pairs(labels,groups,np.random.default_rng(42))
        np.testing.assert_array_equal(q,q2);np.testing.assert_array_equal(r,r2)

    def test_metric_boundaries_zero_strength_and_acceptance(self):
        y=np.array([4.,4.1,9.9,10.])
        b=np.array([5.,5.1,8.9,8.])
        result=metrics(y,b)
        self.assertEqual(result['acid']['count'],1)
        self.assertEqual(result['alkaline']['count'],1)
        self.assertEqual(result['core']['count'],2)
        self.assertTrue(acceptance(metrics(y,y),result)['passed'])
        self.assertFalse(acceptance(result,result)['passed'])
        best,_=select_strength(y,b,np.array([12.,12.,1.,1.]),np.zeros(4),np.ones(4,bool))
        self.assertEqual(best['strength'],0.)

    def test_per_seed_metrics_are_not_ensemble_metrics(self):
        y=np.array([3.,6.,8.,11.])
        p=np.array([y+1,y-1])
        summary=seed_summary(y,p)
        self.assertEqual(summary['mean']['all']['rmse'],1.)
        self.assertEqual(summary['mean']['all']['abs_bias'],1.)
        self.assertEqual(metrics(y,p.mean(0))['all']['rmse'],0.)

    def test_paired_bootstrap_preserves_models_seeds_and_family_units(self):
        y=np.tile([3.,6.,8.,11.],16)
        groups=np.array([f'g{i//4}' for i in range(len(y))])
        b=np.vstack([y+1,y-1]);p=np.vstack([y+.5,y-.5])
        result=paired_family_bootstrap(y,p,{'baseline':b},groups,draws=200)
        acid=result['comparisons']['baseline']['acid']['rmse']
        np.testing.assert_allclose(acid['ci95'],[-.5,-.5])
        self.assertTrue(acid['supported_improvement'])
        zero=paired_family_bootstrap(y,b,{'same':b},groups,draws=200)
        np.testing.assert_allclose(zero['comparisons']['same']['alkaline']['mae']['ci95'],[0,0])

    def test_exclusion_certificate_rejects_meta_feature_leakage(self):
        with self.assertRaises(ValueError):assert_disjoint(['a'],['b'],['same'],['same'])
        valid={'fit_keys':['a','b'],'query_keys':['c'],'excluded_folds':[0,2]}
        validate_certificate(valid,['a','b'],['c'],[2,0])
        with self.assertRaises(ValueError):validate_certificate(valid,['a','b'],['c'],[0])
        with self.assertRaises(ValueError):validate_certificate(valid,['a','c'],['c'],[0,2])

    def test_fixed_refit_is_independent_of_query_labels(self):
        n=48
        x=self.rng.normal(size=(n,10));labels=np.tile([3.,7.,11.],16)
        keys=np.array([f'train::{i}' for i in range(n)]);groups=keys.copy()
        config={'model':{'dropout':.1,'references_per_bin':3},'training':{
            'device':'cpu','threads':2,'learning_rate':.0003,'weight_decay':.001,
            'max_epochs':3,'patience':2,'batch_size':128,'pairs_per_query':8,'natural_partners':4,'gradient_clip':1.}}
        data=DevelopmentData([],keys,x,x,labels,groups,np.arange(n)%5,np.full(n,'train'),config,{'fixture':True})
        fit=np.arange(36);query=np.arange(36,48);recipe={'name':'test','kind':'pair','width':8,'power':.5}
        with tempfile.TemporaryDirectory() as tmp:
            a,ra=fit_subset(data,fit,query,None,recipe,Path(tmp)/'a',fixed_epochs=2)
            changed=copy.deepcopy(data);changed.labels[query]=0.
            b,rb=fit_subset(changed,fit,query,None,recipe,Path(tmp)/'b',fixed_epochs=2)
            np.testing.assert_allclose(a.transfer(x[query])[0],b.transfer(x[query])[0],atol=0,rtol=0)


if __name__=='__main__':
    unittest.main()
