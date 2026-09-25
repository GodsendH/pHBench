import unittest
from types import SimpleNamespace
import torch
from phgeofuse.engine import _configure_regression_weight_normalizer
from phgeofuse.model import compute_loss, compute_low_homology_loss


class WeightedMSETests(unittest.TestCase):
    def inputs(self):
        prediction=torch.tensor([7.,8.],requires_grad=True)
        outputs={'mean':prediction,'logits':torch.zeros(2,41),'ec_logits':torch.zeros(2,2)}
        batch={'labels':torch.tensor([7.,10.]),'weights':torch.tensor([1.,9.]),'ec_labels':torch.tensor([-1,-1])}
        return outputs,batch

    def test_weights_affect_main_and_low_homology_gradients(self):
        for loss_fn in (compute_loss,compute_low_homology_loss):
            outputs,batch=self.inputs()
            loss,parts=loss_fn(outputs,batch,{'loss':{'mse_weighting':'sample','distribution_weight':0}})
            self.assertAlmostEqual(float(loss),3.6,places=6)
            self.assertAlmostEqual(float(parts['mse']),2.0)
            loss.backward()
            torch.testing.assert_close(outputs['mean'].grad,torch.tensor([0.,-3.6]))

    def test_historical_default_and_uniform_weight_equivalence(self):
        for loss_fn in (compute_loss,compute_low_homology_loss):
            out,batch=self.inputs()
            old,_=loss_fn(out,batch,{'loss':{'distribution_weight':0}})
            self.assertEqual(float(old),2.)
            batch['weights']=torch.ones(2)
            weighted,_=loss_fn(out,batch,{'loss':{'distribution_weight':0,'mse_weighting':'sample'}})
            torch.testing.assert_close(old,weighted)

    def test_mse_clipping_is_independent_of_distribution_clipping(self):
        out,batch=self.inputs()
        loss,_=compute_loss(out,batch,{'loss':{'mse_weighting':'sample','distribution_weight':0,
            'max_sample_weight':1,'mse_max_sample_weight':3}})
        self.assertEqual(float(loss),3.)

    def test_invalid_weights_and_modes_fail(self):
        out,batch=self.inputs()
        for invalid in (torch.tensor([1.,float('nan')]),torch.tensor([1.,-1.]),torch.ones(2,1)):
            batch['weights']=invalid
            with self.assertRaises(ValueError):
                compute_low_homology_loss(out,batch,{'loss':{'mse_weighting':'sample'}})
        with self.assertRaises(ValueError):
            compute_low_homology_loss(out,batch,{'loss':{'mse_weighting':'typo'}})

    def test_fixed_normalizer_preserves_weights_across_microbatches(self):
        # A rare sample must retain its gradient multiplier even at batch=1.
        # Compare sample-count-weighted accumulation with one full batch.
        for loss_fn in (compute_loss,compute_low_homology_loss):
            cfg={'loss':{'mse_weighting':'sample_global','distribution_weight':0,
                         'mse_training_weight_mean':5.}}
            out,batch=self.inputs()
            full,_=loss_fn(out,batch,cfg); full.backward()
            expected=out['mean'].grad.clone()
            out,batch=self.inputs()
            accumulated=out['mean'].sum()*0
            for i in range(2):
                micro_out={k:v[i:i+1] for k,v in out.items()}
                micro_batch={k:v[i:i+1] for k,v in batch.items()}
                loss,_=loss_fn(micro_out,micro_batch,cfg)
                accumulated=accumulated+loss/2
            accumulated.backward()
            torch.testing.assert_close(accumulated,full)
            torch.testing.assert_close(out['mean'].grad,expected)
            self.assertAlmostEqual(float(expected[1]),-3.6,places=6)

    def test_normalizer_fits_only_ready_training_rows_after_clipping(self):
        rows=[SimpleNamespace(split='train',status='ready',sample_weight=w) for w in (1.,9.)]
        cfg={'loss':{'mse_weighting':'sample_global','mse_max_sample_weight':3.}}
        _configure_regression_weight_normalizer(rows,cfg)
        self.assertEqual(cfg['loss']['mse_training_weight_mean'],2.)
        self.assertEqual(cfg['loss']['mse_training_weight_rows'],2)
        heldout=SimpleNamespace(split='validation',status='ready',sample_weight=100.)
        with self.assertRaises(ValueError):
            _configure_regression_weight_normalizer(rows+[heldout],cfg)
        with self.assertRaises(ValueError):
            _configure_regression_weight_normalizer(rows[:1],cfg)

    def test_fixed_normalizer_must_be_fitted_not_guessed_from_a_batch(self):
        out,batch=self.inputs()
        for value in (None,0.,-1.,float('nan'),float('inf')):
            cfg={'loss':{'mse_weighting':'sample_global'}}
            if value is not None: cfg['loss']['mse_training_weight_mean']=value
            with self.assertRaises(ValueError):
                compute_low_homology_loss(out,batch,cfg)


if __name__=='__main__':
    unittest.main()
