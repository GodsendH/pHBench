import unittest
import numpy as np
import torch
from localph.environment_transfer import EnvironmentTransfer, training_bin_weights, enzyme_objective


class EnvironmentTransferTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        torch.set_num_threads(2)

    def test_zero_initialization_and_actual_output_extrapolation(self):
        model=EnvironmentTransfer(input_dim=8,width=3).eval()
        x,mask=torch.randn(2,5,8),torch.ones(2,5,dtype=torch.bool)
        baseline=torch.tensor([3.,9.])
        self.assertTrue(torch.equal(model(x,mask,baseline),baseline))
        with torch.no_grad(): model.enzyme_head.bias[0]=3.
        torch.testing.assert_close(model(x,mask,baseline),torch.tensor([6.,12.]))

    def test_padding_and_reordering_do_not_change_evaluation(self):
        model=EnvironmentTransfer(input_dim=8,width=3).eval()
        x,mask=torch.randn(2,5,8),torch.ones(2,5,dtype=torch.bool)
        expected=model.predict_environment(x,mask)
        padded=torch.cat([x,torch.full((2,4,8),float('nan'))],1)
        pmask=torch.cat([mask,torch.zeros(2,4,dtype=torch.bool)],1)
        torch.testing.assert_close(model.predict_environment(padded,pmask),expected)
        torch.testing.assert_close(model.predict_environment(x.flip(1),mask),expected)

    def test_enzyme_training_keeps_frozen_environment_deterministic(self):
        model=EnvironmentTransfer(input_dim=8,width=3,dropout=.8).freeze_environment().train()
        x,mask=torch.randn(3,4,8),torch.ones(3,4,dtype=torch.bool)
        env=model.predict_environment(x,mask).clone()
        before={k:v.clone() for k,v in model.encoder.state_dict().items()}
        baseline=torch.tensor([7.,7.,7.]); labels=torch.tensor([4.,7.,11.])
        optimizer=torch.optim.SGD(model.enzyme_head.parameters(),lr=.1)
        loss,_=enzyme_objective(model(x,mask,baseline),baseline,labels,torch.ones(3))
        loss.backward(); optimizer.step()
        self.assertTrue(torch.equal(model.predict_environment(x,mask),env))
        for k,v in model.encoder.state_dict().items(): self.assertTrue(torch.equal(v,before[k]))
        self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
        self.assertFalse(torch.equal(model(x,mask,baseline),baseline))

    def test_weights_use_only_training_labels_and_protect_core(self):
        w,fit=training_bin_weights([3.,7.,7.,7.,10.],mix=1)
        self.assertEqual(fit['counts'],[1,3,1])
        self.assertAlmostEqual(float(w.mean()),1.,places=6)
        self.assertAlmostEqual(float(w[0]),float(w[1]*3),places=6)
        p=torch.tensor([4.,8.,11.],requires_grad=True)
        baseline=torch.tensor([4.,7.,11.]); labels=torch.tensor([4.,8.,11.])
        loss,parts=enzyme_objective(p,baseline,labels,torch.ones(3),core_strength=2)
        self.assertEqual(float(loss),2.)
        loss.backward()
        torch.testing.assert_close(p.grad,torch.tensor([0.,4.,0.]))


if __name__=='__main__': unittest.main()
