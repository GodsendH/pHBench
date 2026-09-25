import unittest
import numpy as np
from phgeofuse.tail_priority import priority_weights,meta_inputs,fit_priority_residual


class TailPriorityTests(unittest.TestCase):
    def test_asymmetric_objective_identity_and_hard_acid(self):
        y=np.array([2.,3.,4.,6.,7.,8.,10.,11.]);guide=np.full(len(y),7.)
        prediction=np.array([5.,5.,5.,7.,7.,7.,8.,9.]);e=(prediction-y)**2
        for hard in [False,True]:
            w,h,meta=priority_weights(y,guide,.075,.15,hard)
            expected=.775*e.mean()+.075*np.average(e[y<=4],weights=h[y<=4])+.15*e[y>=10].mean()
            self.assertAlmostEqual(np.mean(w*e),expected)
            self.assertAlmostEqual(w.mean(),1.)
            if hard:self.assertGreater(h[0],h[2])

    def test_cap_and_no_input_mutation(self):
        y=np.r_[3.,np.full(2000,7.),10.];p=np.full(len(y),7.);copy=y.copy()
        w,h,meta=priority_weights(y,p,.1,.2,True)
        self.assertLessEqual(w.max(),40.+1e-12);self.assertAlmostEqual(w.mean(),1.)
        self.assertLess(meta['alkaline_mass_effective'],.2)
        np.testing.assert_array_equal(y,copy)

    def test_meta_kernel_and_inference_features(self):
        n=6;r=np.ones((n,15));s=np.arange(n,dtype=float)+4;q=s+.1;c=np.zeros((n,25))
        x,a=meta_inputs(r,s,q,c)
        xx,aa=meta_inputs(r,s,q,c,s+1)
        self.assertEqual(x.shape,(n,41));self.assertEqual(xx.shape,(n,44))
        np.testing.assert_allclose(aa,.5*a+.5*(s+1))

    def test_iteration_keeps_best_actual_asymmetric_objective(self):
        rng=np.random.default_rng(42);y=np.r_[np.full(12,3.),np.full(60,7.),np.full(12,11.)]
        x=np.column_stack([y+rng.normal(size=len(y)),rng.normal(size=len(y))]);a=np.full(len(y),7.);r=a.copy()
        recipe=dict(acid_mass=.05,alkaline_mass=.1,hard_acid=True,under_multiplier=2.)
        params=dict(max_iter=10,max_leaf_nodes=3,min_samples_leaf=5,early_stopping=False,random_state=42)
        model,arrays,metadata=fit_priority_residual(x,y,a,r,a,recipe,params)
        self.assertEqual(len(metadata['iteration_history']),3)
        objective=min(t['objective'] for t in metadata['iteration_history'])
        p=.5*r+.5*(a+model.predict(x));factor=np.where((y>=10)&(p<y),2.,1.)
        self.assertAlmostEqual(np.mean(arrays['weight']*factor*(p-y)**2),objective)


if __name__=='__main__':unittest.main()
