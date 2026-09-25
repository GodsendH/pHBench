import unittest
import numpy as np
from phgeofuse.kernel_expert import KernelExpert

class KernelExpertTests(unittest.TestCase):
    def test_matches_direct_kernel_solution_and_batching(self):
        rng=np.random.default_rng(42)
        x=rng.normal(size=(12,5));x/=np.linalg.norm(x,axis=1,keepdims=True)
        y=rng.normal(size=12);mu=y.mean();gamma=2.;scale=1.2
        k=np.exp(-gamma*np.maximum(2-2*x@x.T,0)/scale)
        dual=np.linalg.solve(k+.3*np.eye(len(x)),y-mu)
        expert=KernelExpert(x,dual,mu,gamma,scale)
        np.testing.assert_allclose(expert.predict(x,batch_size=3),mu+k@dual,atol=1e-12)
        np.testing.assert_allclose(expert.predict(x*3),expert.predict(x),atol=1e-12)
        with self.assertRaises(ValueError):expert.predict(np.zeros((2,5)))
        with self.assertRaises(ValueError):expert.predict(np.ones((2,6)))

if __name__=='__main__':unittest.main()
