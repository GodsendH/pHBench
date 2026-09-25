import unittest
import numpy as np
from phgeofuse.structure_pooling import regional_pool


class RegionalPoolingTests(unittest.TestCase):
    def test_joint_permutation_and_missing_regions(self):
        rng=np.random.default_rng(42)
        h=rng.normal(size=(6,10));q=rng.normal(size=(10,3));sequence='DEAKRA'
        rsa=np.linspace(0,1,6);confidence=np.linspace(40,90,6)
        first=regional_pool(sequence,h,rsa,confidence,q)
        idx=np.array([5,3,1,2,4,0])
        second=regional_pool(''.join(sequence[i] for i in idx),h[idx],rsa[idx],confidence[idx],q)
        for a,b in zip(first,second):np.testing.assert_allclose(a,b,atol=1e-6)
        # Histidine and cysteine regions are absent: zero embedding, explicit missing flag.
        for i in (4,5):
            np.testing.assert_array_equal(first[2][i*5:i*5+5], [0,0,0,0,1])

    def test_reject_misalignment(self):
        with self.assertRaises(ValueError):
            regional_pool('AA',np.ones((3,10)),np.ones(2),np.ones(2),np.ones((10,3)))
