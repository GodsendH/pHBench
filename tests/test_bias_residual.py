import unittest
import numpy as np
from phgeofuse.bias_residual import BiasConstrainedResidual


class BiasResidualTest(unittest.TestCase):
    def test_bias_penalty_reduces_training_group_mean_squared_error(self):
        rng = np.random.default_rng(42)
        n = 240
        meta = rng.normal(size=(n, 41))
        labels = np.repeat([5., 7., 9.], [30, 160, 50])
        meta[:, 0] = labels * .3 + rng.normal(size=n)
        meta[:, 1] = labels * .2 + rng.normal(size=n)
        meta[:, 15] = 7.
        meta[:, 7:9] = 1.
        target = labels - 7.
        masks = [labels < 6, (labels >= 6) & (labels < 8), labels >= 8]
        unpenalized = BiasConstrainedResidual(100., 0.).fit(meta, target, labels=labels)
        penalized = BiasConstrainedResidual(100., 1.).fit(meta, target, labels=labels)
        def bias(model):
            residual = model.predict(meta) - target
            return sum(residual[m].mean()**2 for m in masks)
        self.assertLess(bias(penalized), bias(unpenalized))
        self.assertTrue(np.isfinite(penalized.predict(meta[:4])).all())

    def test_rejects_nonfinite_labels(self):
        with self.assertRaises(ValueError):
            BiasConstrainedResidual().fit(np.ones((3, 41)), np.zeros(3), labels=[5, np.nan, 9])


if __name__ == '__main__':
    unittest.main()
