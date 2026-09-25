import unittest
import numpy as np
from localph.kernel import ContextRidge
from localph.source_kernel import SourceContextRidge


class SourceContextTests(unittest.TestCase):
    def test_zero_external_weight_excludes_external_labels_and_features(self):
        rng = np.random.default_rng(10)
        x, s, y = rng.normal(size=(15, 6)), rng.normal(size=(15, 4)), rng.uniform(2, 12, 15)
        ext, es, ey = rng.normal(size=(8, 6)), rng.normal(size=(8, 4)), rng.uniform(2, 12, 8)
        model = SourceContextRidge(.5, .2, .5, 0).fit(x, s, y, ext, es, ey)
        changed = SourceContextRidge(.5, .2, .5, 0).fit(x, s, y, ext * 1e4, es * 1e4, ey + 100)
        direct = ContextRidge(.5, .2, .5).fit(x, s, y)
        np.testing.assert_allclose(model.predict(x, s), changed.predict(x, s), atol=1e-12)
        np.testing.assert_allclose(model.predict(x, s), direct.predict(x, s), atol=1e-10)

    def test_source_offset_can_absorb_systematic_shift(self):
        x = np.linspace(-1, 1, 30)[:, None]
        s = np.zeros((30, 2))
        y = 7 + 2 * x[:, 0]
        model = SourceContextRidge(0, .001, 0, 1).fit(x, s, y, x, s, y + 2)
        self.assertLess(np.max(abs(model.predict(x, s) - y)), .001)
        self.assertAlmostEqual(model.coef[-1], 2, places=3)


if __name__ == "__main__":
    unittest.main()
