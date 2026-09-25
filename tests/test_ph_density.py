import unittest
import numpy as np
from localph.density import DensityRidge, cosine_targets, decode_density


class DensityTests(unittest.TestCase):
    def test_gaussian_peak_tracks_acid_neutral_alkaline(self):
        y = np.array([2., 7., 12.])
        c = cosine_targets(y)
        p, negative = decode_density(c, c.mean(0), .5, 0, "mode")
        np.testing.assert_allclose(p, y, atol=.05)
        self.assertTrue(np.isfinite(negative).all())

    def test_uniform_density_tie_returns_middle(self):
        p, _ = decode_density(np.zeros((2, 24)), np.zeros(24), .5, 0, "mode")
        np.testing.assert_allclose(p, [7, 7], atol=.05)

    def test_target_shift_does_not_modify_query_features(self):
        from sklearn.linear_model import Ridge
        rng = np.random.default_rng(11)
        x, y, query = rng.normal(size=(12, 18)), rng.uniform(2, 12, 12), rng.normal(size=(4, 18))
        model = DensityRidge(.2).fit(x, y)
        expected = Ridge(alpha=.2, solver="cholesky").fit(x, cosine_targets(y)).predict(query)
        np.testing.assert_allclose(model.moments(query), expected, atol=1e-10)

    def test_decode_uses_supplied_training_prior_without_refitting(self):
        prior = np.linspace(-.1, .1, 24)
        before = prior.copy()
        p, _ = decode_density(cosine_targets([3, 11]), prior, .5, .5, "mean")
        np.testing.assert_array_equal(prior, before)
        self.assertTrue(((p >= 0) & (p <= 14)).all())


if __name__ == "__main__":
    unittest.main()
