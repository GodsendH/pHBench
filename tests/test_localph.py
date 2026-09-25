import unittest
import numpy as np
import torch
from localph.features import ion_context, projection, SiteScaler
from localph.kernel import ContextRidge, training_weights


class SiteFeaturesTests(unittest.TestCase):
    def test_bos_eos_or_length_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            ion_context("DEH", torch.ones(5, 8), projection(8, 4))

    def test_absent_types_are_zero_and_flags_explicit(self):
        x = ion_context("DDD", torch.ones(3, 8), projection(8, 4))
        self.assertTrue(np.isfinite(x).all())
        np.testing.assert_array_equal(x[:84], np.zeros(84))
        np.testing.assert_array_equal(x[-14:], [1, 1] + [0, 0] * 6)

    def test_invariant_to_global_embedding_shift_and_sequence_reversal(self):
        torch.manual_seed(0)
        h = torch.randn(12, 8)
        sequence = "DEHKRYYACCHG"
        p = projection(8, 4)
        a = ion_context(sequence, h, p)
        b = ion_context(sequence, h + 10, p)
        c = ion_context(sequence[::-1], h.flip(0), p)
        np.testing.assert_allclose(a, b, atol=3e-6)
        np.testing.assert_allclose(a, c, atol=3e-6)

    def test_scaler_does_not_learn_from_queries(self):
        s = SiteScaler().fit(np.array([[0, 1], [2, 1]]))
        before = s.mean.copy(), s.scale.copy()
        s.transform(np.array([[1000, -1000]]))
        np.testing.assert_array_equal(before[0], s.mean)
        np.testing.assert_array_equal(before[1], s.scale)

    def test_weighted_kernel_matches_sklearn_primal_and_dual(self):
        from sklearn.linear_model import Ridge
        rng = np.random.default_rng(17)
        for rows, width in [(20, 4), (8, 15)]:
            gx, sx = rng.normal(size=(rows, width)), rng.normal(size=(rows, width))
            y = rng.uniform(2, 12, rows)
            model = ContextRidge(.5, .2, .5).fit(gx, sx, y)
            expected = Ridge(alpha=.2, solver="cholesky").fit(
                model.features(gx, sx), y, sample_weight=training_weights(y, .5))
            qg, qs = rng.normal(size=(3, width)), rng.normal(size=(3, width))
            np.testing.assert_allclose(model.predict(qg, qs),
                                       expected.predict(model.features(qg, qs)), atol=1e-10)

    def test_zero_site_weight_ignores_local_features(self):
        rng = np.random.default_rng(42)
        gx, sx = rng.normal(size=(20, 8)), rng.normal(size=(20, 10))
        y = rng.uniform(2, 12, 20)
        model = ContextRidge(0, .2, 0).fit(gx, sx, y)
        np.testing.assert_allclose(model.predict(gx, sx), model.predict(gx, sx + 500), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
