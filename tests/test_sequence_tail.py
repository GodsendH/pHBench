import inspect
import unittest

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

from phgeofuse.sequence_tail import (
    KernelReadout, SequenceTailFusion, complete_expert, fit_expert, fit_transform, transform,
)


class SequenceTailTest(unittest.TestCase):
    def test_scaler_uses_training_only_and_query_batch_is_invariant(self):
        x = np.arange(36, dtype=float).reshape(12, 3)
        z, state = fit_transform(x)
        q = np.array([[100., -200., 300.], [1., 2., 3.]])
        before = state['mean'].copy()
        np.testing.assert_allclose(transform(q, state)[0], transform(q[:1], state)[0])
        np.testing.assert_array_equal(state['mean'], before)
        np.testing.assert_allclose(z.mean(0), 0., atol=1e-14)

    def test_weighted_kernel_solution_matches_normal_equations(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(18, 5)); z, state = fit_transform(x)
        kernel = rbf_kernel(z, gamma=state['gamma'])
        y = np.array([2., 3., 4., 10., 11., 12.] + [7.] * 12)
        guide = np.full(18, 6.5)
        recipe = dict(acid_mass=.05, alkaline_mass=.1, target='residual', method='krr', regularization=.3)
        model, weights, target, _ = fit_expert(kernel, y, guide, recipe)
        expected = np.linalg.solve(kernel + .3*np.diag(1/weights), target)
        np.testing.assert_allclose(model.dual_coef_, expected, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(complete_expert(model.predict(kernel), guide, 'residual'),
                                   guide+kernel@expected, atol=1e-10)

    def test_compact_svr_and_krr_match_native_prediction(self):
        rng = np.random.default_rng(12)
        z, state = fit_transform(rng.normal(size=(30, 8)))
        kernel = rbf_kernel(z, gamma=state['gamma'])
        query = rbf_kernel(transform(rng.normal(size=(7, 8)), state), z, gamma=state['gamma'])
        y = np.array([3., 11.] * 3 + [7.] * 24)
        for method in ['svr', 'krr']:
            recipe = dict(acid_mass=.05, alkaline_mass=.1, target='direct', method=method, regularization=1.)
            model, _, _, _ = fit_expert(kernel, y, np.full(30, 7.), recipe)
            np.testing.assert_allclose(KernelReadout(model, len(y)).predict(query),
                                       model.predict(query), atol=1e-12)

    def test_serving_signature_does_not_accept_labels(self):
        parameters = inspect.signature(SequenceTailFusion.predict).parameters
        self.assertFalse({'y', 'labels', 'ph_opt', 'fold'}.intersection(parameters))


if __name__ == '__main__':
    unittest.main()
