import unittest
import numpy as np

from phgeofuse.tail_weighting import tail_weights, residual_target, crossfit_rows


class TailWeightingTest(unittest.TestCase):
    def test_weighted_loss_equals_region_objective(self):
        y = np.array([3., 4., 5., 7., 8., 9., 10.])
        p = np.array([5., 5., 6., 7., 7., 8., 8.])
        w, info = tail_weights(y, .1)
        e = (p-y)**2
        expected = .9*e.mean()+.05*e[y<=4].mean()+.05*e[y>=10].mean()
        self.assertAlmostEqual(np.mean(w*e), expected)
        self.assertAlmostEqual(w.mean(), 1.)
        self.assertEqual(info['regions']['extreme_acid']['count'], 2)
        np.testing.assert_array_equal(tail_weights(y, 0)[0], np.ones(len(y)))

    def test_cap_reduces_lambda_and_preserves_mixture(self):
        y = np.r_[3., np.full(1000, 7.), 10.]
        w, info = tail_weights(y, .1, max_weight=10.)
        self.assertLess(info['strength_effective'], .1)
        self.assertLessEqual(w.max(), 10.+1e-12)
        self.assertAlmostEqual(w.mean(), 1.)
        self.assertAlmostEqual(w[1], 1-info['strength_effective'])

    def test_rejects_invalid_or_missing_tail(self):
        for labels in ([5., 7., 11.], [3., 7.], [], [3., np.nan, 11.]):
            with self.assertRaises(ValueError):
                tail_weights(labels, .1)
        self.assertEqual(tail_weights([7.], 0)[0][0], 1.)

    def test_complete_target_recovers_final_prediction(self):
        y = np.array([3., 7., 11.]); r = np.array([5., 7., 8.]); a = np.array([6., 7., 8.])
        for c in [.3, .5, 1.]:
            t = residual_target(y, a, r, target='complete', dual_weight=c)
            np.testing.assert_allclose((1-c)*r+c*(a+t), y)
            h = np.array([-2., .2, 2.])
            np.testing.assert_allclose(((1-c)*r+c*(a+h)-y)**2, c*c*(h-t)**2)

    def test_heldout_label_perturbation_leaves_meta_training_unchanged(self):
        folds = np.repeat(np.arange(5), 3)
        y = np.linspace(2., 12., len(folds))
        def assemble(labels):
            calls = []
            def fetch(excluded):
                calls.append(excluded)
                q = np.flatnonzero(np.isin(folds, excluded))
                fitted_mean = labels[~np.isin(folds, excluded)].mean()
                values = {'ridge': np.full(len(q), fitted_mean),
                          'robust': np.full(len(q), fitted_mean+.1),
                          'retrieval': np.full((len(q), 15), fitted_mean)}
                return q, values
            fit, fields = crossfit_rows(folds, 2, fetch)
            target = residual_target(labels[fit], fields['ridge'], fields['robust'], target='complete')
            return fit, fields, target, calls
        first = assemble(y)
        altered = y.copy(); altered[folds==2] = 999.
        second = assemble(altered)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[2], second[2])
        for name in first[1]:
            np.testing.assert_array_equal(first[1][name], second[1][name])
        self.assertTrue(all(2 in pair and len(pair)==2 for pair in first[3]))


if __name__ == '__main__':
    unittest.main()
