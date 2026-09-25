import unittest
from pathlib import Path
import importlib.util

import numpy as np
from sklearn.svm import SVR

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('svr_control', ROOT / 'scripts/train_ephod_svr_control.py')
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


class EpHodSVRControlTests(unittest.TestCase):
    def test_precomputed_search_matches_native_svr_predictions(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(40, 8))
        query = rng.normal(size=(9, 8))
        y = x[:, 0] * 2 + x[:, 2] ** 2
        weight = rng.uniform(0.2, 3, len(x))
        for kind in ['poly', 'rbf']:
            for setting in ['scale', 'auto']:
                with self.subTest(kernel=kind, gamma=setting):
                    gamma = 1 / (x.shape[1] * x.var()) if setting == 'scale' else 1 / x.shape[1]
                    native = SVR(kernel=kind, gamma=setting, C=10).fit(x, y, sample_weight=weight)
                    cached = SVR(kernel='precomputed', C=10).fit(
                        control.kernel_values(x, x, kind, gamma), y, sample_weight=weight,
                    )
                    np.testing.assert_allclose(native.predict(query), cached.predict(
                        control.kernel_values(query, x, kind, gamma)), rtol=1e-7, atol=1e-7)

    def test_official_weight_boundaries_and_complete_grid(self):
        fn = control.weight_function(ROOT / 'docs/extreme_ph_review_20260916/ephod_official_ephod_training_trainutils.py')
        y = np.array([3., 4., 5., 7., 7., 8., 9., 10.])
        np.testing.assert_allclose(fn(y, 'bin_inv'), [4 / 3, 4 / 3, 2 / 3, 2 / 3, 2 / 3, 2 / 3, 4 / 3, 4 / 3])
        recipes = control.grid()
        self.assertEqual(len(recipes), 200)
        self.assertEqual(len({tuple(r.items()) for r in recipes}), 200)


if __name__ == '__main__':
    unittest.main()
