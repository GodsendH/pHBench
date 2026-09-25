import unittest
import numpy as np
import torch
from localph.residue_field import ResidueField, field_loss, label_prior
from localph.residue_objectives import smooth_rarity_weights, weighted_label_prior, rarity_loss


class ResidueObjectiveTests(unittest.TestCase):
    def test_equal_weights_recover_original_objective_and_prior(self):
        torch.manual_seed(7)
        y = torch.tensor([2., 7., 12.])
        for kind in ("direct", "sparse"):
            model = ResidueField(kind=kind).eval()
            mask = torch.ones(3, 9, dtype=torch.bool)
            output = model(torch.randn(3, 9, 128), mask, mask)
            original = label_prior(y, model.grid)
            weighted = weighted_label_prior(y, torch.ones(3), model.grid)
            torch.testing.assert_close(original, weighted)
            torch.testing.assert_close(field_loss(output, y, model.grid, original, kind),
                                       rarity_loss(output, y, torch.ones(3), model.grid, weighted, kind))

    def test_rare_labels_gain_finite_weight_with_normalized_mass(self):
        y = np.array([3., *([7.] * 100), 11.])
        weights = smooth_rarity_weights(y)
        self.assertTrue(np.isfinite(weights).all())
        self.assertAlmostEqual(float(weights.mean()), 1., places=6)
        self.assertGreater(float(weights[0]), float(weights[1]))
        self.assertGreater(float(weights[-1]), float(weights[1]))
        model = ResidueField()
        natural = label_prior(y, model.grid)
        weighted = weighted_label_prior(y, weights, model.grid)
        before = natural.clone()
        self.assertGreater(float(weighted[12]), float(natural[12]))
        torch.testing.assert_close(natural, before, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
