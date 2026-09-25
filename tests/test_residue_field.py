import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch
from localph.residue_field import ResidueField, label_prior, field_loss, decode
from localph.residue_training import choose, standalone_objective, fit


class ResidueFieldTests(unittest.TestCase):
    def test_masked_padding_does_not_change_predictions(self):
        torch.manual_seed(2)
        x = torch.randn(2, 9, 128)
        mask = torch.ones(2, 9, dtype=torch.bool)
        ion = torch.rand(2, 9) > .5
        for kind in ("sparse", "global", "direct"):
            m = ResidueField(kind=kind).eval()
            with torch.no_grad():
                # Avoid testing only the deliberately zero initialized head.
                m.global_head[-1].weight.normal_(std=.1)
                p = m(x, mask, ion)["prediction"]
                padded = torch.cat([x, torch.randn(2, 13, 128) * 100], 1)
                false = torch.zeros(2, 13, dtype=torch.bool)
                other = m(padded, torch.cat([mask, false], 1), torch.cat([ion, false], 1))["prediction"]
                torch.testing.assert_close(p, other, atol=1e-5, rtol=1e-5)

    def test_short_and_no_ionizable_sequences_have_finite_gradients(self):
        m = ResidueField()
        x = torch.randn(3, 2, 128)
        mask = torch.tensor([[1, 1], [1, 0], [1, 0]], dtype=torch.bool)
        ion = torch.tensor([[1, 0], [0, 0], [1, 0]], dtype=torch.bool)
        y = torch.tensor([2., 7., 12.])
        prior = label_prior(y, m.grid)
        loss = field_loss(m(x, mask, ion), y, m.grid, prior, "sparse")
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_label_prior_and_decode_stay_fixed_for_queries(self):
        m = ResidueField()
        prior = label_prior([3, 7, 10], m.grid)
        before = prior.clone()
        result = decode(torch.randn(5, 57), prior, m.grid)
        self.assertEqual(len(result), 6)
        self.assertTrue(all(np.isfinite(v).all() and ((v >= 0) & (v <= 14)).all() for v in result.values()))
        torch.testing.assert_close(prior, before)

    def test_fallback_tie_can_retain_a_later_improving_head(self):
        y = np.array([3., 7., 11.])
        baseline = y.copy()
        initial, later = np.array([7., 7., 7.]), np.array([5., 7., 9.])
        # Use a non-perfect baseline so relative tail ratios are defined.
        baseline = np.array([3.1, 7.1, 10.9])
        c1 = choose(y, baseline, {"direct": initial})
        c2 = choose(y, baseline, {"direct": later})
        self.assertEqual(c1["strength"], 0)
        self.assertEqual(c2["strength"], 0)
        self.assertEqual(c1["rank"], c2["rank"])
        self.assertLess(standalone_objective(y, later), standalone_objective(y, initial))

    def test_fixed_refit_is_independent_of_outer_query_labels(self):
        rng = np.random.default_rng(21)
        packed = {"tokens": rng.normal(size=(12 * 5, 128)).astype(np.float16),
                  "offsets": np.arange(13) * 5, "ionizable": np.ones(12 * 5, dtype=bool)}
        y = np.tile([3., 7., 11.], 4)
        fit_idx, query = np.arange(9), np.arange(9, 12)
        changed = y.copy()
        changed[query] = [4, 6, 12]
        with tempfile.TemporaryDirectory() as d:
            a, _ = fit(packed, y, fit_idx, query, None, "sparse", Path(d) / "a", fixed_epochs=2, device="cpu")
            b, _ = fit(packed, changed, fit_idx, query, None, "sparse", Path(d) / "b", fixed_epochs=2, device="cpu")
            for name in a:
                np.testing.assert_array_equal(a[name], b[name])
            sa = torch.load(Path(d) / "a/weights.pt", weights_only=True)
            sb = torch.load(Path(d) / "b/weights.pt", weights_only=True)
            for name in sa["state_dict"]:
                torch.testing.assert_close(sa["state_dict"][name], sb["state_dict"][name], atol=0, rtol=0)
            torch.testing.assert_close(sa["prior"], sb["prior"], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
