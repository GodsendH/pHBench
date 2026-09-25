import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from localph.residue_probes import fit_probe
from localph.residue_training import fit


class ResidueProbeTests(unittest.TestCase):
    def inputs(self, width=128):
        rng = np.random.default_rng(71)
        return ({"tokens": rng.normal(size=(15 * 3, width)).astype(np.float16),
                 "offsets": np.arange(16) * 3, "ionizable": np.ones(15 * 3, bool)},
                np.tile([3., 7., 11.], 5), np.arange(9), np.arange(9, 12), np.array([4., 7., 10.]))

    def test_natural_probe_matches_existing_fitter(self):
        packed, y, tr, q, base = self.inputs()
        with tempfile.TemporaryDirectory() as tmp:
            original, report = fit(packed, y, tr, q, base, "sparse", Path(tmp) / "a", max_epochs=2, device="cpu")
            other = fit_probe(packed, y, tr, q, base, Path(tmp) / "b", max_epochs=2, device="cpu")
            self.assertEqual(report["selected_epoch"], other["selected_epoch"])
            with np.load(Path(tmp) / "b/predictions.npz", allow_pickle=False) as z:
                for k in original:
                    np.testing.assert_allclose(original[k], z[k], atol=2e-5, rtol=0)

    def test_full_width_probe_does_not_consume_excluded_outer_labels(self):
        packed, y, tr, q, base = self.inputs(1280)
        changed = y.copy()
        changed[12:] = [0., 14., 13.]
        with tempfile.TemporaryDirectory() as tmp:
            for name, labels in (("a", y), ("b", changed)):
                fit_probe(packed, labels, tr, q, base, Path(tmp) / name, weighted=True, max_epochs=2, device="cpu")
            a = torch.load(Path(tmp) / "a/weights.pt", map_location="cpu", weights_only=True)
            b = torch.load(Path(tmp) / "b/weights.pt", map_location="cpu", weights_only=True)
            for k in a["state_dict"]:
                torch.testing.assert_close(a["state_dict"][k], b["state_dict"][k], atol=0, rtol=0)
            for k in ("prior", "training_prior"):
                torch.testing.assert_close(a[k], b[k], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
