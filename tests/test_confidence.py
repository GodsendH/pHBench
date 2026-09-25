import unittest

from phgeofuse.confidence import confidence_scale, normalize_plddt, normalize_pdb_confidence


class ConfidenceTests(unittest.TestCase):
    def test_provenance_not_observed_magnitude_selects_units(self):
        self.assertEqual(confidence_scale({'source': 'alphafold_db'}), '0_100')
        self.assertEqual(normalize_plddt(.9, '0_100'), .9)
        self.assertEqual(normalize_plddt(.9, '0_1'), 90)
        with self.assertRaises(ValueError):
            confidence_scale({'source': 'experimental_pdb'})

    def test_corrected_esmfold_is_not_scaled_twice(self):
        legacy = {'source': 'esmfold', 'model': 'facebook/esmfold_v1'}
        self.assertEqual(confidence_scale(legacy), '0_1')
        self.assertEqual(confidence_scale({**legacy, 'plddt_scale': '0_100'}), '0_100')
        with self.assertRaises(ValueError):
            normalize_plddt(90, '0_1')
        with self.assertRaises(ValueError):
            normalize_plddt(float('nan'), '0_100')

    def test_pdb_geometry_and_unknown_zero_unchanged(self):
        line = f'ATOM      1  CA  ALA A   1    {1:8.3f}{2:8.3f}{3:8.3f}{1:6.2f}{.9:6.2f}           C\n'
        out = normalize_pdb_confidence(line, '0_1')
        self.assertEqual(line[:60], out[:60])
        self.assertEqual(line[66:], out[66:])
        self.assertEqual(float(out[60:66]), 90)
        self.assertEqual(normalize_pdb_confidence(out, '0_100'), out)
        self.assertEqual(normalize_plddt(0, '0_1'), 0)


if __name__ == '__main__':
    unittest.main()
