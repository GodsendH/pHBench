import unittest

from phgeofuse.structures import mask_unknown_residue_confidence, sequence_matches


class UnknownResidueTests(unittest.TestCase):
    def test_unknown_positions_keep_atoms_but_have_zero_confidence(self):
        lines = [
            f"ATOM  {i:5d}  CA  ALA A{i:4d}    {0:8.3f}{0:8.3f}{0:8.3f}{1:6.2f}{90:6.2f}           C\n"
            for i in range(1, 4)
        ]
        result = mask_unknown_residue_confidence("".join(lines), "AXA").splitlines()
        self.assertEqual(len(result), 3)
        self.assertEqual([float(line[60:66]) for line in result], [90, 0, 90])
        self.assertTrue(sequence_matches("AAA", "AXA"))
        self.assertFalse(sequence_matches("AA", "AXA"))
        self.assertFalse(sequence_matches("ACA", "ADA"))


if __name__ == "__main__":
    unittest.main()
