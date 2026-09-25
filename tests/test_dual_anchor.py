import unittest
import numpy as np

from phgeofuse.dual_fusion import retrieval_sequence_anchor


class AnchorTest(unittest.TestCase):
    def test_input_precision_does_not_change_arithmetic(self):
        r = np.zeros((3, 15), dtype=np.float32)
        r[:, :2] = [[7.812345, 8.567891], [5.234567, 9.123456], [0, 0]]
        r[:, 7:9] = [[1, 1], [1, 0], [0, 0]]
        s = np.array([6.1, 7.2, 8.3])
        actual = retrieval_sequence_anchor(r, s)
        np.testing.assert_array_equal(actual, retrieval_sequence_anchor(r.astype(np.float64), s))
        expected = [(float(r[0, 0]) + float(r[0, 1]) + s[0]) / 3,
                    (float(r[1, 0]) + s[1]) / 2, s[2]]
        np.testing.assert_array_equal(actual, expected)

    def test_rejects_invalid_availability(self):
        r = np.zeros((1, 15))
        r[0, 7] = -1
        with self.assertRaises(ValueError):
            retrieval_sequence_anchor(r, np.array([7.]))


if __name__ == '__main__':
    unittest.main()
