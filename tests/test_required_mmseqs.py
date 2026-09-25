import subprocess
import unittest
from unittest.mock import patch

from phgeofuse.retrieval import _mmseqs_hits


class RequiredMMseqsTest(unittest.TestCase):
    def test_missing_required_binary_is_not_a_low_homology_result(self):
        with patch('phgeofuse.retrieval.shutil.which', return_value=None):
            with self.assertRaises(FileNotFoundError):
                _mmseqs_hits([], [], {'retrieval': {'require_mmseqs': True}})

    def test_failed_required_search_is_not_a_low_homology_result(self):
        failure = subprocess.CompletedProcess([], 1, '', 'search failed')
        with patch('phgeofuse.retrieval.shutil.which', return_value='/bin/mmseqs'), \
                patch('phgeofuse.retrieval.subprocess.run', return_value=failure):
            with self.assertRaisesRegex(RuntimeError, 'search failed'):
                _mmseqs_hits([], [], {'retrieval': {'require_mmseqs': True}})

    def test_optional_missing_binary_keeps_existing_fallback(self):
        with patch('phgeofuse.retrieval.shutil.which', return_value=None):
            self.assertEqual(_mmseqs_hits([], [], {}), {})


if __name__ == '__main__':
    unittest.main()
