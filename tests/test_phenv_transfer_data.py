import csv
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from localph.phenv_data import audit, parse_row


class PhenvDataTests(unittest.TestCase):
    def test_full_audit_preserves_rows_and_detects_normalized_conflicts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                ['a','org1',1,4,'Training','A'*32],
                ['b','org1',1,4,'Validation','A'*32],
                ['c','org2',1,6,'Training','B'+'C'*31],
                ['d','org3',1,9,'Training','Z'+'C'*31],
                ['e','org4',1,10,'Validation','D'*32],
                ['f','org5',1,7,'Training','E'*1023],
                ['g','org6',1,'nan','Training','F'*32],
            ]
            source, phopt = root/'source.csv', root/'phopt.csv'
            with source.open('w', newline='') as f:
                w=csv.writer(f); w.writerow(['Accession','Organism','Sample Weight','pHenv','Split','Sequence']); w.writerows(rows)
            with phopt.open('w', newline='') as f:
                w=csv.writer(f); w.writerow(['protein_id','split','sequence','ph_opt']); w.writerow(['q','test','D'*32,'not a numeric label'])
            report=audit(source,phopt,root/'audit')
            self.assertEqual(report['rows'],7)
            self.assertEqual(report['organisms_shared_between_published_splits'],1)
            self.assertEqual(report['normalized_sequence_groups_conflicting_labels'],1)
            self.assertEqual(report['exact_phopt_overlap_rows'],1)
            self.assertEqual(report['homology_search_candidates'],1)
            self.assertFalse(report['training_ready'])
            self.assertFalse(report['phopt']['labels_consumed'])
            with sqlite3.connect(root/'audit/manifest.sqlite') as db:
                self.assertEqual(db.execute('SELECT length FROM records WHERE accession="f"').fetchone()[0],1023)
                self.assertEqual(db.execute('SELECT COUNT(*) FROM records').fetchone()[0],7)
            self.assertEqual((root/'audit/phenv_search.fasta').read_text(),'>env_1\n'+'A'*32+'\n')
            self.assertEqual(json.loads((root/'audit/report.json').read_text())['source_sha256'], report['source_sha256'])
            with self.assertRaises(FileExistsError):
                audit(source,phopt,root/'audit')

    def test_bad_weight_and_sequence_are_flagged_not_silently_repaired(self):
        row={'Accession':'a','Organism':'org','Sample Weight':-1,'pHenv':7,'Split':'Training','Sequence':'a'*32}
        raw, normalized, y, weight, issues=parse_row(row)
        self.assertEqual(raw,'a'*32)
        self.assertIn('invalid_sequence',issues)
        self.assertIn('invalid_Sample_Weight',issues)
        self.assertIsNone(weight)


if __name__=='__main__':
    unittest.main()
