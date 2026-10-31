"""Tests for metric semantics and checkpoint durability without ASR downloads."""
import json
import gc
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from run_baseline import Journal
from score_predictions import aggregate, counts


class BaselineTests(unittest.TestCase):
    def test_korean_metrics_and_empty_prediction(self):
        self.assertEqual(counts('안녕 세상', '안녕, 세상!')['cer'], 0)
        self.assertEqual(counts('안녕 세상', '')['wer'], 1)
        self.assertEqual(counts('가 나', '가 다')['char_errors'], 1)
        self.assertEqual(counts('가 나', '가 다')['word_errors'], 1)
        self.assertEqual(counts('가', '가 나 다')['wer'], 2)

    def test_micro_aggregation_and_failures(self):
        rows = []
        for ref, hyp in [('가', ''), ('가나다라마바사아자', '가나다라마바사아자')]:
            rows.append(dict(status='success',speaker_id='s',disability_group='g',duration_sec=1,
                             inference_sec=1,loading_sec=0,peak_allocated_mib=1,peak_reserved_mib=1,
                             scores={'normalized':counts(ref,hyp),'raw':counts(ref,hyp,False)}))
        rows.append(dict(status='failed',speaker_id='f',disability_group='g'))
        result = aggregate(rows)['overall']
        self.assertEqual(result['failed'], 1)
        self.assertAlmostEqual(result['normalized']['cer'], .1)

    def test_durable_commit_after_process_exit_and_retry_history(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'checkpoint.sqlite3'
            # Exit without connection close: only the committed attempt survives.
            script = ('import os; from run_baseline import Journal; '
                      'j=Journal('+repr(str(path))+', {"test":1}); '
                      'j.save({"utterance_id":"a","status":"failed"}); '
                      'j.db.execute("INSERT INTO attempts (utterance,payload) VALUES (?,?)",'
                      '("uncommitted", "{}")); os._exit(7)')
            process = subprocess.run([sys.executable,'-B','-c',script],cwd=Path(__file__).parent)
            self.assertEqual(process.returncode, 7)
            journal = Journal(path, {'test':1})
            self.assertEqual(set(journal.latest()), {'a'})
            journal.save({'utterance_id':'a','status':'success'})
            self.assertEqual(journal.latest()['a']['status'], 'success')
            self.assertEqual(journal.db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0], 2)
            journal.db.close()
            with self.assertRaises(ValueError):
                Journal(path, {'test':2})
            gc.collect()  # Release the rejected constructor's SQLite handle on Windows.


if __name__ == '__main__':
    unittest.main()
