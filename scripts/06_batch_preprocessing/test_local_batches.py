import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('sequence', Path(__file__).with_name('14_run_local_batches.py'))
sequence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sequence)


class LocalBatchTests(unittest.TestCase):
    def test_wav_loading_preserves_full_frames_and_source(self):
        import struct
        import wave
        from pydub import AudioSegment
        runner = sequence.load_runner()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'odd.wav'
            pcm = b'\x01\x00' * 44100
            with wave.open(str(path), 'wb') as wav:
                wav.setparams((1, 2, 44100, 0, 'NONE', 'not compressed'))
                wav.writeframes(pcm)
            clean = path.read_bytes()
            loader = runner.frame_aligned_audio_class(AudioSegment)
            self.assertEqual(loader.from_file(path).raw_data, pcm)
            malformed = bytearray(clean + b'\xff')
            struct.pack_into('<I', malformed, 4, len(malformed) - 8)
            struct.pack_into('<I', malformed, 40, len(pcm) + 1)
            path.write_bytes(malformed)
            audio = loader.from_file(path)
            self.assertEqual(audio.raw_data, pcm)
            self.assertEqual((audio.channels, audio.sample_width, audio.frame_rate), (1, 2, 44100))
            self.assertGreater(audio[:500].rms, 0)
            self.assertEqual(path.read_bytes(), malformed)

    def test_prefetch_retries_drive_read_error_but_not_other_failures(self):
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from local_prefetch import LocalPrefetch
        prefetch = LocalPrefetch.__new__(LocalPrefetch)
        row = {'recording_id': 'recording'}
        paths = {'wav': Path('staged.wav')}
        with patch.object(prefetch, '_stage_once', side_effect=[OSError(22, 'Drive read failed'), (paths, 0)]) as stage:
            with patch('local_prefetch.time.sleep'):
                self.assertEqual(prefetch.stage(row)[0], paths)
                self.assertEqual(stage.call_count, 2)
        with patch.object(prefetch, '_stage_once', side_effect=OSError(22, 'Drive read failed')) as stage:
            with patch('local_prefetch.time.sleep'):
                with self.assertRaises(OSError):
                    prefetch.stage(row)
                self.assertEqual(stage.call_count, 3)
        with patch.object(prefetch, '_stage_once', side_effect=OSError(28, 'Disk full')) as stage:
            with self.assertRaises(OSError):
                prefetch.stage(row)
            self.assertEqual(stage.call_count, 1)

    def test_prefetch_preserves_sources_and_rejects_bad_existing_copy(self):
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from local_prefetch import LocalPrefetch
        runner = sequence.load_runner()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root/'source'
            (source/'speaker').mkdir(parents=True)
            (source/'speaker/a.wav').write_bytes(b'abc')
            (source/'speaker/a.json').write_bytes(b'{}')
            rows = [dict(speaker_id='speaker', wav_filename='a.wav', json_filename='a.json',
                         wav_size_bytes='3', json_size_bytes='2')]
            with patch('local_prefetch.shutil.disk_usage', return_value=SimpleNamespace(free=10**12)):
                prefetch = LocalPrefetch(rows, source, root/'staged', runner)
                try:
                    paths, _, _ = prefetch.take(0)
                    self.assertEqual(paths['wav'].read_bytes(), b'abc')
                    self.assertEqual((source/'speaker/a.wav').read_bytes(), b'abc')
                finally:
                    prefetch.close()
                paths['wav'].write_bytes(b'wrong size')
                prefetch = LocalPrefetch(rows, source, root/'staged', runner)
                try:
                    with self.assertRaisesRegex(ValueError, 'invalid local source'):
                        prefetch.take(0)
                finally:
                    prefetch.close()

    def test_local_space_check_skips_remote_stats_but_keeps_budget_guard(self):
        runner = sequence.load_runner()
        rows = [dict(wav_size_bytes='10', json_size_bytes='2', duration_sec='1')]
        root = Path.cwd()
        with patch.object(runner, 'paths_for', side_effect=AssertionError('Repeated source inspection')):
            with patch.object(runner.shutil, 'disk_usage', return_value=SimpleNamespace(free=10**12)):
                result = runner.check_space(rows, root, root, inspect_sources=False)
                self.assertEqual(result['output_budget_bytes'], runner.output_bound(rows))
            with patch.object(runner.shutil, 'disk_usage', return_value=SimpleNamespace(free=runner.RESERVE)):
                with self.assertRaisesRegex(ValueError, 'Insufficient space'):
                    runner.check_space(rows, root, root, inspect_sources=False)
        runner.DOWNLOAD_BACKEND = 'drive-api'
        with self.assertRaisesRegex(ValueError, 'only be skipped'):
            runner.check_space(rows, root, root, inspect_sources=False)

    def test_current_source_size_is_checked_after_initial_validation(self):
        runner = sequence.load_runner()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'source.wav'
            path.write_bytes(b'abc')
            row = dict(wav_size_bytes='3')
            runner.download(row, 'wav', path)
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'invalid local source'):
                runner.download(row, 'wav', path)

    def test_drive_layout_preserves_manifest_subfolders(self):
        runner = sequence.load_runner()
        runner.SOURCE_LAYOUT = 'drive-path'
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for prefix in ('VS01_group', 'VL01_group'):
                (root/prefix/'category').mkdir(parents=True)
            row = dict(wav_drive_path='VS01/category/a.wav',
                       json_drive_path='VL01/category/a.json')
            # A derived 16 kHz dataset must not compete with the source folder.
            (root/'VS01_wav_16k').mkdir()
            paths = runner.paths_for(row, root)
            self.assertEqual(paths['wav'], root/'VS01_group/category/a.wav')
            self.assertEqual(paths['json'], root/'VL01_group/category/a.json')
            runner.SOURCE_FOLDERS.clear()
            (root/'VS01_other').mkdir()
            with self.assertRaisesRegex(ValueError, 'Expected one'):
                runner.paths_for(row, root)
            row['wav_drive_path'] = 'VS01/../escape.wav'
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                runner.paths_for(row, root)

    def test_local_sources_never_download_and_reject_missing(self):
        runner = sequence.load_runner()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'speaker').mkdir()
            (root/'speaker/a.wav').write_bytes(b'abc')
            row = dict(speaker_id='speaker', wav_filename='a.wav', json_filename='a.json',
                       wav_size_bytes='3', json_size_bytes='2', duration_sec='1')
            with patch.object(runner, 'download_gdown', side_effect=AssertionError('Network')), patch.object(runner, 'download_drive_api', side_effect=AssertionError('Network')):
                runner.download(row, 'wav', runner.paths_for(row, root)['wav'])
                with self.assertRaisesRegex(ValueError, 'Missing local source'):
                    runner.check_space([row], root, root/'out')
            row['local_wav_path'] = '../escape.wav'
            with self.assertRaisesRegex(ValueError, 'outside input root'):
                runner.paths_for(row, root)

    def test_archive_resume_and_tamper_detection(self):
        runner = sequence.load_runner()
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'output'
            output.mkdir()
            (output/'result.txt').write_text('processed')
            dest = Path(folder)/'sync'
            with patch.object(sequence.shutil, 'disk_usage', return_value=SimpleNamespace(free=10**12)):
                report = sequence.handoff(runner, output, dest, 'BATCH_001')
                self.assertFalse(report['cloud_upload_verified'])
                self.assertEqual(report['sha256'], sequence.handoff(runner, output, dest, 'BATCH_001')['sha256'])
                (output/'result.txt').write_text('changed')
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    sequence.handoff(runner, output, dest, 'BATCH_001')

    def test_handoff_precedes_next_batch_and_failure_stops(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = SimpleNamespace(plan=root/'plan.csv', input_root=root/'input',
                                   output_dir=root/'output', sync_dir=root/'sync',
                                   batch_ids=['BATCH_001'], dry_run=False, processed_dir=root/'processed',
                                   archive_dir=root/'archives', upload_token=None, staging_dir=None)
            calls = []
            def batch(a):
                calls.append(('process', a.batch_id))
                return {'status': 'COMPLETE'}
            runner = SimpleNamespace(read=lambda _: [dict(batch_id=b, planning_status='PLANNED') for b in ['BATCH_002', 'BATCH_001']],
                                     batch_run=batch, LOCK_HANDLES=[], new_json=lambda *a: None)
            def transfer(r, out, dest, b):
                calls.append(('handoff', b))
                dest.mkdir(exist_ok=True)
                archive = dest/(b+'.zip')
                archive.write_bytes(b'zip')
                return {'archive':str(archive), 'cloud_upload_verified':False}
            with patch.object(sequence, 'extract_success', return_value=root/'processed'/'BATCH_001'), patch.object(sequence, 'handoff', side_effect=transfer):
                sequence.run(args, runner)
            self.assertEqual(calls, [('process','BATCH_001'), ('handoff','BATCH_001'), ('handoff','BATCH_001')])
            calls.clear()
            with patch.object(sequence, 'extract_success', return_value=root/'processed'/'BATCH_001'), patch.object(sequence, 'handoff', side_effect=ValueError('failed')):
                with self.assertRaises(ValueError):
                    sequence.run(args, runner)
            self.assertEqual(calls, [('process','BATCH_001')])
            calls.clear()
            args.batch_ids = ['BATCH_002', 'BATCH_001']
            with patch.object(sequence, 'extract_success', return_value=root/'processed'/'BATCH_002'), patch.object(sequence, 'handoff', side_effect=transfer):
                sequence.run(args, runner)
            self.assertEqual(calls, [('process','BATCH_002'), ('handoff','BATCH_002'), ('handoff','BATCH_002'), ('process','BATCH_001'), ('handoff','BATCH_001'), ('handoff','BATCH_001')])
            calls.clear()
            args.dry_run = True
            sequence.run(args, runner)
            self.assertEqual(calls, [('process','BATCH_002'), ('process','BATCH_001')])


if __name__ == '__main__':
    unittest.main()
