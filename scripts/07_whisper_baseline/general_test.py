"""Read the complete manifest-backed General Test without extracting archives."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path, PureWindowsPath
import wave
import zipfile

DEFAULT_ROOT = Path(r'H:\.shortcut-targets-by-id\15z9OYaGiLG_KO6Fuatrkgs1DNbD-D50q\ieum_project\00_data')
REQUIRED = {'batch', 'audio_zip', 'audio_member', 'transcript_member', 'text',
            'speaker_id', 'disability_group', 'split', 'duration_sec', 'recording_id'}


def read_csv(path):
    payload = path.read_bytes()
    reader = csv.DictReader(io.StringIO(payload.decode('utf-8-sig'), newline=''))
    return list(reader), reader.fieldnames, hashlib.sha256(payload).hexdigest()


class GeneralTest:
    def __init__(self, root=DEFAULT_ROOT):
        self.root = Path(root)
        mapping_path = self.root / 'archive_locations.json'
        self.archive_locations = json.loads(mapping_path.read_text(encoding='utf-8')) if mapping_path.exists() else None
        self.rows, self.columns, self.manifest_sha256 = read_csv(self.root / 'manifest_v1/all_utterances.csv')
        self.speakers, self.speaker_columns, self.speaker_sha256 = read_csv(self.root / 'split_v1/speaker_split.csv')
        if not REQUIRED.issubset(self.columns or []):
            raise ValueError('Missing manifest columns')
        if not {'speaker_id', 'split', 'disability_group'}.issubset(self.speaker_columns or []):
            raise ValueError('Missing speaker split columns')
        lookup = {}
        for row in self.speakers:
            if row['speaker_id'] in lookup:
                raise ValueError('Duplicate speaker in speaker_split: ' + row['speaker_id'])
            lookup[row['speaker_id']] = row
        seen = set()
        for row in self.rows:
            speaker = lookup.get(row['speaker_id'])
            if speaker is None or any(row[k] != speaker[k] for k in ('split', 'disability_group')):
                raise ValueError('Manifest/speaker split mismatch: ' + row['speaker_id'])
            key = (row['batch'], row['audio_member'])
            if key in seen:
                raise ValueError('Duplicate utterance: ' + repr(key))
            seen.add(key)
            self.archive_path(row)  # Validate relocation for every manifest row.
        if set(lookup) != {r['speaker_id'] for r in self.rows}:
            raise ValueError('Speaker sets differ')
        self.test = [r for r in self.rows if r['split'] == 'test']
        if len(self.test) != 46834 or len({r['speaker_id'] for r in self.test}) != 90:
            raise ValueError('Expected 46,834 General Test utterances / 90 speakers')

    def archive_path(self, row):
        # Manifest stores historic C:\ieum paths. Relocate only the exact batch ZIP basename.
        name = PureWindowsPath(row['audio_zip']).name
        if name != row['batch'] + '.zip' or '/' in name or '\\' in name:
            raise ValueError('Archive/batch mismatch: ' + row['audio_zip'])
        if self.archive_locations is not None:
            path = Path(self.archive_locations[name])
            if not path.is_absolute() or path.name != name:
                raise ValueError('Invalid local archive mapping: ' + name)
            return path
        return self.root / 'processed_candidate2_archives' / name

    def iter_audio(self):
        """Yield (manifest row, WAV bytes) for ALL test rows, opening one ZIP at a time.

        No extraction, resampling, VAD, or source mutation. Exhaust iterator to cover
        the full test set. ZIP reads verify CRC; failures propagate rather than skip.
        """
        grouped = defaultdict(list)
        for row in self.test:
            grouped[self.archive_path(row)].append(row)
        for path, rows in grouped.items():
            with zipfile.ZipFile(path) as archive:
                for row in rows:
                    yield dict(row), archive.read(row['audio_member'])

    def audit(self):
        report = dict(manifest_columns=self.columns, speaker_columns=self.speaker_columns,
                      manifest_sha256=self.manifest_sha256, speaker_split_sha256=self.speaker_sha256,
                      manifest_rows=len(self.rows), speaker_rows=len(self.speakers),
                      split_utterances=dict(Counter(r['split'] for r in self.rows)),
                      test_utterances=len(self.test), test_speakers=len({r['speaker_id'] for r in self.test}),
                      split_and_group_mismatches=0, duplicate_utterances=0, archives=[])
        # Validate per-speaker declared segment counts as well as split labels.
        counts = Counter(r['speaker_id'] for r in self.rows)
        if 'segments' in self.speaker_columns:
            if any(counts[r['speaker_id']] != int(r['segments']) for r in self.speakers):
                raise ValueError('Speaker segment count mismatch')
        grouped = defaultdict(list)
        for row in self.rows:
            grouped[self.archive_path(row)].append(row)
        for path, rows in grouped.items():
            with zipfile.ZipFile(path) as archive:
                names = Counter(archive.namelist())
                for row in rows:
                    for field in ('audio_member', 'transcript_member'):
                        if names[row[field]] != 1:
                            raise ValueError('Missing/duplicate ZIP member: ' + str(path) + ':' + row[field])
                        if archive.getinfo(row[field]).file_size == 0:
                            raise ValueError('Empty ZIP member')
                samples = [r for r in rows if r['split'] == 'test'][:1]
                sample_reports = []
                for row in samples:
                    text = archive.read(row['transcript_member']).decode('utf-8-sig')
                    if text.strip() != row['text'].strip():
                        raise ValueError('Sample transcript differs from manifest')
                    audio = archive.read(row['audio_member'])
                    with wave.open(io.BytesIO(audio)) as wav:
                        duration = wav.getnframes() / wav.getframerate()
                        if abs(duration - float(row['duration_sec'])) > 1e-5:
                            raise ValueError('Sample WAV duration mismatch')
                        sample_reports.append(dict(speaker_id=row['speaker_id'], audio_member=row['audio_member'],
                                                   sample_rate=wav.getframerate(), channels=wav.getnchannels(),
                                                   duration_sec=duration, transcript_matches=True, crc_read_ok=True))
                report['archives'].append(dict(path=str(path), manifest_rows=len(rows),
                                               test_rows=sum(r['split'] == 'test' for r in rows),
                                               member_links_verified=True, samples=sample_reports))
            print('Verified ' + path.name, flush=True)
        report['status'] = 'PASS'
        report['scope'] = 'All manifest audio/transcript member links; one test WAV/TXT CRC read per test-bearing ZIP. Full payload CRC scan and ASR not performed.'
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--report', type=Path, required=True, help='New JSON file; existing files are refused')
    args = parser.parse_args()
    if args.report.exists():
        parser.error('Report already exists')
    dataset = GeneralTest(args.data_root)
    report = dataset.audit()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != 'archives'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
