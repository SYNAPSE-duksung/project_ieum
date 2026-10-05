"""Prepare the fixed pilot manifest and runner; never execute VAD or read WAV audio."""
import ast
import csv
import importlib.util
import json
from pathlib import Path

ROOT = Path(r'C:\ieum')
SCRIPTS = ROOT / 'scripts/03_preprocessing_pilot'
RESULTS = ROOT / 'results/preprocessing_pilot'
IDS = 'LKE_M_65 CMJ_F_76 LHJ_F_79 KWY_M_64 BCS_F_65 KMJ_F_25 JHK_M_34 JSM_M_35 HJI_F_49 LUA_F_47 KYY_M_58 LYB_F_22 SYR_F_41 HMR_F_28 HMJ_F_30'.split()

def write_new(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        f.write(content)

def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def main():
    pilots = csv_rows(ROOT / 'results/pilot_selection/pilot_speakers.csv')
    assert len(pilots) == 15 and {p['speaker_id'] for p in pilots} == set(IDS)
    audit = json.loads((SCRIPTS / 'pilot_drive_audit.json').read_text(encoding='utf-8-sig'))
    live = json.loads((Path(__file__).parent / 'drive_verified_files.json').read_text(encoding='utf-8-sig'))
    live_by_id = {f['id']: f for f in live}
    assert len(live) == len(live_by_id) == 278
    metadata = csv_rows(SCRIPTS / 'all_metadata_drive.csv')
    rows = []
    for pilot in pilots:
        sid = pilot['speaker_id']
        item = next(a for a in audit if a['speaker_id'] == sid)
        md = [m for m in metadata if f"{m['speaker']}_{m['sex']}_{int(float(m['age']))}" == sid
              and m['dataset'] == {'VL01':'VS01','TL02':'TS02','TL03':'TS03'}[pilot['dataset']]]
        expected = {m['file_id'] for m in md}
        assert len(md) == len(expected) == int(pilot['n_files'])
        wavs = {f['name']: f for f in item['audio']}
        labels = {f['name']: f for f in item['labels']}
        assert len(wavs) == len(item['audio']) and len(labels) == len(item['labels'])
        assert set(wavs) == expected == {Path(n).with_suffix('.wav').name for n in labels}
        for wavname in sorted(expected):
            labelname = Path(wavname).with_suffix('.json').name
            wav, label = wavs[wavname], labels[labelname]
            row = dict(speaker_id=sid, disability_group=pilot['disability_group'], dataset=pilot['dataset'],
                       recording_id=Path(wavname).stem, wav_filename=wavname, json_filename=labelname)
            for kind, file in [('wav', wav), ('json', label)]:
                current = live_by_id[file['id']]
                assert current['name'] == file['name'] and current['size'] == int(file['size']) > 0
                assert file['folder_id'] in current['parents']
                row[kind+'_drive_path'] = 'https://drive.google.com/file/d/'+file['id']+'/view'
                row[kind+'_drive_id'] = file['id']
                row[kind+'_drive_folder_id'] = file['folder_id']
                row[kind+'_size_bytes'] = current['size']
                row['local_'+kind+'_path'] = str(ROOT / 'data/pilot_15_inputs' / sid / file['name'])
            rows.append(row)
    assert len(rows) == 139 and len({r['recording_id'] for r in rows}) == 139
    for kind in ('wav', 'json'):
        assert len({r[kind+'_drive_id'] for r in rows}) == 139
    manifest = RESULTS / 'pilot_15_manifest.csv'
    runner = SCRIPTS / '05_run_preprocessing_pilot15.py'
    snapshot = SCRIPTS / 'pilot15_drive_verified_files.json'
    assert all(not p.exists() for p in (manifest, runner, snapshot)), 'Refusing to overwrite existing files'
    source = (SCRIPTS / '04_run_preprocessing_smoke_test.py').read_text(encoding='utf-8-sig')
    tree = ast.parse(source)
    replacements = {
        'load_manifest': '''def load_manifest(path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        needed = {'speaker_id','disability_group','dataset','recording_id','wav_filename','json_filename','local_wav_path','local_json_path'}
        if not needed.issubset(reader.fieldnames or []):
            raise ValueError('Missing manifest columns: '+', '.join(sorted(needed-set(reader.fieldnames or []))))
        rows = list(reader)
    if len(rows) != 139 or {r['speaker_id'] for r in rows} != set(PILOT_SPEAKERS):
        raise ValueError('Manifest must contain the fixed 15 speakers and 139 recordings.')
    for key in ('recording_id','wav_filename','json_filename','local_wav_path','local_json_path'):
        if len({r[key] for r in rows}) != len(rows):
            raise ValueError('Duplicate manifest value: '+key)
    for row in rows:
        if row['disability_group'] != PILOT_SPEAKERS[row['speaker_id']]:
            raise ValueError('Manifest disability_group mismatch')
        match = re.fullmatch(r'ID-\\d{2}-\\d{2}-N-([A-Za-z0-9]+)-(?:\\d+-)*([FM])-(\\d+)-[^/\\\\]+\\.wav', row['wav_filename'])
        if not match or f'{match[1]}_{match[2]}_{int(match[3])}' != row['speaker_id']:
            raise ValueError('Filename speaker identity mismatch')
        if Path(row['wav_filename']).stem != row['recording_id'] or row['json_filename'] != Path(row['wav_filename']).with_suffix('.json').name:
            raise ValueError('Manifest WAV/JSON basename mismatch')
        row['wav_path'], row['json_path'] = row['local_wav_path'], row['local_json_path']
    return rows
''',
        'pair_inputs': '''def pair_inputs(row, manifest):
    wav = input_path(row['wav_path'], manifest)
    label = input_path(row['json_path'], manifest)
    if wav.name != row['wav_filename'] or label.name != row['json_filename']:
        raise ValueError('Input basename differs from explicit manifest filename')
    if not wav.is_file() or not label.is_file():
        raise FileNotFoundError(f'Missing WAV/JSON: {wav} | {label}')
    if wav.stat().st_size == 0 or label.stat().st_size == 0:
        raise ValueError('Zero-byte WAV/JSON')
    with label.open('r', encoding='utf-8-sig') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError('JSON must be an object.')
    if data.get('File_id') != wav.name:
        raise ValueError(f"File_id mismatch: {data.get('File_id')!r} != {wav.name!r}")
    if 'Transcript' not in data or not isinstance(data['Transcript'], str):
        raise ValueError('JSON Transcript must exist and be a string.')
    return wav, label, data
''',
        'make_result': '''def make_result(row):
    result = dict.fromkeys(RESULT_FIELDS, '')
    result.update(speaker_id=row['speaker_id'], disability_group=row['disability_group'],
                  recording_id=row['recording_id'], final_status='ERROR',
                  implementation_safety_fix=SAFETY_FIX, retry_needed=False,
                  pairing_method='manifest+basename+JSON.File_id')
    return result
'''
    }
    lines = source.splitlines(keepends=True)
    for node in sorted([n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in replacements], key=lambda n:n.lineno, reverse=True):
        lines[node.lineno-1:node.end_lineno] = [replacements[node.name]]
    source = ''.join(lines)
    start = source.index('EXPECTED = {')
    end = source.index('\nSAFETY_FIX', start)
    groups = {p['speaker_id']:p['disability_group'] for p in pilots}
    source = source[:start] + 'PILOT_SPEAKERS = '+repr(groups)+'\n' + source[end:]
    source = source.replace('Three-recording VAD/transcript feasibility test', 'Fixed pilot-15 manifest VAD/transcript feasibility test')
    source = source.replace('results/preprocessing_pilot/smoke_test', 'results/preprocessing_pilot/pilot15_run1')
    source = source.replace('exact_three_recording_allowlist;', 'explicit_pilot_manifest_only;')
    source = source.replace('logging.getLogger("smoke_test")', 'logging.getLogger("pilot15")')
    source = source.replace('LOG.info("[%d/3] %s", index, row["speaker_id"])', 'LOG.info("[%d/139] %s", index, row["speaker_id"])')
    source = source.replace('EXPECTED[row["speaker_id"]][1]', 'row["wav_filename"]')
    source = source.replace('CSV with the three explicit WAV/JSON pairs.', 'CSV with the fixed pilot 139 explicit WAV/JSON pairs.')
    source = source.replace('    args = parser.parse_args()', "    parser.add_argument('--validate-manifest', action='store_true', help='Validate paths, sizes and JSON File_id only; never decode WAV or run VAD.')\n    args = parser.parse_args()")
    source = source.replace('        from pydub import AudioSegment\n    except Exception as exc:', '''        if args.validate_manifest:
            failures = []
            for row in rows:
                try:
                    wav, label, _ = pair_inputs(row, manifest)
                    for kind, path in [('wav', wav), ('json', label)]:
                        if kind+'_size_bytes' in row and path.stat().st_size != int(row[kind+'_size_bytes']):
                            raise ValueError('File size mismatch: '+kind)
                except Exception as exc:
                    failures.append((row['recording_id'], type(exc).__name__+': '+str(exc)))
            for recording, reason in failures:
                print(recording+': '+reason, flush=True)
            print(f'Manifest validation: speakers=15 recordings=139 valid_pairs={139-len(failures)} failures={len(failures)}; no WAV decoding or VAD', flush=True)
            return 2 if failures else 0
        from pydub import AudioSegment
    except Exception as exc:''')
    compile(source, str(runner), 'exec')
    newtree = ast.parse(source)
    for name in ('vad_segment_by_energy','split_transcript','clean_transcript','save_pass'):
        before = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        after = next(n for n in newtree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert ast.dump(before, include_attributes=False) == ast.dump(after, include_attributes=False), name+' changed'
    assert 'EXPECTED' not in source and 'exact_three_recording_allowlist' not in source
    RESULTS.mkdir(parents=True, exist_ok=True)
    with manifest.open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    write_new(runner, source)
    write_new(snapshot, json.dumps(live, ensure_ascii=False, indent=2))
    print('PASS: 15 speakers / 139 recordings / 139 WAV / 139 JSON / no duplicate / basename pairs / live ID-size-parent checks')
    print('PASS: unchanged VAD, transcript split/clean, segment export AST; no audio processing')

if __name__ == '__main__':
    main()
