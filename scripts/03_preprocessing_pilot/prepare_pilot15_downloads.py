"""Download only validated manifest file references; no audio decoding or preprocessing."""
import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(r'C:\ieum')
MANIFEST = ROOT / 'results/preprocessing_pilot/pilot_15_manifest.csv'
INPUT = ROOT / 'data/pilot_15_inputs'
RESULT = ROOT / 'results/preprocessing_pilot'

def rows():
    with MANIFEST.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def write_json_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)

def validate_one(row):
    for kind in ('wav','json'):
        path = Path(row['local_'+kind+'_path'])
        if not path.is_file() or path.stat().st_size != int(row[kind+'_size_bytes']) or path.stat().st_size <= 0:
            raise ValueError('Missing/empty/wrong-size '+kind)
        if path.name != row[kind+'_filename']:
            raise ValueError('Basename mismatch '+kind)
    data = json.loads(Path(row['local_json_path']).read_text(encoding='utf-8-sig'))
    if data.get('File_id') != row['wav_filename']:
        raise ValueError('JSON File_id mismatch')
    if not isinstance(data.get('Transcript'), str):
        raise ValueError('JSON Transcript missing/not string')

def reuse():
    selected = {'ID-01-15-N-LHJ-03-F-79-KK','ID-02-27-N-JSM-01-01-M-35-KK','ID-03-31-N-KYY-01-01-M-58-KK2'}
    count = 0
    for row in rows():
        if row['recording_id'] not in selected:
            continue
        originals = {kind:ROOT / 'data/smoke_test_inputs' / row[kind+'_filename'] for kind in ('wav','json')}
        original = dict(row, local_wav_path=str(originals['wav']), local_json_path=str(originals['json']))
        validate_one(original)
        for kind, source in originals.items():
            target = Path(row['local_'+kind+'_path'])
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open('rb') as incoming, target.open('xb') as outgoing:
                shutil.copyfileobj(incoming, outgoing, 8*1024*1024)
            count += 1
        validate_one(row)
    print('Reused smoke-test files:', count, flush=True)

def download_job(job):
    target = Path(job['local_path']).resolve()
    record = dict(drive_id=job['drive_id'], filename=target.name, local_path=str(target), status='ERROR', bytes=0, sha256='', failure_reason='')
    temporary = target.with_name(target.name+'.partial_'+uuid.uuid4().hex)
    started = time.perf_counter()
    try:
        if not target.is_relative_to(INPUT.resolve()):
            raise ValueError('Destination outside pilot input directory')
        allowed = {(r[k+'_drive_id'], str(Path(r['local_'+k+'_path']).resolve()), int(r[k+'_size_bytes'])) for r in rows() for k in ('wav','json')}
        if (job['drive_id'],str(target),job['expected_size']) not in allowed:
            raise ValueError('Download not in validated manifest')
        if target.exists():
            raise FileExistsError('Refusing to overwrite existing target')
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        request = urllib.request.Request(job['download_url'], headers={'User-Agent':'ieum-pilot15-download'})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open('xb') as output:
            while True:
                block = response.read(8*1024*1024)
                if not block:
                    break
                output.write(block); digest.update(block); record['bytes'] += len(block)
            output.flush(); os.fsync(output.fileno())
        if record['bytes'] != job['expected_size'] or record['bytes'] == 0:
            raise ValueError('Downloaded byte count differs from Drive metadata')
        if target.suffix == '.json':
            data = json.loads(temporary.read_text(encoding='utf-8-sig'))
            if data.get('File_id') != target.with_suffix('.wav').name:
                raise ValueError('JSON File_id mismatch')
        if target.exists():
            raise FileExistsError('Target appeared during download')
        os.rename(temporary, target)  # Windows rename refuses an existing destination.
        record['status'] = 'DOWNLOADED'; record['sha256'] = digest.hexdigest()
    except Exception as exc:
        # Do not put signed download URLs in result logs.
        record['failure_reason'] = type(exc).__name__+': '+(str(exc.code) if hasattr(exc,'code') else str(exc).split('https://')[0])
    record['elapsed_sec'] = time.perf_counter()-started
    print(record['status'], target.name, record['bytes'], record['failure_reason'], flush=True)
    return record

def download(jobs_path):
    jobs = json.loads(jobs_path.read_text(encoding='utf-8-sig'))
    report = RESULT / 'pilot15_download_batches' / (jobs_path.stem+'.csv')
    if report.exists():
        raise FileExistsError('Refusing to overwrite batch report')
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open('x', encoding='utf-8-sig', newline='') as f:
        writer = None
        with ThreadPoolExecutor(max_workers=4) as executor:
            for record in executor.map(download_job, jobs):
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(record)); writer.writeheader()
                writer.writerow(record); f.flush()

def verify():
    manifest = rows()
    failures=[]; wav=0; label=0; total=0
    for row in manifest:
        for kind in ('wav','json'):
            path=Path(row['local_'+kind+'_path'])
            if path.is_file() and path.stat().st_size == int(row[kind+'_size_bytes']) > 0:
                total+=path.stat().st_size
                if kind=='wav': wav+=1
                else: label+=1
        try:
            validate_one(row)
        except Exception as exc:
            failures.append(dict(speaker_id=row['speaker_id'],recording_id=row['recording_id'],failure_reason=type(exc).__name__+': '+str(exc)))
    assert len(manifest)==139 and len({r['recording_id'] for r in manifest})==139 and len({r['speaker_id'] for r in manifest})==15
    report=dict(speakers=15, recordings=139, wav_present=wav, json_present=label, local_bytes=total, pairing_failures=failures,
                preprocessing_executed=False)
    write_json_new(RESULT / 'pilot15_download_verification.json', report)
    failure_path=RESULT / 'pilot15_download_failures.csv'
    with failure_path.open('x', encoding='utf-8-sig', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=['speaker_id','recording_id','failure_reason']);writer.writeheader();writer.writerows(failures)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if wav==label==139 and not failures else 2

def static_check():
    script=ROOT / 'scripts/03_preprocessing_pilot/05_run_preprocessing_pilot15.py'
    original=ROOT / 'scripts/03_preprocessing_pilot/04_run_preprocessing_smoke_test.py'
    source=script.read_text(encoding='utf-8-sig');compile(source,str(script),'exec')
    before=ast.parse(original.read_text(encoding='utf-8-sig'));after=ast.parse(source)
    for name in ('vad_segment_by_energy','split_transcript','clean_transcript','save_pass'):
        left=next(n for n in before.body if isinstance(n,ast.FunctionDef) and n.name==name)
        right=next(n for n in after.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(left,include_attributes=False)==ast.dump(right,include_attributes=False),name
    spec=importlib.util.spec_from_file_location('pilot15_static',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert len(module.load_manifest(MANIFEST))==139
    assert module.VAD1==dict(frame_ms=500,silence_duration_sec=5,alpha=.3)
    assert module.VAD2==dict(frame_ms=600,silence_duration_sec=3,alpha=.4)
    print('PASS syntax / module import / fixed manifest / unchanged algorithm AST; no audio access',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--reuse',action='store_true');group.add_argument('--jobs',type=Path)
    group.add_argument('--verify',action='store_true');group.add_argument('--static-check',action='store_true')
    args=parser.parse_args()
    if args.reuse: reuse()
    elif args.jobs: download(args.jobs)
    elif args.verify: raise SystemExit(verify())
    else: static_check()
