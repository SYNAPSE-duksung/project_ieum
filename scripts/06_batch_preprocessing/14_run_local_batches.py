"""Process local batches sequentially and hand archives to a sync folder.

Extract only SUCCESS 1..30s pairs using the existing extractor.
Run exactly one pilot batch. Optional API upload verifies remote size/MD5.
Without a token only desktop sync handoff is measured. No files are deleted.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import uuid
import zipfile
import time
from types import SimpleNamespace


def load_runner():
    path = Path(__file__).with_name('10_run_candidate2_batch.py')
    spec = importlib.util.spec_from_file_location('local_batch_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_success(runner, state, destination, batch_id):
    path = Path(__file__).with_name('11_extract_candidate2_usable.py')
    spec = importlib.util.spec_from_file_location('usable_extractor', path)
    extractor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extractor)
    source = Path(state['usable_csv'])
    target = destination / batch_id
    if target.exists():
        summary = json.loads((target / 'validation_summary.json').read_text(encoding='utf-8'))
        if not summary.get('validation_success') or summary.get('input_csv_sha256') != runner.digest(source):
            raise ValueError('Existing SUCCESS dataset has different provenance; preserved')
        rows = runner.read(target / 'metadata.csv')
        expected = runner.read(source)
        if len(rows) != len(expected):
            raise ValueError('Existing SUCCESS dataset row count differs')
        for row, original in zip(rows, expected):
            if any(row[k] != original[k] for k in extractor.FIELDS if k not in ('wav_path', 'transcript_path')):
                raise ValueError('Existing SUCCESS metadata differs')
            wav, txt = Path(row['wav_path']), Path(row['transcript_path'])
            if not wav.resolve().is_relative_to(target.resolve()) or not txt.resolve().is_relative_to(target.resolve()):
                raise ValueError('SUCCESS paths outside dataset')
            extractor.validate_pair(row, wav, txt)
        expected_files = {Path(r[k]).resolve() for r in rows for k in ('wav_path', 'transcript_path')}
        expected_files |= {(target / n).resolve() for n in ('metadata.csv', 'validation_summary.json')}
        if {p.resolve() for p in target.rglob('*') if p.is_file()} != expected_files:
            raise ValueError('Unexpected files in SUCCESS dataset')
    else:
        if extractor.run(SimpleNamespace(batch_id=batch_id, input_csv=source,
                output_root=destination, execute=True, report=None,
                expected_segments=state['usable_segment_count'])) != 0:
            raise ValueError('SUCCESS extraction failed; partial files preserved')
    return target


def handoff(runner, output, destination, batch_id):
    """Verify an existing archive or publish a new ZIP without overwriting."""
    files = sorted(p for p in output.rglob('*') if p.is_file())
    inventory = {p.relative_to(output).as_posix(): runner.digest(p) for p in files}
    archive = destination / (batch_id + '.zip')
    destination.mkdir(parents=True, exist_ok=True)
    partial = destination / (batch_id + '.partial_' + uuid.uuid4().hex)
    if not archive.exists():
        required = sum(p.stat().st_size for p in files) + len(files)*4096 + runner.RESERVE
        if shutil.disk_usage(destination).free < required:
            raise ValueError('Insufficient sync-folder space including reserve')
        with zipfile.ZipFile(partial, 'x', compression=zipfile.ZIP_STORED, allowZip64=True) as z:
            for path in files:
                z.write(path, path.relative_to(output).as_posix())
        candidate = partial
    else:
        candidate = archive
    import hashlib
    with zipfile.ZipFile(candidate) as z:
        if len(z.namelist()) != len(inventory) or set(z.namelist()) != set(inventory):
            raise ValueError('Archive inventory mismatch; existing files preserved')
        for name, expected in inventory.items():
            h = hashlib.sha256()
            with z.open(name) as stream:
                for chunk in iter(lambda: stream.read(8*1024*1024), b''):
                    h.update(chunk)
            if h.hexdigest() != expected:
                raise ValueError('Archive checksum mismatch: '+name)
    if inventory != {p.relative_to(output).as_posix(): runner.digest(p) for p in files}:
        raise ValueError('Output changed during archive creation')
    if candidate == partial:
        # Windows rename refuses an existing target and works on Drive streams,
        # which may not support hard links. Preserve the no-overwrite guarantee.
        if os.name == 'nt':
            os.rename(partial, archive)
        else:
            os.link(partial, archive)
            partial.unlink()
    return dict(batch_id=batch_id, status='LOCAL_HANDOFF_VERIFIED',
                archive=str(archive), sha256=runner.digest(archive),
                cloud_upload_verified=False, local_files_deleted=False)


def run(args, runner):
    batches = sorted({r['batch_id'] for r in runner.read(args.plan)
                      if r['planning_status'] == 'PLANNED'})
    if any(not re.fullmatch(r'BATCH_\d+', b) for b in batches):
        raise ValueError('Invalid planned batch ID')
    if args.batch_ids:
        requested = args.batch_ids
        if len(set(requested)) != len(requested) or not set(requested) <= set(batches):
            raise ValueError('Duplicate or unknown requested batches')
        batches = list(requested)
    if not batches:
        raise ValueError('No planned batches')
    source = args.input_root.resolve()
    output = args.output_dir.resolve()
    destination = args.sync_dir.resolve()
    processed = args.processed_dir.resolve()
    archives = args.archive_dir.resolve()
    roots = [source, output, destination, processed, archives]
    if args.staging_dir:
        roots.append(args.staging_dir.resolve())
    if any(a.is_relative_to(b) or b.is_relative_to(a)
           for i, a in enumerate(roots) for b in roots[i+1:]):
        raise ValueError('Input, output and sync roots must be separate')
    for batch_id in batches:
        args.batch_id = batch_id
        lock_start = len(runner.LOCK_HANDLES)
        try:
            timings = {}
            started = time.perf_counter()
            state = runner.batch_run(args)
            timings['preprocessing_seconds'] = time.perf_counter() - started
            if args.dry_run:
                print(json.dumps(dict(batch_id=batch_id, next_step='EXTRACT_SUCCESS_THEN_ZIP',
                                      sync_dir=str(destination), cloud_upload_verified=False)))
                continue
            if state['status'] != 'COMPLETE':
                raise ValueError('Incomplete batch; stopping sequence')
            started = time.perf_counter()
            dataset = extract_success(runner, state, processed, batch_id)
            timings['selection_seconds'] = time.perf_counter() - started
            started = time.perf_counter()
            report = handoff(runner, dataset, archives, batch_id)
            timings['archive_and_verification_seconds'] = time.perf_counter() - started
            report['archive_bytes'] = Path(report['archive']).stat().st_size
            if args.upload_token:
                path = Path(__file__).with_name('13_backup_batch_archive_to_drive.py')
                spec = importlib.util.spec_from_file_location('archive_uploader', path)
                uploader = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(uploader)
                started = time.perf_counter()
                md5, sha256 = uploader.calculate_hashes(Path(report['archive']))
                upload_report = {}
                uploader.upload_archive(SimpleNamespace(token=args.upload_token,
                    drive_folder=args.drive_folder, batch_id=batch_id),
                    Path(report['archive']), report['archive_bytes'], md5, sha256, upload_report)
                timings['upload_and_verification_seconds'] = time.perf_counter() - started
                report['upload_report'] = upload_report
                report['cloud_upload_verified'] = upload_report['backup_verified']
            else:
                started = time.perf_counter()
                handoff(runner, dataset, destination, batch_id)
                timings['sync_handoff_seconds'] = time.perf_counter() - started
            report['timings'] = timings
            # Outside batch directory: resume archives retain identical contents.
            runner.new_json(output/(batch_id+'_handoff_'+uuid.uuid4().hex+'.json'), report)
            print(json.dumps(report, ensure_ascii=False, indent=2),flush=True)
        finally:
            for handle in runner.LOCK_HANDLES[lock_start:]:
                handle.close()
            del runner.LOCK_HANDLES[lock_start:]


def main():
    runner = load_runner()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--source-layout', choices=['speaker','drive-path'], default='speaker')
    parser.add_argument('--sync-dir', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, default=runner.ROOT/'results/full_expansion/candidate_exclude_all_code_overlap_resolved.csv')
    parser.add_argument('--plan', type=Path, default=runner.ROOT/'results/full_expansion/candidate2_batch_plan.csv')
    parser.add_argument('--output-dir', type=Path, default=runner.ROOT/'results/candidate2_batch_preprocessing')
    parser.add_argument('--batch-ids', nargs='+', required=True, help='Explicit batch IDs, executed in the supplied order')
    parser.add_argument('--staging-dir', type=Path, help='Optional local directory: prefetch one recording ahead')
    parser.add_argument('--processed-dir', type=Path, default=runner.ROOT/'data/processed_candidate2')
    parser.add_argument('--archive-dir', type=Path, default=runner.ROOT/'data/processed_candidate2_archives')
    parser.add_argument('--upload-token', type=Path, help='Existing drive.file token; enables verified API upload')
    parser.add_argument('--drive-folder', default='processed_candidate2_archives')
    parser.add_argument('--retry-errors', action='store_true')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    args.download_backend = 'local'
    args.download_dir = args.input_root
    args.drive_token = None
    args.dry_run_report = None
    args.resume = True
    try:
        if args.execute:
            import pydub
            if args.upload_token:
                if not args.upload_token.is_file():
                    raise ValueError('Upload token not found')
                import googleapiclient.discovery
                import google.auth
        run(args, runner)
        return 0
    except Exception as exc:
        print('STOPPED:', type(exc).__name__, str(exc))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
