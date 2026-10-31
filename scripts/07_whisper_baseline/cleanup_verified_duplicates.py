"""Remove only content-verified duplicates in the explicitly authorized pilot folders.

Retains archives, batch_runs, logs, metadata and checkpoints. Does not follow links.
Each deletion is preceded by a durable evidence record. No recursive deletion.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import stat
import time
import zipfile

PILOT = Path(r'D:\ieum_local_pilot')
SOURCE = Path(r'H:\.shortcut-targets-by-id\15z9OYaGiLG_KO6Fuatrkgs1DNbD-D50q\ieum_project\00_data\mini_dataset')
STAGES = ('staged_batch023','staged_batch024','staged_sources','staged_retry9')


def snapshot(path):
    info = path.stat()
    return (info.st_size, info.st_mtime_ns)


def digest_stream(stream, limit=None):
    h = hashlib.sha256()
    count = 0
    while limit is None or count < limit:
        chunk = stream.read(min(8*1024*1024,limit-count) if limit is not None else 8*1024*1024)
        if not chunk:
            break
        h.update(chunk)
        count += len(chunk)
    if limit is not None and count != limit:
        raise ValueError('Short read')
    return h.hexdigest()


def digest(path,limit=None):
    with path.open('rb') as stream:
        return digest_stream(stream,limit)


def safe_file(path,allowed):
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(allowed.resolve(strict=True)) or resolved == allowed.resolve():
        raise ValueError('Path outside allowed folder')
    for parent in (path,*path.parents):
        if parent == allowed.parent:
            break
        if parent.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise ValueError('Reparse point refused')
    if not path.is_file():
        raise ValueError('Not a file')
    return resolved


class Cleanup:
    def __init__(self,args):
        self.args = args
        args.report_dir.mkdir(parents=True,exist_ok=False)
        self.log = (args.report_dir/'evidence.jsonl').open('x',encoding='utf-8')
        self.counts = Counter()
        self.bytes = 0
        self.started = time.time()

    def event(self,record):
        record['utc_epoch'] = time.time()
        self.log.write(json.dumps(record,ensure_ascii=False)+'\n')
        self.log.flush()
        os.fsync(self.log.fileno())

    def remove(self,path,allowed,before,evidence,source_guard):
        safe_file(path,allowed)
        if snapshot(path)!=before or not source_guard():
            raise ValueError('File or retained source changed during validation')
        self.event(dict(action='VERIFIED_DELETE' if self.args.execute else 'VERIFIED_KEEP_DRY_RUN',
                        path=str(path),bytes=before[0],sha256=evidence['sha256'],evidence=evidence))
        if self.args.execute:
            path.unlink()  # Exact checked file only; no wildcard or recursive deletion.
            self.bytes += before[0]
            self.counts['deleted'] += 1
        else:
            self.counts['verified'] += 1

    def processed(self):
        base = PILOT/'processed_candidate2'
        for batch in sorted(base.iterdir()):
            if not batch.is_dir():
                continue
            archive_path = PILOT/'archives'/(batch.name+'.zip')
            records = list((PILOT/'batch_runs').glob(batch.name+'_handoff_*.json'))
            try:
                safe_file(archive_path,PILOT/'archives')
                before_archive = snapshot(archive_path)
                expected = {json.loads(p.read_text(encoding='utf-8-sig'))['sha256'] for p in records}
                archive_sha = digest(archive_path)
                if expected != {archive_sha}:
                    raise ValueError('ZIP SHA256 does not agree with retained handoff records')
                self.event(dict(action='ZIP_HASH_VERIFIED',path=str(archive_path),sha256=archive_sha,bytes=before_archive[0]))
                with zipfile.ZipFile(archive_path) as archive:
                    names = Counter(archive.namelist())
                    prepared = []
                    for folder in ('audio','transcripts'):
                        directory = batch/folder
                        if not directory.exists():
                            continue
                        for path in sorted(directory.rglob('*')):
                            if path.is_dir():
                                continue
                            safe_file(path,base)
                            member = path.relative_to(batch).as_posix()
                            before = snapshot(path)
                            if names[member]!=1 or archive.getinfo(member).file_size!=before[0]:
                                raise ValueError('Missing/duplicate/size mismatch ZIP member: '+member)
                            with archive.open(member) as stream:
                                checksum = digest_stream(stream)  # Full read verifies ZIP CRC too.
                            if digest(path)!=checksum or snapshot(path)!=before:
                                raise ValueError('ZIP/local content mismatch: '+member)
                            prepared.append((path,before,dict(source=str(archive_path),member=member,
                                                              archive_sha256=archive_sha,sha256=checksum)))
                    # Validate the complete batch before deleting any of its duplicate payloads.
                    for path,before,evidence in prepared:
                        self.remove(path,base,before,evidence,lambda:snapshot(archive_path)==before_archive)
                print('PROCESSED '+batch.name+' verified payloads='+str(len(prepared)),flush=True)
            except Exception as exc:
                self.counts['batch_errors'] += 1
                self.event(dict(action='RETAIN_BATCH_ERROR',path=str(batch),error=repr(exc)))
                print('RETAIN '+batch.name+' '+repr(exc),flush=True)

    def stages(self):
        for name in STAGES:
            base = PILOT/name
            if not base.exists():
                continue
            for n,path in enumerate(sorted(base.rglob('*')),1):
                if path.is_dir():
                    continue
                try:
                    safe_file(path,base)
                    relative = path.relative_to(base)
                    original_name = path.name.split('.partial_',1)[0]
                    source = SOURCE/relative.parent/original_name
                    safe_file(source,SOURCE)
                    before = snapshot(path)
                    source_before = snapshot(source)
                    partial = '.partial_' in path.name
                    if (not partial and before[0]!=source_before[0]) or before[0]>source_before[0]:
                        raise ValueError('Size mismatch')
                    checksum = digest(path)
                    if digest(source,before[0] if partial else None)!=checksum:
                        raise ValueError('Source/staged content mismatch')
                    evidence = dict(source=str(source),source_size=source_before[0],sha256=checksum,
                                    comparison='exact_prefix' if partial else 'full_sha256')
                    self.remove(path,base,before,evidence,lambda:snapshot(source)==source_before)
                except Exception as exc:
                    self.counts['retained_files'] += 1
                    self.event(dict(action='RETAIN_FILE_ERROR',path=str(path),error=repr(exc)))
                if n%25==0:
                    print('STAGING '+name+' entries='+str(n)+' deleted_GiB='+str(round(self.bytes/2**30,2)),flush=True)
            print('STAGING_DONE '+name,flush=True)

    def finish(self):
        summary = dict(mode='execute' if self.args.execute else 'dry_run',counts=dict(self.counts),
                       deleted_bytes=self.bytes,deleted_gib=self.bytes/2**30,seconds=time.time()-self.started,
                       preserved=['archives','batch_runs','all logs','processed metadata.csv and validation_summary.json'],
                       scope='processed_candidate2 audio/transcripts matching retained verified ZIPs; staged_* matching Drive originals, partials require exact prefix equality')
        self.event(dict(action='SUMMARY',**summary))
        self.log.close()
        with (self.args.report_dir/'summary.json').open('x',encoding='utf-8') as stream:
            json.dump(summary,stream,ensure_ascii=False,indent=2)
        print(json.dumps(summary,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--phase',choices=['processed','stages','all'],default='all')
    p.add_argument('--report-dir',type=Path,required=True)
    args=p.parse_args()
    cleanup=Cleanup(args)
    try:
        if args.phase in ('processed','all'):
            cleanup.processed()
        if args.phase in ('stages','all'):
            cleanup.stages()
    finally:
        cleanup.finish()


if __name__=='__main__':
    main()
