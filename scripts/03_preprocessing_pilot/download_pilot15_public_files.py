"""Download only missing fixed-pilot manifest files via normal public Drive downloads.

No folder crawling, browser cookies, permission changes, audio decoding or VAD.
gdown documentation: https://github.com/wkentaro/gdown
"""
import argparse
import csv
import hashlib
import importlib.metadata
import inspect
import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(r'C:\ieum')
MANIFEST = ROOT / 'results/preprocessing_pilot/pilot_15_manifest.csv'
INPUT = ROOT / 'data/pilot_15_inputs'

def download_one(job):
    import gdown
    target=Path(job['local_path']).resolve()
    result=dict(speaker_id=job['speaker_id'],recording_id=job['recording_id'],drive_id=job['drive_id'],filename=target.name,
                status='ERROR',bytes=0,sha256='',failure_reason='',elapsed_sec=0)
    started=time.perf_counter()
    stop=threading.Event()
    def heartbeat():
        while not stop.wait(10):
            print('Downloading:',target.name,'elapsed:',round(time.perf_counter()-started,1),'s',flush=True)
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    try:
        if not target.is_relative_to(INPUT.resolve()): raise ValueError('Outside pilot input root')
        if target.exists(): raise FileExistsError('Refusing to overwrite')
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_name(target.name+'.public_partial_'+uuid.uuid4().hex)
        print('Public Drive download:',target.name,job['expected_size'],'bytes',flush=True)
        options=dict(id=job['drive_id'],output=str(temporary),quiet=True,use_cookies=False,verify=True,resume=False)
        if 'timeout' in inspect.signature(gdown.download).parameters: options['timeout']=120
        obtained=gdown.download(**options)
        if not obtained or not temporary.is_file(): raise ValueError('No downloaded file returned')
        result['bytes']=temporary.stat().st_size
        if result['bytes'] != job['expected_size'] or result['bytes']==0: raise ValueError('Drive size mismatch')
        if target.suffix=='.json':
            data=json.loads(temporary.read_text(encoding='utf-8-sig'))
            if data.get('File_id')!=target.with_suffix('.wav').name: raise ValueError('JSON File_id mismatch')
        digest=hashlib.sha256()
        with temporary.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''): digest.update(block)
        if target.exists(): raise FileExistsError('Destination appeared')
        os.rename(temporary,target)
        result['status']='DOWNLOADED';result['sha256']=digest.hexdigest()
    except Exception as exc:
        result['failure_reason']=type(exc).__name__+': '+str(exc)
    finally:
        stop.set();thread.join(timeout=1)
        result['elapsed_sec']=time.perf_counter()-started
    print(result['status'],target.name,result['bytes'],result['failure_reason'],flush=True)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only-drive-id')
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4,choices=range(1,5))
    args=parser.parse_args()
    import gdown
    print('gdown version:',importlib.metadata.version('gdown'),flush=True)
    with MANIFEST.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    assert len(rows)==139 and len({r['speaker_id'] for r in rows})==15
    assert len({r['recording_id'] for r in rows})==139
    jobs=[]
    for row in rows:
        for kind in ('wav','json'):
            target=Path(row['local_'+kind+'_path'])
            if args.only_drive_id and row[kind+'_drive_id']!=args.only_drive_id: continue
            if target.exists():
                if target.stat().st_size != int(row[kind+'_size_bytes']) > 0: raise ValueError('Existing target size mismatch')
                continue
            jobs.append(dict(speaker_id=row['speaker_id'],recording_id=row['recording_id'],drive_id=row[kind+'_drive_id'],
                             local_path=str(target),expected_size=int(row[kind+'_size_bytes'])))
    print('Missing manifest files to download:',len(jobs),flush=True)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    statuses=[]
    with args.report.open('x',encoding='utf-8-sig',newline='') as f:
        writer=None
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for result in executor.map(download_one,jobs):
                if writer is None:
                    writer=csv.DictWriter(f,fieldnames=list(result));writer.writeheader()
                writer.writerow(result);f.flush();os.fsync(f.fileno());statuses.append(result['status'])
    return 0 if all(s=='DOWNLOADED' for s in statuses) else 2

if __name__=='__main__': raise SystemExit(main())
