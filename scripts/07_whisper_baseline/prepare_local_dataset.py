"""Stage complete baseline ZIPs on D:, reusing the seven original archives.

Sequential source reads; no deletions. ZIP central-directory identity is checked
after transfer, and each evaluated payload's CRC is checked during inference.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid
import zipfile

from general_test import DEFAULT_ROOT, GeneralTest


def write_json(path,value):
    temp=path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(temp,path)


def index(path):
    with zipfile.ZipFile(path) as archive:
        return [(i.filename,i.CRC,i.file_size,i.compress_size) for i in archive.infolist()]


def prepare(source,target,status):
    target.mkdir(parents=True,exist_ok=True)
    archive_dir=target/'processed_candidate2_archives'
    archive_dir.mkdir(exist_ok=True)
    for relative in ('manifest_v1/all_utterances.csv','split_v1/speaker_split.csv'):
        payload=(source/relative).read_bytes()
        destination=target/relative
        destination.parent.mkdir(exist_ok=True)
        if destination.exists():
            if destination.read_bytes()!=payload:
                raise ValueError('Local manifest differs; no overwrite: '+str(destination))
        else:
            with destination.open('xb') as stream:
                stream.write(payload)
    dataset=GeneralTest(source)
    names=sorted({r['batch']+'.zip' for r in dataset.rows})
    mapping={name:str((Path(r'D:\ieum_local_pilot\archives')/name) if
                       (Path(r'D:\ieum_local_pilot\archives')/name).is_file() else archive_dir/name) for name in names}
    map_path=target/'archive_locations.json'
    if map_path.exists() and json.loads(map_path.read_text(encoding='utf-8'))!=mapping:
        raise ValueError('Existing archive map differs')
    write_json(map_path,mapping)
    completed=[]
    for n,name in enumerate(names,1):
        original=source/'processed_candidate2_archives'/name
        destination=Path(mapping[name])
        size=original.stat().st_size
        expected=index(original)
        if destination.is_file():
            if destination.stat().st_size!=size or index(destination)!=expected:
                raise ValueError('Existing ZIP differs: '+str(destination))
            completed.append(dict(name=name,path=str(destination),status='reused',bytes=size))
            print('REUSED '+name,flush=True)
            write_json(status,dict(stage='download',completed=completed,current=None))
            continue
        if shutil.disk_usage(target).free<size+10*2**30:
            raise ValueError('Insufficient external disk space with 10 GiB reserve')
        partial=destination.with_name(destination.name+'.partial_'+uuid.uuid4().hex)
        copied=0
        checksum=hashlib.sha256()
        started=time.perf_counter()
        last=0
        with original.open('rb') as src,partial.open('xb') as dst:
            while chunk:=src.read(8*1024*1024):
                dst.write(chunk)
                checksum.update(chunk)
                copied+=len(chunk)
                if time.perf_counter()-last>=5:
                    write_json(status,dict(stage='download',completed=completed,current=dict(name=name,bytes=copied,total=size,
                               mib_per_sec=copied/2**20/max(.001,time.perf_counter()-started))))
                    print('COPY '+name+' '+str(round(copied/size*100,1))+'%',flush=True)
                    last=time.perf_counter()
            dst.flush()
            os.fsync(dst.fileno())
        if copied!=size or original.stat().st_size!=size or index(partial)!=expected:
            raise ValueError('Transfer size/ZIP index verification failed; partial retained')
        if destination.exists():
            raise ValueError('Destination appeared; no overwrite')
        partial.rename(destination)
        completed.append(dict(name=name,path=str(destination),status='downloaded',bytes=size,
                              transfer_sha256=checksum.hexdigest(),seconds=time.perf_counter()-started))
        write_json(status,dict(stage='download',completed=completed,current=None))
        print('READY '+name,flush=True)
    result=dict(stage='data_ready',completed=completed,total_archives=len(completed),
                verification='size and complete ZIP central-directory identity; payload CRC checked during inference; transfer SHA256 recorded while copying')
    write_json(target/'download_report.json',result)
    write_json(status,result)
    return target


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--target',type=Path,default=Path(r'D:\ieum_whisper_baseline'))
    p.add_argument('--status',type=Path,required=True)
    args=p.parse_args()
    prepare(args.source,args.target,args.status)


if __name__=='__main__':
    main()
