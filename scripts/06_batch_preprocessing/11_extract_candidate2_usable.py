"""Validate/copy only explicitly indexed Candidate 2 usable WAV/TXT pairs.
Dry-run is the default. Actual copying requires --execute. Never overwrites/deletes.
"""
import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import sys
import time
import wave
from pathlib import Path

ELIGIBILITY='SUCCESS_AND_INCLUSIVE_1_30_SECONDS'
FIELDS=['recording_id','speaker_id','disability_group','wav_path','transcript_path','duration_sec','wav_bytes','transcript_sha256','eligibility']

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def write_json(path,obj):
    with path.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2)

def seconds(path):
    with wave.open(str(path),'rb') as f:
        if f.getcomptype()!='NONE' or f.getframerate()<=0:raise ValueError('Unsupported WAV: '+str(path))
        if f.getnframes()*f.getnchannels()*f.getsampwidth()>path.stat().st_size:raise ValueError('Truncated WAV: '+str(path))
        return f.getnframes()/f.getframerate()

def validate_pair(row,wav,txt):
    if row['eligibility']!=ELIGIBILITY:raise ValueError('Disallowed eligibility')
    if not wav.is_file() or not txt.is_file():raise ValueError('Missing WAV/TXT: '+str(wav)+' / '+str(txt))
    if wav.suffix.lower()!='.wav' or txt.suffix.lower()!='.txt':raise ValueError('Invalid WAV/TXT suffix')
    if wav.stat().st_size!=int(row['wav_bytes']) or wav.stat().st_size<=0:raise ValueError('WAV byte size mismatch: '+str(wav))
    if sha(txt)!=row['transcript_sha256']:raise ValueError('Transcript SHA256 mismatch: '+str(txt))
    duration=float(row['duration_sec']);actual=seconds(wav)
    if not math.isfinite(duration) or not 1<=actual<=30 or not math.isclose(duration,actual,abs_tol=1e-6,rel_tol=0):raise ValueError('WAV duration mismatch/outside inclusive 1..30 seconds: '+str(wav))

def copy_exclusive(source,dest):
    with source.open('rb') as src,dest.open('xb') as dst:shutil.copyfileobj(src,dst,1024*1024)

def run(args):
    start=time.perf_counter()
    target=(args.output_root/args.batch_id).resolve()
    summary=dict(batch_id=args.batch_id,input_csv=str(args.input_csv.resolve()),output_directory=str(target),mode='COPY' if args.execute else 'DRY_RUN',status='FAILED',validation_success=False,copy_performed=False)
    created=False
    try:
        if not re.fullmatch(r'BATCH_(00[1-9]|01[0-9]|02[0-8])',args.batch_id):raise ValueError('Expected BATCH_001..BATCH_028')
        if target.exists():raise ValueError('Output already exists; no overwrite/resume/delete: '+str(target))
        if args.report and args.report.exists():raise ValueError('Report already exists; no overwrite')
        if args.report and args.report.resolve().is_relative_to(target):raise ValueError('Dry-run report must be outside final output directory')
        input_sha=sha(args.input_csv)
        with args.input_csv.open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f)
            if not set(FIELDS)<=set(reader.fieldnames or []):raise ValueError('Required CSV fields missing')
            rows=list(reader)
        if not rows:raise ValueError('Empty usable CSV')
        if args.expected_segments is not None and len(rows)!=args.expected_segments:raise ValueError('Unexpected CSV row count: '+str(len(rows)))
        prepared=[];seen_wav=set();seen_txt=set();seen_names=set()
        for index,row in enumerate(rows,1):
            if any(not row[k] for k in FIELDS):raise ValueError('Empty required value at row '+str(index))
            wav=Path(row['wav_path']);txt=Path(row['transcript_path'])
            if not wav.is_absolute() or not txt.is_absolute():raise ValueError('Source paths must be absolute')
            wav=wav.resolve();txt=txt.resolve()
            if wav.is_relative_to(target) or txt.is_relative_to(target):raise ValueError('Source/output overlap')
            if args.batch_id not in wav.parts or args.batch_id not in txt.parts:raise ValueError('Source path outside selected batch at row '+str(index))
            if wav.stem!=txt.stem:raise ValueError('WAV/TXT source basename mismatch')
            if wav in seen_wav or txt in seen_txt:raise ValueError('Duplicate source WAV/TXT at row '+str(index))
            seen_wav.add(wav);seen_txt.add(txt)
            validate_pair(row,wav,txt)
            prefix=re.sub(r'[^\w.-]','_',row['recording_id'])[:100]
            stem=f'{prefix}__{hashlib.sha256(row["recording_id"].encode()).hexdigest()[:12]}__{index:06d}'
            if stem in seen_names:raise ValueError('Final filename collision')
            seen_names.add(stem)
            prepared.append((row,wav,txt,target/'audio'/(stem+'.wav'),target/'transcripts'/(stem+'.txt')))
            if index%1000==0:print(f'Validated {index}/{len(rows)} source pairs',flush=True)
        if sha(args.input_csv)!=input_sha:raise ValueError('Input CSV changed during validation')
        size=sum(w.stat().st_size+t.stat().st_size for _,w,t,_,_ in prepared)
        summary.update(input_csv_sha256=input_sha,csv_rows=len(rows),validated_wav_count=len(seen_wav),validated_txt_count=len(seen_txt),total_duration_sec=sum(float(r['duration_sec']) for r in rows),source_pair_bytes=size,transcript_sha256_verified=True,wav_bytes_verified=True,inclusive_1_30s_verified=True,duplicate_source_count=0,filename_collision_count=0)
        if args.execute:
            ancestor=target.parent
            while not ancestor.exists():ancestor=ancestor.parent
            if shutil.disk_usage(ancestor).free<size+max(1024*1024,len(rows)*2048):raise ValueError('Insufficient free space for copies and metadata')
            target.mkdir(parents=True,exist_ok=False);created=True
            (target/'audio').mkdir();(target/'transcripts').mkdir()
            output=[]
            for index,(row,wav,txt,dw,dt) in enumerate(prepared,1):
                validate_pair(row,wav,txt)
                copy_exclusive(wav,dw);copy_exclusive(txt,dt)
                validate_pair(row,dw,dt)
                output.append(dict(row,wav_path=str(dw),transcript_path=str(dt),source_wav_path=str(wav),source_transcript_path=str(txt)))
                if index%1000==0:print(f'Copied and verified {index}/{len(rows)} pairs',flush=True)
            fields=list(dict.fromkeys(FIELDS+[k for r in output for k in r]))
            with (target/'metadata.csv').open('x',encoding='utf-8-sig',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(output)
            with (target/'metadata.csv').open(encoding='utf-8-sig',newline='') as f:final=list(csv.DictReader(f))
            audio=list((target/'audio').iterdir());texts=list((target/'transcripts').iterdir())
            if len(final)!=len(rows) or len(audio)!=len(rows) or len(texts)!=len(rows):raise ValueError('Final CSV/WAV/TXT counts differ')
            if {x.stem for x in audio}!={x.stem for x in texts}:raise ValueError('Final WAV/TXT one-to-one mismatch')
            for row in final:validate_pair(row,Path(row['wav_path']),Path(row['transcript_path']))
            if sha(args.input_csv)!=input_sha:raise ValueError('Input CSV changed during copy')
            summary.update(copy_performed=True,status='COMPLETE',output_csv_rows=len(final),output_wav_count=len(audio),output_txt_count=len(texts),metadata_sha256=sha(target/'metadata.csv'))
        else:summary.update(status='DRY_RUN_VALIDATED',output_files_created=0)
        summary['validation_success']=True
    except Exception as exc:
        summary.update(status='FAILED',validation_success=False,error=type(exc).__name__+': '+str(exc),partial_output_retained=created)
    summary['elapsed_sec']=time.perf_counter()-start
    if created:write_json(target/'validation_summary.json',summary)
    if args.report and not args.report.exists():write_json(args.report,summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    return 0 if summary['validation_success'] else 2

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch-id',required=True)
    p.add_argument('--input-csv',required=True,type=Path,help='Explicit usable_success_segments CSV; never chooses/merges snapshots automatically')
    p.add_argument('--output-root',type=Path,default=Path('C:/ieum/data/processed_candidate2'))
    p.add_argument('--expected-segments',type=int)
    p.add_argument('--report',type=Path,help='Optional NEW validation report path outside output batch directory')
    mode=p.add_mutually_exclusive_group();mode.add_argument('--dry-run',action='store_true');mode.add_argument('--execute',action='store_true')
    return run(p.parse_args())

if __name__=='__main__':sys.exit(main())
