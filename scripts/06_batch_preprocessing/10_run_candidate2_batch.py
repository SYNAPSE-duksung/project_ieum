"""Plan or run ONE capacity-limited Candidate 2 batch.

Preparation: --make-plan. Read-only verification: --batch-id BATCH_001 --dry-run.
Real downloads/VAD require explicit --execute. No automatic source deletion.
Dependencies for --execute: existing .venv_smoke (pydub==0.25.1, gdown).
Pilot core is imported unchanged; only explicit input pairing is adapted to
the five already-investigated physical filename aliases. No alignment changes.
Outputs/checkpoints are immutable. Resume reads completed validated checkpoints;
incomplete attempts are retained and a NEW attempt directory is used.
"""
import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import logging
import math
import os
import shutil
import sys
import time
import uuid
import wave
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path('C:/ieum')
CORE=ROOT/'scripts/03_preprocessing_pilot/05_run_preprocessing_pilot15.py'
CORE_SHA='a79154da1fa27c5c5a2f29dbb76d9eabaa9d95e3b77566f547b24bd4cce53aef'
GB=10**9
RESERVE=100*GB
CORE_FUNCTIONS=('vad_segment_by_energy','split_transcript','clean_transcript','save_pass','process_recording')
LOCK_HANDLES=[]
DRIVE_SESSION=None
DOWNLOAD_BACKEND="gdown"
DRIVE_SCOPE="https://www.googleapis.com/auth/drive.readonly"


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def new_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.flush();os.fsync(f.fileno())


def new_csv(path,rows):
    with path.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())


def core_module():
    if digest(CORE)!=CORE_SHA:
        raise ValueError('Pinned pilot core hash changed; execution refused')
    tree=ast.parse(CORE.read_text(encoding='utf-8-sig'))
    evidence={n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()
              for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in CORE_FUNCTIONS}
    if set(evidence)!=set(CORE_FUNCTIONS):
        raise ValueError('Missing pinned core functions')
    spec=importlib.util.spec_from_file_location('verified_pilot_core',CORE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if module.VAD1!=dict(frame_ms=500,silence_duration_sec=5,alpha=.3) or module.VAD2!=dict(frame_ms=600,silence_duration_sec=3,alpha=.4):
        raise ValueError('Unexpected VAD settings')
    return module,evidence


def checked_rows(rows):
    for key in ('recording_id','wav_drive_id','json_drive_id'):
        vals=[r[key] for r in rows]
        if len(vals)!=len(set(vals)) or not all(vals):
            raise ValueError('Duplicate/empty '+key)
    for r in rows:
        if r['recovered_relation']!='C_new_no_overlap' or r['speaker_id']=='CYZ_F_65':
            raise ValueError('Outside confirmed Candidate 2 / CYZ HOLD')
        for key in ('speaker_id','recording_id','wav_filename','json_filename'):
            if Path(r[key]).name!=r[key] or r[key] in ('.','..') or '/' in r[key] or '\\' in r[key]:
                raise ValueError('Unsafe path component')
        if any(int(r[k+'_size_bytes'])<=0 for k in ('wav','json')):
            raise ValueError('Zero/invalid source size')
        if not math.isfinite(float(r['duration_sec'])) or float(r['duration_sec'])<0:
            raise ValueError('Invalid duration')
        if r.get('pairing_verified')!='True':
            raise ValueError('Unverified source pair')


def source_size(rows,kind=None):
    return sum(int(r[k+'_size_bytes']) for r in rows for k in ([kind] if kind else ['wav','json']))


def output_bound(rows):
    # Each pass exports disjoint PCM intervals within the source duration;
    # Two passes retain at most 2x payload; pydub promotes 24-bit PCM to
    # 32-bit, so reserve 3x WAV bytes instead of assuming byte-for-byte export.
    # Allow WAV headers, metadata/transcript duplication, and 1GB extra margin.
    return 3*source_size(rows,'wav')+20*source_size(rows,'json')+int(sum(float(r['duration_sec']) for r in rows)*512)+GB


def pilot_measurement():
    manifest=read(ROOT/'results/preprocessing_pilot/pilot_15_manifest.csv')
    results=read(ROOT/'results/preprocessing_pilot/pilot15_run1/preprocessing_results.csv')
    audit=read(ROOT/'results/preprocessing_analysis/pilot15/segment_duration_audit.csv')
    if len(results)!=139 or {r['recording_id'] for r in manifest}!={r['recording_id'] for r in results}:
        raise ValueError('Pilot provenance differs')
    if Counter(r['final_status'] for r in results)!=Counter(SUCCESS=84,NEEDS_MANUAL_REVIEW=55):
        raise ValueError('Pilot results differ from validated run')
    success={r['recording_id'] for r in results if r['final_status']=='SUCCESS'}
    valid=[r for r in audit if r['valid_1_30s']=='True']
    if len(valid)!=5094 or any(r['recording_id'] not in success for r in valid):
        raise ValueError('Pilot usable audit differs')
    wav_bytes=0;transcript_bytes=0
    for r in valid:
        path=Path(r['wav_path']);txt=path.parent/'transcript_segments'/path.with_suffix('.txt').name
        if not path.is_file() or not txt.is_file():
            raise ValueError('Missing pilot SUCCESS usable artifacts')
        with wave.open(str(path),'rb') as audio:
            seconds=audio.getnframes()/audio.getframerate()
        if not 1<=seconds<=30 or not math.isclose(seconds,float(r['duration_sec']),abs_tol=1e-6):
            raise ValueError('Pilot artifact duration audit differs')
        wav_bytes+=path.stat().st_size;transcript_bytes+=txt.stat().st_size
    source=0
    for r in manifest:
        size=Path(r['local_wav_path']).stat().st_size
        if size!=int(r['wav_size_bytes']):raise ValueError('Pilot actual source size differs')
        source+=size
    all_artifact_bytes=sum(p.stat().st_size for p in (ROOT/'results/preprocessing_pilot/pilot15_run1').rglob('*') if p.is_file())
    return dict(pilot_speakers=sorted({r['speaker_id'] for r in manifest}),pilot_recording_ids=sorted(success|{r['recording_id'] for r in results}),
                pilot_source_wav_bytes=source,usable_wav_bytes=wav_bytes,usable_transcript_bytes=transcript_bytes,
                usable_wav_to_all_source_wav_ratio=wav_bytes/source,
                usable_wav_and_transcript_ratio=(wav_bytes+transcript_bytes)/source,
                pilot_all_attempt_artifact_bytes=all_artifact_bytes,
                all_attempt_artifact_to_source_wav_ratio=all_artifact_bytes/source,
                pilot_results_sha256=digest(ROOT/'results/preprocessing_pilot/pilot15_run1/preprocessing_results.csv'))


def make_plan(args):
    target=ROOT/'results/full_expansion'
    names=['candidate2_batch_plan.csv','candidate2_batch_summary.csv','candidate2_storage_estimate.json']
    if any((target/n).exists() for n in names):
        raise ValueError('Existing plan target; refusing overwrite')
    rows=read(args.candidate);checked_rows(rows)
    if len(rows)!=6107 or len({r['speaker_id'] for r in rows})!=682:
        raise ValueError('Expected confirmed 682/6107 Candidate 2')
    pilot=pilot_measurement()
    pilot_ids=set(pilot['pilot_recording_ids']);pilot_speakers=set(pilot['pilot_speakers'])
    if {r['recording_id'] for r in rows if r['speaker_id'] in pilot_speakers}!=pilot_ids:
        raise ValueError('Pilot/Candidate recording set mismatch')
    grouped=defaultdict(list)
    for r in rows:
        grouped[r['speaker_id']].append(r)
    candidate_sha=digest(args.candidate)
    plan=[];batch=1;current=0;limit=int(args.max_source_gb*GB)
    for speaker in sorted(grouped):
        rs=sorted(grouped[speaker],key=lambda r:r['recording_id'])
        size=source_size(rs)
        if speaker in pilot_speakers:
            bid='PILOT_ALREADY_PROCESSED';status='ALREADY_PROCESSED_PILOT'
        elif size>limit:
            bid='OVERSIZED_'+speaker;status='OVERSIZED_SPEAKER'
        else:
            if current and current+size>limit:
                batch+=1;current=0
            bid=f'BATCH_{batch:03d}';status='PLANNED';current+=size
        for r in rs:
            plan.append(dict(r,batch_id=bid,planning_status=status,
                             pilot_provenance=str(ROOT/'results/preprocessing_pilot/pilot15_run1') if speaker in pilot_speakers else '',
                             candidate_manifest_sha256=candidate_sha))
    summaries=[]
    for bid in sorted({r['batch_id'] for r in plan}):
        rs=[r for r in plan if r['batch_id']==bid];size=source_size(rs)
        expected=source_size(rs,'wav')*pilot['usable_wav_and_transcript_ratio']
        bound=output_bound(rs)
        summaries.append(dict(batch_id=bid,planning_status=rs[0]['planning_status'],speaker_count=len({r['speaker_id'] for r in rs}),
                              recording_count=len(rs),source_bytes=size,source_GB=size/GB,
                              total_duration_sec=sum(float(r['duration_sec']) for r in rs),
                              estimated_success_usable_output_GB=expected/GB,
                              conservative_output_budget_GB=bound/GB,
                              estimated_source_plus_usable_peak_GB=(size+expected)/GB,
                              conservative_source_plus_output_peak_GB=(size+bound)/GB,
                              safe_under_user_279GB_free_100GB_reserve=size+bound<=179*GB))
    newrows=[r for r in plan if r['planning_status']!='ALREADY_PROCESSED_PILOT']
    storage=dict(**pilot,planning_speaker_count=682,planning_recording_count=6107,
                 pilot_already_processed_speaker_count=15,pilot_already_processed_recording_count=139,
                 new_speaker_count=667,new_recording_count=5968,
                 expected_total_usable_output_GB=source_size(rows,'wav')*pilot['usable_wav_and_transcript_ratio']/GB,
                 expected_new_usable_output_GB=source_size(newrows,'wav')*pilot['usable_wav_and_transcript_ratio']/GB,
                 estimated_all_attempt_retention_GB=source_size(newrows,'wav')*pilot['all_attempt_artifact_to_source_wav_ratio']/GB,
                 current_C_free_GB=shutil.disk_usage(ROOT.anchor).free/GB,
                 max_source_GB=args.max_source_gb,reserve_GB=100,
                 assumptions=['Pilot extrapolation, not a storage guarantee.',
                              'Core retains both VAD attempts including mismatch WAVs; SUCCESS usable estimate is smaller than actual retained artifacts.',
                              'Reserve checks use conservative three-times-WAV export budget (two passes plus 24-to-32-bit promotion), not empirical usable ratio.',
                              'Free space is checked again each recording and batch; accumulated output may block later batches.',
                              'No source cleanup performed automatically; CYZ remains HOLD.'])
    new_csv(target/names[0],plan);new_csv(target/names[1],summaries);new_json(target/names[2],storage)
    print(json.dumps(dict(storage=storage,batches=summaries),ensure_ascii=False,indent=2))


def paths_for(row,base):
    return {k:(base/row['speaker_id']/row[k+'_filename']).resolve() for k in ['wav','json']}


def pair_adapter(row,manifest):
    wav,label=Path(row['wav_path']),Path(row['json_path'])
    for kind,path in [('wav',wav),('json',label)]:
        if path.name!=row[kind+'_filename'] or not path.is_file() or path.stat().st_size!=int(row[kind+'_size_bytes']) or path.stat().st_size==0:
            raise ValueError('Explicit manifest file/size mismatch: '+kind)
    obj=json.loads(label.read_text(encoding='utf-8-sig'))
    expected=row.get('json_File_id') or row['recording_id']+'.wav'
    if obj.get('File_id')!=expected or not isinstance(obj.get('Transcript'),str):
        raise ValueError('Exact JSON File_id/Transcript validation failed')
    pat=obj.get('Patient_info',{})
    if pat.get('Sex')!=row['sex'] or int(pat.get('Age','-1'))!=int(row['age']):
        raise ValueError('JSON Patient_info differs from canonical manifest')
    # wave reads headers only: restrict to native uncompressed PCM so the
    # conservative export bound is meaningful; no conversion or resampling.
    with wave.open(str(wav),'rb') as audio:
        if audio.getcomptype()!='NONE' or audio.getnframes()==0:
            raise ValueError('Unsupported/empty PCM WAV')
        if audio.getnframes()*audio.getnchannels()*audio.getsampwidth()>wav.stat().st_size:
            raise ValueError('Truncated/inconsistent PCM WAV header')
    return wav,label,obj


def verify_artifacts(result,core_output):
    if result['final_status']!='SUCCESS':
        return []
    folder=core_output/result['recording_id']/('vad'+str(result['final_pass']))
    rows=read(folder/'segments.csv')
    if len(rows)!=int(result['total_segments']) or len(rows)!=int(result['text_segment_count']):
        raise ValueError('SUCCESS artifact counts differ')
    if len({r['wav_name'] for r in rows})!=len(rows):
        raise ValueError('Duplicate artifact filename')
    usable=[]
    for r in rows:
        wav=(folder/r['wav_name']).resolve();txt=(folder/r['transcript_path']).resolve()
        if not wav.is_relative_to(folder.resolve()) or not txt.is_relative_to(folder.resolve()) or not txt.is_file():
            raise ValueError('Unsafe/missing SUCCESS artifact')
        with wave.open(str(wav),'rb') as audio:
            seconds=audio.getnframes()/audio.getframerate()
        valid=1<=seconds<=30
        if not math.isclose(seconds,float(r['duration_sec']),abs_tol=1e-6) or valid!=(r['valid_1_30s']=='True'):
            raise ValueError('Artifact duration differs')
        if valid:
            usable.append(dict(recording_id=result['recording_id'],speaker_id=result['speaker_id'],
                               disability_group=result['disability_group'],wav_path=str(wav),transcript_path=str(txt),
                               duration_sec=seconds,wav_bytes=wav.stat().st_size,transcript_sha256=digest(txt),
                               eligibility='SUCCESS_AND_INCLUSIVE_1_30_SECONDS'))
    if len(usable)!=int(result['valid_1_30s_segments']):
        raise ValueError('Valid artifact count differs')
    return usable


def check_space(rows,base,pending_output):
    missing=0
    for r in rows:
        for kind,path in paths_for(r,base).items():
            if path.exists():
                if not path.is_file() or path.stat().st_size!=int(r[kind+'_size_bytes']):
                    raise ValueError('Existing source size mismatch; will not overwrite '+str(path))
            else:
                missing+=int(r[kind+'_size_bytes'])
    drives={Path(ROOT.anchor)}
    for root in [base,pending_output]:
        drives.add(Path(root.anchor))
    required=missing+output_bound(rows)+RESERVE
    free={str(d):shutil.disk_usage(d).free for d in drives}
    if any(value<required for value in free.values()):
        raise ValueError(f'Insufficient space: need {required/GB:.3f} GB including 100GB reserve; free={free}')
    return dict(download_missing_bytes=missing,output_budget_bytes=output_bound(rows),reserve_bytes=RESERVE,free_bytes=free)


class LimitedDownloadWriter:
    """Bound every download write by manifest bytes and live reserve."""
    def __init__(self,stream,expected,drive):
        self.stream=stream;self.expected=expected;self.drive=drive

    def write(self,data):
        if self.stream.tell()+len(data)>self.expected:
            raise ValueError('Download exceeds manifest size; partial retained')
        if shutil.disk_usage(self.drive).free-len(data)<RESERVE or shutil.disk_usage(Path(ROOT.anchor)).free-len(data)<RESERVE:
            raise ValueError('Download would breach 100GB reserve; partial retained')
        return self.stream.write(data)

    def __getattr__(self,name):
        return getattr(self.stream,name)


def download_gdown(row,kind,path):
    import gdown
    if path.exists():
        if path.stat().st_size!=int(row[kind+'_size_bytes']) or path.stat().st_size==0:
            raise ValueError('Existing source invalid; no overwrite')
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    partial=path.with_name(path.name+'.partial_'+uuid.uuid4().hex)
    with partial.open('xb') as stream:
        guarded=LimitedDownloadWriter(stream,int(row[kind+'_size_bytes']),Path(path.anchor))
        obtained=gdown.download(id=row[kind+'_drive_id'],output=guarded,quiet=False,use_cookies=False,resume=False)
    if not obtained or partial.stat().st_size!=int(row[kind+'_size_bytes']) or partial.stat().st_size==0:
        raise ValueError('Partial download retained: '+str(partial))
    if shutil.disk_usage(Path(ROOT.anchor)).free<RESERVE:
        raise ValueError('Reserve breached; partial retained')
    if path.exists():
        raise ValueError('Destination appeared; no overwrite')
    # Windows os.rename fails if the destination exists; do not use replace.
    os.rename(partial,path)


class DriveDownloadError(RuntimeError):
    pass


def authorize_drive(args):
    """Explicit user-run OAuth only; never downloads or preprocesses."""
    if not args.drive_client_secrets or not args.drive_token:
        raise ValueError('--authorize-drive requires --drive-client-secrets and --drive-token')
    if args.drive_token.exists():
        raise ValueError('Existing token preserved; choose a NEW token path')
    from google_auth_oauthlib.flow import InstalledAppFlow
    flow=InstalledAppFlow.from_client_secrets_file(str(args.drive_client_secrets),[DRIVE_SCOPE])
    credentials=flow.run_local_server(port=0)
    new_json(args.drive_token,json.loads(credentials.to_json()))
    print('OAuth authorized; token saved. No file downloads or preprocessing.',flush=True)


def drive_session(token):
    if not token or not token.is_file():
        raise ValueError('Drive API requires user-authorized --drive-token; no fallback to public links')
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request,AuthorizedSession
    credentials=Credentials.from_authorized_user_file(str(token),scopes=[DRIVE_SCOPE])
    if not credentials.valid:
        credentials.refresh(Request())
    # Token refresh is in memory; existing token file is never overwritten.
    return AuthorizedSession(credentials)


def checked_response(response):
    if response.status_code==200:
        return
    try:
        error=response.json().get('error',{})
        reasons=[x.get('reason','unknown') for x in error.get('errors',[])]
    except (ValueError,AttributeError):
        reasons=['non_json_response']
    # Do not expose Authorization headers/tokens or full response bodies.
    reason=','.join(reasons) or 'unspecified'
    transient=(response.status_code==429 or response.status_code>=500 or
               (response.status_code==403 and any(x in ('rateLimitExceeded','userRateLimitExceeded','backendError') for x in reasons)))
    exc=DriveDownloadError(f'Drive API HTTP {response.status_code}: {reason}')
    exc.transient=transient
    raise exc


def download_drive_api(row,kind,path):
    import random
    import requests
    if path.exists():
        if not path.is_file() or path.stat().st_size!=int(row[kind+'_size_bytes']) or path.stat().st_size==0:
            raise ValueError('Existing source invalid; no overwrite')
        return
    if DRIVE_SESSION is None:
        raise ValueError('Authenticated Drive session missing')
    file_id=row[kind+'_drive_id']
    if not file_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in file_id):
        raise ValueError('Invalid exact Drive file ID')
    url='https://www.googleapis.com/drive/v3/files/'+file_id
    expected=int(row[kind+'_size_bytes'])
    path.parent.mkdir(parents=True,exist_ok=True)
    for attempt in range(4):
        try:
            with DRIVE_SESSION.get(url,params={'fields':'id,name,size,md5Checksum,capabilities/canDownload','supportsAllDrives':'true'},timeout=(20,120)) as response:
                checked_response(response);meta=response.json()
            if meta.get('id')!=file_id or meta.get('name')!=row[kind+'_filename'] or int(meta.get('size',-1))!=expected:
                raise ValueError('Drive metadata ID/name/size differs from pinned manifest')
            if not meta.get('capabilities',{}).get('canDownload',False):
                raise ValueError('Authenticated account cannot download this file')
            partial=path.with_name(path.name+'.partial_'+uuid.uuid4().hex)
            md5=hashlib.md5()
            with DRIVE_SESSION.get(url,params={'alt':'media','supportsAllDrives':'true'},stream=True,timeout=(20,120)) as response:
                checked_response(response)
                with partial.open('xb') as stream:
                    guarded=LimitedDownloadWriter(stream,expected,Path(path.anchor))
                    for chunk in response.iter_content(chunk_size=1024*1024):
                        if chunk:
                            guarded.write(chunk);md5.update(chunk)
                    stream.flush();os.fsync(stream.fileno())
            if partial.stat().st_size!=expected or expected==0:
                raise ValueError('Partial download retained: '+str(partial))
            if meta.get('md5Checksum') and md5.hexdigest()!=meta['md5Checksum']:
                raise ValueError('Drive MD5 mismatch; partial retained: '+str(partial))
            if shutil.disk_usage(Path(ROOT.anchor)).free<RESERVE:
                raise ValueError('Reserve breached; partial retained')
            if path.exists():raise ValueError('Destination appeared; no overwrite')
            os.rename(partial,path)
            return
        except (DriveDownloadError,requests.exceptions.RequestException) as exc:
            transient=isinstance(exc,requests.exceptions.RequestException) or getattr(exc,'transient',False)
            if not transient or attempt==3:
                raise DriveDownloadError('Authenticated download failed; prior partials retained: '+
                                         (str(exc) if isinstance(exc,DriveDownloadError) else type(exc).__name__)) from None
            delay=min(2**(attempt+1)+random.random(),30)
            print(f'Download transient failure; retry {attempt+1}/3 in {delay:.1f}s; partial retained',flush=True)
            time.sleep(delay)


def download(row,kind,path):
    if DOWNLOAD_BACKEND=='drive-api':
        return download_drive_api(row,kind,path)
    return download_gdown(row,kind,path)


def batch_run(args):
    if not args.batch_id or not args.batch_id.startswith('BATCH_'):
        raise ValueError('Choose one numeric planned batch; pilot/oversized groups cannot execute')
    allrows=read(args.plan);checked_rows(allrows)
    rows=[r for r in allrows if r['batch_id']==args.batch_id]
    if not rows or any(r['planning_status']!='PLANNED' for r in rows) or source_size(rows)>50*GB:
        raise ValueError('Unknown/nonplanned/oversized batch')
    base=(args.download_dir/args.batch_id).resolve();out=(args.output_dir/args.batch_id).resolve()
    protected=[ROOT/'data/pilot_15_inputs',ROOT/'data/smoke_test_inputs',ROOT/'results/preprocessing_pilot']
    if base==out or base.is_relative_to(out) or out.is_relative_to(base) or base in (ROOT,ROOT/'data') or out in (ROOT,ROOT/'results'):
        raise ValueError('Use separate dedicated batch input/output roots')
    if any(base.is_relative_to(p.resolve()) or out.is_relative_to(p.resolve()) for p in protected):
        raise ValueError('Pilot/source provenance must remain untouched')
    core,ast_hashes=core_module()
    if any(r['speaker_id'] in core.PILOT_SPEAKERS for r in rows):
        raise ValueError('Pilot already processed; repeat download/preprocessing refused')
    candidate=read(args.candidate);checked_rows(candidate)
    candidate_sha=digest(args.candidate)
    expected={r['recording_id']:r for r in candidate}
    for r in rows:
        if r['candidate_manifest_sha256']!=candidate_sha or any(r[k]!=expected[r['recording_id']][k] for k in expected[r['recording_id']]):
            raise ValueError('Plan/candidate provenance conflict')
    binding=dict(batch_id=args.batch_id,plan_sha256=digest(args.plan),core_sha256=CORE_SHA,
                 download_directory=str(base),output_directory=str(out),core_function_ast_sha256=ast_hashes)
    if out.exists():
        marker=out/'batch_binding.json'
        if not marker.exists() or json.loads(marker.read_text(encoding='utf-8'))!=binding:
            raise ValueError('Existing output does not belong to this exact batch/plan/core')
        if not args.resume and not args.dry_run:
            raise ValueError('Existing batch output: use --resume; nothing overwritten')
    if args.execute and not args.dry_run:
        marker=out/'batch_binding.json'
        if not marker.exists():
            check_space(rows,base,out)
            out.mkdir(parents=True,exist_ok=True)
            new_json(marker,binding)
        handle=marker.open('rb')
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except Exception:
            handle.close();raise ValueError('Another runner owns this batch')
        LOCK_HANDLES.append(handle)
    outcomes=[];usable=[];pending=[]
    for r in rows:
        checkpoints=sorted((out/'records'/r['recording_id']).glob('attempt_*/result.json')) if out.exists() else []
        previous=json.loads(checkpoints[-1].read_text(encoding='utf-8')) if checkpoints else None
        if previous and (previous['result']['final_status']!='ERROR' or not args.retry_errors):
            u=verify_artifacts(previous['result'],Path(previous['core_output']))
            if u!=previous['usable_artifacts']:
                raise ValueError('Completed artifact changed; refusing resume')
            outcomes.append(previous['result']);usable+=u
        else:
            pending.append(r)
    space=check_space(pending,base,out)
    report=dict(**binding,recording_count=len(rows),speaker_count=len({r['speaker_id'] for r in rows}),
                source_GB=source_size(rows)/GB,completed_skipped=len(outcomes),pending_recordings=len(pending),
                partial_download_count=len(list(base.rglob('*.partial_*'))) if base.exists() else 0,
                zero_byte_partial_count=sum(p.stat().st_size==0 for p in base.rglob('*.partial_*')) if base.exists() else 0,
                download_backend=args.download_backend,authentication_checked=False,
                **space,mode='DRY_RUN_NO_DOWNLOAD_NO_VAD_NO_CLEANUP')
    if args.dry_run:
        if args.dry_run_report:new_json(args.dry_run_report,report)
        print(json.dumps(report,ensure_ascii=False,indent=2));return
    if not args.execute:
        raise ValueError('Real run requires --execute; use --dry-run for preparation')
    global DOWNLOAD_BACKEND,DRIVE_SESSION
    DOWNLOAD_BACKEND=args.download_backend
    if DOWNLOAD_BACKEND=="drive-api":
        DRIVE_SESSION=drive_session(args.drive_token)
    else:
        import gdown
    from pydub import AudioSegment
    out.mkdir(parents=True,exist_ok=True)
    if not (out/'batch_binding.json').exists():new_json(out/'batch_binding.json',binding)
    run=uuid.uuid4().hex
    core.LOG.setLevel(logging.INFO);core.LOG.addHandler(logging.StreamHandler(sys.stdout))
    core.pair_inputs=pair_adapter
    for index,r in enumerate(pending,1):
        # Account again for accumulating retained output before each recording.
        rec=out/'records'/r['recording_id'];rec.mkdir(parents=True,exist_ok=True)
        attempts=list(rec.glob('attempt_*'));attempt=rec/f'attempt_{len(attempts)+1:04d}'
        attempt.mkdir(exist_ok=False);coreout=attempt/'core';coreout.mkdir()
        u=[];result=core.make_result(r)
        try:
            check_space(pending[index-1:],base,out)
            local=paths_for(r,base)
            for kind,path in local.items():
                if not path.is_relative_to(base):raise ValueError('Unsafe input path')
                download(r,kind,path)
            inputrow=dict(r,wav_path=str(local['wav']),json_path=str(local['json']))
            pair_adapter(inputrow,args.plan)
            print(f'[{index}/{len(pending)}] {r["speaker_id"]}: {r["recording_id"]}',flush=True)
            result=core.process_recording(inputrow,args.plan,coreout,AudioSegment,index)
            u=verify_artifacts(result,coreout)
        except Exception as exc:
            result['final_status']='ERROR';result['failure_reason']=type(exc).__name__+': '+str(exc)
            print('Recording ERROR:',result['failure_reason'],flush=True)
        result.update(download_backend=args.download_backend,batch_id=args.batch_id,source_manifest_sha256=candidate_sha,core_sha256=CORE_SHA,
                      batch_pairing_adapter='explicit physical filenames + canonical JSON File_id + Patient_info; no fuzzy matching')
        new_json(attempt/'result.json',dict(result=result,core_output=str(coreout),usable_artifacts=u))
        outcomes.append(result);usable+=u
    counts=Counter(r['final_status'] for r in outcomes)
    complete=len(outcomes)==len(rows) and counts['ERROR']==0
    resultfile=out/f'preprocessing_results_{run}.csv';new_csv(resultfile,outcomes)
    usablefile=out/f'usable_success_segments_{run}.csv'
    if usable:new_csv(usablefile,usable)
    if complete and not (out/'preprocessing_results.csv').exists():new_csv(out/'preprocessing_results.csv',outcomes)
    state=dict(batch_id=args.batch_id,status='COMPLETE' if complete else 'FAILED',cleanup_ready=complete,
               recording_count=len(outcomes),status_counts=dict(counts),results_csv=str(resultfile),
               results_sha256=digest(resultfile),SUCCESS_artifacts_verified=complete,
               usable_segment_count=len(usable),source_directory=str(base),core_sha256=CORE_SHA,
               source_cleanup_performed=False,cleanup_policy='Only explicit user cleanup after COMPLETE/result/artifact verification; runner never deletes sources')
    state['speaker_outcomes']={s:dict(Counter(r['final_status'] for r in outcomes if r['speaker_id']==s)) for s in sorted({r['speaker_id'] for r in outcomes})}
    state['disability_group_outcomes']={g:dict(Counter(r['final_status'] for r in outcomes if r['disability_group']==g)) for g in sorted({r['disability_group'] for r in outcomes})}
    new_json(out/f'batch_summary_{run}.json',state)
    if complete and not (out/'batch_summary.json').exists():new_json(out/'batch_summary.json',state)
    print(json.dumps(state,ensure_ascii=False,indent=2))


def self_test():
    """No network, WAV decoding/VAD, cleanup or output directory creation."""
    import io
    from unittest.mock import patch
    from types import SimpleNamespace
    core,evidence=core_module()
    fixture=[dict(speaker_id='SYNTHETIC_CHECK',wav_filename='a.wav',json_filename='a.json',
                  wav_size_bytes='10',json_size_bytes='10',duration_sec='1')]
    with patch.object(shutil,'disk_usage',return_value=SimpleNamespace(free=RESERVE+20+output_bound(fixture)-1)):
        try:check_space(fixture,ROOT/'data/__NONEXISTENT_SAFETY_CHECK__',ROOT/'results/__NONEXISTENT_SAFETY_CHECK__')
        except ValueError:pass
        else:raise AssertionError('Reserve guard did not refuse')
    with patch.object(shutil,'disk_usage',return_value=SimpleNamespace(free=RESERVE+100)):
        stream=io.BytesIO();writer=LimitedDownloadWriter(stream,3,ROOT)
        writer.write(b'abc')
        try:writer.write(b'd')
        except ValueError:pass
        else:raise AssertionError('Download byte limit did not refuse')
        if stream.getvalue()!=b'abc':raise AssertionError('Oversize bytes were written')
    with patch.object(shutil,'disk_usage',return_value=SimpleNamespace(free=RESERVE)):
        stream=io.BytesIO();writer=LimitedDownloadWriter(stream,3,ROOT)
        try:writer.write(b'a')
        except ValueError:pass
        else:raise AssertionError('Live reserve guard did not refuse')
        if stream.getvalue():raise AssertionError('Reserve-breaching bytes written')
    rs=read(ROOT/'results/full_expansion/candidate_exclude_all_code_overlap_resolved.csv')
    try:checked_rows([rs[0],rs[0]])
    except ValueError:pass
    else:raise AssertionError('Duplicate guard did not refuse')
    print(json.dumps(dict(status='PASS',checks=['core hash/AST integrity','reserve threshold refusal',
                     'download byte cap','live download reserve refusal','duplicate recording refusal'],
                     network_calls=0,preprocessing_calls=0,cleanup_calls=0),indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--download-backend',choices=['gdown','drive-api'],default='gdown')
    p.add_argument('--drive-token',type=Path,help='User-authorized OAuth token; never printed or overwritten')
    p.add_argument('--drive-client-secrets',type=Path,help='Desktop OAuth client JSON; only used with --authorize-drive')
    p.add_argument('--authorize-drive',action='store_true',help='User-operated browser OAuth only; no batch execution')
    p.add_argument('--make-plan',action='store_true')
    p.add_argument('--candidate',type=Path,default=ROOT/'results/full_expansion/candidate_exclude_all_code_overlap_resolved.csv')
    p.add_argument('--plan',type=Path,default=ROOT/'results/full_expansion/candidate2_batch_plan.csv')
    p.add_argument('--batch-id')
    p.add_argument('--download-dir',type=Path,default=ROOT/'data/candidate2_batch_inputs')
    p.add_argument('--output-dir',type=Path,default=ROOT/'results/candidate2_batch_preprocessing')
    p.add_argument('--max-source-gb',type=float,default=40,help='<=50 GB; default 40 because conservative two-pass export peak at 50GB exceeds the stated 279GB free minus 100GB reserve')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--dry-run-report',type=Path,help='Optional NEW metadata report; never creates batch input/output directories')
    p.add_argument('--self-test',action='store_true',help='Pure safety checks; no download or VAD')
    p.add_argument('--execute',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--retry-errors',action='store_true')
    args=p.parse_args()
    if not 0<args.max_source_gb<=50:p.error('Source limit must be <=50 decimal GB')
    if args.make_plan and (args.execute or args.batch_id or args.dry_run):p.error('Plan generation is separate from batch execution/dry-run')
    if args.dry_run_report and not args.dry_run:p.error('--dry-run-report requires --dry-run')
    if args.self_test and (args.execute or args.make_plan or args.batch_id):p.error('Self-test cannot execute a batch')
    if args.authorize_drive and (args.execute or args.batch_id or args.make_plan or args.self_test or args.dry_run):p.error('OAuth authorization must run separately from any batch')
    try:
        if args.authorize_drive:authorize_drive(args)
        elif args.self_test:self_test()
        elif args.make_plan:make_plan(args)
        else:batch_run(args)
        return 0
    except Exception as exc:
        print('REFUSED/FAILED:',type(exc).__name__,str(exc),file=sys.stderr,flush=True)
        if args.execute and not args.dry_run and LOCK_HANDLES:
            new_json(args.output_dir/args.batch_id/f'batch_summary_failure_{uuid.uuid4().hex}.json',
                     dict(batch_id=args.batch_id,status='FAILED',cleanup_ready=False,
                          failure_reason=type(exc).__name__+': '+str(exc),source_cleanup_performed=False))
        return 2
    finally:
        for handle in LOCK_HANDLES:
            handle.close()
        LOCK_HANDLES.clear()


if __name__=='__main__':
    raise SystemExit(main())
