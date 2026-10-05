"""Back up validated Candidate 2 usable data to app-owned Google Drive folders.
Default: local-only dry-run. --execute uploads; --authorize is separate OAuth.
Never deletes local/remote data or overwrites existing remote file contents.
Restart resume skips MD5/size-verified files; interrupted file restarts with same
pre-generated Drive ID. Local immutable checkpoints prevent duplicate creates.
"""
import argparse,csv,hashlib,importlib.util,json,mimetypes,os,re,sys,time,uuid
from pathlib import Path

ROOT=Path('C:/ieum')
SCOPE='https://www.googleapis.com/auth/drive.file'
APP='ieum_candidate2_backup_v1'
FOLDER='application/vnd.google-apps.folder'
FIELDS='id,name,mimeType,size,md5Checksum,parents,appProperties,trashed'

def hashes(path):
    md5=hashlib.md5();sha=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):md5.update(chunk);sha.update(chunk)
    return md5.hexdigest(),sha.hexdigest()

def new_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())

def inventory(args):
    folder=(args.local_root/args.batch_id).resolve()
    summary=json.loads((folder/'validation_summary.json').read_text(encoding='utf-8-sig'))
    if summary.get('status')!='COMPLETE' or summary.get('validation_success') is not True or summary.get('batch_id')!=args.batch_id:raise ValueError('Local COMPLETE/validation_success/batch_id validation failed')
    extractor=ROOT/'scripts/06_batch_preprocessing/11_extract_candidate2_usable.py'
    spec=importlib.util.spec_from_file_location('usable_validator',extractor)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with (folder/'metadata.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    if not rows:raise ValueError('Empty metadata')
    expected=set();wavs=set();texts=set()
    for row in rows:
        wav=Path(row['wav_path']).resolve();txt=Path(row['transcript_path']).resolve()
        if wav.parent!=folder/'audio' or txt.parent!=folder/'transcripts' or wav.stem!=txt.stem:raise ValueError('Metadata path/pair outside final dataset')
        if wav in wavs or txt in texts:raise ValueError('Duplicate metadata WAV/TXT')
        module.validate_pair(row,wav,txt);wavs.add(wav);texts.add(txt);expected.update([wav,txt])
    if set((folder/'audio').iterdir())!=wavs or set((folder/'transcripts').iterdir())!=texts:raise ValueError('Local directory/metadata counts or extra entries differ')
    if {p.name for p in folder.iterdir()}!={'audio','transcripts','metadata.csv','validation_summary.json'}:raise ValueError('Unexpected local dataset root entry')
    for key in ('csv_rows','output_csv_rows','output_wav_count','output_txt_count'):
        if summary.get(key)!=len(rows):raise ValueError('Local validation summary count mismatch: '+key)
    expected.update([folder/'metadata.csv',folder/'validation_summary.json'])
    files=[]
    for index,path in enumerate(sorted(expected),1):
        if path.is_symlink() or not path.is_file():raise ValueError('Symlink/non-file input refused')
        md5,sha=hashes(path)
        files.append(dict(relative_path=path.relative_to(folder).as_posix(),size=path.stat().st_size,md5=md5,sha256=sha))
        if index%2000==0:print(f'Hashed {index}/{len(expected)} files',flush=True)
    if summary.get('metadata_sha256')!=next(x['sha256'] for x in files if x['relative_path']=='metadata.csv'):raise ValueError('Metadata SHA mismatch against local summary')
    fingerprint=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return folder,files,fingerprint,len(rows)

def token_safe(path):
    if path.name.lower()=='drive_readonly_token.json':raise ValueError('Readonly download token cannot be used or modified')

def auth(args):
    token_safe(args.token)
    if args.token.exists():raise ValueError('Upload token exists; no overwrite. Use a new token path')
    if not args.client_secrets:raise ValueError('--authorize requires --client-secrets Desktop OAuth JSON')
    from google_auth_oauthlib.flow import InstalledAppFlow
    creds=InstalledAppFlow.from_client_secrets_file(str(args.client_secrets),[SCOPE]).run_local_server(port=0)
    new_json(args.token,json.loads(creds.to_json()))
    print('Upload-only app-file OAuth token created; no uploads.',flush=True)

def service(args):
    token_safe(args.token)
    obj=json.loads(args.token.read_text(encoding='utf-8'))
    if set(obj.get('scopes',[]))!={SCOPE}:raise ValueError('Upload token must have exactly drive.file scope')
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    creds=Credentials.from_authorized_user_file(str(args.token),scopes=[SCOPE])
    if not creds.valid:creds.refresh(Request())
    # Refresh in memory only: no existing token file is overwritten.
    return build('drive','v3',credentials=creds,cache_discovery=False)

def listing(api,parent):
    items=[];page=None
    while True:
        result=api.files().list(q=f"'{parent}' in parents and trashed=false",fields='nextPageToken,files('+FIELDS+')',pageSize=1000,pageToken=page).execute(num_retries=5)
        items+=result.get('files',[]);page=result.get('nextPageToken')
        if not page:return items

def get_file(api,fid):
    from googleapiclient.errors import HttpError
    try:return api.files().get(fileId=fid,fields=FIELDS).execute(num_retries=5)
    except HttpError as exc:
        if exc.resp.status==404:return None
        raise

def stable_id(api,state,key):
    path=state/(hashlib.sha256(key.encode()).hexdigest()+'.json')
    if path.exists():
        obj=json.loads(path.read_text());
        if obj['key']!=key:raise ValueError('Checkpoint key mismatch')
        return obj['id']
    fid=api.files().generateIds(count=1,space='drive',type='files').execute(num_retries=5)['ids'][0]
    new_json(path,dict(key=key,id=fid));return fid

def folder(api,state,parent,name,key,props,allow_existing):
    matches=[x for x in listing(api,parent) if x['name']==name]
    if len(matches)>1:raise ValueError('Duplicate Drive folder name: '+name)
    if matches:
        x=matches[0]
        if not allow_existing:raise ValueError('Batch folder already exists: use --resume; no overwrite')
        if x['mimeType']!=FOLDER or x.get('appProperties')!=props:raise ValueError('Existing folder not owned by this backup/binding')
        return x['id']
    fid=stable_id(api,state,key)
    existing=get_file(api,fid)
    if existing:raise ValueError('Checkpoint folder ID exists but is absent from expected parent/name listing')
    api.files().create(body=dict(id=fid,name=name,mimeType=FOLDER,parents=[parent],appProperties=props),fields='id').execute(num_retries=5)
    return fid

def matches(remote,local):
    return remote.get('mimeType')!=FOLDER and int(remote.get('size',-1))==local['size'] and remote.get('md5Checksum')==local['md5'] and remote.get('trashed') is not True

def upload(args,local,files,fingerprint,report):
    from googleapiclient.http import MediaFileUpload
    api=service(args)
    account=api.about().get(fields='user(permissionId)').execute(num_retries=5)['user']['permissionId']
    state=args.state_root/args.batch_id
    binding=dict(batch_id=args.batch_id,local_directory=str(local),inventory_sha256=fingerprint,account_permission_id=account,client_id=json.loads(args.token.read_text())['client_id'],drive_root_id=args.drive_root_id)
    marker=state/'binding.json'
    if marker.exists():
        if json.loads(marker.read_text())!=binding:raise ValueError('Existing checkpoint binding differs; no resume')
        if not args.resume:raise ValueError('Existing backup state: use --resume')
    else:new_json(marker,binding)
    handle=marker.open('rb')
    try:
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        rootprops={'backup_app':APP}
        if args.drive_root_id:
            remote=get_file(api,args.drive_root_id)
            if not remote or remote['name']!='processed_candidate2' or remote['mimeType']!=FOLDER or remote.get('appProperties')!=rootprops:raise ValueError('Explicit Drive root must be an app-owned processed_candidate2 folder')
            rootid=remote['id']
        else:rootid=folder(api,state,'root','processed_candidate2','root',rootprops,True)
        batchprops={'backup_app':APP,'batch_id':args.batch_id,'inventory_sha256':fingerprint}
        batchid=folder(api,state,rootid,args.batch_id,'batch',batchprops,args.resume)
        children=listing(api,batchid)
        if any(x['name'] not in ('audio','transcripts','metadata.csv','validation_summary.json') for x in children):raise ValueError('Unexpected remote batch entry')
        parents={'':batchid}
        for name in ['audio','transcripts']:parents[name]=folder(api,state,batchid,name,name,batchprops,True)
        report.update(drive_root_id=rootid,drive_batch_id=batchid,uploaded_files=0,skipped_verified_files=0)
        maps={}
        for kind,parent in parents.items():
            entries=listing(api,parent);names={}
            for x in entries:
                if x['name'] in names:raise ValueError('Duplicate remote filename')
                names[x['name']]=x
            maps[kind]=names
        allowed={kind:{Path(x['relative_path']).name for x in files if str(Path(x['relative_path']).parent).replace('\\','/')== (kind or '.')} for kind in parents}
        for kind,names in maps.items():
            extra=set(names)-allowed[kind]-(set(['audio','transcripts']) if kind=='' else set())
            if extra:raise ValueError('Unexpected remote files: '+str(extra))
        for index,item in enumerate(files,1):
            relative=Path(item['relative_path']);kind='' if relative.parent==Path('.') else str(relative.parent)
            path=local/relative;remote=maps[kind].get(relative.name)
            if remote:
                if not matches(remote,item) or remote.get('appProperties',{}).get('inventory_sha256')!=fingerprint:raise ValueError('Remote file mismatch; no overwrite: '+item['relative_path'])
                report['skipped_verified_files']+=1;continue
            fid=stable_id(api,state,'file/'+item['relative_path'])
            remote=get_file(api,fid)
            if remote:
                if remote.get('name')!=relative.name or parents[kind] not in remote.get('parents',[]) or not matches(remote,item):raise ValueError('Checkpoint remote file mismatch')
                report['skipped_verified_files']+=1;continue
            if hashes(path)!=(item['md5'],item['sha256']):raise ValueError('Local file changed before upload')
            media=MediaFileUpload(str(path),mimetype=mimetypes.guess_type(str(path))[0] or 'application/octet-stream',chunksize=8*1024*1024,resumable=True)
            request=api.files().create(body=dict(id=fid,name=relative.name,parents=[parents[kind]],appProperties=batchprops),media_body=media,fields=FIELDS)
            response=None
            try:
                while response is None:_,response=request.next_chunk(num_retries=5)
            finally:
                # MediaFileUpload owns a local open stream, not a dataset mutation.
                media.stream().close()
            if not matches(response,item):raise ValueError('Uploaded file checksum/size mismatch')
            report['uploaded_files']+=1
            if index%500==0:print(f'Uploaded/verified {index}/{len(files)}',flush=True)
        count=0;groupcounts={}
        for kind,parent in parents.items():
            actual=[x for x in listing(api,parent) if x['mimeType']!=FOLDER]
            expected=[x for x in files if ('' if Path(x['relative_path']).parent==Path('.') else str(Path(x['relative_path']).parent))==kind]
            if len(actual)!=len(expected) or len({x['name'] for x in actual})!=len(actual):raise ValueError('Remote file count/duplicates mismatch')
            byname={x['name']:x for x in actual}
            for item in expected:
                x=byname.get(Path(item['relative_path']).name)
                if not x or not matches(x,item):raise ValueError('Final remote size/MD5 verification failed')
                if hashes(local/item['relative_path'])!=(item['md5'],item['sha256']):raise ValueError('Local file changed during backup')
            groupcounts[kind or 'batch_root']=len(actual);count+=len(actual)
        if count!=len(files):raise ValueError('Final local/Drive file counts differ')
        report.update(status='COMPLETE',backup_verified=True,drive_file_count=count,drive_group_counts=groupcounts,all_sizes_and_md5_verified=True)
    finally:handle.close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch-id')
    p.add_argument('--local-root',type=Path,default=ROOT/'data/processed_candidate2')
    p.add_argument('--token',type=Path,default=ROOT/'credentials/drive_upload_token.json')
    p.add_argument('--client-secrets',type=Path)
    p.add_argument('--authorize',action='store_true')
    p.add_argument('--drive-root-id',help='Optional ID of the processed_candidate2 folder previously created by this app')
    p.add_argument('--state-root',type=Path,default=ROOT/'results/candidate2_drive_backup/checkpoints')
    p.add_argument('--report',type=Path,help='New report file; default unique JSON under results/candidate2_drive_backup')
    p.add_argument('--resume',action='store_true')
    mode=p.add_mutually_exclusive_group();mode.add_argument('--dry-run',action='store_true');mode.add_argument('--execute',action='store_true')
    args=p.parse_args()
    if args.authorize:
        if args.batch_id or args.execute or args.dry_run or args.resume:p.error('Authorization must be separate from a batch')
        auth(args);return 0
    if not args.batch_id or not re.fullmatch(r'BATCH_(00[1-9]|01[0-9]|02[0-8])',args.batch_id):p.error('Choose BATCH_001..BATCH_028')
    token_safe(args.token)
    reportpath=args.report or ROOT/'results/candidate2_drive_backup'/f'{args.batch_id}_validation_{uuid.uuid4().hex}.json'
    if reportpath.exists():p.error('Existing report cannot be overwritten')
    if reportpath.resolve().is_relative_to((args.local_root/args.batch_id).resolve()):p.error('Report must be outside the backed-up dataset')
    report=dict(batch_id=args.batch_id,status='FAILED',backup_verified=False,local_files_deleted=False,mode='UPLOAD' if args.execute else 'DRY_RUN',network_calls_permitted=args.execute,oauth_scope=SCOPE)
    start=time.perf_counter()
    try:
        local,files,fingerprint,segments=inventory(args)
        report.update(local_validation_success=True,local_file_count=len(files),audio_files=segments,transcript_files=segments,root_files=2,total_bytes=sum(x['size'] for x in files),inventory_sha256=fingerprint)
        if args.execute:upload(args,local,files,fingerprint,report)
        else:report.update(status='DRY_RUN_VALIDATED',remote_verification_performed=False,upload_performed=False)
    except Exception as exc:
        # No request headers, OAuth token or resumable session URL logged.
        report.update(status='FAILED',backup_verified=False,error_type=type(exc).__name__,error=str(exc) if isinstance(exc,ValueError) else 'Upload/auth/API failed; checkpoints retained. Inspect account/network and retry with --resume.')
    report['elapsed_sec']=time.perf_counter()-start
    new_json(reportpath,report)
    print(json.dumps(report,ensure_ascii=False,indent=2));print('Report:',reportpath)
    return 0 if report['status'] in ('COMPLETE','DRY_RUN_VALIDATED') else 2

if __name__=='__main__':sys.exit(main())
