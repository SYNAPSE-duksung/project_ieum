"""Record final download state and static checks; no WAV decoding or VAD."""
import ast
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT=Path(r'C:\ieum')
SCRIPTS=ROOT/'scripts/03_preprocessing_pilot'
RESULT=ROOT/'results/preprocessing_pilot'

def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    manifest=read_csv(RESULT/'pilot_15_manifest.csv')
    attempts=[]
    for file in sorted((RESULT/'pilot15_download_batches').glob('*.csv')):
        for row in read_csv(file):
            if row['status']!='DOWNLOADED':
                attempts.append(dict(filename=row['filename'],source_report=str(file),failure_reason=row['failure_reason'],resolved=Path(row['local_path']).is_file()))
    for file in sorted(RESULT.glob('pilot15_public_download_*.csv')):
        for row in read_csv(file):
            if row['status']!='DOWNLOADED':
                target=ROOT/'data/pilot_15_inputs'/row['speaker_id']/row['filename']
                attempts.append(dict(filename=row['filename'],source_report=str(file),failure_reason=row['failure_reason'],resolved=target.is_file()))
    live=json.loads((SCRIPTS/'pilot15_drive_verified_files.json').read_text(encoding='utf-8'))
    for file in live:
        if file['mime']=='audio/wav' and file['size']>268435456:
            matching=next(r for r in manifest if r['wav_drive_id']==file['id'])
            attempts.append(dict(filename=file['name'],source_report='Drive connector 256 MiB limit',failure_reason='Connector raw fetch exceeds 268435456-byte limit; public Drive download used',resolved=Path(matching['local_wav_path']).is_file()))
    report=RESULT/'pilot15_download_attempt_failures.csv'
    with report.open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['filename','source_report','failure_reason','resolved']);writer.writeheader();writer.writerows(attempts)
    runner=SCRIPTS/'05_run_preprocessing_pilot15.py'
    source=runner.read_text(encoding='utf-8-sig');compile(source,str(runner),'exec')
    spec=importlib.util.spec_from_file_location('pilot15_final_static',runner)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    normalized=module.load_manifest(RESULT/'pilot_15_manifest.csv')
    valid=[];failures=[];total=0;wav_count=json_count=0
    for row in normalized:
        try:
            wav,label,_=module.pair_inputs(row,RESULT/'pilot_15_manifest.csv')
            for kind,path in [('wav',wav),('json',label)]:
                if path.stat().st_size!=int(row[kind+'_size_bytes']):raise ValueError('Size mismatch '+kind)
            valid.append(row['recording_id']);total+=wav.stat().st_size+label.stat().st_size
            wav_count+=1;json_count+=1
        except Exception as exc:failures.append(dict(recording_id=row['recording_id'],reason=str(exc)))
    report_data=dict(speakers=len({r['speaker_id'] for r in manifest}),recordings=len(manifest),drive_wav=139,drive_json=139,
        wav_count=wav_count,json_count=json_count,valid_pairs=len(valid),total_input_bytes=total,
        reused_smoke_test_bytes=167142996,failures=failures,syntax=True,module_import=True,
        preprocessing_executed=False,source_sha256=hashlib.sha256((SCRIPTS/'04_run_preprocessing_smoke_test.py').read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),gdown_reference='https://github.com/wkentaro/gdown')
    with (RESULT/'pilot15_preparation_report.json').open('x',encoding='utf-8') as f:json.dump(report_data,f,ensure_ascii=False,indent=2)
    print(json.dumps(report_data,ensure_ascii=False),flush=True)
    return 2 if failures else 0

if __name__=='__main__':raise SystemExit(main())
