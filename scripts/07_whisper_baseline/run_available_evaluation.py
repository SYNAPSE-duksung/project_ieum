"""Evaluate complete external-drive ZIPs now; defer missing ZIPs in the full selection."""
from collections import Counter
import json
from pathlib import Path
import os
import subprocess
import sys
import traceback
import zipfile

from general_test import GeneralTest
from prepare_inputs import prepare
from prepare_local_dataset import write_json

BASE=Path(__file__).resolve().parents[2]
ROOT=Path(r'D:\ieum_whisper_baseline')
RESULTS=BASE/'results/whisper_baseline'
STATUS=RESULTS/'available_evaluation_status.json'
OUTPUT=RESULTS/'general_test_external_20261028'


def main():
    os.environ['PYTHONIOENCODING']='utf-8'
    os.environ['HF_HUB_DISABLE_XET']='1'
    dataset=GeneralTest(ROOT)
    by_path={dataset.archive_path(r):[] for r in dataset.test}
    for row in dataset.test:
        by_path[dataset.archive_path(row)].append(row)
    ready=[]
    available_rows=[]
    for path,rows in by_path.items():
        if not path.is_file():
            continue
        with zipfile.ZipFile(path) as archive:
            names=Counter(archive.namelist())
            for row in rows:
                for field in ('audio_member','transcript_member'):
                    if names[row[field]]!=1 or archive.getinfo(row[field]).file_size==0:
                        raise ValueError('Missing/empty/duplicate ZIP member: '+str(path)+':'+row[field])
        available_rows.extend(rows)
        ready.append(dict(batch=rows[0]['batch'],path=str(path),test_utterances=len(rows)))
    if not ready:
        raise ValueError('No complete local archives')
    selection=prepare(ROOT,True)
    selection_path=RESULTS/'general_test_selection_external.json'
    if selection_path.exists():
        if json.loads(selection_path.read_text(encoding='utf-8'))!=selection:
            raise ValueError('Existing selection differs')
    else:
        with selection_path.open('x',encoding='utf-8') as stream:
            json.dump(selection,stream,ensure_ascii=False)
    preflight=dict(stage='evaluating_available',archives=ready,available_test_utterances=len(available_rows),
                   available_pseudo_speakers=len({r['speaker_id'] for r in available_rows}),
                   total_test_utterances=len(dataset.test),deferred_utterances=len(dataset.test)-len(available_rows),
                   output=str(OUTPUT))
    write_json(STATUS,preflight)
    print('PREFLIGHT '+json.dumps(preflight,ensure_ascii=False),flush=True)
    result=subprocess.run([sys.executable,'-u','-B',str(Path(__file__).with_name('run_baseline.py')),
                           '--data-root',str(ROOT),'--selection',str(selection_path),'--output',str(OUTPUT),
                           '--revision','973afd24965f72e36ca33b3055d56a652f456b4d',
                           '--allow-full-test','--available-only','--export-every','500','--retry-failures'])
    preflight.update(stage='available_finished' if result.returncode==0 else 'available_finished_with_failures',
                     exit_code=result.returncode)
    write_json(STATUS,preflight)
    raise SystemExit(result.returncode)


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        write_json(STATUS,dict(stage='failed',error=repr(exc),traceback=traceback.format_exc()))
        raise
