"""Durable workflow: external ZIP preparation -> local smoke -> full General Test."""
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from general_test import DEFAULT_ROOT
from prepare_local_dataset import prepare,write_json
from prepare_inputs import prepare as select

BASE=Path(__file__).resolve().parents[2]
SCRIPTS=Path(__file__).parent
RESULTS=BASE/'results/whisper_baseline'
STATUS=RESULTS/'external_evaluation_status.json'
ROOT=Path(r'D:\ieum_whisper_baseline')
REVISION='973afd24965f72e36ca33b3055d56a652f456b4d'


def run():
    os.environ['PYTHONIOENCODING']='utf-8'
    os.environ['HF_HUB_DISABLE_XET']='1'
    root=prepare(DEFAULT_ROOT,ROOT,STATUS)
    common=[sys.executable,'-B',str(SCRIPTS/'run_baseline.py'),'--data-root',str(root),'--revision',REVISION]
    write_json(STATUS,dict(stage='local_smoke',data_root=str(root)))
    subprocess.run(common+['--selection',str(RESULTS/'smoke_selection_20261027.json'),
                           '--output',str(RESULTS/'smoke_external_20261027')],check=True)
    full=RESULTS/'general_test_selection_external.json'
    selection=select(root,True)
    if full.exists():
        if json.loads(full.read_text(encoding='utf-8'))!=selection:
            raise ValueError('Existing full selection differs')
    else:
        with full.open('x',encoding='utf-8') as stream:
            json.dump(selection,stream,ensure_ascii=False)
    write_json(STATUS,dict(stage='full_evaluation',utterances=len(selection['rows']),data_root=str(root),
                           output=str(RESULTS/'general_test_external_20261027')))
    subprocess.run(common+['--selection',str(full),'--output',str(RESULTS/'general_test_external_20261027'),
                           '--allow-full-test','--export-every','500','--retry-failures'],check=True)
    write_json(STATUS,dict(stage='complete',output=str(RESULTS/'general_test_external_20261027')))


if __name__=='__main__':
    try:
        run()
    except BaseException as exc:
        write_json(STATUS,dict(stage='failed',error=repr(exc),traceback=traceback.format_exc()))
        raise
