"""Real checkpoints + Test WAV API smoke, no training or client-code imports."""
import os
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import csv
import hashlib
import io
import json
import sys
from dataclasses import replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient
from src.api.config import load_settings
from src.api.main import create_app
from src.api.model_loader import ModelLoader
from src.api.whisper_inference import ASRInference
from src.personalization.evaluator import normalize_text

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    dest=ROOT/'outputs/e1_demo/verification'
    dest.mkdir(parents=True,exist_ok=True)
    cfg=load_settings()
    protected=list((ROOT/'src/personalization').glob('*.py'))+list((ROOT/'scripts/07_personalization').rglob('*'))
    protected=[p for p in protected if p.is_file() and '__pycache__' not in p.parts]
    protected += [cfg.general_model/'model.safetensors',cfg.personalized_model/'model.safetensors',ROOT/'data/splits/personalization_split.csv']
    for folder in ('E1','general_baseline'):
        protected += list((ROOT/f'outputs/personalization/{folder}/KSM_M_45').glob('test_*'))
    before={str(p):sha(p) for p in protected}
    manifests=list(csv.DictReader((ROOT/'data/splits/personalization_split.csv').open(encoding='utf-8-sig')))
    rows=[r for r in manifests if r['speaker_id']=='KSM_M_45' and r['personal_split']=='personal_test' and r['audio_member'].endswith(('003250.wav','003254.wav'))]
    assert len(rows)==2
    historical={}
    for model,folder in [('general','general_baseline'),('personalized','E1')]:
        historical[model]={Path(r['audio_path']).name:r['hypothesis'] for r in map(json.loads,(ROOT/f'outputs/personalization/{folder}/KSM_M_45/test_predictions.jsonl').read_text(encoding='utf-8').splitlines())}
    checks=[]
    with TestClient(create_app(cfg)) as client:
        app=client.app
        response=client.get('/health')
        assert response.status_code==200 and response.json()['general_model_loaded']
        health=response.json()
        for r in rows:
            path=ROOT.parent/'ieum_datasets'/r['batch']/r['audio_member']
            content=path.read_bytes()
            direct=app.state.inference.transcribe_compare(content,'KSM_M_45',path.name)
            response=client.post('/transcribe/compare',data={'speaker_id':'KSM_M_45'},files={'audio':(path.name,content,'audio/wav')})
            assert response.status_code==200,response.text
            assert direct==response.json()
            assert set(direct)=={'speaker_id','filename','duration_seconds','general_text','personalized_text'}
            for model in ('general','personalized'):
                assert normalize_text(direct[model+'_text'])==historical[model][path.name],(path.name,model,direct)
            checks.append(dict(id=path.stem,reference=r['text'],**direct,direct_equals_api=True,saved_predictions_match=True))
        for route,expected in [('/transcribe',direct['general_text']),('/transcribe/personalized',direct['personalized_text'])]:
            response=client.post(route,data={'speaker_id':'KSM_M_45'},files={'audio':(path.name,content,'audio/wav')})
            assert response.status_code==200 and response.json()['text']==expected
        errors=[]
        for description,kwargs,status in [
            ('missing file',{},422),
            ('empty WAV',dict(files={'audio':('empty.wav',b'','audio/wav')},data={'speaker_id':'KSM_M_45'}),400),
            ('corrupt WAV',dict(files={'audio':('bad.wav',b'not audio','audio/wav')},data={'speaker_id':'KSM_M_45'}),400),
            ('unsupported speaker',dict(files={'audio':(path.name,content,'audio/wav')},data={'speaker_id':'OTHER'}),400),
            ('wrong filename',dict(files={'audio':('x.txt',content,'audio/wav')},data={'speaker_id':'KSM_M_45'}),400),
        ]:
            response=client.post('/transcribe/compare',**kwargs)
            assert response.status_code==status,(description,response.text)
            errors.append(dict(case=description,status=status))
        response=client.post('/tts',json={'text':' '});assert response.status_code==400
        if not health['tts_ready']:
            assert client.post('/tts',json={'text':'안녕하세요'}).status_code==503
        # Validate decoding independently of model output for varied channels/rates.
        for sample_rate,channels in [(44100,2),(16000,1)]:
            buffer=io.BytesIO();sf.write(buffer,np.zeros((sample_rate,channels),dtype=np.float32),sample_rate,format='WAV',subtype='PCM_16')
            decoded=app.state.inference.decode_audio(buffer.getvalue());assert len(decoded)==16000 and decoded.ndim==1
        buffer=io.BytesIO();sf.write(buffer,np.zeros(480001,dtype=np.float32),16000,format='WAV')
        response=client.post('/transcribe',files={'audio':('long.wav',buffer.getvalue(),'audio/wav')});assert response.status_code==400
        limit=app.state.settings.max_upload_bytes
        app.state.settings=replace(cfg,max_upload_bytes=8)
        assert client.post('/transcribe',files={'audio':('big.wav',content,'audio/wav')}).status_code==413
        app.state.settings=cfg
    try:
        with TestClient(create_app(replace(cfg,general_model=ROOT/'missing_demo_model'))):
            pass
    except FileNotFoundError:
        missing_model=True
    else:
        raise AssertionError('Missing model did not stop startup')
    assert all(sha(Path(p))==h for p,h in before.items())
    import importlib.metadata as metadata
    result=dict(comparisons=checks,health=health,input_errors=errors,over30_rejected=True,oversize_rejected=True,stereo_resample_passed=True,missing_model_fails_startup=missing_model,protected_files_unchanged=len(before),hardware_tested=False,piper_synthesis_tested=False,packages={p:metadata.version(p) for p in ['torch','transformers','fastapi','uvicorn','python-multipart','httpx','librosa','soundfile']})
    (dest/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
