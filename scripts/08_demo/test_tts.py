"""Direct Korean Piper synthesis and real loopback HTTP /tts verification."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request
import urllib.error
import wave

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import soundfile as sf
from src.api.config import load_settings
from src.tts.piper_tts import PiperTTS

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def properties(path):
    audio,sr=sf.read(path,dtype='float32')
    info=sf.info(path)
    with wave.open(str(path),'rb') as wav:
        assert wav.getsampwidth()==2 and wav.getnchannels()==1
    assert sr==22050 and len(audio)>0 and np.isfinite(audio).all()
    peak=float(np.max(np.abs(audio)))
    rms=float(np.sqrt(np.mean(audio**2)))
    assert peak>0.001 and rms>0.0001, 'Silent or near-silent synthesis'
    return dict(path=str(path),format=info.format,subtype=info.subtype,sample_rate=sr,channels=info.channels,frames=info.frames,seconds=info.duration,bytes=path.stat().st_size,peak=peak,rms=rms,nonzero_fraction=float(np.mean(audio!=0)),sha256=sha(path),decodable=True,silent=False)

def main():
    settings=load_settings()
    tts=PiperTTS(settings)
    assert tts.ready,(tts.executable,settings.piper_model)
    dest=ROOT/'outputs/e1_demo/tts_verification_20261110'
    dest.mkdir(parents=True,exist_ok=True)
    result_path=dest/'results.json'
    if result_path.exists():
        raise FileExistsError('Refusing to overwrite previous verification')
    text='안녕하세요. 이음 음성 안내입니다. 오늘 저녁에 약속 있어.'
    start=time.perf_counter()
    direct=tts.synthesize(text)
    direct_seconds=time.perf_counter()-start
    direct_saved=dest/'direct_korean.wav'
    if direct_saved.exists():
        raise FileExistsError(direct_saved)
    direct.replace(direct_saved)
    direct_info=properties(direct_saved)
    print('DIRECT',json.dumps(direct_info),flush=True)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        port=sock.getsockname()[1]
    url=f'http://127.0.0.1:{port}'
    env=dict(os.environ,HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
    log_path=dest/'server.log'
    with log_path.open('w',encoding='utf-8') as log:
        process=subprocess.Popen([sys.executable,'-m','uvicorn','src.api.main:app','--host','127.0.0.1','--port',str(port),'--workers','1'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        try:
            deadline=time.monotonic()+90
            while True:
                if process.poll() is not None:
                    raise RuntimeError('Server startup failed; inspect '+str(log_path))
                try:
                    with urllib.request.urlopen(url+'/health',timeout=2) as response:
                        health=json.load(response)
                    break
                except (urllib.error.URLError,TimeoutError):
                    if time.monotonic()>deadline:
                        raise TimeoutError('Server did not become ready')
                    time.sleep(0.25)
            assert health['tts_ready'] is True and health['general_model_loaded'] is True
            headers={'Content-Type':'application/json; charset=utf-8'}
            request=urllib.request.Request(url+'/tts',data=json.dumps({'text':text},ensure_ascii=False).encode('utf-8'),headers=headers,method='POST')
            previous=set(settings.tts_output.glob('*.wav'))
            start=time.perf_counter()
            with urllib.request.urlopen(request,timeout=70) as response:
                status=response.status
                content_type=response.headers.get('Content-Type')
                disposition=response.headers.get('Content-Disposition')
                content=response.read()
            api_seconds=time.perf_counter()-start
            assert status==200 and content_type.startswith('audio/wav') and 'attachment' in disposition
            saved=dest/'api_korean.wav'
            saved.write_bytes(content)
            api_info=properties(saved)
            # Streaming completion triggers server-side temporary-file removal.
            for _ in range(40):
                if set(settings.tts_output.glob('*.wav'))==previous:
                    break
                time.sleep(0.1)
            assert set(settings.tts_output.glob('*.wav'))==previous
            bad=urllib.request.Request(url+'/tts',data=b'{"text":" "}',headers=headers,method='POST')
            try:
                urllib.request.urlopen(bad,timeout=10)
                raise AssertionError('Empty text accepted')
            except urllib.error.HTTPError as exc:
                assert exc.code==400
            with urllib.request.urlopen(url+'/openapi.json',timeout=5) as response:
                specification=json.load(response)
            assert 'application/json' in specification['paths']['/tts']['post']['requestBody']['content']
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    protected=json.loads((dest/'protected_before.json').read_text(encoding='utf-8'))
    assert all(sha(ROOT/p)==h for p,h in protected['hashes'].items())
    packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
    assert packages==protected['asr_packages'], 'ASR environment package list changed'
    freeze=subprocess.check_output([str(ROOT/'.venv_tts/Scripts/python.exe'),'-m','pip','freeze'],text=True)
    (dest/'tts_packages.txt').write_text(freeze,encoding='utf-8')
    result=dict(text=text,executable=tts.executable,model=str(settings.piper_model),config=str(settings.piper_config),direct=direct_info,direct_seconds=direct_seconds,api=api_info,api_seconds=api_seconds,http_status=status,content_type=content_type,content_disposition=disposition,health=health,real_loopback_http=True,temporary_file_cleanup=True,empty_text_status=400,protected_files_unchanged=len(protected['hashes']),asr_packages_unchanged=True,listening_verified=False,raspberry_pi_tested=False,voice_provenance=json.loads((ROOT/'piper/voice_provenance.json').read_text(encoding='utf-8')))
    result_path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    main()
